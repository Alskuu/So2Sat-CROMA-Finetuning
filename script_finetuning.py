import argparse
import sys
import torch
from pathlib import Path
import logging
from torch.amp import GradScaler

from torch.utils.data import DataLoader, DistributedSampler

# Imports depuis vos modules personnalisés
# (On suppose que dataset.py, finetuning.py et sample.py sont dans le même dossier)
from dataset import So2SatDataset
from finetuning import (
    EMBED_DIM,
    FineTuneConfig,
    CROMAFineTuning,
    build_optimizer,
    load_croma_backbone,
    setup_hardware_and_distributed,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    save_confusion_matrix,
    PreemptionFlag,
)
from sampler import SequentialDistributedSampler


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.StreamHandler()  # Affiche aussi dans le terminal
                    ])
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def build_argparser():
    p = argparse.ArgumentParser()
    # Chemin vers les données So2Sat
    p.add_argument("--data-dir", type=str, required=True)
    # Dossier où le script va sauvegarder ses checkpoints
    p.add_argument("--checkpoint-dir", type=str,
                   required=True, default="checkpoint")
    # Chemin vers le fichier des poids préentraînés de notre modèle
    p.add_argument("--backbone-checkpoint", type=str, required=True)
    p.add_argument("--backbone-size", type=str, default="base")
    p.add_argument("--image-resolution", type=int, default=120)
    p.add_argument("--interpolation-mtd", type=str, default="nearest",
                   choices=["bilinear", "nearest"])
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--early-stop-epsilon", type=float, default=0.001)
    p.add_argument("--min-epochs", type=int, default=15)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--backbone-lr", type=float, default=5e-5)
    p.add_argument("--head-lr", type=float, default=2e-4)
    p.add_argument("--layer-decay", type=float, default=0.65)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=6800)
    p.add_argument("--freeze-n-layers", type=int, default=0)
    DTYPE_MAP = {"bfloat16": torch.bfloat16,
                 "float16": torch.float16, "float32": torch.float32}
    p.add_argument("--mixed-precision",
                   type=lambda s: DTYPE_MAP[s], default=torch.float16, choices=DTYPE_MAP.values())
    p.add_argument("--no-gradient-checkpointing",
                   action="store_false", dest="gradient_checkpointing")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    return p


def main(urban_only: bool):
    args = build_argparser().parse_args()

    cfg = FineTuneConfig(**vars(args))
    if cfg.head_init_path is None:
        cfg.head_init_path = f"{cfg.checkpoint_dir}/best_linear_probe_both_{cfg.interpolation_mtd}_{cfg.image_resolution}.pt"

    torch.manual_seed(cfg.seed)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    is_distributed, device, local_rank, rank, world_size = setup_hardware_and_distributed()

    logger.info(f"Taille du World Size ")

    backbone = load_croma_backbone(
        cfg.backbone_checkpoint, size=cfg.backbone_size, image_resolution=cfg.image_resolution)
    backbone = backbone.to(device)

    if urban_only:
        # Noms officiels des 10 classes LCZ urbaines de So2Sat
        LCZ_CLASSES = [
            "1. Compact high-rise", "2. Compact mid-rise", "3. Compact low-rise",
            "4. Open high-rise", "5. Open mid-rise", "6. Open low-rise",
            "7. Lightweight low-rise", "8. Large low-rise", "9. Sparsely built",
            "10. Heavy industry"
        ]
    else:
        # Noms officiels des 17 classes LCZ de So2Sat
        LCZ_CLASSES = [
            "1. Compact high-rise", "2. Compact mid-rise", "3. Compact low-rise",
            "4. Open high-rise", "5. Open mid-rise", "6. Open low-rise",
            "7. Lightweight low-rise", "8. Large low-rise", "9. Sparsely built",
            "10. Heavy industry", "A. Dense trees", "B. Scattered trees",
            "C. Bush, scrub", "D. Low plants", "E. Bare rock/paved",
            "F. Bare soil/sand", "G. Water"
        ]
    model = CROMAFineTuning(
        backbone=backbone,
        num_classes=len(LCZ_CLASSES),
        embed_dim=EMBED_DIM,
        freeze_n_layers=cfg.freeze_n_layers,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
    ).to(device)

    if Path(cfg.head_init_path).exists():
        model.load_pretrained_head(str(cfg.head_init_path))
    else:
        if rank == 0:
            logger.warning(
                f"Linear head weights not found at {cfg.head_init_path}. Initializing randomly.")

    # CRITICAL: Build optimizer BEFORE wrapping with DDP.
    # Layer-Wise Rate Decay (LLRD) needs direct access to model.backbone and model.head.
    # Once wrapped in DDP, these attributes move under 'model.module' and would throw an AttributeError.
    optimizer = build_optimizer(model, cfg)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)

    train_ds = So2SatDataset(
        h5_path=cfg.data_dir+"/training.h5", interpolation_mtd=cfg.interpolation_mtd, image_resolution=cfg.image_resolution, urban_only=urban_only)
    val_ds = So2SatDataset(
        h5_path=cfg.data_dir+"/validation.h5", interpolation_mtd=cfg.interpolation_mtd, image_resolution=cfg.image_resolution, urban_only=urban_only)

    train_sampler = DistributedSampler(train_ds) if is_distributed else None
    val_sampler = SequentialDistributedSampler(
        val_ds, world_size, rank) if is_distributed else None
    # Dans la commande Slurm ajouter une commande pour allouer suffisamment de ces ressources
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    scaler = GradScaler(
        enabled=(cfg.mixed_precision == torch.float16))
    killer = PreemptionFlag()

    start_epoch, global_step, best_metric, epochs_without_improvement = 0, 0, 0.0, 0
    last_ckpt = Path(cfg.checkpoint_dir) / "last.pt"
    if cfg.resume and last_ckpt.exists():
        start_epoch, global_step, best_metric, epochs_without_improvement = load_checkpoint(
            last_ckpt, model, optimizer, scaler)

    total_steps = cfg.epochs * len(train_loader)

    # Compte le nombre d'apparition de chaque label dans l'ordre officiel LCZ
    if not urban_only:
        train_counts = torch.tensor([
            5068, 24431, 31693, 8651, 16493, 35290, 3269, 39326,
            13584, 11954, 42902, 9514, 9165, 41377, 2392, 7898, 49359
        ], dtype=torch.float32)
    else:
        train_counts = torch.tensor([
            5068, 24431, 31693, 8651, 16493, 35290, 3269, 39326,
            13584, 11954
        ], dtype=torch.float32)

    for epoch in range(start_epoch+1, cfg.epochs):

        if is_distributed:
            train_sampler.set_epoch(epoch)

        global_step, train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg, global_step, total_steps, killer, rank, is_distributed, train_counts, urban_only
        )
        if rank == 0:
            logger.info(f"Epoch {epoch} | train_loss {train_loss:.4f}")
            # Synchronize signal status across all ranks
        is_preempted = killer.synchronize(is_distributed)

        if epoch % cfg.eval_every == 0 and not is_preempted:
            oa, aa, all_preds, all_targets = evaluate(model, val_loader, device,
                                                      world_size, len(val_ds), is_distributed)

            logger.info(f"Epoch {epoch} | val OA {oa:.4f} | val AA {aa:.4f}")
            if aa > best_metric + cfg.early_stop_epsilon:
                best_metric = aa
                epochs_without_improvement = 0
                if rank == 0:
                    save_confusion_matrix(all_targets, all_preds, LCZ_CLASSES, Path(
                        cfg.checkpoint_dir) / "figures/confusion_best.png")
                    save_checkpoint(Path(cfg.checkpoint_dir) / "best.pt",
                                    model, optimizer, scaler, epoch, global_step, best_metric, epochs_without_improvement)
            else:
                epochs_without_improvement += 1
                if rank == 0:
                    logger.info(
                        f"Pas d'amélioration <= {cfg.early_stop_epsilon} de l'AA \n ({epochs_without_improvement}/{cfg.patience} étapes sans améliorations)"
                    )
                    save_confusion_matrix(all_targets, all_preds, LCZ_CLASSES, Path(
                        cfg.checkpoint_dir) / f"figures/confusion_{epoch}.png")
        if rank == 0:
            save_checkpoint(last_ckpt, model, optimizer, scaler,
                            epoch, global_step, best_metric, epochs_without_improvement)

        if is_preempted:
            if rank == 0:
                logger.info(
                    "Exiting cleanly after SIGTERM (job should requeue with --requeue).")
            if is_distributed:
                # Very important because we use NCCL and elsewise we can have problem in the stopping of the script with the memory...
                torch.distributed.destroy_process_group()
                try:
                    train_ds.close()
                    val_ds.close()
                except Exception as e:
                    logger.warning(
                        f"Erreur lors de la fermeture des datasets : {e}")
            # L'erreur 99 permet de créer une erreur spécifique compréhensible par mon script slurm pour obliger le --requeue..
            sys.exit(99)

        if epoch + 1 >= cfg.min_epochs and epochs_without_improvement >= cfg.patience:
            if rank == 0:
                logger.info(
                    f"Arrêt anticipé : AA sans amélioration depuis {epochs_without_improvement} epochs"
                )
                logger.info(
                    f"évaluations (patience={cfg.patience}, min_epochs={cfg.min_epochs}).")
            break
        if is_distributed:
            torch.distributed.barrier()   # tout le monde attend que rank 0 ait fini d'écrire

    try:
        train_ds.close()
        val_ds.close()
    except Exception as e:
        logger.warning(f"Erreur lors de la fermeture des datasets : {e}")

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(True)
