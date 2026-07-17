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
    NUM_CLASSES,
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
    PreemptionFlag,
)
from sampler import SequentialDistributedSampler


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        # Sauvegarde dans les fichiers
                        logging.FileHandler("training_croma.log"),
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
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--early-stop-epsilon", type=float, default=0.001)
    p.add_argument("--min-epochs", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--layer-decay", type=float, default=0.75)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--freeze-n-layers", type=int, default=0)
    DTYPE_MAP = {"bfloat16": torch.bfloat16,
                 "float16": torch.float16, "float32": torch.float32}
    p.add_argument("--mixed-precision",
                   type=lambda s: DTYPE_MAP[s], default=torch.bfloat16, choices=DTYPE_MAP.values())
    p.add_argument("--no-gradient-checkpointing",
                   action="store_false", dest="gradient_checkpointing")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    return p


def main():
    args = build_argparser().parse_args()

    cfg = FineTuneConfig(**vars(args))
    if cfg.head_init_path is None:
        cfg.head_init_path = f"{cfg.checkpoint_dir}/best_linear_probe_both_{cfg.interpolation_mtd}_{cfg.image_resolution}.pt"

    torch.manual_seed(cfg.seed)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    is_distributed, device, local_rank, rank, world_size = setup_hardware_and_distributed()

    backbone = load_croma_backbone(
        cfg.backbone_checkpoint, size=cfg.backbone_size, image_resolution=cfg.image_resolution)
    backbone = backbone.to(device)

    model = CROMAFineTuning(
        backbone=backbone,
        num_classes=NUM_CLASSES,
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
            model, device_ids=[local_rank])

    train_ds = So2SatDataset(
        h5_path=cfg.data_dir+"/training.h5", interpolation_mtd=cfg.interpolation_mtd, image_resolution=cfg.image_resolution)
    val_ds = So2SatDataset(
        h5_path=cfg.data_dir+"/validation.h5", interpolation_mtd=cfg.interpolation_mtd, image_resolution=cfg.image_resolution)

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

    for epoch in range(start_epoch, cfg.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        global_step, train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg, global_step, total_steps, killer, rank, is_distributed
        )
        if rank == 0:
            logger.info(f"Epoch {epoch} | train_loss {train_loss:.4f}")
            # Synchronize signal status across all ranks
        is_preempted = killer.synchronize(is_distributed)

        if (epoch + 1) % cfg.eval_every == 0 and not is_preempted:
            oa, aa = evaluate(model, val_loader, device, world_size, is_distributed,
                              num_val_samples=len(val_ds))

            logger.info(f"Epoch {epoch} | val OA {oa:.4f} | val AA {aa:.4f}")
            if aa > best_metric + cfg.early_stop_epsilon:
                best_metric = aa
                epochs_without_improvement = 0
                if rank == 0:
                    save_checkpoint(Path(cfg.checkpoint_dir) / "best.pt",
                                    model, optimizer, scaler, epoch, global_step, best_metric, epochs_without_improvement)
            else:
                epochs_without_improvement += 1
                if rank == 0:
                    logger.info(
                        f"Pas d'amélioration <= {cfg.early_stop_epsilon} de l'AA \n ({epochs_without_improvement}/{cfg.patience} étapes sans améliorations)"
                    )
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
            sys.exit(0)

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
    main()
