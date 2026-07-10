"""
croma_finetuning.py

Full fine-tuning of the CROMA foundation model for LCZ classification on
So2Sat LCZ42 (Sentinel-1 + Sentinel-2), following on from the linear/MLP
probing phase.

Designed for two environments:
    - Local workstation -> mixed precision + gradient checkpointing
    - French supercomputer (Jean Zay-style SLURM cluster) -> multi-GPU via DDP,
      and checkpoint/resume on SIGTERM so a preempted/requeued job picks back up

=====================================================================
ASSUMPTIONS - I don't have your exact model-loading / dataset code from
the probing session, so three things below are marked TODO. Paste your
`load_croma(...)` function and `So2SatDataset` and I'll wire them in
exactly instead of guessing.
=====================================================================
    1. `load_croma_backbone(checkpoint_path, size)` loads the pretrained
       CROMA encoder the SAME way your feature-extraction script did.
    2. `So2SatDataset` yields a dict {"s1": Tensor[2,H,W], "s2": Tensor[12,H,W],
       "label": int} - s2 already zero-padded to CROMA's 12-band format,
       exactly as you built it for probing.
    3. CROMA's forward() returns a dict; `JOINT_EMBED_KEY` below must match
       the key you used when caching features for probing.py (same embed_dim
       you already used as input to your linear/MLP head).

"""

import os
import sys
import math
import time
import signal
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import multiprocessing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.optim import AdamW
from torch.cuda.amp import GradScaler


from use_croma import PretrainedCROMA
from dataset import So2SatDataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        # Sauvegarde dans les fichiers
                        logging.FileHandler("training_croma.log"),
                        logging.StreamHandler()  # Affiche aussi dans le terminal
                    ])
logger = logging.getLogger(__name__)

# c'est l'encoding que l'on utilise avec ce modèle
JOINT_EMBED_KEY = "joint_GAP"
# On a deux choix entre la concaténation du SAR et de l'optique (dim=1536) ou bien la jointure des deux (dim=768)
EMBED_DIM = 768
NUM_CLASSES = 17  # So2Sat LCZ42: 10 construites + 7 classes LCZ naturelles


@dataclass
class FineTuneConfig:
    data_dir: str
    checkpoint_dir: str
    backbone_checkpoint: str
    backbone_size: str = "base"
    image_resolution: int = 120
    # dir containing best_linear_probe_{mtd}_{res}.pt
    probe_checkpoint_dir: str = ""
    # "nearest" | "bilinear"
    interpolation_mtd: str = "nearest"
    modality: str = "both"
    # explicit override; bypasses probe_checkpoint_dir/interpolation_mtd
    head_init_path: Optional[str] = f"{checkpoint_dir}/best_linear_probe_{modality}_{interpolation_mtd}_{image_resolution}.pt"
    epochs: int = 30
    batch_size: int = 32
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    layer_decay: float = 0.75          # LLRD factor, <1 => earlier layers get smaller LR
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    # 0 = full fine-tuning; >0 for partial fine-tuning ablation
    freeze_n_layers: int = 0
    mixed_precision: torch.dtype = torch.bfloat16
    gradient_checkpointing: bool = True
    num_workers: int = 8
    seed: int = 42
    # contrôle si le scipt doit reprendre l'entraînement là où il
    # s'était arrêté en rechargeant un checkpoint existant
    resume: bool = True
    eval_every: int = 1
    log_every: int = 50

    # Early stopping (on the validation AA)
    patience: int = 5
    early_stop_epsilon: float = 0.001
    min_epochs: int = 8


# -------------------Modèle------------------------------------------------


class CROMAFineTuning(nn.Module):
    """
    Wraps a pretrained CROMA backbone with a classification head and
    (by default) unfreezes the entire backbone for full fine-tuning.

        Head is deliberately kept linear (LayerNorm -> Dropout -> Linear), not an
    MLP: the backbone itself has plenty of capacity to reshape representations
    during full fine-tuning, and a bigger randomly-initialized head produces
    noisier early gradients into the pretrained weights. See load_pretrained_head
    for initializing this Linear layer from your linear-probe checkpoint (LP-FT),
    which matters far more than the head's architecture.

    Set freeze_n_layers > 0 to freeze the first N transformer blocks instead
    (useful as a cheap ablation vs. full fine-tuning for your methodology
    section, without having to also implement LoRA).
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = EMBED_DIM,
        freeze_n_layers: int = 0,
        use_gradient_checkpointing: bool = True,
        joint_embed_key: str = JOINT_EMBED_KEY,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.joint_embed_key = joint_embed_key
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        if freeze_n_layers > 0:
            self._freeze_first_n_layers(freeze_n_layers)

    def _freeze_first_n_layers(self, n: int):
        """
        Freezes the first n transformer layers in each tower (s1_encoder,
        s2_encoder, cross_encoder). CROMA uses x-transformers style naming:
        `<tower>.transformer.layers.<idx>.<0|1>...` where <0|1> distinguishes
        attention (0) from feedforward (1) sub-blocks within the same layer idx.
        """
        frozen_layers = set()
        for name, p in self.backbone.named_parameters():
            parts = name.split(".")
            if "layers" in parts:
                idx_pos = parts.index("layers") + 1
                if idx_pos < len(parts) and parts[idx_pos].isdigit():
                    layer_idx = int(parts[idx_pos])
                    if layer_idx < n:
                        p.requires_grad = False
                        frozen_layers.add(
                            (parts[0], layer_idx))  # (tower, idx)
        logger.info(
            f"Froze layers < {n} across towers: {sorted(frozen_layers)}")

    def load_pretrained_head(self, path: str):
        """
        LP-FT initialization: loads the already-trained linear-probe weights
        into the head's final Linear layer, instead of starting from random
        init. Preserves pretrained backbone features much better than naive
        full fine-tuning (Kumar et al., ICLR 2022).
        """
        state = torch.load(path, map_location="cpu")
        missing = {"weight", "bias"} - state.keys()
        if missing:
            raise KeyError(
                f"Expected a plain nn.Linear state_dict with 'weight'/'bias', "
                f"got keys {list(state.keys())} (missing {missing}) from {path}"
            )
        linear = self.head[-1]
        if linear.weight.shape != state["weight"].shape:
            raise ValueError(
                f"Shape mismatch loading {path}: head Linear is "
                f"{tuple(linear.weight.shape)}, checkpoint is {tuple(state['weight'].shape)}. "
                f"(embed_dim/num_classes mismatch between probing and fine-tuning configs?)"
            )
        with torch.no_grad():
            linear.weight.copy_(state["weight"])
            linear.bias.copy_(state["bias"])
        logger.info(f"Initialized fine-tuning head from linear probe: {path}")

    def _backbone_forward(self, sar, optical):
        return self.backbone(SAR_images=sar, optical_images=optical)

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        # Helps to gain a lot of VRAM because you don't have to register all
        # activations for the backward that is coming : the checkpointing is calculating
        # TODO : Verify if it is useful with my 32 GBs of VRAM for the finetuning...
        if self.use_gradient_checkpointing and self.training:
            outputs = grad_checkpoint(
                self._backbone_forward, sar, optical, use_reentrant=False)
        else:
            outputs = self._backbone_forward(sar, optical)
        embed = outputs[self.joint_embed_key]
        return self.head(embed)


def load_croma_backbone(checkpoint_path: str, size: str, image_resolution: int = 120) -> nn.Module:
    return PretrainedCROMA(
        pretrained_path=checkpoint_path,
        size=size,
        modality="both",
        image_resolution=image_resolution
    )


# --------------------------------------------------------------------------- #
# Layer-wise LR decay optimizer - critical to avoid catastrophic forgetting
# --------------------------------------------------------------------------- #
def build_llrd_param_groups(model: CROMAFineTuning, cfg: FineTuneConfig):
    """
    Groups params so the head gets cfg.head_lr, and backbone layers get
    progressively smaller LR the earlier they are (layer_decay^depth) on their 
    RELATIVE depth within their respective tour (S1/S2 encoder).
    """
    param_groups = []

    # We distingusish the parameters in the head that need a head decay and the
    # ones that don't need one.
    head_decay = []
    head_no_decay = []
    for p in model.head.parameters():
        if not p.requires_grad:
            continue
        # Dimensions rule : 1D or less -> no decay
        # (biais or layer norm for example so don't influence the overfitting)
        if p.ndim <= 1:
            head_no_decay.append(p)
        else:
            head_decay.append(p)

    if head_decay:
        param_groups.append(
            {"params": head_decay, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay})
    if head_no_decay:
        param_groups.append(
            {"params": head_no_decay, "lr": cfg.head_lr, "weight_decay": 0.0}
        )

    # Sectorisation by tours and calculation from the max_depth per tour
    max_depths = {}
    for name, p in model.backbone.named_parameters():
        if not p.requires_grad:
            continue

        parts = name.split(".")
        if "layers" in parts:
            idx_pos = parts.index("layers") + 1
            if idx_pos < len(parts) and parts[idx_pos].isdigit():
                depth = int(parts[idx_pos])

                # We determinate the name from the tour ('s1_encoder', 's2_encoder' and so on...)
                # Generally it is the first element in the backbone arborescence
                tower_name = parts[0]
                max_depths[tower_name] = max(
                    max_depths.get(tower_name, 0), depth)

    # print(max_depths)

    # 2nd crossing to order the parameters in the buckets
    # The dictionnary key becomes : (tower_name, has_decay)
    # bucket backbone params by inferred depth
    depth_buckets = {}
    max_depth = 0
    for name, p in model.backbone.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split(".")
        tower_name = parts[0]
        if "layers" in parts:
            idx_pos = parts.index("layers") + 1
            depth = int(parts[idx_pos]) if idx_pos < len(
                parts) and parts[idx_pos].isdigit() else 0

        # norm_out : LR max (équivalent depth = max_depth + 1, decay=0)
        elif "norm_out" in parts:
            depth = None  # traité à part, cf. ci-dessous
        else:
            depth = 0  # linear_input, embeddings, etc.

        # Security decay : Determinate if the parameter need a weight decay
        # We verify its dimension (p.ndim <= 1) and key words in its name by safety
        if p.ndim <= 1 or "bias" in parts or "norm" in parts:
            has_decay = False
        else:
            has_decay = True

        bucket_key = (tower_name, depth, has_decay)
        depth_buckets.setdefault(bucket_key, []).append(p)

    # Handling the "norm_out" case (depth=None) per tour
    # By definition, norm_out contains biases and norm weights so no decay
    # We extract the final normalisation layers in which block
    towers = list(set([key[0] for key in depth_buckets.keys()]))
    for tower in towers:
        norm_out_params = []
        for has_decay in [True, False]:
            if (tower, None, has_decay) in depth_buckets:
                norm_out_params.extend(
                    depth_buckets.pop((tower, None, has_decay)))

        if norm_out_params:
            param_groups.append({
                "params": norm_out_params,
                "lr": cfg.backbone_lr,
                "weight_decay": 0.0,
            })
    # 4 : Final groups creation with the LLRD calculation per block
    for (tower_name, depth, has_decay), params in depth_buckets.items():
        tower_max_depth = max_depths.get(tower_name, 0)
        # deeper layers (closer to output) get less decay, i.e. higher LR
        scale = cfg.layer_decay ** (tower_max_depth - depth)
        lr = cfg.backbone_lr * scale
        weight_decay_value = cfg.weight_decay if has_decay else 0.0
        param_groups.append({
            "params": params,
            "lr": lr,
            "weight_decay": weight_decay_value,
        })

    return param_groups


def build_optimizer(model: CROMAFineTuning, cfg: FineTuneConfig) -> AdamW:
    groups = build_llrd_param_groups(model, cfg)
    optimizer = AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    # snapshot base LR per group, needed to apply lr_scale multiplicatively later
    for g in optimizer.param_groups:
        g["_base_lr"] = g["lr"]
    return optimizer


# It is a learning rate scheduler, helps to adapt ou training to the knowing of the data
# At first the pretrained model is not adapted to our data, that is why there is a warmup
# and after we continue...
def cosine_warmup_lr_lambda(step: int, warmup_steps: int, total_steps: int):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# Metrics: OA / AA, as decided during your probing phase
# --------------------------------------------------------------------------- #
def compute_oa_aa(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = NUM_CLASSES):
    correct = (preds == labels).sum().item()
    total = labels.numel()
    oa = correct / total if total > 0 else 0.0

    per_class_acc = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc.append((preds[mask] == c).float().mean().item())
    aa = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0
    return oa, aa

# ---------------------------------------------------------------------------- #
# Get access to the set up and do some loggings to show it
# ---------------------------------------------------------------------------- #


def setup_hardware_and_distributed():
    # Detect the distributed mode
    # Call the Linux environment variable showing if there are multiple GPUs
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    # Get the uniques ids
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if is_distributed:
        # Initialisation of Pytorch's communication's protocol (NCCL is the fastest on NVIDIA's GPUs)
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging about the Hardware informations (Just on the RANK 0 to avoid duplicates)
    if rank == 0:
        logger.info("=" * 60)
        logger.info(
            "       CONFIGURATION DU MATÉRIEL & ENVIRONNEMENT       ")
        logger.info("=" * 60)

        # Execution mode
        logger.info(
            f"Mode d'exécution    : {'DISTRIBUÉ (Multi-GPU)' if is_distributed else 'MONO-PROCESSUS'}")

        # CPU Informations (Number of global hearts available on the node)
        total_cpus = multiprocessing.cpu_count()
        logger.info(
            f"Cœurs CPU disponibles : {total_cpus} cœurs au total sur ce nœud")

        # Informations about the GPU
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Modèle de GPU        : {gpu_name}")
            logger.info(f"GPU visibles (local) : {gpu_count}")
        else:
            logger.warning(
                "ATTENTION : Aucun GPU détecté par PyTorch ! Le code tourne sur CPU.")

        # Distributed Slurm/ PyTorch variables details
        if is_distributed:
            logger.info("-" * 40)
            logger.info(
                f"Taille du Monde (WORLD_SIZE) : {world_size} GPU(s) au total")
            logger.info(
                f"Nom du nœud Slurm            : {os.environ.get('SLURM_NODENAME', 'Inconnu')}")
            logger.info(
                f"ID du Job Slurm              : {os.environ.get('SLURM_JOB_ID', 'Inconnu')}")
            logger.info("-" * 40)

        logger.info(f"Périphérique cible principal : {device}")
        logger.info("=" * 60 + "\n")

    return is_distributed, device, local_rank, rank

# --------------------------------------------------------------------------- #
# Graceful SLURM preemption handling
# signal is the way of communiccating for all Linux objects, with scripts and all of this...
# IMPORTANT ::: `#SBATCH --signal=B:SIGTERM@120
# needs to be written in the terminal to the sbatch script so SLURM sends
# SIGTERM ~120s before killing the job, giving time to checkpoint & exit
# cleanly (combine with `#SBATCH --requeue` to auto-resume).
# elsewise it would be a direct SIGKILL after the consecutive hours needed consumed
# for example...
# --------------------------------------------------------------------------- #


class PreemptionFlag:
    def __init__(self):
        self.should_stop = False
        # SIGTERM is the polite asking of closing given
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        logger.warning(
            "Received SIGTERM - will checkpoint and exit after this step.")
        self.should_stop = True


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path, model, optimizer, scaler, epoch, global_step, best_metric, epochs_without_improvement=0):
    state_dict = model.module.state_dict() if hasattr(
        model, "module") else model.state_dict()
    torch.save({
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "epochs_without_improvement": epochs_without_improvement,
    }, path)
    logger.info(
        f"Checkpoint saved: {path} (epoch={epoch}, step={global_step})")


def load_checkpoint(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu")
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    logger.info(
        f"Resumed from {path} (epoch={ckpt['epoch']}, step={ckpt['global_step']})")
    return ckpt["epoch"], ckpt["global_step"], ckpt["best_metric"], ckpt.get("epochs_without_improvement", 0)


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def prepare_batch(batch, device: bool):
    # non_blocking=True asks PyTorch to do the transfer CPU to GPU asynchronycally
    # without waiting to get all the the copy of the batch charged on the CPU
    # It helps to gain a little bit of time
    sar = batch["s1"].to(device, non_blocking=True)
    optical = batch["s2"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return sar, optical, labels


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, global_step, total_steps, killer, rank):
    model.train()
    running_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    # build the flat param list once instead of on every optimizer step
    # the tensors here are stable across the epoch even though their .grad
    # changes; only used for clip_grad_norm_.

    all_params = [p for g in optimizer.param_groups for p in g["params"]]
    is_fp16 = (cfg.mixed_precision == torch.float16)

    for i, batch in enumerate(loader):
        sar, optical, labels = prepare_batch(batch, device)

        with torch.amp.autocast(device_type="cuda", dtype=cfg.mixed_precision):
            logits = model(sar, optical)
            loss = loss_fn(logits, labels)

        if is_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        """Unscale, clip, apply LR schedule, and step. Returns nothing;
        mutates optimizer/scaler in place. With an end-of-epoch/SIGTERM flush."""

        global_step += 1
        # If the hardware is not compatible with bfloat16 and we want some mixed_precision
        if is_fp16:
            scaler.unscale_(optimizer)

        # It helps to get a total norm of all params that is slower than 1
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        # Update the Learning Rate Scheduler
        lr_scale = cosine_warmup_lr_lambda(
            global_step, cfg.warmup_steps, total_steps)

        for g in optimizer.param_groups:
            g["lr"] = g["_base_lr"] * lr_scale

        if is_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            # For the BF16 or the FP32, direct step
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # loss.item() helps to extract the numerical value from the tensor loss free up
        # some memory for the device
        running_loss += loss.item()

        if (i + 1) % cfg.log_every == 0 and rank == 0:
            logger.info(
                f"step {global_step} | loss  pour le rank 0 {running_loss / (i + 1):.4f}")

        if killer.should_stop:
            break

    return global_step, running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        sar, optical, labels = prepare_batch(batch, device)
        with torch.amp.autocast(device_type="cuda", dtype=cfg.mixed_precision):
            logits = model(sar, optical)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    oa, aa = compute_oa_aa(all_preds, all_labels)
    return oa, aa


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_argparser():
    p = argparse.ArgumentParser()
    # Chemin vers les données So2Sat
    p.add_argument("--data-dir", type=str, required=True)
    # Dossier où le script va sauvegarder ses checkpoints
    p.add_argument("--checkpoint-dir", type=str, required=True)
    # Chemin vers le fichier des poids préentraînés de notre modèle
    p.add_argument("--backbone-checkpoint", type=str, required=True)
    p.add_argument("--backbone-size", type=str, default="base")
    p.add_argument("--image-resolution", type=int, default=120)
    p.add_argument("--interpolation-mtd", type=str, default="nearest",
                   choices=["bilinear", "nearest"])
    p.add_argument("--probe-checkpoint-dir", type=str, default="checkpoints")
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

    torch.manual_seed(cfg.seed)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    is_distributed, device, local_rank, rank = setup_hardware_and_distributed()

    backbone = load_croma_backbone(
        cfg.backbone_checkpoint, size=cfg.backbone_size)
    backbone = backbone.to(device)

    model = CROMAFineTuning(
        backbone=backbone,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        freeze_n_layers=cfg.freeze_n_layers,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
    ).to(device)

    model.load_pretrained_head(str(cfg.head_init_path))

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
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    scaler = torch.amp.GradScaler(
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
            model, train_loader, optimizer, scaler, device, cfg, global_step, total_steps, killer, rank
        )
        if rank == 0:
            logger.info(f"Epoch {epoch} | train_loss {train_loss:.4f}")

        if (epoch + 1) % cfg.eval_every == 0:
            oa, aa = evaluate(model, val_loader, device, cfg)
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

        if killer.should_stop:
            if rank == 0:
                logger.info(
                    "Exiting cleanly after SIGTERM (job should requeue with --requeue).")
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


if __name__ == "__main__":
    main()
