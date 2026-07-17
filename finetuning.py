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
import math
import signal
import logging
from dataclasses import dataclass
from typing import Optional
import multiprocessing

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.optim import AdamW
from torch.amp import autocast


from use_croma import PretrainedCROMA

# Configuration du logger (détaillée ci-dessous)
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
    # "nearest" | "bilinear"
    interpolation_mtd: str = "nearest"
    # explicit override; bypasses checkpoint_dir/interpolation_mtd
    head_init_path: Optional[str] = None
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
    num_workers: int = 6
    seed: int = 42
    # contrôle si le scipt doit reprendre l'entraînement là où il
    # s'était arrêté en rechargeant un checkpoint existant
    resume: bool = True
    eval_every: int = 1
    log_every: int = 50

    # Early stopping (on the validation AA)
    patience: int = 8
    early_stop_epsilon: float = 0.001
    min_epochs: int = 18


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
    for name, p in model.backbone.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split(".")
        tower_name = parts[0]
        if "layers" in parts:
            idx_pos = parts.index("layers") + 1
            depth = int(parts[idx_pos]) if idx_pos < len(
                parts) and parts[idx_pos].isdigit() else 0

        # norm_out : LR max
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

    return is_distributed, device, local_rank, rank, world_size

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
        self.should_stop = torch.tensor(
            0, dtype=torch.int32, device="cuda" if torch.cuda.is_available() else "cpu")

        # SIGTERM is the polite asking of closing given
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        logger.warning(
            "Received SIGTERM - will checkpoint and exit after this step.")
        self.should_stop.fill_(1)

    def synchronize(self, is_distributed: bool):
        if is_distributed:
            # Broadcast the signal from any rank that caught it (usually Slurm signals all ranks, but better safe than sorry)
            torch.distributed.all_reduce(
                self.should_stop, op=torch.distributed.ReduceOp.MAX)
        return self.should_stop.item() == 1

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


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, global_step, total_steps, killer, rank, is_distributed):
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

        # Periodically check preemption
        if i % 10 == 0 and killer.synchronize(is_distributed):
            break

        with autocast(device_type="cuda", dtype=cfg.mixed_precision):
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

    return global_step, running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, world_size, num_val_samples, is_distributed):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sar, optical, labels in loader:
            sar, optical = sar.to(device), optical.to(device)
            outputs = model(sar, optical)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds)
            all_targets.append(labels.to(device))

    preds_local = torch.cat(all_preds)
    targets_local = torch.cat(all_targets)

    if is_distributed:
        # We gather all predictions and labels across GPUs to calculate the exact validation score
        full_preds = torch.empty(
            preds_local.numel() * world_size,
            dtype=preds_local.dtype,
            device=device
        )
        full_targets = torch.empty(
            targets_local.numel() * world_size,
            dtype=targets_local.dtype,
            device=device
        )

        # DIRECT COLLECTIVE COMMUNICATION GPU-TO-GPU (NCCL)
        # Very fast exchange between tensors, without going through the CPU neither with "pickle"
        torch.distributed.all_gather_into_tensor(full_preds, preds_local)
        torch.distributed.all_gather_into_tensor(full_targets, targets_local)
    else:
        full_preds = preds_local
        full_targets = targets_local

    # Thanks to the SequentialDistributedSampler, all the padding is at the end of full_preds.
    # We truncate to the exact size for rigorous and duplicate-free metric computation.
    full_preds = full_preds[:num_val_samples]
    full_targets = full_targets[:num_val_samples]

    # Calculate the Global Exactitude (OA - Overall Accuracy)
    correct = (full_preds == full_targets).float().sum().item()
    oa = correct / num_val_samples

    # Calculate the Mean Exactitude (AA - Average Accuracy)
    # We recover all the unique classes that are in the validation dataset
    unique_classes = torch.unique(full_targets)
    class_accuracies = []
    for c in unique_classes:
        class_mask = (full_targets == c)
        if class_mask.sum() > 0:
            class_acc = (full_preds[class_mask] == c).float().mean().item()
            class_accuracies.append(class_acc)

    aa = sum(class_accuracies) / \
        len(class_accuracies) if class_accuracies else 0.0
    return oa, aa
