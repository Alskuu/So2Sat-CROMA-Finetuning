"""
LR Range Test (Smith, 2017) adapté au pipeline de fine-tuning CROMA,
avec comparaison automatique de plusieurs ratios head_lr / backbone_lr (LLRD)

Principe :
- On ne teste PAS une grille de LR tenus fixes pendant N steps chacun.
- On fait UNE SEULE run courte pendant laquelle le LR augmente
  exponentiellement à CHAQUE step (1 step par valeur de LR).
- On multiplie tous les groupes de param de l'optimizer par le MÊME facteur
  multiplicatif à chaque step -> Durant chaque run les ratios LLRD (head_lr / 
  backbone_lr / layer_decay) restent strictement identiques à ceux du run réel, 
  seule l'échelle globale du LR varie. On sweep donc une seule dimension.
- On enregistre la loss brute + une loss lissée par EMA (avec correction de
  biais façon Adam) à chaque step, on trace loss lissée vs. LR (échelle log),
  avec la loss brute affichée en transparence en dessous pour visualiser le
  bruit lissé,
  puis on suggère automatiquement un LR de départ basé sur la pente la plus
  raide (PAS sur la loss minimale absolue, qui serait trop proche de la
  divergence) avec une marge de sécurité (division par 10) pour s'éloigner
  un peu quand même de cette divergence.
  Une loss lissée est très intéressante car les batchs sont très différents entre
  eux, donc pour réduire ce bruit entre les batchs on fait cela..
  On répète ce protocole pour plusieurs valeurs du ratio head_lr / backbone_lr (--ratios)
  chacunn run étant indépendant.

/!\ ADAPTATIONS A FAIRE DE TON COTE (marquées ci-dessous) :
- La forme exacte des batches retournés par ton DataLoader (dict vs tuple,
  clés s1/s2/labels...) : reprends exactement ce que fait `train_one_epoch`.
- La loss utilisée (CrossEntropyLoss standard ici) : remplace si tu utilises
  une pondération de classes, du label smoothing, etc. dans train_one_epoch.
"""

from finetuning import (
    NUM_CLASSES,
    EMBED_DIM,
    FineTuneConfig,
    CROMAFineTuning,
    build_optimizer,
    load_croma_backbone,
)
from dataset import So2SatDataset
import matplotlib.pyplot as plt
import argparse
import itertools
import math
import csv
import dataclasses
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")


def parse_ratios(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--backbone-checkpoint", type=str, required=True)
    p.add_argument("--backbone-size", type=str, default="base")
    p.add_argument("--image-resolution", type=int, default=120)
    p.add_argument("--interpolation-mtd", type=str, default="nearest",
                   choices=["bilinear", "nearest"])
    p.add_argument("--batch-size", type=int, default=32)
    # Ces LR servent de point "multiplicateur = 1.0" : la rampe les fait
    # varier ensemble, la structure LLRD (ratios) reste donc inchangée.
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--ratios", type=parse_ratios, default=[10.0, 100.0],
                   help="Ratio head_lr/backbone_lr")
    p.add_argument("--layer-decay", type=float, default=0.75)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--freeze-n-layers", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # --- Spécifique au range test ---
    p.add_argument("--num-iters", type=int, default=800,
                   help="Nombre total de steps sur toute la rampe exponentielle "
                   "(1 step = 1 valeur de LR, pas de répétition par valeur).")
    p.add_argument("--start-mult", type=float, default=1e-2,
                   help="Multiplicateur initial appliqué aux LR de base.")
    p.add_argument("--end-mult", type=float, default=200.0,
                   help="Multiplicateur final appliqué aux LR de base.")
    p.add_argument("--smoothing-beta", type=float, default=0.98,
                   help="Coefficient EMA pour lisser la loss (proche de 1 = plus lisse).")
    p.add_argument("--diverge-factor", type=float, default=4.0,
                   help="Arrêt si loss_lissée > diverge_factor * meilleure_loss_lissée.")

    DTYPE_MAP = {"bfloat16": torch.bfloat16,
                 "float16": torch.float16, "float32": torch.float32}
    p.add_argument("--mixed-precision", type=lambda s: DTYPE_MAP[s],
                   default=torch.bfloat16, choices=DTYPE_MAP.values())
    p.add_argument("--head-init-path", type=str, default=None)
    return p


def build_fresh_model(cfg, device):
    """Recharge backbone + tête à neuf avant chaque test de ratio."""
    backbone = load_croma_backbone(
        cfg.backbone_checkpoint, size=cfg.backbone_size,
        image_resolution=cfg.image_resolution,
    )
    backbone = backbone.to(device)

    model = CROMAFineTuning(
        backbone=backbone,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        freeze_n_layers=cfg.freeze_n_layers,
        use_gradient_checkpointing=False,
    ).to(device)

    if Path(cfg.head_init_path).exists():
        model.load_pretrained_head(str(cfg.head_init_path))
    else:
        print(
            f"[WARN] Poids de tête introuvables à {cfg.head_init_path}. Initialisation aléatoire.")

    model.train()
    return model


def run_single_lr_range_test(args, cfg, ratio, device, train_loader, criterion):
    """Exécute UN range test complet pour un ratio head_lr/backbone_lr donné.
    Retourne l'historique des steps ainsi que le LR suggéré."""

    print(f"\n{'='*70}")
    print(f"  RATIO head_lr/backbone_lr = {ratio:g}  "
          f"(head_lr={cfg.head_lr:.2e}, backbone_lr={cfg.backbone_lr:.2e})")
    print(f"{'='*70}")

    torch.manual_seed(cfg.seed)

    model = build_fresh_model(cfg, device)
    optimizer = build_optimizer(model, cfg)
    # Gestion de FP16 avec GradScaler si nécessaire (miroir de train_one_epoch)
    is_fp16 = (cfg.mixed_precision == torch.float16)
    scaler = GradScaler("cuda", enabled=is_fp16)

    # Récupération de tous les paramètres pour clip_grad_norm_
    all_params = [p for g in optimizer.param_groups for p in g["params"]]

    data_iter = itertools.cycle(train_loader)

    # Calcul des multiplicateurs exponentiels
    lr_mults = [
        args.start_mult *
        (args.end_mult / args.start_mult) ** (i / (args.num_iters - 1))
        for i in range(args.num_iters)
    ]

    history = []
    ema_loss = None
    best_ema = float("inf")

    for step, mult in enumerate(lr_mults):
        # Application de l'échelle exponentielle sur le _base_lr de chaque groupe
        for g in optimizer.param_groups:
            g["lr"] = g["_base_lr"] * mult

        batch = next(data_iter)
        sar, optical, labels = batch
        sar = sar.to(device, non_blocking=True)
        optical = optical.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward Pass
        with autocast(device_type="cuda", dtype=cfg.mixed_precision):
            logits = model(sar, optical)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            print(
                f"[STOP] Loss non finie au step {step} (mult={mult:.2e}). Arrêt.")
            break

        # Backward Pass & Gradient Clipping (miroir de train_one_epoch)
        if is_fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

        # Calcul du lissage EMA avec correction de biais
        raw_loss = loss.item()
        ema_loss = raw_loss if ema_loss is None else (
            args.smoothing_beta * ema_loss +
            (1 - args.smoothing_beta) * raw_loss
        )
        smoothed = ema_loss / (1 - args.smoothing_beta ** (step + 1))

        history.append({
            "ratio": ratio,
            "step": step,
            "lr_mult": mult,
            "head_lr_eff": cfg.head_lr * mult,
            "backbone_lr_eff": cfg.backbone_lr * mult,
            "raw_loss": raw_loss,
            "smoothed_loss": smoothed,
        })

        if step % 10 == 0:
            print(f"step {step:4d} | mult {mult:8.4f} | head_lr_eff {cfg.head_lr*mult:.2e} "
                  f"| loss(brut) {raw_loss:.4f} | loss(lissée) {smoothed:.4f}")

        # Arrêt anticipé en cas de divergence
        if smoothed < best_ema:
            best_ema = smoothed
        elif smoothed > args.diverge_factor * best_ema:
            print(
                f"[STOP] Divergence détectée au step {step} (mult={mult:.2e}). Arrêt.")
            break

    # Nettoyage mémoire
    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return history


def analyze_history(history, cfg, args):
    """Calcule le LR optimal selon la pente la plus raide."""
    if len(history) < 5:
        return None

    head_lrs = [h["head_lr_eff"] for h in history]
    raw_losses = [h["raw_loss"] for h in history]
    smoothed_losses = [h["smoothed_loss"] for h in history]

    slopes = [
        (smoothed_losses[i + 1] - smoothed_losses[i]) /
        (math.log10(head_lrs[i + 1]) - math.log10(head_lrs[i]))
        for i in range(len(head_lrs) - 1)
    ]
    steepest_idx = slopes.index(min(slopes))  # pente la plus négative
    suggested_head_lr = head_lrs[steepest_idx] / 10.0  # marge de sécurité ÷10
    ratio = history[0]["ratio"]
    suggested_backbone_lr = suggested_head_lr / ratio

    return {
        "ratio": ratio,
        "n_steps": len(history),
        "min_smoothed_loss": min(smoothed_losses),
        "lr_at_steepest_slope": head_lrs[steepest_idx],
        "suggested_head_lr": suggested_head_lr,
        "suggested_backbone_lr": suggested_backbone_lr,
        "head_lrs": head_lrs,
        "raw_losses": raw_losses,
        "smoothed_losses": smoothed_losses,
    }


def main():
    args = build_argparser().parse_args()
    base_cfg = FineTuneConfig(
        **{k: v for k, v in vars(args).items() if k != "ratios"},
        backbone_lr=args.head_lr,  # placeholder, réécrit ci-dessous par ratio
    )
    if base_cfg.head_init_path is None:
        base_cfg.head_init_path = (
            f"{base_cfg.checkpoint_dir}/best_linear_probe_both_"
            f"{base_cfg.interpolation_mtd}_{base_cfg.image_resolution}.pt"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data : partagé entre tous les runs, seul le seed avant chaque run
    #     change la séquence effective de batches consommée. ---
    train_ds = So2SatDataset(
        h5_path=Path(base_cfg.data_dir) / "training.h5",
        interpolation_mtd=base_cfg.interpolation_mtd,
        image_resolution=base_cfg.image_resolution,
    )
    train_loader = DataLoader(
        train_ds, batch_size=base_cfg.batch_size, shuffle=True,
        num_workers=base_cfg.num_workers, pin_memory=True,
        persistent_workers=base_cfg.num_workers > 0,
    )

    criterion = torch.nn.CrossEntropyLoss()

    out_dir = Path(base_cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ratio in args.ratios:
        cfg = dataclasses.replace(
            base_cfg,
            head_lr=args.head_lr,
            backbone_lr=args.head_lr / ratio,
        )

        history = run_single_lr_range_test(
            args, cfg, ratio, device, train_loader, criterion)

        if len(history) < 5:
            print(f"[WARN] Ratio {ratio:g} : trop peu de steps valides "
                  f"({len(history)}), run ignoré dans la comparaison.")
            continue

        # Sauvegarde CSV
        ratio_tag = str(ratio).replace(".", "p")
        csv_path = out_dir / \
            f"lr_range_test_ratio{ratio_tag}_{base_cfg.batch_size}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"Historique sauvegardé dans {csv_path}")

        result = analyze_history(history, cfg, args)
        all_results.append(result)

        # --- Plot individuel : loss brute en transparence sous la lissée ---
        plt.figure(figsize=(8, 5))
        plt.plot(result["head_lrs"], result["raw_losses"],
                 alpha=0.25, linewidth=0.8, color="tab:blue",
                 label="loss brute", zorder=1)
        plt.plot(result["head_lrs"], result["smoothed_losses"],
                 linewidth=2.0, color="tab:blue",
                 label="loss lissée (EMA)", zorder=2)
        plt.xscale("log")
        plt.xlabel("head_lr effectif (échelle log)")
        plt.ylabel("Loss")
        plt.title(f"LR Range Test — ratio head/backbone = {ratio:g}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        fig_path = out_dir / \
            f"lr_range_test_ratio{ratio_tag}_{base_cfg.batch_size}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Graphe sauvegardé dans {fig_path}")

    if not all_results:
        print("\nAucun run n'a produit assez de steps valides. "
              "Vérifie --start-mult / la stabilité du modèle.")
        return

    # Plot de comparaison
    plt.figure(figsize=(9, 6))
    color_cycle = itertools.cycle(
        plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for result in all_results:
        color = next(color_cycle)
        plt.plot(result["head_lrs"], result["raw_losses"],
                 color=color, alpha=0.15, linewidth=0.8, zorder=1)
        plt.plot(result["head_lrs"], result["smoothed_losses"],
                 color=color, linewidth=2.0, zorder=2,
                 label=f"ratio = {result['ratio']:g}")
    plt.xscale("log")
    plt.xlabel(
        "head_lr effectif (échelle log) — backbone_lr suit le ratio indiqué")
    plt.ylabel("Loss (brute en transparence, lissée en trait plein)")
    plt.title("LR Range Test — comparaison des ratios head/backbone (LLRD)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    comparison_path = out_dir / \
        f"lr_range_test_comparison_{base_cfg.batch_size}.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGraphe de comparaison sauvegardé dans {comparison_path}")

    # --- Tableau récapitulatif + candidat indicatif ---
    print(f"\n{'='*90}")
    print("RÉCAPITULATIF")
    print(f"{'='*90}")
    header = (f"{'ratio':>8} | {'n_steps':>7} | {'min_loss_lissée':>16} | "
              f"{'lr_pente_max':>13} | {'head_lr_sugg.':>13} | {'backbone_lr_sugg.':>17}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['ratio']:>8.3g} | {r['n_steps']:>7d} | "
              f"{r['min_smoothed_loss']:>16.4f} | {r['lr_at_steepest_slope']:>13.2e} | "
              f"{r['suggested_head_lr']:>13.2e} | {r['suggested_backbone_lr']:>17.2e}")

    # Candidat indicatif = celui qui atteint la loss lissée minimale la plus
    # basse. C'est un signal FAIBLE et automatique, pas un verdict : à
    # confirmer visuellement sur lr_range_test_comparison.png (forme du
    # plateau, régularité de la descente, distance avant divergence...).
    best = min(all_results, key=lambda r: r["min_smoothed_loss"])
    print(f"\nCandidat indicatif (loss lissée minimale) : "
          f"ratio = {best['ratio']:g}")
    print("/!\\ Ceci est un simple tri automatique sur un seul critère "
          "(min de la loss lissée). Vérifie impérativement le graphe de "
          "comparaison avant de trancher : un ratio peut atteindre une "
          "loss un peu plus basse tout en étant plus instable ou en "
          "divergeant plus tôt, ce qui n'apparaît pas dans ce seul chiffre.")


if __name__ == "__main__":
    main()
