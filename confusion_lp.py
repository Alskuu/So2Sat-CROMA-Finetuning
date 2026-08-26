import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from dataset import So2SatDataset
from use_model import CROMAClassifier
from utils.naming_utils import num_classes_for

# ------------------------------------------------------------------
# Activation des Tensor Cores Ampere/Ada/Hopper
# ------------------------------------------------------------------
torch.set_float32_matmul_precision('high')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Noms officiels des 17 classes LCZ (Local Climate Zones) de So2Sat.
# Les 10 premières sont les classes urbaines (mode urban_only=True) : le
# dataset filtre toujours les classes urbaines en tête, donc LCZ_CLASSES[:10]
# reste valide dans ce mode (cf. dataset.py::So2SatDataset).
LCZ_CLASSES = [
    "1. Compact high-rise", "2. Compact mid-rise", "3. Compact low-rise",
    "4. Open high-rise", "5. Open mid-rise", "6. Open low-rise",
    "7. Lightweight low-rise", "8. Large low-rise", "9. Sparsely built",
    "10. Heavy industry", "A. Dense trees", "B. Scattered trees",
    "C. Bush, scrub", "D. Low plants", "E. Bare rock/paved",
    "F. Bare soil/sand", "G. Water"
]


def class_names_for(urban_only: bool) -> list:
    """Noms de classe LCZ attendus, cohérents avec naming_utils.num_classes_for."""
    return LCZ_CLASSES[:num_classes_for(urban_only)]


@torch.inference_mode()
def evaluate_and_get_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list,
):
    """
    Inférence rapide accélérée par Tensor Cores en bfloat16.
    Fonctionne avec n'importe quel DataLoader (train/validation/test), tant
    que `class_names` correspond au filtrage de classes du dataset sous-jacent.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    for sar, optical, labels in dataloader:
        sar = sar.to(device, non_blocking=True)
        optical = optical.to(device, non_blocking=True)

        # Utilisation des Tensor Cores avec bfloat16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(sar=sar, optical=optical)

        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.cpu())
        all_targets.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    n_classes = len(class_names)

    pred_unique, pred_counts = np.unique(all_preds, return_counts=True)
    pred_dist = dict(zip(pred_unique, pred_counts))

    target_unique, target_counts = np.unique(all_targets, return_counts=True)
    target_dist = dict(zip(target_unique, target_counts))

    print("Compte de toutes les prédictions qui ont été faites :")
    for c in range(n_classes):
        count = pred_dist.get(c, 0)
        pct = 100 * count / len(all_preds)
        print(f"  {class_names[c]:25s} : {count:6d} ({pct:5.2f}%)")

    print("\nCompte de tous les labels qu'il y a en réalité :")
    for c in range(n_classes):
        count = target_dist.get(c, 0)
        pct = 100 * count / len(all_targets)
        print(f"  {class_names[c]:25s} : {count:6d} ({pct:5.2f}%)")

    return all_targets, all_preds


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str,
    normalize: bool = True
):
    """
    Affiche une matrice de confusion normalisée ou brute.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = '.2f'
        title = 'Matrice de Confusion Normalisée (%)'
    else:
        fmt = 'd'
        title = 'Matrice de Confusion (Comptes Bruts)'

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Vraie Classe (Ground Truth)', fontsize=12)
    plt.xlabel('Classe Prédite', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrice de confusion sauvegardée ici : {save_path}")

    plt.show()


def build_argparser():
    p = argparse.ArgumentParser(
        description="Matrice de confusion d'un linear probe CROMA sur un split So2Sat.")
    p.add_argument("--split", type=str, default="validation",
                   choices=["train", "validation", "test"],
                   help="Split So2Sat à évaluer.")
    p.add_argument("--data-dir", type=str, default="D:/LCZ42")
    p.add_argument("--pretrained-path", type=str,
                   default="D:/CROMA_weights/CROMA_base.pt")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--head-path", type=str, default=None,
                   help="Chemin du checkpoint de tête de linear probe. "
                        "Par défaut, dérivé de --checkpoint-dir/--modality/--interpolation-mtd/--image-resolution.")
    p.add_argument("--modality", type=str, default="both",
                   choices=["both", "SAR", "optical"])
    p.add_argument("--interpolation-mtd", type=str, default="nearest",
                   choices=["bilinear", "nearest"])
    p.add_argument("--image-resolution", type=int, default=120)
    p.add_argument("--urban-only", action="store_true",
                   help="Évalue uniquement sur les 10 classes LCZ urbaines au lieu des 17 classes complètes.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-path", type=str,
                   default="checkpoints/figures/confusion_lps.png")
    p.add_argument("--no-normalize", action="store_false", dest="normalize")
    return p


def main():
    args = build_argparser().parse_args()

    if args.head_path is None:
        args.head_path = (
            f"{args.checkpoint_dir}/best_linear_probe_{args.modality}_"
            f"{args.interpolation_mtd}_{args.image_resolution}.pt"
        )

    h5_path = f"{args.data_dir}/{args.split}.h5"

    print(f"Initialisation du dataset '{args.split}'")
    dataset = So2SatDataset(
        h5_path=h5_path,
        interpolation_mtd=args.interpolation_mtd,
        image_resolution=args.image_resolution,
        urban_only=args.urban_only,
    )

    print("Initialisation du loader")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,  # Important pour l'accélération GPU
    )

    model = CROMAClassifier(
        args.pretrained_path,
        args.head_path,
        modality=args.modality,
        lp=True,
        use_joint_gap=True,
    )

    class_names = class_names_for(args.urban_only)

    y_true, y_pred = evaluate_and_get_predictions(
        model, loader, device, class_names)

    plot_confusion_matrix(
        y_true, y_pred, class_names=class_names,
        save_path=args.save_path, normalize=args.normalize)


if __name__ == "__main__":
    main()
