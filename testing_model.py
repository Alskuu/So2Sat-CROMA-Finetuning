import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from use_model import CROMAClassifier
from finetuning import save_confusion_matrix
from dataset import So2SatDataset
from utils.naming_utils import num_classes_for

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

JOINT_EMBED_KEY = "joint_GAP"
EMBED_DIM = 768
NUM_CLASSES = 17
BUILT_CLASSES_MAX_LABEL = 10

LCZ_CLASSES_17 = [
    "1. Compact high-rise", "2. Compact mid-rise", "3. Compact low-rise",
    "4. Open high-rise", "5. Open mid-rise", "6. Open low-rise",
    "7. Lightweight low-rise", "8. Large low-rise", "9. Sparsely built",
    "10. Heavy industry", "A. Dense trees", "B. Scattered trees",
    "C. Bush, scrub", "D. Low plants", "E. Bare rock/paved",
    "F. Bare soil/sand", "G. Water",
]
LCZ_CLASSES_10 = LCZ_CLASSES_17[:10]

# Etapes d'ablation "urban_only" : le modele n'est entraine/evalue que sur
# les 10 classes LCZ urbaines (LCZ 1-10, indices 0-9) plutot que les 17.
URBAN_ONLY_STEPS = {4}

# ============================================================
# Fichiers de test, global + par continent
# (produits par 02_split_by_continent.py pour les continents)
# ============================================================
H5_GLOBAL = "D:/LCZ42/testing.h5"
H5_DIR_CONTINENTS = "D:/LCZ42/per_continent/"

CONTINENT_TO_H5 = {
    "global": H5_GLOBAL,
    "europe": H5_DIR_CONTINENTS + "testing_europe.h5",
    "afrique": H5_DIR_CONTINENTS + "testing_afrique.h5",
    "asie_ouest": H5_DIR_CONTINENTS + "testing_asie_ouest.h5",
    "asie_est": H5_DIR_CONTINENTS + "testing_asie_est.h5",
    "ameriquenord": H5_DIR_CONTINENTS + "testing_ameriquenord.h5",
    "ameriquesud": H5_DIR_CONTINENTS + "testing_ameriquesud.h5",
    "oceanie": H5_DIR_CONTINENTS + "testing_oceanie.h5",
}

# Ordre d'execution : tous les continents seront testes systematiquement
# a chaque lancement du script.
ALL_CONTINENTS = list(CONTINENT_TO_H5.keys())


def load_checkpoint(path, model, scaler):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model"])
    logger.info(
        f"Resumed from {path} (epoch={ckpt['epoch']}, step={ckpt['global_step']})")
    return ckpt["epoch"], ckpt["best_metric"]


def export_backbone_head(arg, k, ckpt_path, out_dir):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    full_sd = ckpt["model"]
    backbone_sd = {k[len("backbone."):]: v for k,
                   v in full_sd.items() if k.startswith("backbone.")}
    head_sd = {k[len("head."):]: v for k, v in full_sd.items()
               if k.startswith("head.")}
    backbone_path = f"{arg}_backbone_etape_{k}.pt"
    head_path = f"{arg}_head_etape_{k}.pt"
    out_backbone = out_dir + backbone_path
    out_head = out_dir + head_path
    torch.save(backbone_sd, out_backbone)
    torch.save(head_sd, out_head)
    return out_backbone, out_head


def prepare_batch(batch, device: bool):
    sar, optical, labels = batch
    sar = sar.to(device, non_blocking=True)
    optical = optical.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return sar, optical, labels


def build_test_loader(h5_path, k, batch_size=64, num_workers=4):
    """
    Pour k in URBAN_ONLY_STEPS, le modele ne predit que les 10 classes urbaines
    (LCZ 1-10, indices 0-9). So2SatDataset supporte deja ce filtrage nativement
    via `urban_only=True`.
    """
    urban_only = k in URBAN_ONLY_STEPS
    dataset = So2SatDataset(
        h5_path=h5_path,
        interpolation_mtd="nearest",
        urban_only=urban_only,
    )
    if urban_only:
        logger.info(
            f"[k={k}] Filtrage aux classes urbaines actif : "
            f"{len(dataset)} echantillons."
        )

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    return loader, len(dataset)


@torch.inference_mode()
def test(model, loader, device, num_test_samples, k, continent="global", save_examples=False):
    model.eval()
    all_preds = []
    all_targets = []

    # {classe_vraie: {"correct": idx | None, "wrong": (idx, pred_c) | None}}
    examples = {} if save_examples else None
    sample_ptr = 0  # position sequentielle dans le dataset (shuffle=False)

    for sar, optical, labels in loader:
        sar, optical = sar.to(device), optical.to(device)
        outputs = model(sar, optical)
        preds = torch.argmax(outputs, dim=1)

        all_preds.append(preds)
        all_targets.append(labels.to(device))

        if save_examples:
            preds_cpu = preds.cpu()
            labels_cpu = labels.cpu()
            for i in range(labels_cpu.shape[0]):
                global_idx = sample_ptr + i
                true_c = int(labels_cpu[i].item())
                pred_c = int(preds_cpu[i].item())
                cls_dict = examples.setdefault(
                    true_c, {"correct": None, "wrong": None})
                if pred_c == true_c and cls_dict["correct"] is None:
                    cls_dict["correct"] = global_idx
                elif pred_c != true_c and cls_dict["wrong"] is None:
                    cls_dict["wrong"] = (global_idx, pred_c)
            sample_ptr += labels_cpu.shape[0]

    full_preds = torch.cat(all_preds)[:num_test_samples]
    full_targets = torch.cat(all_targets)[:num_test_samples]
    full_preds_cpu = full_preds.cpu()
    full_targets_cpu = full_targets.cpu()

    LCZ_CLASSES = LCZ_CLASSES_10 if k in URBAN_ONLY_STEPS else LCZ_CLASSES_17

    save_confusion_matrix(
        full_targets_cpu, full_preds_cpu, LCZ_CLASSES,
        f"figures/test_etape_{k}_{continent}.png"
    )

    correct = (full_preds == full_targets).float().sum().item()
    oa = correct / num_test_samples

    unique_classes = torch.unique(full_targets)
    class_accuracies = []
    for c in unique_classes:
        mask = (full_targets == c)
        if mask.sum() > 0:
            class_accuracies.append(
                (full_preds[mask] == c).float().mean().item())
    aa = sum(class_accuracies) / \
        len(class_accuracies) if class_accuracies else 0.0

    macro_f1 = f1_score(
        full_targets_cpu, full_preds_cpu,
        average="macro", labels=list(range(len(LCZ_CLASSES)))
    )

    return oa, aa, macro_f1, examples, LCZ_CLASSES


def run_one(arg: str, k: int, continent: str, save_examples: bool, examples_dir: str,
            on_examples=None):
    """
    Evalue le checkpoint `arg` a l'etape `k` sur le continent donne.

    `on_examples`, si fourni, recoit les exemples corrects/faux collectes
    par `test()` (signature : `on_examples(examples, LCZ_CLASSES, k,
    continent, examples_dir)`) -- c'est le point d'extension utilise par
    create_gpkg_kml.py pour exporter de vrais GeoTIFF/GPKG/KML, sans
    dupliquer le chargement du modele, la construction du dataset ou la
    boucle de test. Ce module n'a plus d'exporteur par defaut (l'ancien CSV
    n'avait plus de consommateur depuis la suppression de
    create_good_bad_gpkgs.py, cf. reco_upgrade.txt) : sans `on_examples`,
    `save_examples=True` ne fait que collecter les exemples en memoire sans
    rien ecrire sur le disque.
    """
    h5_path = CONTINENT_TO_H5[continent]
    if not Path(h5_path).exists():
        logger.warning(
            f"[continent={continent}] fichier introuvable : {h5_path} -> ignore.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(f"D:/CROMA_weights/poids_etape_{k}/{arg}.pt")
    out_dir = "D:/CROMA_weights/separation/"
    backbone_path, head_path = export_backbone_head(
        arg, k, checkpoint_path, out_dir)

    num_classes = num_classes_for(k in URBAN_ONLY_STEPS)
    model = CROMAClassifier(
        backbone_path=backbone_path, num_classes=num_classes,
        head_path=head_path, use_joint_gap=True
    )
    model.to(device)

    test_loader, num_test_samples = build_test_loader(h5_path, k)
    if num_test_samples == 0:
        logger.warning(
            f"[continent={continent}, k={k}] 0 echantillon -> ignore.")
        return None

    logger.info(
        f"[continent={continent}, k={k}] {num_test_samples} echantillons de test.")
    oa, aa, macro_f1, examples, LCZ_CLASSES = test(
        model, test_loader, device, num_test_samples, k,
        continent=continent, save_examples=save_examples
    )

    print(f"\n--- RESULTATS -- continent={continent}, etape={k} ---")
    print(f"Overall Accuracy (OA) : {oa * 100:.2f}%")
    print(f"Average Accuracy (AA) : {aa * 100:.2f}%")
    print(f"Macro-F1 : {macro_f1 * 100:.2f}%")

    if save_examples:
        if on_examples is not None:
            on_examples(examples, LCZ_CLASSES, k, continent, examples_dir)
        else:
            logger.info(
                "save_examples=True mais aucun on_examples fourni -- rien "
                "n'est ecrit sur le disque (cf. create_gpkg_kml.py pour un "
                "export GeoTIFF/GPKG/KML reel)."
            )

    return {"continent": continent, "k": k, "oa": oa, "aa": aa, "macro_f1": macro_f1, "n": num_test_samples}


def write_summary_csv(summary_rows, summary_path):
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in summary_rows:
            writer.writerow(row)
    logger.info(f"Resume multi-continents sauvegarde dans {summary_path}")


def run_loop(k: int, continents: list, save_examples: bool, examples_dir: str,
             on_examples=None, summary_path: str = "figures/resume_test_continents.csv"):
    """Boucle `run_one` sur `continents` puis ecrit un CSV de synthese OA/AA/F1."""
    summary_rows = [
        ("continent", "etape_k", "n_echantillons", "OA", "AA", "macro_F1")]

    for continent in continents:
        result = run_one(
            "best", k, continent, save_examples, examples_dir, on_examples=on_examples)
        if result is not None:
            summary_rows.append((
                result["continent"], result["k"], result["n"],
                f"{result['oa']*100:.2f}", f"{result['aa']*100:.2f}", f"{result['macro_f1']*100:.2f}"
            ))

    write_summary_csv(summary_rows, summary_path)
    return summary_rows


def build_argparser():
    p = argparse.ArgumentParser(
        description="Evaluation CROMA - boucle systematique sur tous les continents.")
    # store_false + default implicite True : contrairement a
    # `action="store_true", default=True` (bug precedent, cf. reco_upgrade.txt),
    # cette forme permet reellement de desactiver l'option depuis la CLI.
    p.add_argument(
        "--no-save-examples", action="store_false", dest="save_examples",
        help="Desactive la collecte d'un exemple correct + un exemple faux par classe LCZ "
             "(sans effet ici sans exporteur externe, cf. create_gpkg_kml.py). Actif par defaut."
    )
    p.add_argument("--examples-dir", type=str, default="figures/exemples")
    return p


def parse_args():
    return build_argparser().parse_args()


def main(k):
    args = parse_args()
    run_loop(k, ALL_CONTINENTS, args.save_examples, args.examples_dir)


if __name__ == "__main__":
    for k in [3, 4, 6]:
        main(k)
