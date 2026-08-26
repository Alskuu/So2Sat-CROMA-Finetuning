"""
Utilitaires de nommage partagés pour garder cohérents entre eux les noms de
fichiers de cache de features (cache_to_disk.py, script_uno.py), de têtes de
probing (script_probing.py) et de tête pré-entraînée pour le fine-tuning
(script_finetuning.py), selon qu'on travaille sur les 10 classes urbaines
(urban_only=True) ou les 17 classes complètes (urban_only=False).

Avant ce module, chaque script reconstruisait ses propres noms de fichiers
via des f-strings dupliquées sans jamais encoder ce choix, ce qui provoquait
des mélanges silencieux (ex. une tête de probing entraînée en 17 classes
rechargée par erreur pour du fine-tuning en 10 classes urban_only, ou
l'inverse).
"""

from pathlib import Path
from typing import Optional

NUM_CLASSES_URBAN = 10
NUM_CLASSES_FULL = 17


def num_classes_for(urban_only: bool) -> int:
    """Nombre de classes LCZ attendu selon le mode urban_only."""
    return NUM_CLASSES_URBAN if urban_only else NUM_CLASSES_FULL


def urban_only_tag(urban_only: bool) -> str:
    """Suffixe court et sans ambiguïté à inclure dans tout nom de fichier
    dérivé d'un dataset/modèle dépendant du mode urban_only."""
    return "urban10" if urban_only else "full17"


def build_probe_checkpoint_name(
    kind: str,
    modality: str,
    interpolation_mtd: Optional[str],
    image_resolution: int,
    urban_only: bool,
) -> str:
    """
    Construit un nom de fichier de checkpoint de probe cohérent avec
    build_cache_path (cache_to_disk.py).

    Args:
        kind: préfixe du type de probe, ex. "best_linear_probe" ou
            "best_non_linear_probe".
        modality: "both", "SAR" ou "optical".
        interpolation_mtd: méthode d'interpolation ou None.
        image_resolution: résolution des images en entrée du modèle.
        urban_only: True si le probe a été entraîné sur les 10 classes
            urbaines uniquement, False s'il couvre les 17 classes.

    Returns:
        Nom de fichier (sans dossier), ex.
        "best_linear_probe_both_nearest_120_urban10.pt".
    """
    parts = [kind, modality]
    if interpolation_mtd is not None:
        parts.append(interpolation_mtd)
    parts.append(str(image_resolution))
    parts.append(urban_only_tag(urban_only))
    return "_".join(parts) + ".pt"


def build_probe_checkpoint_path(
    checkpoint_dir: str,
    kind: str,
    modality: str,
    interpolation_mtd: Optional[str],
    image_resolution: int,
    urban_only: bool,
) -> Path:
    """Comme build_probe_checkpoint_name, mais renvoie le chemin complet sous
    checkpoint_dir."""
    return Path(checkpoint_dir) / build_probe_checkpoint_name(
        kind, modality, interpolation_mtd, image_resolution, urban_only
    )
