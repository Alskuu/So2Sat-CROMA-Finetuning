import torch
import torch.nn.functional as F
# loading datasets and helps in managing data parallelism
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from pathlib import Path
import json
import hashlib
from tqdm import tqdm

from use_croma import PretrainedCROMA


def build_cache_path(
    cache_dir: str,
    cache_name: str,
    cache_metadata: Dict,
    method: Optional[str],
    image_resolution: int,
    modality: str
) -> Path:
    """
    Build a deterministic cache path from metadata.

    Args:
        cache_dir: Root cache directory.
        cache_name: Human-readable cache prefix.
        cache_metadata: Metadata describing the extraction setup.

    Returns:
        Full cache file path.
    """
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    if method is not None:
        return cache_dir_path / f"{cache_name}_{modality}_{method}_{image_resolution}.pt"
    else:
        return cache_dir_path / f"{cache_name}_{modality}_{image_resolution}.pt"


# ADAPTER L'ETAPE DE CROMA BATCH NORMALIZE A MA NORMALISATION A MOI, ENSUITE ON TESTERA DANS UN DEUXIEME TEMPS AVEC CELLE CI
@torch.no_grad()
def extract_features_from_loader(
    dataloader: DataLoader,
    pretrained_path: str,
    device: str,
    method: Optional[str],
    cache_name: str,
    size: str = "base",
    modality: str = "both",
    image_resolution: int = 120,
    feature_type: str = "GAP",
    cache_dir: Optional[str] = None,
    cache_metadata: Optional[Dict] = None,
    force_recompute: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from a dataloader using PretrainedCROMA.

    Supported outputs:
        - modality="both", feature_type="GAP"        -> joint_GAP
        - modality="both", feature_type="encodings"  -> joint_encodings
        - modality="SAR", feature_type="GAP"         -> SAR_GAP
        - modality="SAR", feature_type="encodings"   -> SAR_encodings
        - modality="optical", feature_type="GAP"     -> optical_GAP
        - modality="optical", feature_type="encodings" -> optical_encodings

    If a cache path is configured and the cache exists, the function loads it
    directly instead of recomputing.

    Args:
        dataloader: Loader yielding (sar, optical, labels).
        pretrained_path: Path to converted CROMA weights.
        device: Device string.
        size: CROMA model size.
        modality: One of {"both", "SAR", "optical"}.
        image_resolution: Input image resolution.
        feature_type: One of {"GAP", "encodings"}.
        cache_dir: Optional cache directory.
        cache_name: Cache file prefix.
        cache_metadata: Optional metadata for deterministic cache naming.
        force_recompute: Whether to ignore existing cache.

    Returns:
        features: Tensor of shape (N, D) or (N, P, D).
        labels: Tensor of shape (N,).
    """
    if modality not in {"both", "SAR", "optical"}:
        raise ValueError(
            f"Unsupported modality '{modality}'. "
            "Expected one of {'both', 'SAR', 'optical'}."
        )

    if feature_type not in {"GAP", "encodings"}:
        raise ValueError(
            f"Unsupported feature_type '{feature_type}'. "
            "Expected one of {'GAP', 'encodings'}."
        )
    print("Test")
    cache_path = None
    if cache_dir is not None:
        metadata = cache_metadata or {}
        full_metadata = {
            "pretrained_path": pretrained_path,
            "size": size,
            "modality": modality,
            "image_resolution": image_resolution,
            "feature_type": feature_type,
            **metadata,
        }
        cache_path = build_cache_path(
            cache_dir=cache_dir,
            cache_name=cache_name,
            cache_metadata=full_metadata,
            image_resolution=image_resolution,
            method=method,
            modality=modality
        )

        if cache_path.exists() and not force_recompute:
            payload = torch.load(cache_path, map_location="cpu")
            print(f"Loaded cached features from: {cache_path}")
            return payload["features"], payload["labels"]

    model = PretrainedCROMA(
        pretrained_path=pretrained_path,
        size=size,
        modality=modality,
        image_resolution=image_resolution
    ).to(device)
    model.eval()

    if modality == "both":
        output_key = "joint_GAP" if feature_type == "GAP" else "joint_encodings"
    elif modality == "SAR":
        output_key = "SAR_GAP" if feature_type == "GAP" else "SAR_encodings"
    else:
        output_key = (
            "optical_GAP"
            if feature_type == "GAP"
            else "optical_encodings"
        )

    # PRE-ALLOCATION pour optimiser le code : on crée un tensor empty pour lui ajouter des features
    # sans jouer en mémoire sur liste et tensors
    num_samples = len(dataloader.dataset)

    # Création de tenseurs qui vont m'être utiles pour la préallocation à suivre
    features_tensor: Optional[torch.Tensor] = None
    labels_tensor: Optional[torch.Tensor] = None

    # --- OPTIMISATION 2 : MIXED PRECISION & BOUCLE ---
    current_idx = 0

    for sar, optical, labels in tqdm(dataloader, desc=f"Extracting {modality}/{feature_type}", unit="batch"):
        batch_size = labels.size(0)
        # Pourquoi non_blocking= True ?
        sar = sar.to(device, non_blocking=True)
        optical = optical.to(device, non_blocking=True)

        # sar = croma_batch_normalize(sar)
        # optical = croma_batch_normalize(optical)

        # Autocast permet de passer les calculs du modèle en BF16 sur GPU (j'ai de la chance d'avoir un GPU qui tient bien le BF16)
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            if modality == "both":
                outputs = model(SAR_images=sar, optical_images=optical)
            elif modality == "SAR":
                outputs = model(SAR_images=sar)
            else:
                outputs = model(optical_images=optical)

            features = outputs[output_key]

        # Allocation différée : on connaît feature_shape seulement après
        # le premier forward. On alloue le tensor de destination en
        # pinned memory pour permettre un transfert GPU->CPU non bloquant.
        if features_tensor is None:
            feature_shape = features.shape[1:]
            pin = (device == "cuda")
            features_tensor = torch.empty(
                (num_samples, *feature_shape),
                dtype=torch.float32,
                pin_memory=pin,
            )
            labels_tensor = torch.empty(
                (num_samples,), dtype=torch.long, pin_memory=pin
            )

        # Stockage direct par indexation (évite le torch.cat final)
        next_idx = current_idx + batch_size
        features_tensor[current_idx:next_idx] = features.detach().cpu()
        labels_tensor[current_idx:next_idx] = labels
        current_idx = next_idx

    if cache_path is not None:
        payload = {
            "features": features_tensor,
            "labels": labels_tensor,
            "metadata": full_metadata,
        }
        torch.save(payload, cache_path)
        print(f"Saved features to: {cache_path}")

    return features_tensor, labels_tensor
