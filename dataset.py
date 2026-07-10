import torch
import torch.nn.functional as F
# loading datasets and helps in managing data parallelism
from torch.utils.data import Dataset
import numpy as np
import h5py
from typing import Iterable, Optional, Tuple, Optional

TARGET_SIZE = 120  # résolution native CROMA


class So2SatDataset(Dataset):
    """ 
    Inclue un filtrage optionnel et un remapping des labels
    Retourne (sar, optical, label_index)
    """

    def __init__(self, h5_path: str, interpolation_mtd: Optional[str], image_resolution: int = TARGET_SIZE, urban_only: bool = False,
                 selected_classes: Optional[Iterable[int]] = None,
                 remap_labels: bool = True) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.image_resolution = image_resolution
        self.urban_only = urban_only
        self.remap_labels = remap_labels
        self._h5_file = self._sen1 = self._sen2 = self._labels = None
        self.method = interpolation_mtd

        self.selected_classes = None
        self.label_mapping = None
        if selected_classes is not None:
            self.selected_classes = sorted(
                set(int(c) for c in selected_classes))
            if self.remap_labels:
                self.label_mapping = {
                    orig: new for new, orig in enumerate(self.selected_classes)
                }

        with h5py.File(self.h5_path, "r") as f:
            class_idx = np.argmax(f["label"][:], axis=1)
            # np.ones : Return an array of ones with the same shape and type as a given array
            mask = np.ones_like(class_idx, dtype=bool)
            if self.urban_only:
                mask &= class_idx < 10
            if self.selected_classes is not None:
                mask &= np.isin(class_idx, self.selected_classes)
            self.indices = np.where(mask)[0]
            self._length = len(self.indices)

    # Ouverture persistante par worker
    def _lazy_open(self) -> None:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
            self._sen1 = self._h5_file["sen1"]
            self._sen2 = self._h5_file["sen2"]
            self._labels = self._h5_file["label"]

    def __len__(self):
        return self._length

    @staticmethod
    def _build_sar(sen1: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Conversion du signal linéaire en décibel
        sar = sen1[4:6]
        return 10*torch.log10(sar.clamp_min(eps))

    @staticmethod
    def _build_optical(sen2: torch.Tensor) -> torch.Tensor:
        _, H, W = sen2.shape
        z = torch.zeros(1, H, W, dtype=sen2.dtype)
        return torch.cat([
            z,           # B1  (absent dans So2Sat)
            sen2[0:1],   # B2
            sen2[1:2],   # B3
            sen2[2:3],   # B4
            sen2[3:4],   # B5
            sen2[4:5],   # B6
            sen2[5:6],   # B7
            sen2[9:10],  # B8
            sen2[6:7],   # B8A
            z,           # B9  (absent dans So2Sat)
            sen2[7:8],   # B11
            sen2[8:9],   # B12
        ], dim=0)

    # Normalisation par canal
    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        """
        Z-score par canal sur une image individuelle, comme SatMAE/CROMA.
        x : (C, H, W)
        """
        mean = x.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        std = x.std(dim=(1, 2), keepdim=True)   # (C, 1, 1)
        # On ajoute 1e-6 au dénominateur au cas où l'écart-type est nul pour ne pas avoir une forme indéterminée
        return (x - mean) / (std + 1e-6)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # BESOIN DE DB pour passer dans CROMA
        # REGARDER le Github pour voir le preprocessing complet

        self._lazy_open()
        real_index = int(self.indices[idx])

        sen1 = torch.from_numpy(
            self._sen1[real_index]).permute(2, 0, 1).float()
        sen2 = torch.from_numpy(
            self._sen2[real_index]).permute(2, 0, 1).float()

        sar = self._build_sar(sen1)
        optical = self._build_optical(sen2)

        # COMPARER EN ESSAYANT DE PASSER DES IMAGES DE TAILLE DE 32*32
        # Upsample 32×32 → image_resolution x image_resolution
        # Le mode bilinéaire permet de faire en sorte que chaque pixel soit la moyenne pondérée de ses 4 voisins les plus
        # proches avec des poids propotionnels à la distance : résultat lisse et continu, avec malheureusement aussi ajout
        # un peu lissé de valeurs qui ne devraient pas être présentes...
        # Le mode nearest permet de reprendre les valeurs des pixels les plus proches de ce qu'on crée : enlève cet effet de
        # lissage, mais dans notre cas où l'on passe de 32 à 120 (120/32=3,75), il y a alors une distorsion spatiale non négligeable
        # qui a lieu qui pose également problème à notre modèle
        if self.method is not None:
            sar = F.interpolate(
                sar.unsqueeze(0),
                size=(self.image_resolution, self.image_resolution),
                mode=self.method,
            ).squeeze(0)
            optical = F.interpolate(
                optical.unsqueeze(0),
                size=(self.image_resolution, self.image_resolution),
                mode=self.method,
            ).squeeze(0)

        # Normalisation après interpolation : on normalise les pixels
        # à la résolution finale, pas les pixels 32×32 originaux
        sar = self._normalize(sar)
        optical = self._normalize(optical)

        label_index = int(
            np.argmax(self._labels[real_index]).astype(np.int64))
        if self.label_mapping is not None:
            label_index = self.label_mapping[label_index]

        return sar, optical, label_index

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = self._sen1 = self._sen2 = self._labels = None
