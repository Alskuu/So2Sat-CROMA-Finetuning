import torch
from torch import nn  # provides classes and functions for building neural networks
from use_croma import PretrainedCROMA
from dataset import TARGET_SIZE


class CROMALinearProbe(nn.Module):
    """
    Linear probing : encodeur CROMA gelé + couche linéaire finale.
    modality='both' → concat optical_GAP + SAR_GAP (1536 dim).
    Les images reçues doivent être déjà normalisées (z-score par canal).
    """

    def __init__(self, pretrained_path: str, image_resolution: int = TARGET_SIZE, modality: str = 'both', joint: bool = False, num_classes: int = 17) -> None:
        super().__init__()
        assert modality in {"both", "SAR", "optical"}, \
            f"modality invalide : {modality!r}"
        self.modality = modality
        self.encoder = PretrainedCROMA(
            pretrained_path=pretrained_path,
            modality=modality,
            image_resolution=image_resolution
        )
        # On gèle ces paramètres avec requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Dimension d'entrée selon la modalité choisie
        if modality == 'both' and joint == False:
            # on concatène optical_GAP et SAR_GAP → 768 + 768 = 1536
            in_dim = 768 * 2
        else:
            in_dim = 768

        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, 768) — joint_GAP extrait via extract_features_from_loader notamment
        Returns:
            logits: (B, num_classes)
        """
        return self.classifier(features)


class CROMAMLProbe(nn.Module):
    def __init__(self, pretrained_path: str, image_resolution: int = TARGET_SIZE, modality: str = 'both', joint: bool = False, num_classes: int = 17, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        assert modality in {"both", "SAR", "optical"}, \
            f"modality invalide : {modality!r}"
        self.modality = modality
        self.encoder = PretrainedCROMA(
            pretrained_path=pretrained_path,
            modality=modality,
            image_resolution=image_resolution
        )
        # On gèle ces paramètres avec requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Dimension d'entrée selon la modalité choisie
        if modality == 'both' and joint == False:
            # on concatène optical_GAP et SAR_GAP → 768 + 768 = 1536
            in_dim = 768 * 2
        else:
            in_dim = 768

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, 768) — joint_GAP extrait via extract_features_from_loader notamment
        Returns:
            logits: (B, num_classes)
        """
        return self.classifier(features)
