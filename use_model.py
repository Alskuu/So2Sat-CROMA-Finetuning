import torch
import torch.nn as nn
from typing import Optional
from use_croma import PretrainedCROMA


class CROMAClassifier(nn.Module):
    def __init__(
        self,
        backbone_path: str,
        head_path: str,
        lp: bool = False,
        num_classes: int = 17,
        modality: str = "both",
        use_joint_gap: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.modality = modality
        self.use_joint_gap = use_joint_gap

        if modality == "both":
            embed_dim = 768 if use_joint_gap else 768 * 2
        else:
            embed_dim = 768

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        self.backbone = PretrainedCROMA(
            pretrained_path="D:/CROMA_weights/CROMA_base.pt",
            modality=modality
        )

        backbone_state = torch.load(
            backbone_path, map_location="cpu", weights_only=True
        )
        self.backbone.load_state_dict(backbone_state, strict=False)

        head_state = torch.load(
            head_path, map_location="cpu", weights_only=True)

        if lp:
            # Remap "weight"/"bias" -> "1.weight"/"1.bias" si le checkpoint vient
            # d'un nn.Linear brut plutôt que d'un nn.Sequential(Dropout, Linear)
            # cas de mon linear probing notamment
            if "weight" in head_state and "1.weight" not in head_state:
                head_state = {
                    f"1.{k}": v for k, v in head_state.items()}

        self.head.load_state_dict(head_state, strict=True)

    def forward(self, sar: torch.Tensor = None, optical: torch.Tensor = None) -> torch.Tensor:
        out = self.backbone(SAR_images=sar, optical_images=optical)

        if self.modality == "both":
            features = out["joint_GAP"] if self.use_joint_gap else torch.cat(
                [out["optical_GAP"], out["SAR_GAP"]], dim=-1)
        elif self.modality == "SAR":
            features = out["SAR_GAP"]
        else:
            features = out["optical_GAP"]

        return self.head(features)
