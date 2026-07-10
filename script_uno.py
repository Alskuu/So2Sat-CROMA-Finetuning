from dataset import So2SatDataset
from torch.utils.data import DataLoader
import torch
from typing import Optional
import time

from optimisation import extract_features_from_loader


def script(train: bool, interpolation_mtd: Optional[str], image_resolution: int, modality: str, force_recompute: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_SIZE = image_resolution
    if train:
        train_dataset = So2SatDataset(h5_path="D:/LCZ42/training.h5", image_resolution=TARGET_SIZE,
                                      urban_only=False,
                                      selected_classes=None,
                                      remap_labels=False,
                                      interpolation_mtd=interpolation_mtd
                                      )

        val_dataset = So2SatDataset(h5_path="D:/LCZ42/validation.h5", image_resolution=TARGET_SIZE,
                                    urban_only=False,
                                    selected_classes=None,
                                    remap_labels=False,
                                    interpolation_mtd=interpolation_mtd
                                    )

        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

        print("Passage au val_loader")

        val_loader = DataLoader(val_dataset,   batch_size=64,
                                shuffle=False, num_workers=0, pin_memory=True)
        print("Passage à l'extraction de features depuis le loader")

        train_features, train_labels = extract_features_from_loader(
            dataloader=train_loader,
            pretrained_path="D:/CROMA_weights/CROMA_base.pt",
            device=device,
            method=interpolation_mtd,
            image_resolution=image_resolution,
            modality=modality,
            feature_type="GAP",
            cache_dir="cache/",
            cache_name="train",
            force_recompute=force_recompute
        )
        '''
        val_features, val_labels = extract_features_from_loader(
            dataloader=val_loader,
            pretrained_path="D:/CROMA_weights/CROMA_base.pt",
            device=device,
            method=interpolation_mtd,
            image_resolution=image_resolution,
            modality=modality,
            feature_type="GAP",
            cache_dir="cache/",
            cache_name="val",
            force_recompute=force_recompute
        )
        '''
        print("Extraction terminée, features sauvegardées dans cache/")
    else:
        test_dataset = So2SatDataset(h5_path="D:/LCZ42/testing.h5", image_resolution=TARGET_SIZE,
                                     urban_only=False,
                                     selected_classes=None,
                                     remap_labels=False,
                                     interpolation_mtd=interpolation_mtd
                                     )
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        print("Passage à l'extraction de features depuis le loader")

        test_features, test_labels = extract_features_from_loader(
            dataloader=test_loader,
            pretrained_path="D:/CROMA_weights/CROMA_base.pt",
            device=device,
            image_resolution=image_resolution,
            method=interpolation_mtd,
            modality=modality,
            feature_type="GAP",
            cache_dir="cache/",
            cache_name="test",
            force_recompute=force_recompute
        )
        print("Extraction terminée, features sauvegardées dans cache/")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    start_time = time.time()
    # script(False, "nearest", image_resolution=120, modality="both", force_recompute=True)
    script(True, "nearest", image_resolution=120,
           modality="both", force_recompute=True)
    # script(True, "bilinear", image_resolution=120,
    #       modality="both", force_recompute=True)
    # script(False, "bilinear", image_resolution=120,
    #       modality="both", force_recompute=True)
    end_time = time.time() - start_time
    print(
        f"Temps pour faire tourner tous les scripts en minutes : {end_time/60}")
