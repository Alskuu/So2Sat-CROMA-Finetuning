import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)
import torch
from torch import nn  # provides classes and functions for building neural networks
# loading datasets and helps in managing data parallelism
from torch.utils.data import DataLoader, TensorDataset
import time
from typing import Optional


from probes import CROMALinearProbe, CROMAMLProbe


def probing(train: bool, linear: bool, interpolation_mtd: Optional[str], image_resolution: int, modality: str):
    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_path = "D:/CROMA_weights/CROMA_base.pt"

    if train:
        # Charger les features déjà extraites
        if interpolation_mtd is not None:
            train_payload = torch.load(
                f"cache/train_{modality}_{interpolation_mtd}_{image_resolution}.pt", map_location="cpu")
            val_payload = torch.load(
                f"cache/val_{modality}_{interpolation_mtd}_{image_resolution}.pt",   map_location="cpu")
        else:
            train_payload = torch.load(
                f"cache/train_{modality}_{image_resolution}.pt", map_location="cpu")
            val_payload = torch.load(
                f"cache/val_{modality}_{image_resolution}.pt", map_location="cpu")

        train_features = train_payload["features"]  # (N, 768)
        train_labels = train_payload["labels"]    # (N,)
        val_features = val_payload["features"]
        val_labels = val_payload["labels"]

        print(train_features.shape)

        # DataLoaders légers, num_workers=0, tout en RAM
        train_features_loader = DataLoader(
            TensorDataset(train_features, train_labels),
            batch_size=512,   # on peut augmenter, c'est juste des vecteurs
            shuffle=True,
            num_workers=0,
        )
        val_features_loader = DataLoader(
            TensorDataset(val_features, val_labels),
            batch_size=512,
            shuffle=False,
            num_workers=0,
        )
        if linear:
            probe = CROMALinearProbe(
                pretrained_path=pretrained_path,
                modality="both",
                joint=True,
                num_classes=17,
            ).to(device)
        else:
            probe = CROMAMLProbe(
                pretrained_path=pretrained_path, joint=True).to(device)

        optimizer = torch.optim.Adam(probe.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience = 30  # nombre d'epochs sans amélioration avant d'arrêter
        epochs_without_improvement = 0

        epoch = 0
        epsilon = 0
        while epochs_without_improvement < patience:
            print("Training")
            train_loss = 0.0
            correct = 0
            total = 0
            # entraînement
            probe.train()
            for features, labels in train_features_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = probe(features)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

            train_loss /= len(train_features_loader)
            train_acc = correct / total

            # validation
            print("Validation")
            probe.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for features, labels in val_features_loader:
                    logits = probe(features.to(device))
                    val_loss += criterion(logits, labels.to(device)).item()
                    val_correct += (logits.argmax(dim=1) ==
                                    labels.to(device)).sum().item()
                    val_total += labels.size(0)
            val_loss /= len(val_features_loader)
            val_acc = val_correct / val_total

            print(
                f"Epoch {epoch:3d} | "
                f"train loss: {train_loss:.4f} | train acc: {train_acc:.3f} | "
                f"val loss: {val_loss:.4f} | val acc: {val_acc:.3f} | "
                f"patience: {epochs_without_improvement}/{patience}"
                + (" validé : best" if val_loss < best_val_loss else "")
            )

            if val_loss < best_val_loss - epsilon:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                print("Meilleur modèle sauvegardé !")
                if linear:
                    if interpolation_mtd is not None:
                        torch.save(probe.classifier.state_dict(),
                                   f"best_linear_probe_{modality}_{interpolation_mtd}_{image_resolution}.pt")
                    else:
                        torch.save(probe.classifier.state_dict(),
                                   f"best_linear_probe_{modality}_{image_resolution}.pt")
                else:
                    if interpolation_mtd is not None:
                        torch.save(probe.classifier.state_dict(),
                                   f"best_non_linear_probe_{modality}_{interpolation_mtd}_{image_resolution}.pt")
                    else:
                        torch.save(probe.classifier.state_dict(),
                                   f"best_non_linear_probe_{modality}_{image_resolution}.pt")
            else:
                epochs_without_improvement += 1

            epoch += 1

        print(
            f"\nEntraînement terminé — {epoch} epochs | meilleure val loss: {best_val_loss:.4f}")

    else:
        if interpolation_mtd is not None:
            test_payload = torch.load(
                f"cache/test_{modality}_{interpolation_mtd}_{image_resolution}.pt", map_location="cpu")
        else:
            test_payload = torch.load(
                f"cache/test_{modality}_{image_resolution}.pt", map_location="cpu")
        test_features = test_payload["features"]  # (N, 768)
        test_labels = test_payload["labels"]    # (N,)
        # DataLoaders légers, num_workers=0, tout en RAM
        test_features_loader = DataLoader(
            TensorDataset(test_features, test_labels),
            batch_size=512,   # tu peux augmenter, c'est juste des vecteurs
            shuffle=True,
            num_workers=0,
        )
        # ── Chargement du meilleur modèle ────────────────────────────────────
        if linear:
            probe = CROMALinearProbe(pretrained_path=pretrained_path,
                                     modality="both",
                                     joint=True,
                                     num_classes=17,).to(device)
            if interpolation_mtd is not None:
                probe.classifier.load_state_dict(torch.load(
                    f"best_linear_probe_{modality}_{interpolation_mtd}_{image_resolution}.pt", map_location=device))
            else:
                probe.classifier.load_state_dict(torch.load(
                    f"best_linear_probe_{modality}_{image_resolution}.pt", map_location=device))
        else:
            probe = CROMAMLProbe(
                pretrained_path=pretrained_path, joint=True).to(device)
            if interpolation_mtd is not None:
                probe.classifier.load_state_dict(torch.load(
                    f"best_non_linear_probe_{modality}_{interpolation_mtd}_{image_resolution}.pt", map_location=device))
            else:
                probe.classifier.load_state_dict(torch.load(
                    f"best_non_linear_probe_{modality}_{image_resolution}.pt", map_location=device))
        probe.eval()

        # ── Inférence ────────────────────────────────────────────────────────
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels in test_features_loader:
                features = features.to(device)
                logits = probe(features)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        # ── Métriques ────────────────────────────────────────────────────────
        oa = accuracy_score(y_true, y_pred)
        aa = np.diag(confusion_matrix(y_true, y_pred)) / \
            confusion_matrix(y_true, y_pred).sum(axis=1)
        aa = aa.mean()
        macro_f1 = f1_score(y_true, y_pred, average="macro")

        if linear:
            if interpolation_mtd is not None:
                print(
                    f"Résultats du test pour {image_resolution} pixels avec la méthode {interpolation_mtd} en linear probing")
            else:
                print(
                    f"Résultats du test pour {image_resolution} pixels en linear probing")
        else:
            if interpolation_mtd is not None:
                print(
                    f"Résultats du test pour {image_resolution} pixels avec la méthode {interpolation_mtd} en non linear probing")
            else:
                print(
                    f"Résultats du test pour {image_resolution} pixels en non linear probing")

        print("\n══════════════════════════════════════════")
        print("        Résultats — Test set")
        print("══════════════════════════════════════════")
        print(f"  Overall Accuracy (OA)  : {oa:.4f} ({oa*100:.2f}%)")
        print(f"  Average Accuracy (AA)  : {aa:.4f} ({aa*100:.2f}%)")
        print(f"  Macro F1               : {macro_f1:.4f}")
        print("══════════════════════════════════════════\n")
        cm = confusion_matrix(y_true, y_pred)
        per_class = np.diag(cm) / cm.sum(axis=1)

        print("Accuracy par classe :")
        for i, acc in enumerate(per_class):
            print(f"  LCZ {i+1:2d} : {acc:.4f} ({acc*100:.2f}%)")

    total_time = time.time() - start_time

    print(f"Temps total en minutes : {total_time/60}")


if __name__ == "__main__":
    print("On commence par le train : \n")
    probing(train=True, linear=True, interpolation_mtd="nearest",
            image_resolution=120, modality="both")
    print("Passage au test :\n")
    probing(train=False, linear=True,
            interpolation_mtd="nearest", image_resolution=120, modality="both")
