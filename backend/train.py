"""
train.py
Fine-tunes HaemorrhageNet on your dataset.

Expected data layout (ImageFolder style) — produced automatically by
prepare_dataset.py if you're using the vbookshelf CT-ICH Kaggle dataset:

    dataset/
        train/
            Normal/            *.jpg / *.png
            Hemorrhage/
        val/
            Normal/
            Hemorrhage/

(If you switch model.py to full subtype classification later, just add
the matching subfolders — this script is agnostic to class count.)

Usage:
    python train.py --data_dir dataset --epochs 15 --batch_size 16
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CLASS_NAMES, build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224


def get_dataloaders(data_dir: str, batch_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)

    print(f"[train] Detected classes (must match model.py CLASS_NAMES order): "
          f"{train_ds.classes}")
    print(f"[train] model.py CLASS_NAMES:                                    "
          f"{CLASS_NAMES}")
    if train_ds.classes != CLASS_NAMES:
        print("[train] WARNING: folder class names/order don't match "
              "model.py's CLASS_NAMES exactly. Predictions at inference "
              "time will be mislabeled unless these line up. Either rename "
              "your dataset folders or edit CLASS_NAMES in model.py.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def compute_class_weights(train_loader) -> torch.Tensor:
    """
    With a small, imbalanced dataset (e.g. ~2k CT slices where 'Normal'
    slices usually outnumber 'Hemorrhage' slices), weighting the loss by
    inverse class frequency stops the model from just predicting the
    majority class every time.
    """
    counts = torch.zeros(len(CLASS_NAMES))
    for _, labels in train_loader:
        for c in range(len(CLASS_NAMES)):
            counts[c] += (labels == c).sum()
    counts = torch.clamp(counts, min=1)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    print(f"[train] Class counts:  {dict(zip(CLASS_NAMES, counts.tolist()))}")
    print(f"[train] Class weights: {dict(zip(CLASS_NAMES, weights.tolist()))}")
    return weights


def train(data_dir: str, epochs: int, batch_size: int, lr: float, out_path: str):
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)

    model = build_model(pretrained=True).to(DEVICE)
    class_weights = compute_class_weights(train_loader).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    for epoch in range(epochs):
        start = time.time()

        # ---- train ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- validate ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        # Per-class recall is more informative than raw accuracy on an
        # imbalanced medical dataset — track it too.
        class_correct = torch.zeros(len(CLASS_NAMES))
        class_total = torch.zeros(len(CLASS_NAMES))
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                for c in range(len(CLASS_NAMES)):
                    mask = labels == c
                    class_total[c] += mask.sum().item()
                    class_correct[c] += (preds[mask] == c).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        per_class_recall = {
            CLASS_NAMES[c]: (class_correct[c] / class_total[c]).item() if class_total[c] > 0 else float("nan")
            for c in range(len(CLASS_NAMES))
        }

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"recall={per_class_recall} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_path)
            print(f"  -> saved new best model to {out_path} (val_acc={val_acc:.4f})")

    print(f"[train] Done. Best val_acc={best_val_acc:.4f}. Weights at {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_path", type=str, default="models/haemorrhage_model.pth")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.out_path)
