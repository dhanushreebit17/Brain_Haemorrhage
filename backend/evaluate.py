import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CLASS_NAMES, load_trained_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224


def get_val_loader(data_dir: str, batch_size: int = 32):
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    val_ds = datasets.ImageFolder(f"{data_dir}/val", transform=val_tf)

    if val_ds.classes != CLASS_NAMES:
        print(f"[evaluate] WARNING: folder classes {val_ds.classes} don't match "
              f"model.py CLASS_NAMES {CLASS_NAMES}. Results will be mislabeled "
              "unless these match exactly.")

    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2), val_ds.classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--weights", type=str, default="models/haemorrhage_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print(f"[evaluate] Loading model from {args.weights} on {DEVICE}")
    model = load_trained_model(args.weights, device=DEVICE)

    val_loader, class_names = get_val_loader(args.data_dir, args.batch_size)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n[evaluate] Overall accuracy: {acc:.4f}\n")

    print("[evaluate] Per-class precision / recall / F1:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("[evaluate] Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    # Save confusion matrix as an image
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (accuracy={acc:.3f})")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("[evaluate] Saved confusion_matrix.png")


if __name__ == "__main__":
    main()