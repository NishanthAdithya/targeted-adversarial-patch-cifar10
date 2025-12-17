import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from models.classifier import load_classifier
from utils.augment import apply_patch_to_batch
from config import DEVICE, MODEL_CHECKPOINT, TARGET_CLASS, DATA_ROOT

# ---------------------------
# Utility functions
# ---------------------------

def unnormalize(t):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)

def save_image(tensor, path):
    img = unnormalize(tensor).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def save_grid(imgs, filename, rows=5, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            axes[r][c].imshow(imgs[idx])
            axes[r][c].axis("off")
            idx += 1
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_curve(data, name, ylabel):
    plt.figure(figsize=(6,4))
    plt.plot(data, marker='o')
    plt.xlabel("Batch Index")
    plt.ylabel(ylabel)
    plt.title(name)
    plt.savefig(f"./evaluation/{name}.png")
    plt.close()

def confusion_matrix(preds, labels, name):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds, labels=list(range(10)))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(name)
    plt.savefig(f"./evaluation/{name}.png")
    plt.close()

# ---------------------------
# MAIN EVALUATION PIPELINE
# ---------------------------

def main():
    print("Loading classifier...")
    model = load_classifier(DEVICE, MODEL_CHECKPOINT)
    model.eval()

    print("Loading patch...")
    patch = torch.load("./checkpoints/patch.pt", map_location=DEVICE)

    normalize = T.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    test_set = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )
    loader = DataLoader(test_set, batch_size=128, shuffle=False)

    os.makedirs("evaluation/images", exist_ok=True)
    os.makedirs("evaluation/grids", exist_ok=True)

    total = 0
    clean_preds_all = []
    patched_preds_all = []
    labels_all = []

    success_batches = []
    patched_accuracy_batches = []

    sample_originals = []
    sample_patched = []

    print("Evaluating...")

    for batch_idx, (images, labels) in enumerate(tqdm(loader)):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # CLEAN prediction
        clean_logits = model(images)
        clean_preds = clean_logits.argmax(1)

        # APPLY PATCH
        patched_images = apply_patch_to_batch(images, patch)
        patched_logits = model(patched_images)
        patched_preds = patched_logits.argmax(1)

        # Logging
        labels_all.extend(labels.cpu().numpy())
        clean_preds_all.extend(clean_preds.cpu().numpy())
        patched_preds_all.extend(patched_preds.cpu().numpy())

        total += labels.size(0)

        batch_success = (patched_preds == TARGET_CLASS).float().mean().item()
        batch_acc = (patched_preds == labels).float().mean().item()

        success_batches.append(batch_success)
        patched_accuracy_batches.append(batch_acc)

        # SAVE EXAMPLE IMAGES
        if len(sample_originals) < 50:  # save 50 images for report
            for i in range(min(10, images.size(0))):
                sample_originals.append(unnormalize(images[i]).permute(1,2,0).cpu().numpy())
                sample_patched.append(unnormalize(patched_images[i]).permute(1,2,0).cpu().numpy())

    # ---------------------------
    # SAVE SAMPLE GRIDS
    # ---------------------------
    save_grid(sample_originals[:25], "./evaluation/grids/original_grid.png")
    save_grid(sample_patched[:25], "./evaluation/grids/patched_grid.png")

    # ---------------------------
    # METRICS
    # ---------------------------
    clean_acc = (np.array(clean_preds_all) == np.array(labels_all)).mean() * 100
    patched_acc = (np.array(patched_preds_all) == np.array(labels_all)).mean() * 100
    targeted_success = (np.array(patched_preds_all) == TARGET_CLASS).mean() * 100

    print("\n================= RESULTS =================")
    print(f"Clean Accuracy (no patch):       {clean_acc:.2f}%")
    print(f"Accuracy on Patched Images:      {patched_acc:.2f}%")
    print(f"Targeted Attack Success Rate:    {targeted_success:.2f}%")
    print("===========================================\n")

    # ---------------------------
    # PLOTS
    # ---------------------------
    plot_curve(success_batches, "batch_success_rate", "Success %")
    plot_curve(patched_accuracy_batches, "patched_accuracy", "Accuracy")

    # ---------------------------
    # CONFUSION MATRIX
    # ---------------------------
    confusion_matrix(clean_preds_all, labels_all, "confusion_clean")
    confusion_matrix(patched_preds_all, labels_all, "confusion_patched")

    print("Saved all results to ./evaluation/")
    print("You can now use these figures in your report.")

if __name__ == "__main__":
    main()
