import argparse
import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import (
    DEVICE,
    DATA_ROOT,
    BATCH_SIZE,
    NUM_WORKERS,
    PATCH_TENSOR_PATH,
    CHECKPOINT_DIR,
    MODEL_CHECKPOINT,
    TARGET_CLASS,
)
from models.classifier import load_classifier
from utils.augment import apply_patch_to_batch
from utils.visualization import save_image_grid
from utils.misc import ensure_dir

def get_test_loader(data_root: str, batch_size: int, num_workers: int):
    transform = T.Compose([
        T.ToTensor(),
    ])
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_loader

def main(num_images: int):
    ensure_dir(CHECKPOINT_DIR)

    # Load model
    model = load_classifier(DEVICE, ckpt_path=MODEL_CHECKPOINT if os.path.exists(MODEL_CHECKPOINT) else None)
    model.eval()

    # Load patch
    patch = torch.load(PATCH_TENSOR_PATH, map_location=DEVICE)
    patch = patch.to(DEVICE)

    # Data
    test_loader = get_test_loader(DATA_ROOT, BATCH_SIZE, NUM_WORKERS)

    all_clean = []
    all_patched = []
    total = 0
    success = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            patched_images = apply_patch_to_batch(images, patch)

            logits_clean = model(images)
            logits_patched = model(patched_images)

            pred_clean = logits_clean.argmax(dim=1)
            pred_patched = logits_patched.argmax(dim=1)

            # Targeted success: patched prediction == target_class
            success += (pred_patched == TARGET_CLASS).sum().item()
            total += images.size(0)

            if len(all_clean) * images.size(0) < num_images:
                all_clean.append(images.cpu())
                all_patched.append(patched_images.cpu())

            if total >= num_images:
                break

    fooling_rate = success / total
    print(f"Targeted fooling rate on first {total} images: {fooling_rate*100:.2f}%")

    # Save grid for visual inspection
    clean_batch = torch.cat(all_clean, dim=0)[:num_images]
    patched_batch = torch.cat(all_patched, dim=0)[:num_images]

    save_image_grid(clean_batch, f"{CHECKPOINT_DIR}/clean_samples.png", nrow=8)
    save_image_grid(patched_batch, f"{CHECKPOINT_DIR}/patched_samples.png", nrow=8)
    print(f"Saved clean and patched sample grids to {CHECKPOINT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=32)
    args = parser.parse_args()
    main(num_images=args.num_images)

