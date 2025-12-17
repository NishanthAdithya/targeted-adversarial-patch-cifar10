import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import random
from tqdm import tqdm
from utils.augment import apply_patch_to_batch
from utils.misc import ensure_dir

class AdversarialPatchAttack:
    def __init__(self, model, target_class, patch_size, device,
                 tv_lambda=0.0, lr=0.001, checkpoint_dir="./checkpoints"):

        self.model = model.eval()
        self.device = device
        self.target_class = target_class
        self.tv_lambda = tv_lambda
        self.checkpoint_dir = checkpoint_dir

        # Initialize patch
        patch = torch.rand(3, patch_size, patch_size, device=device)
        self.patch = nn.Parameter(patch)

        # Optimizer
        self.optimizer = torch.optim.Adam([self.patch], lr=lr)

        ensure_dir(checkpoint_dir)

        # CIFAR normalization
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(3,1,1)
        self.std  = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(3,1,1)

    # ------------------------------------
    # PATCH TRANSFORMATION (EoT)
    # ------------------------------------
    def transform_patch(self, patch):
        # Random rotation
        angle = random.uniform(-20, 20)
        patch = TF.rotate(patch, angle)

        # Random scaling
        scale = random.uniform(0.8, 1.3)
        new_size = int(patch.shape[1] * scale)
        new_size = max(4, min(new_size, 32))  # safe bounds
        patch = TF.resize(patch, (new_size, new_size))

        # Random brightness
        brightness_factor = random.uniform(0.7, 1.3)
        patch = patch * brightness_factor
        patch = patch.clamp(0, 1)

        return patch

    # ------------------------------------
    # TOTAL VARIATION REGULARIZATION
    # ------------------------------------
    def total_variation(self, p):
        return (
            torch.sum(torch.abs(p[:, :, :-1] - p[:, :, 1:])) +
            torch.sum(torch.abs(p[:, :-1, :] - p[:, 1:, :]))
        )

    # ------------------------------------
    # TRAINING LOOP
    # ------------------------------------
    def train(self, loader, epochs, steps_per_epoch,
              patch_tensor_path, patch_image_path):

        for ep in range(epochs):
            pbar = tqdm(loader, total=steps_per_epoch,
                        desc=f"EP {ep+1}/{epochs}")

            for i, (images, _) in enumerate(pbar):
                if i >= steps_per_epoch:
                    break

                images = images.to(self.device)

                # Target class vector
                target = torch.full(
                    (images.size(0),),
                    self.target_class,
                    dtype=torch.long,
                    device=self.device
                )

                # Apply EoT transformation to patch
                transformed_patch = self.transform_patch(self.patch)

                # Patch the images
                patched_images = apply_patch_to_batch(images, transformed_patch)

                # Normalize patched images
                patched_images = (patched_images - self.mean) / self.std

                # Model prediction
                logits = self.model(patched_images)
                ce_loss = nn.CrossEntropyLoss()(logits, target)

                # Optional TV regularization
                if self.tv_lambda > 0:
                    tv = self.total_variation(self.patch)
                    loss = ce_loss + self.tv_lambda * tv
                else:
                    loss = ce_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.patch.data.clamp_(0, 1)

                # Success calculation
                preds = logits.argmax(1)
                success = (preds == target).float().mean().item()

                pbar.set_postfix(
                    loss=float(loss),
                    success=f"{success*100:.1f}%"
                )

            # Save the patch
            torch.save(self.patch.detach().cpu(), patch_tensor_path)

            from utils.visualization import save_patch
            save_patch(self.patch.detach().cpu(), patch_image_path)

