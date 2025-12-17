import torch
import torchvision.utils as vutils
from pathlib import Path

def save_patch(patch: torch.Tensor, out_path: str):
    """
    Save patch tensor (C, H, W) as an image.
    Patch values must be in [0,1].
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Add batch dimension so save_image works
    vutils.save_image(patch.unsqueeze(0), out_path)

def save_image_grid(images: torch.Tensor, out_path: str, nrow: int = 8, normalize: bool = True):
    """
    Save a grid of images.
    images should be a (B, C, H, W) tensor.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(images, out_path, nrow=nrow, normalize=normalize)

