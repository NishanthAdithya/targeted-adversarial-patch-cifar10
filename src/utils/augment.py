import torch
import random

def random_position(H, W, ph, pw):
    """Return a valid random (y, x) position for placing the patch."""
    y = random.randint(0, H - ph)
    x = random.randint(0, W - pw)
    return y, x

def apply_patch_to_batch(images, patch):
    """
    images: (B, 3, H, W)
    patch:  (3, ph, pw)
    returns patched images of same size (B, 3, H, W)
    """
    B, C, H, W = images.shape
    _, ph, pw = patch.shape

    patched = images.clone()

    for i in range(B):
        y, x = random_position(H, W, ph, pw)

        # apply patch only inside this region
        patched[i, :, y:y+ph, x:x+pw] = patch

    return patched

