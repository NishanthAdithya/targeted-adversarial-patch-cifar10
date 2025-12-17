import torch
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import *
from models.classifier import load_classifier
from attacks.patch_attack import AdversarialPatchAttack
from utils.misc import ensure_dir

def main():
    ensure_dir(CHECKPOINT_DIR)

    # SAME normalization as classifier!
    normalize = T.Normalize(
        mean=[0.4914,0.4822,0.4465],
        std=[0.2023,0.1994,0.2010]
    )

    transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    train = dset.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )

    loader = DataLoader(train, batch_size=128, shuffle=True)

    model = load_classifier(DEVICE, MODEL_CHECKPOINT)

    attack = AdversarialPatchAttack(
        model=model,
        target_class=TARGET_CLASS,
        patch_size=PATCH_SIZE,
        device=DEVICE,
        tv_lambda=TV_LAMBDA,
        lr=LEARNING_RATE,
        checkpoint_dir=CHECKPOINT_DIR
    )

    attack.train(
        loader,
        epochs=NUM_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        patch_tensor_path=f"{CHECKPOINT_DIR}/patch.pt",
        patch_image_path=f"{CHECKPOINT_DIR}/patch.png",
    )

if __name__ == "__main__":
    main()

