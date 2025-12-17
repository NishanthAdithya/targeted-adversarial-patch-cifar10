import torch
import torch.nn as nn
import torchvision.models as models

class CIFAR10ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18(weights=None)

        # CIFAR-10 modifications
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


def load_classifier(device="cpu", ckpt_path=None):
    model = CIFAR10ResNet18()
    if ckpt_path is not None:
        print(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
    return model.to(device).eval()

