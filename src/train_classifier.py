import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.classifier import CIFAR10ResNet18
from utils.misc import ensure_dir
from config import DATA_ROOT, CHECKPOINT_DIR, MODEL_CHECKPOINT, DEVICE

def get_loaders():
    normalize = T.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )

    train_t = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    test_t = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    train = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=train_t
    )
    test = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=test_t
    )

    return (
        DataLoader(train, batch_size=128, shuffle=True, num_workers=4),
        DataLoader(test, batch_size=128, shuffle=False, num_workers=4)
    )

def eval(model, loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    ensure_dir(CHECKPOINT_DIR)

    train_loader, test_loader = get_loaders()
    model = CIFAR10ResNet18().to(DEVICE)

    optimz = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
        for xb, yb in pbar:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimz.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optimz.step()
            pbar.set_postfix(loss=float(loss))

        acc = eval(model, test_loader)
        print(f"ACC = {acc*100:.2f}%")

        torch.save(model.state_dict(), MODEL_CHECKPOINT)
        print("Saved checkpoint.")

if __name__ == "__main__":
    main()

