import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """One train epoch if optimizer is given, else one eval epoch."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []

    torch.set_grad_enabled(is_train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train:
            optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        n += x.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    return {
        "loss": total_loss / max(n, 1),
        "acc": correct / max(n, 1),
        "preds": all_preds,
        "labels": all_labels,
    }


class EarlyStopper:
    def __init__(self, patience=5, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.count = 0

    def step(self, value):
        if self.best is None or (self.mode == "max" and value > self.best) or \
           (self.mode == "min" and value < self.best):
            self.best = value
            self.count = 0
            return False  # not stopping, improved
        self.count += 1
        return self.count >= self.patience
