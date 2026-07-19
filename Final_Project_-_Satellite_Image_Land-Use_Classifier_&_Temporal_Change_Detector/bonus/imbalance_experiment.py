"""
Bonus D: Downsample two classes to 20% of their size, retrain, compare F1.
Apply one mitigation (weighted loss, oversampling, or Mixup) and compare.

Usage:
    python bonus/imbalance_experiment.py --data-root ./data/raw/EuroSAT \
        --classes Highway River --mitigation weighted_loss --epochs 8
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import (EuroSATDataset, spatial_block_split, default_train_transform,
                           default_eval_transform, EUROSAT_CLASSES)
from models.baseline_cnn import BaselineCNN
from evaluation.metrics import per_class_f1, macro_f1
from training.utils import set_seed, get_device, run_epoch


def downsample_classes(samples, target_classes, fraction=0.2, seed=42):
    rng = random.Random(seed)
    target_idx = {EUROSAT_CLASSES.index(c) for c in target_classes}
    by_label = {}
    for path, label in samples:
        by_label.setdefault(label, []).append((path, label))

    new_samples = []
    for label, items in by_label.items():
        if label in target_idx:
            k = max(1, int(len(items) * fraction))
            new_samples.extend(rng.sample(items, k))
        else:
            new_samples.extend(items)
    rng.shuffle(new_samples)
    return new_samples


def mixup(x, y, num_classes, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return mixed_x, mixed_y


def train_with_mitigation(train_samples, val_ds, mitigation, epochs, device, num_classes):
    train_ds = EuroSATDataset(train_samples, transform=default_train_transform())
    labels = [l for _, l in train_samples]

    if mitigation == "oversampling":
        counts = np.bincount(labels, minlength=num_classes)
        weights = 1.0 / np.maximum(counts[labels], 1)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    model = BaselineCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if mitigation == "weighted_loss":
        counts = np.bincount(labels, minlength=num_classes)
        class_weights = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if mitigation == "mixup":
                mixed_x, mixed_y = mixup(x, y, num_classes)
                logits = model(mixed_x)
                loss = -(mixed_y * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            else:
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    result = run_epoch(model, val_loader, nn.CrossEntropyLoss(), None, device)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--classes", nargs=2, default=["Highway", "River"],
                         help="Two classes to downsample to 20%%")
    parser.add_argument("--mitigation", choices=["weighted_loss", "oversampling", "mixup"],
                         default="weighted_loss")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--out", default="../outputs/imbalance_analysis.md")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    splits = spatial_block_split(args.data_root)
    val_ds = EuroSATDataset(splits["val"], transform=default_eval_transform())

    # --- Imbalanced, no mitigation ---
    imbalanced_train = downsample_classes(splits["train"], args.classes, fraction=0.2)
    no_mitigation = train_with_mitigation(imbalanced_train, val_ds, "none", args.epochs, device, len(EUROSAT_CLASSES))

    # --- Imbalanced, with mitigation ---
    with_mitigation = train_with_mitigation(imbalanced_train, val_ds, args.mitigation, args.epochs,
                                             device, len(EUROSAT_CLASSES))

    f1_no_mit = per_class_f1(no_mitigation["labels"], no_mitigation["preds"], EUROSAT_CLASSES)
    f1_with_mit = per_class_f1(with_mitigation["labels"], with_mitigation["preds"], EUROSAT_CLASSES)

    report = ["# Imbalance Experiment\n",
              f"Downsampled classes: **{args.classes}** to 20% of original size.\n",
              f"Mitigation applied: **{args.mitigation}**\n",
              "| Class | F1 (no mitigation) | F1 (with mitigation) |",
              "|---|---|---|"]
    for cls in EUROSAT_CLASSES:
        marker = " **(downsampled)**" if cls in args.classes else ""
        report.append(f"| {cls}{marker} | {f1_no_mit[cls]:.3f} | {f1_with_mit[cls]:.3f} |")
    report.append(f"\n**Macro-F1** — no mitigation: {macro_f1(no_mitigation['labels'], no_mitigation['preds']):.3f}, "
                  f"with mitigation: {macro_f1(with_mitigation['labels'], with_mitigation['preds']):.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(report))
    print("\n".join(report))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
