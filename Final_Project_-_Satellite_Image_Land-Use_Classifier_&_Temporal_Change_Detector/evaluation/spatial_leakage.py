"""
Quantifies the spatial-leakage gap: trains the same model under a naive
random split vs. the spatial-block split and compares test accuracy.

Usage:
    python evaluation/spatial_leakage.py --data-root ./data/raw/EuroSAT --epochs 5
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import (EuroSATDataset, spatial_block_split, random_split_baseline,
                           default_train_transform, default_eval_transform, EUROSAT_CLASSES)
from models.baseline_cnn import BaselineCNN
from training.utils import set_seed, get_device, run_epoch


def train_and_eval(splits, epochs, device, batch_size=64):
    train_ds = EuroSATDataset(splits["train"], transform=default_train_transform())
    test_ds = EuroSATDataset(splits["test"], transform=default_eval_transform())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = BaselineCNN(num_classes=len(EUROSAT_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        run_epoch(model, train_loader, criterion, optimizer, device)

    result = run_epoch(model, test_loader, criterion, None, device)
    return result["acc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out-dir", default="../outputs")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    random_splits = random_split_baseline(args.data_root)
    block_splits = spatial_block_split(args.data_root)

    random_acc = train_and_eval(random_splits, args.epochs, device)
    block_acc = train_and_eval(block_splits, args.epochs, device)

    gap = random_acc - block_acc
    report = f"""# Spatial Leakage Experiment

| Split strategy | Test accuracy |
|---|---|
| Naive random split (file-level) | {random_acc:.4f} |
| Spatial block split | {block_acc:.4f} |

**Gap: {gap:+.4f} ({gap*100:+.2f} pts)**

## Explanation

EuroSAT tiles are cropped from contiguous Sentinel-2 scenes, so tiles that are
spatially adjacent are visually near-duplicates (same field boundary, same
rooftop cluster, same illumination/season). A random file-level split places
some of these near-duplicate neighbours in the training set and their
almost-identical twins in the test set, letting the model "memorize" local
patterns rather than generalise. This inflates the reported test accuracy
relative to how the model would perform on genuinely unseen geography.

The spatial block split groups tiles into blocks (derived from their file
index, a proxy for scene locality) and assigns *whole blocks* to train/val/test,
which prevents this leakage and gives a more honest performance estimate —
this is the split used for every other experiment in this project.
"""
    out_path = os.path.join(args.out_dir, "spatial_leakage_report.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(report)
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
