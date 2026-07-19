"""
Trains the scratch 3-layer CNN baseline.

Usage:
    python training/train_baseline.py --data-root ./data/raw/EuroSAT --epochs 15
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import (EuroSATDataset, spatial_block_split,
                           default_train_transform, default_eval_transform, EUROSAT_CLASSES)
from models.baseline_cnn import BaselineCNN
from evaluation.metrics import per_class_f1, macro_f1
from training.utils import set_seed, get_device, run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="../outputs")
    parser.add_argument("--checkpoint-dir", default="../checkpoints")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    splits = spatial_block_split(args.data_root)
    train_ds = EuroSATDataset(splits["train"], transform=default_train_transform())
    val_ds = EuroSATDataset(splits["val"], transform=default_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = BaselineCNN(num_classes=len(EUROSAT_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device)
        va = run_epoch(model, val_loader, criterion, None, device)
        train_losses.append(tr["loss"])
        val_losses.append(va["loss"])
        print(f"Epoch {epoch}/{args.epochs} | train_loss={tr['loss']:.4f} acc={tr['acc']:.4f} "
              f"| val_loss={va['loss']:.4f} acc={va['acc']:.4f}")

    f1s = per_class_f1(va["labels"], va["preds"], EUROSAT_CLASSES)
    print("\nPer-class F1 (val):")
    for cls, f1 in f1s.items():
        print(f"  {cls:25s} {f1:.3f}")
    print(f"Macro-F1: {macro_f1(va['labels'], va['preds']):.3f}")

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("Baseline CNN loss curves")
    plt.savefig(os.path.join(args.out_dir, "baseline_loss_curve.png"), dpi=150)

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "baseline_cnn.pt"))
    print(f"\nSaved checkpoint to {args.checkpoint_dir}/baseline_cnn.pt")


if __name__ == "__main__":
    main()
