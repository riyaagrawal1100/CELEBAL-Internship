"""
Two-phase fine-tuning of ResNet-18 or EfficientNet-B0 on EuroSAT, then
evaluation on the UC Merced holdout.

Phase 1: freeze backbone, train classifier head only, 3 epochs.
Phase 2: unfreeze last 2 conv blocks, LR / 10, 5 more epochs.

Usage:
    python training/train_transfer.py --backbone resnet18 --data-root ./data/raw/EuroSAT \
        --ucm-root ./data/raw/UCMerced_LandUse --phase1-epochs 3 --phase2-epochs 5
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import (EuroSATDataset, UCMercedHoldout, spatial_block_split,
                           default_train_transform, default_eval_transform, EUROSAT_CLASSES)
from data.download_eurosat import UCM_TO_EUROSAT_MAP
from models.transfer_model import TransferModel
from evaluation.metrics import per_class_f1, macro_f1, plot_confusion_matrix
from training.utils import set_seed, get_device, run_epoch


def train_one_phase(model, loader, val_loader, criterion, optimizer, device, epochs, phase_name):
    history = []
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, loader, criterion, optimizer, device)
        va = run_epoch(model, val_loader, criterion, None, device)
        history.append(va)
        print(f"[{phase_name}] epoch {epoch}/{epochs} | train_loss={tr['loss']:.4f} "
              f"acc={tr['acc']:.4f} | val_loss={va['loss']:.4f} acc={va['acc']:.4f}")
    return history[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["resnet18", "efficientnet_b0"], default="resnet18")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ucm-root", default=None)
    parser.add_argument("--phase1-epochs", type=int, default=3)
    parser.add_argument("--phase2-epochs", type=int, default=5)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--phase2-lr-divisor", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=64)
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

    model = TransferModel(args.backbone, num_classes=len(EUROSAT_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()

    # ---------------- Phase 1: frozen backbone ----------------
    model.freeze_backbone()
    optimizer = torch.optim.Adam(model.trainable_param_groups(args.phase1_lr))
    frozen_val = train_one_phase(model, train_loader, val_loader, criterion, optimizer,
                                  device, args.phase1_epochs, "phase1-frozen")

    # ---------------- Phase 2: unfreeze last 2 blocks ----------------
    model.unfreeze_last_blocks()
    phase2_lr = args.phase1_lr / args.phase2_lr_divisor
    optimizer = torch.optim.Adam(model.trainable_param_groups(phase2_lr))
    unfrozen_val = train_one_phase(model, train_loader, val_loader, criterion, optimizer,
                                    device, args.phase2_epochs, "phase2-unfrozen")

    # ---------------- Ablation table: frozen vs unfrozen ----------------
    print("\n=== Frozen vs Unfrozen ablation (EuroSAT val) ===")
    print(f"{'Phase':<20}{'Val Acc':<12}{'Val Loss':<12}{'Macro-F1'}")
    print(f"{'Phase1 (frozen)':<20}{frozen_val['acc']:<12.4f}{frozen_val['loss']:<12.4f}"
          f"{macro_f1(frozen_val['labels'], frozen_val['preds']):.4f}")
    print(f"{'Phase2 (unfrozen)':<20}{unfrozen_val['acc']:<12.4f}{unfrozen_val['loss']:<12.4f}"
          f"{macro_f1(unfrozen_val['labels'], unfrozen_val['preds']):.4f}")

    # ---------------- Full report on EuroSAT val ----------------
    print("\nPer-class F1 (EuroSAT val, final model):")
    for cls, f1 in per_class_f1(unfrozen_val["labels"], unfrozen_val["preds"], EUROSAT_CLASSES).items():
        print(f"  {cls:25s} {f1:.3f}")
    plot_confusion_matrix(unfrozen_val["labels"], unfrozen_val["preds"], EUROSAT_CLASSES,
                           os.path.join(args.out_dir, f"confusion_matrix_eurosat_{args.backbone}.png"))

    # ---------------- Evaluate on UC Merced holdout ----------------
    if args.ucm_root:
        ucm_ds = UCMercedHoldout(args.ucm_root, UCM_TO_EUROSAT_MAP, transform=default_eval_transform())
        ucm_loader = DataLoader(ucm_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        ucm_result = run_epoch(model, ucm_loader, criterion, None, device)
        print(f"\nUC Merced holdout | acc={ucm_result['acc']:.4f} "
              f"macro-F1={macro_f1(ucm_result['labels'], ucm_result['preds']):.4f}")
        for cls, f1 in per_class_f1(ucm_result["labels"], ucm_result["preds"], EUROSAT_CLASSES).items():
            print(f"  {cls:25s} {f1:.3f}")
        plot_confusion_matrix(ucm_result["labels"], ucm_result["preds"], EUROSAT_CLASSES,
                               os.path.join(args.out_dir, f"confusion_matrix_ucmerced_{args.backbone}.png"))

    ckpt_path = os.path.join(args.checkpoint_dir, f"transfer_{args.backbone}.pt")
    torch.save({"state_dict": model.state_dict(), "backbone": args.backbone,
                "classes": EUROSAT_CLASSES}, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
