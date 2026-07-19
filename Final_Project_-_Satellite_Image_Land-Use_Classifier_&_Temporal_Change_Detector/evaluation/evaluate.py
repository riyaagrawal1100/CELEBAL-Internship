"""
Full evaluation of a trained checkpoint on EuroSAT val and UC Merced holdout.

Usage:
    python evaluation/evaluate.py --checkpoint checkpoints/transfer_resnet18.pt \
        --data-root ./data/raw/EuroSAT --ucm-root ./data/raw/UCMerced_LandUse
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import (EuroSATDataset, UCMercedHoldout, spatial_block_split,
                           default_eval_transform, EUROSAT_CLASSES)
from data.download_eurosat import UCM_TO_EUROSAT_MAP
from models.transfer_model import TransferModel
from evaluation.metrics import per_class_f1, macro_f1, plot_confusion_matrix
from training.utils import get_device, run_epoch


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone = ckpt.get("backbone", "resnet18")
    classes = ckpt.get("classes", EUROSAT_CLASSES)
    model = TransferModel(backbone, num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ucm-root", default=None)
    parser.add_argument("--out-dir", default="../outputs")
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    model, classes = load_model(args.checkpoint, device)
    criterion = nn.CrossEntropyLoss()

    splits = spatial_block_split(args.data_root)
    val_ds = EuroSATDataset(splits["val"], transform=default_eval_transform())
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    result = run_epoch(model, val_loader, criterion, None, device)

    print("=== EuroSAT val ===")
    print(f"Accuracy: {result['acc']:.4f} | Macro-F1: {macro_f1(result['labels'], result['preds']):.4f}")
    for cls, f1 in per_class_f1(result["labels"], result["preds"], classes).items():
        print(f"  {cls:25s} {f1:.3f}")
    plot_confusion_matrix(result["labels"], result["preds"], classes,
                           os.path.join(args.out_dir, "confusion_matrix_eurosat_eval.png"))

    if args.ucm_root:
        ucm_ds = UCMercedHoldout(args.ucm_root, UCM_TO_EUROSAT_MAP, transform=default_eval_transform())
        ucm_loader = DataLoader(ucm_ds, batch_size=64, shuffle=False, num_workers=2)
        ucm_result = run_epoch(model, ucm_loader, criterion, None, device)
        print("\n=== UC Merced holdout ===")
        print(f"Accuracy: {ucm_result['acc']:.4f} | "
              f"Macro-F1: {macro_f1(ucm_result['labels'], ucm_result['preds']):.4f}")
        for cls, f1 in per_class_f1(ucm_result["labels"], ucm_result["preds"], classes).items():
            print(f"  {cls:25s} {f1:.3f}")
        plot_confusion_matrix(ucm_result["labels"], ucm_result["preds"], classes,
                               os.path.join(args.out_dir, "confusion_matrix_ucmerced_eval.png"))


if __name__ == "__main__":
    main()
