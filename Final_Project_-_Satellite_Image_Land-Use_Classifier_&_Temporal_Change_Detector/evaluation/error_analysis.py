"""
Finds the top-5 most-confident misclassifications and saves them with
predicted/true labels for visual inspection + a written hypothesis stub.

Usage:
    python evaluation/error_analysis.py --checkpoint checkpoints/transfer_resnet18.pt \
        --data-root ./data/raw/EuroSAT
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import EuroSATDataset, spatial_block_split, default_eval_transform
from evaluation.evaluate import load_model
from training.utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", default="../outputs/error_analysis")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    model, classes = load_model(args.checkpoint, device)

    splits = spatial_block_split(args.data_root)
    val_ds = EuroSATDataset(splits["val"], transform=default_eval_transform())
    loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    mistakes = []  # (confidence, true, pred, sample_idx)
    idx = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            for i in range(x.size(0)):
                if pred[i].item() != y[i].item():
                    mistakes.append((conf[i].item(), y[i].item(), pred[i].item(), idx))
                idx += 1

    mistakes.sort(key=lambda m: -m[0])  # most confident mistakes first
    top = mistakes[: args.top_k]

    fig, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 4))
    if len(top) == 1:
        axes = [axes]
    for ax, (conf, true_label, pred_label, sample_idx) in zip(axes, top):
        img_path, _ = val_ds.samples[sample_idx]
        from PIL import Image
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f"true={classes[true_label]}\npred={classes[pred_label]}\nconf={conf:.2f}",
                     fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "top5_misclassified.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    with open(os.path.join(args.out_dir, "hypotheses.md"), "w") as f:
        f.write("# Top-5 Misclassified Pairs — Failure Hypotheses\n\n")
        for i, (conf, true_label, pred_label, sample_idx) in enumerate(top, 1):
            f.write(f"## {i}. True: {classes[true_label]} -> Predicted: {classes[pred_label]} "
                    f"(confidence {conf:.2f})\n")
            f.write("- Hypothesis: _fill in after visual inspection — common causes include "
                    "visually similar textures (e.g. Pasture vs HerbaceousVegetation), "
                    "seasonal colour shifts, or partial occlusion by cloud/shadow._\n\n")
    print(f"Saved hypothesis stub to {args.out_dir}/hypotheses.md")


if __name__ == "__main__":
    main()
