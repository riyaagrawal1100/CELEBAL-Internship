"""
Cosine-similarity based change detector.

- Computes cosine similarity per region pair (T1 vs T2 embedding).
- Plots an ROC curve treating simulated "changed" labels as ground truth,
  using (1 - cosine_similarity) as the change score.
- Selects an operating threshold via Youden's J statistic and prints a
  written justification.
- Renders a side-by-side visual "change heatmap" for at least 5 sample
  region pairs: the two tiles plus a per-patch similarity heatmap.

Usage:
    python change_detection/change_detector.py --embeddings ../outputs/region_embeddings.pt
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from evaluation.metrics import plot_roc


def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def patch_similarity_heatmap(model, t1_path, t2_path, transform, device, grid=8):
    """Splits each tile into a grid x grid patch grid, embeds each patch
    independently, and returns a grid x grid cosine-similarity map — this
    gives a spatial sense of *where* change is concentrated within a tile,
    beyond the single whole-tile embedding score."""
    img1 = Image.open(t1_path).convert("RGB")
    img2 = Image.open(t2_path).convert("RGB")
    w, h = img1.size
    pw, ph = w // grid, h // grid

    sim_map = np.zeros((grid, grid))
    with torch.no_grad():
        for i in range(grid):
            for j in range(grid):
                box = (j * pw, i * ph, (j + 1) * pw, (i + 1) * ph)
                p1 = img1.resize((64, 64)).crop((0, 0, 64, 64)) if pw == 0 or ph == 0 else img1.crop(box)
                p2 = img2.crop(box) if pw > 0 and ph > 0 else img2.resize((64, 64))
                x1 = transform(p1.resize((64, 64))).unsqueeze(0).to(device)
                x2 = transform(p2.resize((64, 64))).unsqueeze(0).to(device)
                e1 = model.get_embedding(x1).squeeze(0)
                e2 = model.get_embedding(x2).squeeze(0)
                sim_map[i, j] = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
    return sim_map, img1, img2


def plot_change_heatmap(img1, img2, sim_map, out_path, region_id, threshold, whole_sim, changed_flag):
    change_map = 1 - sim_map  # higher = more change
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(img1); axes[0].set_title("T1 (before)"); axes[0].axis("off")
    axes[1].imshow(img2); axes[1].set_title("T2 (after)"); axes[1].axis("off")

    im = axes[2].imshow(change_map, cmap="hot", vmin=0, vmax=change_map.max() if change_map.max() > 0 else 1)
    axes[2].set_title(f"Change heatmap\nsim={whole_sim:.3f} thr={threshold:.3f}\n"
                       f"{'CHANGED' if changed_flag else 'no change'}")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.suptitle(f"Region {region_id}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="../outputs/region_embeddings.pt")
    parser.add_argument("--checkpoint", required=True, help="needed to compute patch heatmaps")
    parser.add_argument("--out-dir", default="../outputs")
    parser.add_argument("--n-heatmaps", type=int, default=6)
    args = parser.parse_args()

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from evaluation.evaluate import load_model
    from data.dataset import default_eval_transform
    from training.utils import get_device

    os.makedirs(os.path.join(args.out_dir, "heatmaps"), exist_ok=True)
    records = torch.load(args.embeddings)

    similarities, change_scores, labels = [], [], []
    for r in records:
        sim = cosine_similarity(r["t1_embedding"], r["t2_embedding"])
        r["similarity"] = sim
        similarities.append(sim)
        change_scores.append(1 - sim)
        labels.append(int(r["changed"]))

    # ---------------- ROC curve + threshold selection ----------------
    roc_stats = plot_roc(labels, change_scores, os.path.join(args.out_dir, "roc_curve.png"),
                          title="Change Detection ROC (score = 1 - cosine similarity)")
    similarity_threshold = 1 - roc_stats["best_threshold"]

    justification = f"""# Change Detection — Threshold Selection

ROC AUC: **{roc_stats['auc']:.3f}**

Operating point chosen via Youden's J statistic (maximises TPR - FPR):
- change-score threshold = {roc_stats['best_threshold']:.3f}  (i.e. cosine similarity threshold = {similarity_threshold:.3f})
- True Positive Rate at this point: {roc_stats['best_tpr']:.3f}
- False Positive Rate at this point: {roc_stats['best_fpr']:.3f}

**Justification:** Youden's J balances sensitivity and specificity without
assuming a particular cost ratio between missed changes (false negatives) and
false alarms (false positives). For a monitoring dashboard where both change
types carry meaningful but not wildly asymmetric costs (missing a real change
delays action; a false alarm wastes an analyst's review time), this balanced
operating point is a reasonable default. The bonus multi-threshold dashboard
(`bonus/multi_threshold_dashboard.py`) additionally exposes high-recall and
high-precision operating points for users who want to shift this trade-off.

Region pairs with cosine similarity below **{similarity_threshold:.3f}** are
flagged as **changed**.
"""
    with open(os.path.join(args.out_dir, "threshold_justification.md"), "w") as f:
        f.write(justification)
    print(justification)

    # save threshold for the dashboard to reuse
    with open(os.path.join(args.out_dir, "threshold.txt"), "w") as f:
        f.write(str(similarity_threshold))

    # ---------------- Heatmaps for sample region pairs ----------------
    device = get_device()
    model, _ = load_model(args.checkpoint, device)
    transform = default_eval_transform()

    # pick a mix: a few correctly flagged changed, a few correctly flagged unchanged
    changed_records = [r for r in records if r["similarity"] < similarity_threshold]
    unchanged_records = [r for r in records if r["similarity"] >= similarity_threshold]
    sample = (changed_records[: args.n_heatmaps // 2] +
              unchanged_records[: args.n_heatmaps - args.n_heatmaps // 2])

    for r in sample:
        sim_map, img1, img2 = patch_similarity_heatmap(model, r["t1_path"], r["t2_path"], transform, device)
        out_path = os.path.join(args.out_dir, "heatmaps", f"region_{r['region_id']}.png")
        plot_change_heatmap(img1, img2, sim_map, out_path, r["region_id"], similarity_threshold,
                             r["similarity"], r["similarity"] < similarity_threshold)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
