"""
Bonus C: Project all 27,000 EuroSAT embeddings to 2D via t-SNE or UMAP,
coloured by class, comparing scratch-CNN vs fine-tuned embeddings side by side.

Usage:
    python bonus/tsne_visualization.py --checkpoint checkpoints/transfer_resnet18.pt \
        --baseline-checkpoint checkpoints/baseline_cnn.pt --data-root ./data/raw/EuroSAT
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import EuroSATDataset, spatial_block_split, default_eval_transform, EUROSAT_CLASSES
from evaluation.evaluate import load_model
from models.baseline_cnn import BaselineCNN
from training.utils import get_device


def extract_all_embeddings(model, loader, device, use_baseline_penultimate=False):
    embeddings, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if use_baseline_penultimate:
                feats = model.features(x)
                emb = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
            else:
                emb = model.get_embedding(x)
            embeddings.append(emb.cpu().numpy())
            labels.extend(y.tolist())
    return np.concatenate(embeddings, axis=0), np.array(labels)


def reduce_2d(embeddings, method="tsne"):
    if method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    return reducer.fit_transform(embeddings)


def plot_side_by_side(coords_scratch, labels_scratch, coords_fine, labels_fine, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, labels, title in [
        (axes[0], coords_scratch, labels_scratch, "Scratch CNN embeddings"),
        (axes[1], coords_fine, labels_fine, "Fine-tuned backbone embeddings"),
    ]:
        for cls_idx, cls_name in enumerate(EUROSAT_CLASSES):
            mask = labels == cls_idx
            ax.scatter(coords[mask, 0], coords[mask, 1], s=6, label=cls_name, alpha=0.6)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--out", default="../outputs/embedding_comparison.png")
    args = parser.parse_args()

    device = get_device()
    all_samples = []
    splits = spatial_block_split(args.data_root)
    for k in splits:
        all_samples.extend(splits[k])
    ds = EuroSATDataset(all_samples, transform=default_eval_transform())
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)

    fine_model, _ = load_model(args.checkpoint, device)
    fine_embeddings, fine_labels = extract_all_embeddings(fine_model, loader, device)

    baseline_model = BaselineCNN(num_classes=len(EUROSAT_CLASSES)).to(device)
    baseline_model.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device))
    baseline_model.eval()
    scratch_embeddings, scratch_labels = extract_all_embeddings(
        baseline_model, loader, device, use_baseline_penultimate=True)

    coords_scratch = reduce_2d(scratch_embeddings, args.method)
    coords_fine = reduce_2d(fine_embeddings, args.method)

    plot_side_by_side(coords_scratch, scratch_labels, coords_fine, fine_labels, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
