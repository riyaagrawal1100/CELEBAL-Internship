"""
Extracts 512-d embeddings for every T1/T2 tile in region_pairs.json using the
fine-tuned backbone (classifier head stripped).

Usage:
    python change_detection/embeddings.py --checkpoint checkpoints/transfer_resnet18.pt \
        --regions ../outputs/region_pairs.json
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import default_eval_transform
from evaluation.evaluate import load_model
from training.utils import get_device


def embed_image(model, path, transform, device):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.get_embedding(x)
    return emb.squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--regions", default="../outputs/region_pairs.json")
    parser.add_argument("--out", default="../outputs/region_embeddings.pt")
    args = parser.parse_args()

    device = get_device()
    model, _ = load_model(args.checkpoint, device)
    transform = default_eval_transform()

    with open(args.regions) as f:
        regions = json.load(f)

    records = []
    for r in regions:
        t1_emb = embed_image(model, r["t1_path"], transform, device)
        t2_emb = embed_image(model, r["t2_path"], transform, device)
        records.append({
            "region_id": r["region_id"],
            "t1_path": r["t1_path"], "t2_path": r["t2_path"],
            "t1_class": r["t1_class"], "t2_class": r["t2_class"],
            "changed": r["changed"],
            "t1_embedding": t1_emb, "t2_embedding": t2_emb,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(records, args.out)
    print(f"Saved {len(records)} region embeddings to {args.out}")


if __name__ == "__main__":
    main()
