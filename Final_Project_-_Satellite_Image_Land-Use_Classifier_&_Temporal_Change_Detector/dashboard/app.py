"""
Streamlit Geo-Dashboard.

Upload a "before" and "after" satellite tile. Shows:
  - predicted land-use class + confidence for each tile
  - cosine similarity between their embeddings
  - side-by-side view with a change-flag heatmap if similarity < threshold

Run:
    streamlit run dashboard/app.py

Runs fully locally / offline once the checkpoint + threshold files exist
(see CACHE_DIR below) — no internet calls are made at inference time.
"""
import os
import sys

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import default_eval_transform, EUROSAT_CLASSES
from evaluation.evaluate import load_model
from training.utils import get_device

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DEFAULT_CHECKPOINT = os.path.join(CACHE_DIR, "transfer_resnet18.pt")
THRESHOLD_FILE = os.path.join(OUTPUTS_DIR, "threshold.txt")


@st.cache_resource
def get_model(checkpoint_path):
    device = get_device()
    model, classes = load_model(checkpoint_path, device)
    return model, classes, device


def read_default_threshold():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE) as f:
            return float(f.read().strip())
    return 0.85  # sensible fallback if change_detector.py hasn't been run yet


def predict(model, classes, device, img: Image.Image, transform):
    x = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        emb = model.get_embedding(x).squeeze(0).cpu()
    pred_idx = int(np.argmax(probs))
    return classes[pred_idx], float(probs[pred_idx]), emb


def patch_change_map(model, img1, img2, transform, device, grid=8):
    w, h = img1.size
    pw, ph = max(w // grid, 1), max(h // grid, 1)
    sim_map = np.zeros((grid, grid))
    with torch.no_grad():
        for i in range(grid):
            for j in range(grid):
                box = (j * pw, i * ph, (j + 1) * pw, (i + 1) * ph)
                p1 = img1.crop(box).resize((64, 64))
                p2 = img2.crop(box).resize((64, 64))
                e1 = model.get_embedding(transform(p1.convert("RGB")).unsqueeze(0).to(device)).squeeze(0)
                e2 = model.get_embedding(transform(p2.convert("RGB")).unsqueeze(0).to(device)).squeeze(0)
                sim_map[i, j] = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
    return 1 - sim_map  # change map


def main():
    st.set_page_config(page_title="Satellite Change Detection Dashboard", layout="wide")
    st.title("🛰️ Satellite Land-Use Classifier & Change Detector")
    st.caption("Runs fully locally — no internet dependency at inference time.")

    with st.sidebar:
        st.header("Settings")
        checkpoint_path = st.text_input("Checkpoint path", DEFAULT_CHECKPOINT)
        threshold = st.slider("Change threshold (cosine similarity)", 0.0, 1.0,
                               value=read_default_threshold(), step=0.01,
                               help="Tile pairs with similarity below this are flagged as changed.")

    if not os.path.exists(checkpoint_path):
        st.warning(f"Checkpoint not found at `{checkpoint_path}`. Train a model first "
                   f"with `training/train_transfer.py`, or update the path in the sidebar.")
        return

    model, classes, device = get_model(checkpoint_path)
    transform = default_eval_transform()

    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("Upload BEFORE tile", type=["jpg", "jpeg", "png", "tif", "tiff"], key="before")
    with col2:
        after_file = st.file_uploader("Upload AFTER tile", type=["jpg", "jpeg", "png", "tif", "tiff"], key="after")

    if before_file and after_file:
        img1 = Image.open(before_file)
        img2 = Image.open(after_file)

        label1, conf1, emb1 = predict(model, classes, device, img1, transform)
        label2, conf2, emb2 = predict(model, classes, device, img2, transform)
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        changed = similarity < threshold

        c1, c2, c3 = st.columns(3)
        c1.image(img1, caption=f"BEFORE — {label1} ({conf1:.1%} confidence)", use_container_width=True)
        c2.image(img2, caption=f"AFTER — {label2} ({conf2:.1%} confidence)", use_container_width=True)

        with c3:
            st.metric("Cosine similarity", f"{similarity:.3f}")
            if changed:
                st.error(f"⚠️ CHANGE DETECTED (similarity {similarity:.3f} < threshold {threshold:.3f})")
            else:
                st.success(f"✅ No significant change (similarity {similarity:.3f} ≥ threshold {threshold:.3f})")

        st.subheader("Spatial change heatmap")
        change_map = patch_change_map(model, img1, img2, transform, device)
        st.image(
            (change_map / (change_map.max() + 1e-8) * 255).astype(np.uint8),
            caption="Brighter regions = more embedding-level change within the tile",
            clamp=True, use_container_width=True,
        )
    else:
        st.info("Upload both a BEFORE and an AFTER tile to run classification + change detection.")


if __name__ == "__main__":
    main()
