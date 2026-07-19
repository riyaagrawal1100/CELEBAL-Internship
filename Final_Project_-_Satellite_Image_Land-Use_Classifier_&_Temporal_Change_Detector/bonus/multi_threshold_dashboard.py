"""
Bonus B: Multi-threshold toggle dashboard.

Extends dashboard/app.py with 3 preset operating points:
  - High recall     (catches more real changes, more false alarms)
  - Balanced         (Youden's J — same as the default dashboard)
  - High precision   (fewer false alarms, may miss subtle changes)

Run:
    streamlit run bonus/multi_threshold_dashboard.py
"""
import os
import sys

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import default_eval_transform
from evaluation.evaluate import load_model
from training.utils import get_device
from dashboard.app import predict, patch_change_map, DEFAULT_CHECKPOINT, get_model

# These presets are illustrative defaults; recompute them precisely from your
# ROC curve (evaluation/metrics.py::plot_roc) using different points along
# the curve for high-recall / high-precision operating points.
PRESETS = {
    "High recall (catch more changes)": 0.93,
    "Balanced (Youden's J)": 0.85,
    "High precision (fewer false alarms)": 0.70,
}


def main():
    st.set_page_config(page_title="Multi-Threshold Change Dashboard", layout="wide")
    st.title("🛰️ Change Detection — Multi-Threshold Toggle")

    with st.sidebar:
        checkpoint_path = st.text_input("Checkpoint path", DEFAULT_CHECKPOINT)
        preset_name = st.radio("Operating point", list(PRESETS.keys()), index=1)
        threshold = PRESETS[preset_name]
        st.write(f"Similarity threshold: **{threshold:.2f}**")

    if not os.path.exists(checkpoint_path):
        st.warning(f"Checkpoint not found at `{checkpoint_path}`.")
        return

    model, classes, device = get_model(checkpoint_path)
    transform = default_eval_transform()

    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("BEFORE tile", type=["jpg", "jpeg", "png", "tif", "tiff"], key="b2")
    with col2:
        after_file = st.file_uploader("AFTER tile", type=["jpg", "jpeg", "png", "tif", "tiff"], key="a2")

    if before_file and after_file:
        img1, img2 = Image.open(before_file), Image.open(after_file)
        label1, conf1, emb1 = predict(model, classes, device, img1, transform)
        label2, conf2, emb2 = predict(model, classes, device, img2, transform)
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        changed = similarity < threshold

        c1, c2, c3 = st.columns(3)
        c1.image(img1, caption=f"BEFORE — {label1} ({conf1:.1%})", use_container_width=True)
        c2.image(img2, caption=f"AFTER — {label2} ({conf2:.1%})", use_container_width=True)
        with c3:
            st.metric("Cosine similarity", f"{similarity:.3f}")
            st.write(f"Preset: **{preset_name}**")
            st.error("⚠️ CHANGE") if changed else st.success("✅ No change")

        st.subheader("Change heatmap at this operating point")
        change_map = patch_change_map(model, img1, img2, transform, device)
        st.image((change_map / (change_map.max() + 1e-8) * 255).astype(np.uint8),
                  clamp=True, use_container_width=True)

        st.caption("Toggle the operating point in the sidebar to see how the change "
                   "flag and heatmap sensitivity shift.")


if __name__ == "__main__":
    main()
