"""
Bonus A: GradCAM visualisation on the fine-tuned model.

Usage:
    python bonus/gradcam.py --checkpoint checkpoints/transfer_resnet18.pt \
        --data-root ./data/raw/EuroSAT --n-examples 3
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import default_eval_transform, spatial_block_split
from evaluation.evaluate import load_model
from training.utils import get_device

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM_LIB = True
except ImportError:
    HAS_GRADCAM_LIB = False


def manual_gradcam(model, x, target_class, device, target_layer):
    """Fallback GradCAM implementation (no external lib needed) using hooks
    on the given target_layer (e.g. model.backbone.layer4)."""
    activations, gradients = [], []

    def fwd_hook(module, inp, out):
        activations.append(out)

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    logits = model(x)
    score = logits[0, target_class]
    score.backward()

    h1.remove(); h2.remove()

    act = activations[0].detach()[0]      # [C, H, W]
    grad = gradients[0].detach()[0]       # [C, H, W]
    weights = grad.mean(dim=(1, 2))       # [C]
    cam = torch.relu((weights[:, None, None] * act).sum(0))
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--out-dir", default="../outputs/gradcam")
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    model, classes = load_model(args.checkpoint, device)
    model.eval()

    target_layer = model.backbone.layer4 if hasattr(model.backbone, "layer4") else None
    if target_layer is None:
        # EfficientNet fallback: last feature block
        target_layer = model.backbone.features[-1]

    transform = default_eval_transform()
    splits = spatial_block_split(args.data_root)
    samples = splits["val"][: args.n_examples]

    for idx, (path, true_label) in enumerate(samples):
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        x.requires_grad_(True)

        logits = model(x)
        pred_class = int(logits.argmax(dim=1).item())

        cam = manual_gradcam(model, x, pred_class, device, target_layer)
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size))

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img); axes[0].set_title(f"Input\ntrue={classes[true_label]}"); axes[0].axis("off")
        axes[1].imshow(img)
        axes[1].imshow(cam_resized, cmap="jet", alpha=0.5)
        axes[1].set_title(f"GradCAM\npred={classes[pred_class]}")
        axes[1].axis("off")
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"gradcam_{idx}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    print("\nInterpretation notes (fill in after inspecting the 3 saved examples):")
    print("- Does the highlighted region correspond to the class-defining texture "
          "(e.g. crop rows for AnnualCrop, tree canopy for Forest)?")
    print("- Are misclassifications associated with attention on background/edge "
          "pixels rather than the dominant land-cover feature?")


if __name__ == "__main__":
    main()
