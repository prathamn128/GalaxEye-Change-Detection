#!/usr/bin/env python3
"""
Inference script for single image pair prediction.
Usage: python inference.py --pre path/to/pre.png --post path/to/post.png
"""

import argparse
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from src.model import build_model
from src.utils import load_config, get_device, load_checkpoint


def load_and_preprocess(path: str, img_size: int = 256) -> torch.Tensor:
    """Load an image and preprocess for model input."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor, img


@torch.no_grad()
def predict(model, pre_tensor, post_tensor, device, threshold=0.5):
    """Run inference on a pre/post image pair."""
    model.eval()
    fused = torch.cat([pre_tensor, post_tensor], dim=1).to(device)
    logits = model(fused)
    prob = torch.sigmoid(logits).cpu().squeeze().numpy()
    binary = (prob > threshold).astype(np.uint8)
    return prob, binary


def visualize_result(pre_img, post_img, prob_map, binary_map, save_path):
    """Create visualization of inference result."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    if pre_img.shape[-1] == 1:
        axes[0].imshow(pre_img.squeeze(), cmap="gray")
    else:
        axes[0].imshow(pre_img)
    axes[0].set_title("Pre-event", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    if post_img.shape[-1] == 1:
        axes[1].imshow(post_img.squeeze(), cmap="gray")
    else:
        axes[1].imshow(post_img)
    axes[1].set_title("Post-event", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Probability Map", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    cmap = mcolors.ListedColormap(["black", "red"])
    axes[3].imshow(binary_map, cmap=cmap, vmin=0, vmax=1)
    axes[3].set_title("Change Detection", fontsize=13, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Result saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Change Detection Inference")
    parser.add_argument("--pre", required=True, help="Pre-event image path")
    parser.add_argument("--post", required=True, help="Post-event image path")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", default="outputs/inference_result.png")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    img_size = config.get("dataset", {}).get("img_size", 256)

    # Load images
    pre_tensor, pre_img = load_and_preprocess(args.pre, img_size)
    post_tensor, post_img = load_and_preprocess(args.post, img_size)

    # Determine total channels
    total_c = pre_tensor.shape[1] + post_tensor.shape[1]

    # Build and load model
    model = build_model(config, in_channels=total_c)
    model = model.to(device)

    weights_path = args.weights or config.get("output", {}).get("best_model", "weights/best_model.pth")
    checkpoint = load_checkpoint(weights_path, model, device=device)

    threshold = args.threshold or checkpoint.get("threshold", 0.5)
    print(f"Using threshold: {threshold:.2f}")

    # Predict
    prob_map, binary_map = predict(model, pre_tensor, post_tensor, device, threshold)

    # Stats
    change_pct = binary_map.sum() / binary_map.size * 100
    print(f"Change detected: {change_pct:.1f}% of pixels")

    # Visualize
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    visualize_result(pre_img, post_img, prob_map, binary_map, args.output)


if __name__ == "__main__":
    main()
