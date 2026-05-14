#!/usr/bin/env python3
"""Visualization utilities for change detection results."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from pathlib import Path
from typing import List, Optional, Dict
import os


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to displayable numpy array."""
    if t.dim() == 4:
        t = t[0]
    img = t.detach().cpu().numpy()
    if img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
    img = np.clip(img, 0, 1)
    return img


def plot_qualitative(
    pre: torch.Tensor,
    post: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    filename: str,
    save_dir: str = "outputs/qualitative",
    threshold: float = 0.5,
):
    """Plot pre, post, ground truth, and prediction side by side."""
    os.makedirs(save_dir, exist_ok=True)

    pre_np = tensor_to_numpy(pre)
    post_np = tensor_to_numpy(post)
    gt_np = tensor_to_numpy(gt_mask)
    pred_prob = torch.sigmoid(pred_mask).detach().cpu()
    pred_np = (tensor_to_numpy(pred_prob) > threshold).astype(float)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Pre image
    if pre_np.ndim == 2 or (pre_np.ndim == 3 and pre_np.shape[-1] == 1):
        axes[0].imshow(pre_np.squeeze(), cmap="gray")
    else:
        axes[0].imshow(pre_np)
    axes[0].set_title("Pre-event", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Post image
    if post_np.ndim == 2 or (post_np.ndim == 3 and post_np.shape[-1] == 1):
        axes[1].imshow(post_np.squeeze(), cmap="gray")
    else:
        axes[1].imshow(post_np)
    axes[1].set_title("Post-event", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    # Ground truth
    cmap = mcolors.ListedColormap(["black", "red"])
    axes[2].imshow(gt_np.squeeze(), cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title("Ground Truth", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    # Prediction
    axes[3].imshow(pred_np.squeeze(), cmap=cmap, vmin=0, vmax=1)
    axes[3].set_title("Prediction", fontsize=13, fontweight="bold")
    axes[3].axis("off")

    plt.suptitle(f"Sample: {filename}", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str = "outputs/plots/confusion_matrix.png",
    title: str = "Confusion Matrix",
):
    """Plot and save confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    labels = ["No Change", "Change"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        ylabel="True Label", xlabel="Predicted Label",
        title=title,
    )
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,.0f}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_f1s: List[float],
    save_path: str = "outputs/plots/training_curves.png",
):
    """Plot training and validation loss/F1 curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1s, "g-", label="Val F1", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1 Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
