#!/usr/bin/env python3
"""Utility functions: config loading, checkpointing, seeding, early stopping."""

import os
import yaml
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[DEVICE] Using CPU")
    return device


def save_checkpoint(
    model, optimizer, epoch: int, metrics: dict,
    path: str, threshold: float = 0.5,
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "threshold": threshold,
    }
    torch.save(checkpoint, path)
    print(f"[CKPT] Saved checkpoint to {path}")


def load_checkpoint(path: str, model, optimizer=None, device=None):
    """Load model checkpoint."""
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"[CKPT] Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint


class EarlyStopping:
    """Early stopping to halt training when validation metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"[EARLY STOP] No improvement for {self.patience} epochs.")
                return True

        return False


def ensure_dirs(config: dict):
    """Create output directories."""
    for key in ["dir", "weights_dir", "report_dir"]:
        path = config.get("output", {}).get(key)
        if path:
            os.makedirs(path, exist_ok=True)
    os.makedirs("outputs/qualitative", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
