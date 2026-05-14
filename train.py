#!/usr/bin/env python3
"""
Training script for GalaxEye Change Detection.

Features:
  - Mixed precision training (GPU) with CPU fallback
  - Early stopping and model checkpointing (best val F1)
  - Cosine annealing LR scheduler
  - Threshold tuning on validation set
  - Training curve logging
  - Auto-splits val set if train data is missing
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from src.dataset import ChangeDetectionDataset, get_dataloaders
from src.model import build_model
from src.losses import build_loss
from src.metrics import MetricsCalculator, find_best_threshold
from src.utils import (
    load_config, set_seed, get_device,
    save_checkpoint, EarlyStopping, ensure_dirs,
)
from src.visualize import plot_training_curves


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    metrics_calc = MetricsCalculator()

    pbar = tqdm(loader, desc="  Train", ncols=100, leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        metrics_calc.update(logits.detach(), masks.detach())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics = metrics_calc.compute()
    return avg_loss, metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate and return loss, metrics, and collected logits/targets."""
    model.eval()
    total_loss = 0.0
    metrics_calc = MetricsCalculator()
    all_logits = []
    all_targets = []

    pbar = tqdm(loader, desc="  Val  ", ncols=100, leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        metrics_calc.update(logits, masks)
        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

    avg_loss = total_loss / len(loader)
    metrics = metrics_calc.compute()
    return avg_loss, metrics, all_logits, all_targets


def get_train_val_loaders(config):
    """
    Get train and val dataloaders.
    If train data doesn't exist, split the val set 80/20.
    """
    from torch.utils.data import DataLoader, random_split

    data_root = config.get("data", {}).get("root", "data")
    train_dir = Path(data_root) / "train"
    val_dir = Path(data_root) / "val"

    label_remap = config.get("dataset", {}).get("label_remap", {0: 0, 1: 0, 2: 1, 3: 1})
    label_remap = {int(k): int(v) for k, v in label_remap.items()}
    img_size = config.get("dataset", {}).get("img_size", 256)
    batch_size = config.get("training", {}).get("batch_size", 4)
    num_workers = config.get("training", {}).get("num_workers", 0)

    # Check if train data exists
    train_exists = train_dir.exists()
    if train_exists:
        try:
            # Try to detect the subdirectories
            subdirs = [d.name for d in train_dir.iterdir() if d.is_dir()]
            if not any(d in subdirs for d in ["pre-event", "A", "images1", "pre"]):
                train_exists = False
        except Exception:
            train_exists = False

    if train_exists:
        print("[DATA] Using separate train/val directories")
        train_ds = ChangeDetectionDataset(
            root_dir=data_root, split="train",
            img_size=img_size, augment=True, label_remap=label_remap,
        )
        val_ds = ChangeDetectionDataset(
            root_dir=data_root, split="val",
            img_size=img_size, augment=False, label_remap=label_remap,
        )
    else:
        print("[DATA] Train data not found — splitting val set 80/20 for training")
        full_ds = ChangeDetectionDataset(
            root_dir=data_root, split="val",
            img_size=img_size, augment=False, label_remap=label_remap,
        )
        total = len(full_ds)
        train_size = int(0.8 * total)
        val_size = total - train_size

        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)

        # Wrap the train split so augmentation is applied
        class AugmentedSubset:
            """Wrapper to apply augmentation to a subset."""
            def __init__(self, subset, root_dir, img_size, label_remap):
                self.subset = subset
                # Create augmented dataset referencing the same data
                self.aug_ds = ChangeDetectionDataset(
                    root_dir=root_dir, split="val",
                    img_size=img_size, augment=True, label_remap=label_remap,
                )

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                original_idx = self.subset.indices[idx]
                return self.aug_ds[original_idx]

            def get_channel_info(self):
                return self.aug_ds.get_channel_info()

        train_ds = AugmentedSubset(train_ds, data_root, img_size, label_remap)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    return train_loader, val_loader, train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="Train Change Detection Model")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    train_cfg = config.get("training", {})

    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size

    # Setup
    set_seed(train_cfg.get("seed", 42))
    device = get_device()
    ensure_dirs(config)

    use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"
    print(f"[AMP] Mixed precision: {'enabled' if use_amp else 'disabled'}")

    # Data
    print("\n[DATA] Loading datasets...")
    train_loader, val_loader, train_ds, val_ds = get_train_val_loaders(config)
    print(f"  Train: {len(train_ds)} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_ds)} samples ({len(val_loader)} batches)")

    # Auto-detect input channels
    pre_c, post_c, total_c = train_ds.get_channel_info()
    print(f"  Channels: pre={pre_c}, post={post_c}, total={total_c}")

    # Model
    model = build_model(config, in_channels=total_c, pre_channels=pre_c, post_channels=post_c)
    model = model.to(device)

    # Loss
    criterion = build_loss(config)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.get("epochs", 30),
        eta_min=train_cfg.get("scheduler_params", {}).get("eta_min", 1e-6),
    )

    # Early stopping
    es_cfg = train_cfg.get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=es_cfg.get("patience", 8),
        min_delta=es_cfg.get("min_delta", 0.001),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    epochs = train_cfg.get("epochs", 30)
    best_f1 = 0.0
    best_threshold = 0.5
    train_losses, val_losses, val_f1s = [], [], []
    best_model_path = config.get("output", {}).get("best_model", "weights/best_model.pth")

    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{epochs} (lr={optimizer.param_groups[0]['lr']:.6f})")

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )

        # Validate
        val_loss, val_metrics, val_logits, val_targets = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_metrics["f1"])

        epoch_time = time.time() - epoch_start
        print(f"  Train - Loss: {train_loss:.4f} | F1: {train_metrics['f1']:.4f} | IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} | F1: {val_metrics['f1']:.4f} | IoU: {val_metrics['iou']:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Check for best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            # Threshold tuning
            thresh_cfg = config.get("evaluation", {}).get("threshold_search", {})
            if thresh_cfg.get("enabled", True):
                best_threshold, _ = find_best_threshold(
                    val_logits, val_targets,
                    min_t=thresh_cfg.get("min", 0.1),
                    max_t=thresh_cfg.get("max", 0.9),
                    step=thresh_cfg.get("step", 0.05),
                )

            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                best_model_path, threshold=best_threshold,
            )
            print(f"  ** New best F1: {best_f1:.4f} (threshold: {best_threshold:.2f})")

        # Early stopping
        if early_stopping(val_metrics["f1"]):
            print(f"\nEarly stopping at epoch {epoch}")
            break

        print()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time/60:.1f} min")
    print(f"Best F1: {best_f1:.4f} | Best Threshold: {best_threshold:.2f}")
    print(f"Best model: {best_model_path}")
    print(f"{'='*60}")

    # Plot training curves
    curves_path = plot_training_curves(train_losses, val_losses, val_f1s)
    print(f"Training curves saved to {curves_path}")

    # Save training log
    log_path = "outputs/training_log.txt"
    with open(log_path, "w") as f:
        f.write(f"Training completed in {total_time/60:.1f} minutes\n")
        f.write(f"Total epochs: {epoch}\n")
        f.write(f"Best F1: {best_f1:.4f}\n")
        f.write(f"Best Threshold: {best_threshold:.2f}\n")
        f.write(f"Device: {device}\n")
        f.write(f"\nPer-epoch metrics:\n")
        for i in range(len(train_losses)):
            f.write(f"  Epoch {i+1}: train_loss={train_losses[i]:.4f}, val_loss={val_losses[i]:.4f}, val_f1={val_f1s[i]:.4f}\n")
    print(f"Training log saved to {log_path}")

    # Save training results as JSON for report generation
    results = {
        "total_time_min": total_time / 60,
        "total_epochs": epoch,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "device": str(device),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_f1s": val_f1s,
    }
    with open("outputs/training_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
