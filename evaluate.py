#!/usr/bin/env python3
"""
Evaluation script for GalaxEye Change Detection.
Loads the best model and evaluates on validation and test sets.
Reports IoU, Precision, Recall, F1, Confusion Matrix.
Generates qualitative prediction examples.
"""

import os
import argparse
import json
import torch
from tqdm import tqdm
from pathlib import Path

from src.dataset import ChangeDetectionDataset
from src.model import build_model
from src.losses import build_loss
from src.metrics import MetricsCalculator
from src.utils import load_config, set_seed, get_device, load_checkpoint, ensure_dirs
from src.visualize import plot_qualitative, plot_confusion_matrix


@torch.no_grad()
def evaluate_split(model, loader, criterion, device, threshold, split_name, config):
    """Evaluate model on a data split."""
    model.eval()
    metrics_calc = MetricsCalculator(threshold=threshold)
    total_loss = 0.0
    num_qual = config.get("evaluation", {}).get("num_qualitative", 8)
    qual_count = 0
    qual_paths = []

    pbar = tqdm(loader, desc=f"  Evaluating {split_name}", ncols=100)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        filenames = batch["filename"]

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()
        metrics_calc.update(logits, masks)

        # Save qualitative examples
        if qual_count < num_qual:
            for i in range(min(images.size(0), num_qual - qual_count)):
                save_dir = f"outputs/qualitative/{split_name}"
                path = plot_qualitative(
                    pre=batch["pre"][i],
                    post=batch["post"][i],
                    gt_mask=masks[i].cpu(),
                    pred_mask=logits[i].cpu(),
                    filename=filenames[i],
                    save_dir=save_dir,
                    threshold=threshold,
                )
                qual_paths.append(path)
                qual_count += 1

    avg_loss = total_loss / len(loader)
    metrics = metrics_calc.compute()
    cm = metrics_calc.confusion_matrix()

    # Plot confusion matrix
    cm_path = f"outputs/plots/confusion_matrix_{split_name}.png"
    plot_confusion_matrix(cm, save_path=cm_path, title=f"Confusion Matrix - {split_name.upper()}")

    return avg_loss, metrics, cm, qual_paths


def print_metrics(split_name, loss, metrics, cm):
    """Print formatted metrics."""
    print(f"\n{'='*50}")
    print(f"  {split_name.upper()} Results")
    print(f"{'='*50}")
    print(f"  Loss:      {loss:.4f}")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                 Pred No-Change  Pred Change")
    print(f"  True No-Change    {cm[0,0]:>12,.0f}  {cm[0,1]:>11,.0f}")
    print(f"  True Change       {cm[1,0]:>12,.0f}  {cm[1,1]:>11,.0f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Change Detection Model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--weights", default=None, help="Path to model weights")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--split", default="both", choices=["val", "test", "both"])
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("training", {}).get("seed", 42))
    device = get_device()
    ensure_dirs(config)

    # Load data
    print("\n[DATA] Loading datasets...")
    from torch.utils.data import DataLoader

    data_root = config.get("data", {}).get("root", "data")
    label_remap = config.get("dataset", {}).get("label_remap", {0: 0, 1: 0, 2: 1, 3: 1})
    label_remap = {int(k): int(v) for k, v in label_remap.items()}
    img_size = config.get("dataset", {}).get("img_size", 256)
    batch_size = config.get("training", {}).get("batch_size", 4)
    num_workers = config.get("training", {}).get("num_workers", 0)

    val_ds = ChangeDetectionDataset(
        root_dir=data_root, split="val",
        img_size=img_size, augment=False, label_remap=label_remap,
    )
    test_ds = ChangeDetectionDataset(
        root_dir=data_root, split="test",
        img_size=img_size, augment=False, label_remap=label_remap,
    )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"  Val:  {len(val_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")

    # Auto-detect channels
    pre_c, post_c, total_c = val_ds.get_channel_info()

    # Build model
    model = build_model(config, in_channels=total_c, pre_channels=pre_c, post_channels=post_c)
    model = model.to(device)

    # Load weights
    weights_path = args.weights or config.get("output", {}).get("best_model", "weights/best_model.pth")
    checkpoint = load_checkpoint(weights_path, model, device=device)

    # Get threshold
    threshold = args.threshold or checkpoint.get("threshold", 0.5)
    print(f"[EVAL] Using threshold: {threshold:.2f}")

    criterion = build_loss(config)

    results = {}

    if args.split in ["val", "both"]:
        val_loss, val_metrics, val_cm, val_quals = evaluate_split(
            model, val_loader, criterion, device, threshold, "val", config
        )
        print_metrics("Validation", val_loss, val_metrics, val_cm)
        results["val"] = {"loss": val_loss, **val_metrics}

    if args.split in ["test", "both"]:
        test_loss, test_metrics, test_cm, test_quals = evaluate_split(
            model, test_loader, criterion, device, threshold, "test", config
        )
        print_metrics("Test", test_loss, test_metrics, test_cm)
        results["test"] = {"loss": test_loss, **test_metrics}

    # Save results JSON
    results_path = "outputs/evaluation_results.json"
    serializable = {}
    for split_key, m in results.items():
        serializable[split_key] = {k: float(v) for k, v in m.items()}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
