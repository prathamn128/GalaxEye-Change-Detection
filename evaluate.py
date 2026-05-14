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
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy import ndimage

from src.dataset import ChangeDetectionDataset
from src.model import build_model
from src.losses import build_loss
from src.metrics import MetricsCalculator
from src.utils import load_config, set_seed, get_device, load_checkpoint, ensure_dirs
from src.visualize import plot_qualitative, plot_confusion_matrix


def postprocess_mask(pred_binary, min_size=50):
    """Remove small connected components (false positives) from binary mask."""
    struct = ndimage.generate_binary_structure(2, 1)
    cleaned = ndimage.binary_opening(pred_binary, structure=struct, iterations=1)
    labeled, num_features = ndimage.label(cleaned)
    if num_features == 0:
        return cleaned.astype(np.float32)
    component_sizes = ndimage.sum(cleaned, labeled, range(1, num_features + 1))
    for i, size in enumerate(component_sizes):
        if size < min_size:
            cleaned[labeled == (i + 1)] = 0
    return cleaned.astype(np.float32)


@torch.no_grad()
def tta_predict(model, images, device):
    """Test-Time Augmentation: average predictions over flips."""
    logits_orig = model(images)
    probs_orig = torch.sigmoid(logits_orig)
    images_hflip = torch.flip(images, dims=[3])
    probs_hflip = torch.sigmoid(torch.flip(model(images_hflip), dims=[3]))
    images_vflip = torch.flip(images, dims=[2])
    probs_vflip = torch.sigmoid(torch.flip(model(images_vflip), dims=[2]))
    return (probs_orig + probs_hflip + probs_vflip) / 3.0

@torch.no_grad()
def evaluate_split(model, loader, criterion, device, threshold, split_name, config,
                   use_tta=False, use_postprocess=False, min_component_size=50):
    """Evaluate model on a data split with optional TTA and post-processing."""
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

        # Get predictions (with or without TTA)
        if use_tta:
            avg_probs = tta_predict(model, images, device)
            avg_probs_clamped = torch.clamp(avg_probs, 1e-6, 1 - 1e-6)
            logits = torch.log(avg_probs_clamped / (1 - avg_probs_clamped))
        else:
            logits = model(images)

        loss = criterion(logits, masks)
        total_loss += loss.item()

        # Apply post-processing if enabled
        if use_postprocess:
            probs = torch.sigmoid(logits)
            binary = (probs > threshold).float()
            for b_idx in range(binary.size(0)):
                mask_np = binary[b_idx, 0].cpu().numpy()
                cleaned = postprocess_mask(mask_np, min_size=min_component_size)
                binary[b_idx, 0] = torch.from_numpy(cleaned)
            binary_clamped = torch.clamp(binary, 0.01, 0.99)
            clean_logits = torch.log(binary_clamped / (1 - binary_clamped))
            metrics_calc.update(clean_logits.to(device), masks)
        else:
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
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    parser.add_argument("--no-postprocess", action="store_true", help="Disable post-processing")
    parser.add_argument("--min-size", type=int, default=50, help="Min component size")
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
    use_tta = not args.no_tta
    use_pp = not args.no_postprocess
    print(f"[EVAL] Threshold: {threshold:.2f} | TTA: {use_tta} | Post-process: {use_pp} (min_size={args.min_size})")

    criterion = build_loss(config)

    results = {}

    if args.split in ["val", "both"]:
        val_loss, val_metrics, val_cm, val_quals = evaluate_split(
            model, val_loader, criterion, device, threshold, "val", config,
            use_tta=use_tta, use_postprocess=use_pp, min_component_size=args.min_size
        )
        print_metrics("Validation", val_loss, val_metrics, val_cm)
        results["val"] = {"loss": val_loss, **val_metrics}

    if args.split in ["test", "both"]:
        test_loss, test_metrics, test_cm, test_quals = evaluate_split(
            model, test_loader, criterion, device, threshold, "test", config,
            use_tta=use_tta, use_postprocess=use_pp, min_component_size=args.min_size
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
