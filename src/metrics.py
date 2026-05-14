#!/usr/bin/env python3
"""Metrics for binary change detection evaluation."""

import numpy as np
import torch
from typing import Dict, Tuple, List


class MetricsCalculator:
    """Accumulates predictions and computes metrics over an entire dataset."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update with a batch of predictions."""
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()
            t = targets.float()

            self.tp += ((preds == 1) & (t == 1)).sum().item()
            self.fp += ((preds == 1) & (t == 0)).sum().item()
            self.tn += ((preds == 0) & (t == 0)).sum().item()
            self.fn += ((preds == 0) & (t == 1)).sum().item()

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated counts."""
        eps = 1e-7
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "accuracy": accuracy,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }

    def confusion_matrix(self) -> np.ndarray:
        """Return 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])


def find_best_threshold(
    logits_list: List[torch.Tensor],
    targets_list: List[torch.Tensor],
    min_t: float = 0.1,
    max_t: float = 0.9,
    step: float = 0.05,
) -> Tuple[float, Dict[str, float]]:
    """Search for the best threshold based on F1 score."""
    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics = {}

    all_logits = torch.cat(logits_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)

    thresholds = np.arange(min_t, max_t + step, step)
    print(f"\n[THRESHOLD] Searching {len(thresholds)} thresholds [{min_t:.2f} - {max_t:.2f}]...")

    for t in thresholds:
        calc = MetricsCalculator(threshold=t)
        calc.update(all_logits, all_targets)
        metrics = calc.compute()

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = t
            best_metrics = metrics

    print(f"[THRESHOLD] Best: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold, best_metrics
