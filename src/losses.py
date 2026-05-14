#!/usr/bin/env python3
"""
Loss functions for change detection.
Combines BCEWithLogitsLoss and Dice Loss with configurable weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Differentiable Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice Loss.
    Total = bce_weight * BCE + dice_weight * Dice
    pos_weight handles class imbalance by upweighting positive (change) pixels.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=3.0, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.bce.pos_weight.device != logits.device:
            self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def build_loss(config: dict) -> nn.Module:
    """Build loss function from config."""
    loss_cfg = config.get("loss", {})
    loss_fn = CombinedLoss(
        bce_weight=loss_cfg.get("bce_weight", 0.5),
        dice_weight=loss_cfg.get("dice_weight", 0.5),
        pos_weight=loss_cfg.get("pos_weight", 3.0),
    )
    print(f"[LOSS] BCE(w={loss_fn.bce_weight}) + Dice(w={loss_fn.dice_weight}), pos_weight={loss_cfg.get('pos_weight', 3.0)}")
    return loss_fn
