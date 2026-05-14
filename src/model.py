#!/usr/bin/env python3
"""
Model definitions for change detection.
Implements a Siamese U-Net with shared encoder weights and feature-level fusion,
as well as a simpler concatenation-based approach using segmentation_models_pytorch.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for Change Detection.
    
    Architecture:
        - Shared encoder processes pre and post images independently
        - Feature maps from both branches are fused (concatenation + 1x1 conv)
          at each decoder level
        - Decoder produces binary change map
    
    Supports:
        - Variable input channels (EO: 3ch, SAR: 1ch)
        - Pretrained ImageNet encoders (ResNet34, EfficientNet, etc.)
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
    ):
        super().__init__()

        self.in_channels_per_image = in_channels

        # Create a full U-Net and extract encoder/decoder
        # We'll use the encoder in Siamese fashion
        self._base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )

        self.encoder = self._base_model.encoder

        # Fusion layers: reduce concatenated features (2x channels) to original channels
        encoder_channels = self.encoder.out_channels  # e.g., (3, 64, 64, 128, 256, 512) for resnet34

        self.fusion_convs = nn.ModuleList()
        for ch in encoder_channels:
            if ch > 0:
                self.fusion_convs.append(
                    nn.Sequential(
                        nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.fusion_convs.append(nn.Identity())

        self.decoder = self._base_model.decoder
        self.segmentation_head = self._base_model.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Concatenated pre+post image tensor (B, C_pre+C_post, H, W)
        
        Returns:
            Logits tensor (B, 1, H, W)
        """
        # Split into pre and post
        c = self.in_channels_per_image
        pre = x[:, :c, :, :]
        post = x[:, c:, :, :]

        # Shared encoder forward
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)

        # Fuse features at each level
        fused_features = []
        for i, (pf, qf) in enumerate(zip(pre_features, post_features)):
            concat = torch.cat([pf, qf], dim=1)  # (B, 2*C, H, W)
            fused = self.fusion_convs[i](concat)  # (B, C, H, W)
            fused_features.append(fused)

        # Decoder
        decoder_output = self.decoder(*fused_features)

        # Segmentation head
        masks = self.segmentation_head(decoder_output)

        return masks


class ConcatUNet(nn.Module):
    """
    Simple concatenation-based U-Net for Change Detection.
    
    Concatenates pre and post images along channel dimension
    and passes through a standard U-Net.
    
    Simpler than Siamese but still effective as a baseline.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 6,  # pre + post concatenated
        classes: int = 1,
        activation: Optional[str] = None,
    ):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Concatenated pre+post image tensor (B, C_total, H, W)
        
        Returns:
            Logits tensor (B, 1, H, W)
        """
        return self.model(x)


def build_model(config: dict, in_channels: int = None, pre_channels: int = None, post_channels: int = None) -> nn.Module:
    """
    Build the model from config.
    
    Args:
        config: Configuration dictionary
        in_channels: Override for total input channels (auto-detected from dataset)
        pre_channels: Number of channels in pre-event image
        post_channels: Number of channels in post-event image
    
    Returns:
        Model instance
    """
    model_cfg = config.get("model", {})
    encoder_name = model_cfg.get("encoder_name", "resnet34")
    encoder_weights = model_cfg.get("encoder_weights", "imagenet")
    classes = model_cfg.get("classes", 1)
    activation = model_cfg.get("activation", None)

    # Use provided in_channels or config value
    total_channels = in_channels or model_cfg.get("in_channels", 6)

    # Decide between Siamese and Concat based on channel symmetry
    use_siamese = True
    if pre_channels is not None and post_channels is not None:
        if pre_channels != post_channels:
            use_siamese = False
            print(f"[MODEL] Asymmetric channels (pre={pre_channels}, post={post_channels}) -> using ConcatUNet")
    elif total_channels % 2 != 0:
        use_siamese = False
        print(f"[MODEL] Odd total channels ({total_channels}) -> using ConcatUNet")

    if use_siamese and (pre_channels is None or pre_channels == post_channels):
        per_image_channels = total_channels // 2
        print(f"[MODEL] Building Siamese U-Net:")
        print(f"  Encoder: {encoder_name} (weights: {encoder_weights})")
        print(f"  Input channels: {total_channels} ({per_image_channels}+{per_image_channels})")
        print(f"  Output classes: {classes}")

        model = SiameseUNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=per_image_channels,
            classes=classes,
            activation=activation,
        )
    else:
        print(f"[MODEL] Building Concat U-Net:")
        print(f"  Encoder: {encoder_name} (weights: {encoder_weights})")
        print(f"  Input channels: {total_channels}")
        print(f"  Output classes: {classes}")

        model = ConcatUNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=total_channels,
            classes=classes,
            activation=activation,
        )

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    return model
