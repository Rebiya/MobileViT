"""
MobileViT XXS model definition for CIFAR10.

This is a compact variant designed to be small enough for typical
CPU/GPU memory while still exercising the MobileViT pattern of
local CNN + global Transformer-based processing.
"""

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.blocks import ConvBNReLU, MobileViTBlock


class MobileViTXXS(nn.Module):
    """
    A lightweight MobileViT-style network for CIFAR10.

    Architecture (simplified):
    - Stem: 3x3 conv, 3 -> 16
    - Stage 1: downsample + MobileViT block (C=32)
    - Stage 2: downsample + MobileViT block (C=48)
    - Stage 3: downsample + MobileViT block (C=64)
    - Head: global average pooling + linear classifier (10 classes)

    The MobileViT blocks internally perform patch unfolding/folding
    with a Transformer encoder as global feature extractor.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Stem.
        self.stem = ConvBNReLU(3, 16, kernel_size=3, stride=1, padding=1)

        # Stages with downsampling conv followed by MobileViT blocks.
        self.stage1 = nn.Sequential(
            ConvBNReLU(16, 32, kernel_size=3, stride=2, padding=1),
            MobileViTBlock(
                in_channels=32,
                transformer_dim=64,
                depth=2,
                patch_size=(2, 2),
                num_heads=4,
            ),
        )

        self.stage2 = nn.Sequential(
            ConvBNReLU(32, 48, kernel_size=3, stride=2, padding=1),
            MobileViTBlock(
                in_channels=48,
                transformer_dim=80,
                depth=2,
                patch_size=(2, 2),
                num_heads=4,
            ),
        )

        self.stage3 = nn.Sequential(
            ConvBNReLU(48, 64, kernel_size=3, stride=2, padding=1),
            MobileViTBlock(
                in_channels=64,
                transformer_dim=96,
                depth=2,
                patch_size=(2, 2),
                num_heads=4,
            ),
        )

        # Classification head.
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward for classification.

        x: (B, 3, 32, 32)
        returns: (B, num_classes)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x).flatten(1)
        x = self.classifier(x)
        return x

    def forward_features_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward path used for the "Dimension Defense Test".

        This method applies a single MobileViTBlock that preserves the
        input channels and spatial dimensions (after removing padding),
        verifying that the patch-unfold/Transformer/fold pipeline can
        handle arbitrary (B, C, H, W) tensors.
        """
        c = x.shape[1]
        defense_block = MobileViTBlock(
            in_channels=c,
            transformer_dim=max(32, c),
            depth=1,
            patch_size=(2, 2),
            num_heads=2,
        ).to(x.device)
        return defense_block(x)


__all__ = ["MobileViTXXS"]


