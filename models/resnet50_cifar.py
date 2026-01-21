"""
ResNet-50 wrapper adapted for CIFAR10.

We use torchvision's ResNet-50 backbone and replace:
- The first conv layer to be more suitable for 32x32 images.
- The final fully-connected layer to output 10 classes.
"""

from typing import Optional

import torch
from torch import nn
from torchvision import models


def resnet50_cifar(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    Construct a ResNet-50 model for CIFAR10 classification.

    Args:
        num_classes: number of output classes (default: 10).
        pretrained: if True, loads ImageNet-pretrained weights for the backbone.
    """
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.resnet50(weights=weights)

    # Adjust first conv for small CIFAR10 images.
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()

    # Replace final classification layer.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


__all__ = ["resnet50_cifar"]


