"""
Data loading utilities for CIFAR10.

This module provides a helper to construct train and test DataLoaders
with standard augmentations and normalization.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_dataloaders(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for CIFAR10 with standard transforms.

    Transforms (train):
        - RandomCrop(32, padding=4)
        - RandomHorizontalFlip()
        - ToTensor()
        - Normalize with CIFAR10 mean/std

    Transforms (test):
        - ToTensor()
        - Normalize with CIFAR10 mean/std
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


__all__ = ["get_cifar10_dataloaders"]


