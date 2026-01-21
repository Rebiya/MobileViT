"""
Training and evaluation utilities for MobileViT XXS and ResNet-50.

Includes:
- Standard training loop with CrossEntropyLoss.
- Mixed precision support with torch.cuda.amp.
- Simple accuracy computation per epoch.
"""

from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy given raw logits and integer labels."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
) -> Tuple[float, float]:
    """Train the model for a single epoch, returning (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation/test set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    use_amp: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Full training loop for a model on CIFAR10.

    Args:
        model: nn.Module to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation/test data.
        device: computation device.
        epochs: number of training epochs.
        lr: learning rate for Adam optimizer.
        use_amp: whether to use mixed precision (if CUDA is available).
        save_path: where to store the final checkpoint (optional).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, scaler=scaler
        )
        val_loss, val_acc = validate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{epochs:03d} "
            f"| Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% "
            f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
        )

    if save_path is not None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "history": history,
            },
            save_path,
        )

    return history


def plot_accuracy_curves(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """Optional helper to plot accuracy vs epoch."""
    epochs = range(1, len(history["train_acc"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


__all__ = ["train_model", "train_one_epoch", "validate", "plot_accuracy_curves"]


