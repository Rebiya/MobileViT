"""
Benchmark utilities:

- `evaluate_accuracy`: compute test accuracy.
- `measure_fps`: measure inference FPS over multiple iterations.
- `measure_flops`: estimate FLOPs with fvcore.
- `print_benchmark_table`: pretty-print comparison between models.
"""

from typing import Iterable, List, Tuple
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from fvcore.nn import FlopCountAnalysis


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute top-1 accuracy of a model on a given DataLoader."""
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples if total_samples > 0 else 0.0


def measure_fps(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
    warmup_iters: int = 10,
    measure_iters: int = 50,
) -> float:
    """
    Measure inference frames-per-second (FPS) for a single model.

    Uses a dummy input tensor and averages over `measure_iters`.
    """
    model.eval()
    x = torch.randn(*input_shape, device=device)

    # Warmup iterations (to stabilize GPU frequency, caches, etc.).
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(measure_iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    total_time = end - start
    fps = measure_iters / total_time if total_time > 0 else 0.0
    return fps


def measure_flops(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
) -> float:
    """
    Approximate FLOPs (in GigaFLOPs) for a single forward pass using fvcore.
    """
    model.eval()
    x = torch.randn(*input_shape, device=device)
    flop_analyzer = FlopCountAnalysis(model, x)
    flops = flop_analyzer.total()  # raw FLOPs
    return flops / 1e9


def print_benchmark_table(results: Iterable[Tuple[str, float, float, float]]) -> None:
    """
    Pretty-print benchmark results.

    Each element of `results` is (model_name, accuracy, fps, flops_g).
    """
    header = "| Model | Accuracy (%) | FPS | FLOPs (G) |"
    sep = "|-------|--------------|-----|-----------|"
    print(header)
    print(sep)
    for name, acc, fps, flops_g in results:
        print(f"| {name} | {acc*100:>11.2f} | {fps:>7.2f} | {flops_g:>9.3f} |")


__all__ = ["evaluate_accuracy", "measure_fps", "measure_flops", "print_benchmark_table"]


