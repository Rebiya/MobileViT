"""
Benchmark script: compares MobileViT XXS vs ResNet-50 on CIFAR10.
Measures accuracy, FPS, and FLOPs.
"""

import torch

from utils.data_utils import get_cifar10_dataloaders
from utils.benchmark_utils import (
    measure_fps,
    measure_flops,
    evaluate_accuracy,
    print_benchmark_table,
)
from models.mobilevit_xxs import MobileViTXXS
from models.resnet50_cifar import resnet50_cifar


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_dataloaders(
        data_root="./data/cifar10", batch_size=64
    )

    mobilevit = MobileViTXXS(num_classes=10).to(device)
    resnet = resnet50_cifar(num_classes=10).to(device)

    # Quick training (or assume pretrained checkpoints)
    # For simplicity here we just evaluate randomly initialized models;
    # in practice you should first train and load checkpoints.

    models = {
        "MobileViT-XXS": mobilevit,
        "ResNet-50": resnet,
    }

    results = []
    for name, model in models.items():
        acc = evaluate_accuracy(model, test_loader, device)
        fps = measure_fps(model, device=device, input_shape=(1, 3, 32, 32))
        flops_g = measure_flops(model, device=device, input_shape=(1, 3, 32, 32))
        results.append((name, acc, fps, flops_g))

    print_benchmark_table(results)


if __name__ == "__main__":
    main()


