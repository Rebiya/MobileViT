"""
Entry point for training MobileViT XXS on CIFAR10.
"""

import os
import torch

from utils.data_utils import get_cifar10_dataloaders
from utils.train_utils import train_model
from models.mobilevit_xxs import MobileViTXXS


def main() -> None:
    """Train MobileViT XXS on CIFAR10 and save the trained weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_dataloaders(
        data_root="./data/cifar10", batch_size=64
    )

    model = MobileViTXXS(num_classes=10).to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        epochs=20,
        lr=1e-3,
        use_amp=True,
        save_path="./experiments/run_logs/mobilevit_xxs_cifar10.pt",
    )

    # ----- Saving (post-training) -----
    os.makedirs("checkpoints", exist_ok=True)
    model.eval()
    model.cpu()  # ensure checkpoint is CPU-loadable everywhere
    ckpt_path = "checkpoints/mobilevit_xxs_cifar10.pkl"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()



