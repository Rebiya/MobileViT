"""
Dimension Defense Test:
Verifies that MobileViT XXS can handle arbitrary (B, C, H, W) inputs
by dynamically padding and unpadding around patch operations.
"""

import torch

from models.mobilevit_xxs import MobileViTXXS


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MobileViTXXS(num_classes=10).to(device)

    # Unusual input shape requested in the specification.
    x = torch.randn(3, 47, 33, 33, device=device)

    with torch.no_grad():
        y = model.forward_features_only(x)

    assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    print("Dimension Defense Test Passed")


if __name__ == "__main__":
    main()


