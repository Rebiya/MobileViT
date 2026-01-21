"""
Streamlit inference app for MobileViT XXS on CIFAR-10.

This app:
- Loads the trained MobileViT XXS weights from checkpoints/mobilevit_xxs_cifar10.pkl
- Accepts an uploaded image (jpg/png), resizes to 32x32 (CIFAR-10 resolution),
  applies the same normalization used during training, and runs inference.
- Displays the predicted CIFAR-10 class with confidence.

Notes:
- No training occurs here; we only perform inference.
- The model is cached to avoid reloading on every interaction.
- Resizing is required because the model was trained on 32x32 CIFAR-10 inputs.
- Normalization must match training statistics to keep predictions consistent.
- state_dict is used for portability and CPU-safe loading.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import streamlit as st
from torchvision import transforms

from models.mobilevit_xxs import MobileViTXXS


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@st.cache_resource
def load_model() -> torch.nn.Module:
    """
    Load the MobileViT XXS model and weights onto CPU for inference.
    Using state_dict avoids coupling to training-side optimizer/etc.
    """
    model = MobileViTXXS(num_classes=10)
    state = torch.load(
        "checkpoints/mobilevit_xxs_cifar10.pkl", map_location="cpu"
    )
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convert PIL image to normalized tensor matching CIFAR-10 training pipeline.

    Steps:
    - Convert to RGB (ensures 3 channels)
    - Resize to 32x32 (model was trained on CIFAR-10 resolution)
    - ToTensor + Normalize with CIFAR-10 stats
    - Add batch dimension for inference
    """
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ]
    )
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)  # (1, 3, 32, 32)
    return tensor


def predict(model: torch.nn.Module, tensor: torch.Tensor) -> Tuple[str, float]:
    """Run inference and return predicted class and confidence."""
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)
    return CIFAR10_CLASSES[pred_idx.item()], conf.item() * 100.0


def main() -> None:
    st.set_page_config(
        page_title="MobileViT CIFAR-10 Image Classifier",
        layout="centered",
    )
    st.title("MobileViT CIFAR-10 Image Classifier")
    st.write(
        "Upload an image (jpg/png). The model was trained on CIFAR-10 (32×32). "
        "We resize to 32×32 and apply the same normalization as training."
    )

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Classify Image"):
            model = load_model()
            tensor = preprocess_image(image)
            cls, conf = predict(model, tensor)
            st.success(f"Predicted: {cls} (confidence: {conf:.2f}%)")


if __name__ == "__main__":
    main()


