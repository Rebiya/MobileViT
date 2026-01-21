"""
Building blocks for MobileViT XXS:

- ConvBNReLU: local convolutional feature extractor.
- MultiHeadSelfAttention: standard MHA with learnable projections.
- TransformerEncoder: MHA + MLP with residual connections.
- MobileViTBlock: combines local CNN features with global transformer
  features using unfold/fold patch processing.
"""

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from utils.tensor_utils import unfold_patches, fold_patches


class ConvBNReLU(nn.Sequential):
    """A basic Conv2d -> BatchNorm2d -> ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention operating on sequences of tokens.

    Input:  (B, N, D)
    Output: (B, N, D)
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, num_heads, N, head_dim)
        def _reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        q = _reshape(q)
        k = _reshape(k)
        v = _reshape(v)

        # Scaled dot-product attention.
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class TransformerEncoder(nn.Module):
    """
    A simple Transformer encoder block with:
    - LayerNorm
    - Multi-head self-attention
    - MLP (two linear layers + GELU)
    - Residual connections
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """
    MobileViT block:

    1. Local representation:
       - Two 3x3 ConvBNReLU layers extract local features.
    2. Global representation:
       - 1x1 conv to project to transformer dimension.
       - Unfold spatial feature map into non-overlapping patches.
       - Each patch is flattened to a token vector.
       - Learnable positional encoding per patch.
       - Several Transformer encoder layers process tokens globally.
       - Tokens are folded back to the spatial layout using F.fold.
    3. Fusion:
       - 1x1 conv to project back to original channel dimension.
       - Residual connection with the input.

    Dynamic shapes:
    - H and W are automatically padded so that they are divisible by
      the patch size, and padding is removed after folding.
    """

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        depth: int = 2,
        patch_size: Tuple[int, int] = (2, 2),
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.transformer_dim = transformer_dim
        self.patch_size = patch_size

        # Local representation (CNN).
        self.local_conv = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )

        # Channel projection into transformer dimension and back.
        self.conv1x1_in = nn.Conv2d(in_channels, transformer_dim, kernel_size=1)
        self.conv1x1_out = nn.Conv2d(transformer_dim, in_channels, kernel_size=1)

        # A small stack of Transformer encoder blocks.
        self.transformers = nn.ModuleList(
            [
                TransformerEncoder(
                    dim=transformer_dim,
                    num_heads=num_heads,
                    mlp_ratio=2.0,
                    dropout=0.0,
                )
                for _ in range(depth)
            ]
        )

        # Positional encoding for patches. We initialize with a maximum number
        # of patches, but only the required prefix is used at runtime.
        self.max_patches = 1024
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.max_patches, transformer_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        identity = x

        # Local representation.
        x = self.local_conv(x)  # (B, C, H, W)

        # Channel projection for transformer processing.
        b, c, h, w = x.shape
        x = self.conv1x1_in(x)  # (B, D, H, W)

        # Unfold into patches and flatten to tokens.
        # tokens: (B, N, D * Ph * Pw / something) but we reshape to (B, N, D).
        ph, pw = self.patch_size
        tokens, padding = unfold_patches(x, patch_size=self.patch_size)
        # tokens: (B, N, D * Ph * Pw)
        b, n, dim_tokens = tokens.shape
        d = self.transformer_dim

        # Project flattened patch to transformer dim if necessary.
        if dim_tokens != d:
            tokens = nn.Linear(dim_tokens, d, device=x.device)(tokens)
        else:
            d = dim_tokens

        # Add positional encoding (truncate to required length).
        pos = self.pos_embed[:, :n, :]
        tokens = tokens + pos

        # Global representation via Transformer encoders.
        for blk in self.transformers:
            tokens = blk(tokens)  # (B, N, D)

        # Optionally project back to patch dim if we projected earlier.
        if dim_tokens != d:
            tokens = nn.Linear(d, dim_tokens, device=x.device)(tokens)

        # Fold tokens back to spatial layout.
        x = fold_patches(
            tokens=tokens,
            padding=padding,
            original_hw=(h, w),
            patch_size=self.patch_size,
            channels=self.transformer_dim,
        )  # (B, D, H, W)

        # Channel projection back to input channels.
        x = self.conv1x1_out(x)  # (B, C, H, W)

        # Residual fusion.
        x = x + identity
        return x


__all__ = ["ConvBNReLU", "MultiHeadSelfAttention", "TransformerEncoder", "MobileViTBlock"]


