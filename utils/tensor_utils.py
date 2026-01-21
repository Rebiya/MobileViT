"""
Tensor utilities for patch unfolding/folding and dynamic padding.

These helpers implement the patch/token processing described in the
MobileViT paper using `unfold`/`fold` while supporting arbitrary
input spatial dimensions via padding.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def pad_to_multiple(
    x: torch.Tensor, multiple: Tuple[int, int]
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Pad input tensor on the right/bottom so that H and W are divisible
    by `multiple` (h_mult, w_mult).

    Returns padded tensor and padding tuple (pad_left, pad_right, pad_top, pad_bottom).
    """
    b, c, h, w = x.shape
    h_mult, w_mult = multiple

    pad_h = (h_mult - h % h_mult) % h_mult
    pad_w = (w_mult - w % w_mult) % w_mult

    # F.pad uses (pad_left, pad_right, pad_top, pad_bottom)
    padding = (0, pad_w, 0, pad_h)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, padding, mode="reflect")
    return x, padding


def unpad(x: torch.Tensor, padding: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    Remove padding that was previously added with `pad_to_multiple`.
    """
    pad_left, pad_right, pad_top, pad_bottom = padding
    _, _, h, w = x.shape
    h_start = pad_top
    h_end = h - pad_bottom if pad_bottom > 0 else h
    w_start = pad_left
    w_end = w - pad_right if pad_right > 0 else w
    return x[..., h_start:h_end, w_start:w_end]


def unfold_patches(
    x: torch.Tensor, patch_size: Tuple[int, int]
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Unfold feature map into non-overlapping patches and flatten each patch
    to a token vector.

    Steps:
    1. Optionally pad H, W so they are multiples of patch_size.
    2. Use F.unfold to extract patches of size `patch_size` with stride `patch_size`.
    3. Transpose to shape (B, N, C * Ph * Pw), where N is number of patches.
    """
    b, c, h, w = x.shape
    ph, pw = patch_size

    x, padding = pad_to_multiple(x, (ph, pw))
    _, _, h_pad, w_pad = x.shape

    # F.unfold -> (B, C * Ph * Pw, L) where L is number of patches.
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    patches = patches.transpose(1, 2)  # (B, L, C * Ph * Pw)

    num_patches = (h_pad // ph) * (w_pad // pw)
    assert (
        patches.shape[1] == num_patches
    ), "Unexpected number of patches after unfolding."

    return patches, padding


def fold_patches(
    tokens: torch.Tensor,
    padding: Tuple[int, int, int, int],
    original_hw: Tuple[int, int],
    patch_size: Tuple[int, int],
    channels: int,
) -> torch.Tensor:
    """
    Inverse of `unfold_patches`:

    1. Reshape tokens (B, N, C * Ph * Pw) back to (B, C * Ph * Pw, N).
    2. Use F.fold with kernel_size=stride=patch_size to reconstruct a padded
       feature map.
    3. Remove padding to match the original (H, W).
    """
    b, n, dim = tokens.shape
    ph, pw = patch_size
    h_orig, w_orig = original_hw

    patches = tokens.transpose(1, 2)  # (B, C * Ph * Pw, N)

    # Recover padded spatial size.
    pad_left, pad_right, pad_top, pad_bottom = padding
    h_pad = h_orig + pad_top + pad_bottom
    w_pad = w_orig + pad_left + pad_right

    out = F.fold(
        patches,
        output_size=(h_pad, w_pad),
        kernel_size=patch_size,
        stride=patch_size,
    )  # (B, C, H_pad, W_pad)

    # Remove padding to return to original size.
    out = unpad(out, padding)
    return out


__all__ = ["pad_to_multiple", "unpad", "unfold_patches", "fold_patches"]


