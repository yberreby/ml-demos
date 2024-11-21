# utils/masking_patterns.py

from typing import Tuple, Union
import torch
import numpy as np


def create_random_mask(
    num_patches: int,
    mask_ratio: float,
    device: str = 'cuda'
) -> torch.Tensor:
    """Creates a random boolean mask for MAE models."""
    if not 0 <= mask_ratio <= 1:
        raise ValueError(f"Invalid mask_ratio: {mask_ratio}")

    device = 'cpu' if not torch.cuda.is_available() and device == 'cuda' else device

    # Create mask and randomly select patches to mask
    mask = torch.ones(num_patches, dtype=torch.bool, device=device)
    num_masked = min(int(num_patches * mask_ratio), num_patches)
    mask[torch.randperm(num_patches, device=device)[:num_masked]] = False

    return ~mask.unsqueeze(0)  # Add batch dim, invert mask


def create_dual_square_mask(
    num_patches_side: int,
    mask_ratio: float,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """Creates a mask with two non-overlapping visible squares."""
    if not 0 <= mask_ratio <= 1:
        raise ValueError(f"Invalid mask_ratio: {mask_ratio}")

    device = 'cpu' if not torch.cuda.is_available() and device == 'cuda' else device
    total_patches = num_patches_side ** 2

    # Initialize mask and calculate sizes
    mask = torch.ones(1, total_patches, dtype=torch.bool, device=device)
    visible_patches = int(total_patches * (1 - mask_ratio))
    if visible_patches < 2:
        raise ValueError("mask_ratio too high for two squares")

    square_side = int(np.sqrt(visible_patches // 2))
    if square_side == 0:
        raise ValueError("Not enough visible patches for squares")

    def get_random_pos():
        max_pos = num_patches_side - square_side
        return np.random.randint(0, max_pos + 1), np.random.randint(0, max_pos + 1)

    def make_square_visible(x: int, y: int, patches_left: int) -> int:
        visible = 0
        for i in range(square_side):
            for j in range(square_side):
                if patches_left <= 0:
                    break
                idx = (y + i) * num_patches_side + (x + j)
                if mask[0, idx]:  # Only if not already visible
                    mask[0, idx] = False
                    visible += 1
                    patches_left -= 1
        return visible

    # Place first square
    x1, y1 = get_random_pos()
    remaining = visible_patches
    made_visible = make_square_visible(x1, y1, remaining)
    remaining -= made_visible

    # Try to place second square without overlap
    for _ in range(100):
        x2, y2 = get_random_pos()
        if (abs(x2 - x1) >= square_side or abs(y2 - y1) >= square_side):
            made_visible = make_square_visible(x2, y2, remaining)
            remaining -= made_visible

            # Add any remaining visible patches randomly if needed
            if remaining > 0:
                available = torch.where(mask[0])[0]
                indices = available[torch.randperm(len(available))[:remaining]]
                mask[0, indices] = False

            return mask, (x1, y1), (x2, y2)

    raise RuntimeError("Failed to place non-overlapping squares")
