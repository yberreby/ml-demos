# utils/image_processing.py

from typing import Union, List, Optional
import torch

def apply_mask(
    mask_indices: torch.Tensor,
    num_patches_side: int,
    patch_size: int,
    image_tensor: torch.Tensor,
    fill_value: float = 0.5,
    fill_color: Optional[List[float]] = None,
    add_grid: bool = False,
    grid_color: float = 0.8
) -> torch.Tensor:
    """Applies masking to image tensor with optional grid visualization."""
    if len(image_tensor.shape) != 4:
        raise ValueError(f"Expected 4D tensor (B,C,H,W), got shape {image_tensor.shape}")

    batch_size, num_channels = image_tensor.shape[:2]
    device = image_tensor.device
    masked_image = image_tensor.clone()

    # Set fill color/value
    fill = (torch.tensor(fill_color, device=device).view(1, 3, 1, 1)
           if fill_color else torch.full((1, num_channels, 1, 1), fill_value, device=device))

    # Apply mask
    for b in range(batch_size):
        for idx in mask_indices:
            y, x = (idx // num_patches_side).item(), (idx % num_patches_side).item()
            h_start, w_start = y * patch_size, x * patch_size
            masked_image[b, :, h_start:h_start + patch_size, w_start:w_start + patch_size] = fill

    # Add grid if requested
    if add_grid:
        grid = torch.ones_like(masked_image) * grid_color
        for i in range(1, num_patches_side):
            pos = i * patch_size
            masked_image[:, :, :, pos:pos+1] = grid[:, :, :, pos:pos+1]  # Vertical
            masked_image[:, :, pos:pos+1, :] = grid[:, :, pos:pos+1, :]  # Horizontal

    return masked_image
