# utils/visualization.py

from typing import List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt
import torch
import numpy as np


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> torch.Tensor:
    """Denormalizes image tensor with optional clamping."""
    mean_t = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std_t + mean_t, 0, 1)

def visualize_results(
    tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    titles: List[str],
    **kwargs: Any
) -> None:
    """Visualizes original, masked, and reconstructed images."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=kwargs.get('figure_size', (15, 5)))

    for ax, img, title in zip(axes, tensors, titles):
        ax.imshow(img.squeeze().permute(1, 2, 0).cpu())
        ax.set_title(title)
        ax.axis('off')
        if kwargs.get('grid', True):
            ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    if save_path := kwargs.get('save_path'):
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
