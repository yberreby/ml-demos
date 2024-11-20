# utils/image_preprocessing.py

from typing import List, Union
import torch
import numpy as np
from PIL import Image


def preprocess_image(
    image_path: str,
    image_size: int,
    mean: List[float],
    std: List[float],
    device: str = 'cuda'
) -> torch.Tensor:
    """Preprocesses an image for MAE models."""
    device = 'cpu' if not torch.cuda.is_available() and device == 'cuda' else device

    # Load and resize image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)

    # Convert to normalized tensor
    img_tensor = torch.tensor(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # Normalize
    mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (img_tensor - mean_tensor) / std_tensor
