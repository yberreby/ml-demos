# models/vit.py
from typing import Tuple, Dict, Any, Optional, Union, List
import torch
from transformers import AutoImageProcessor, ViTMAEForPreTraining
from PIL import Image
import numpy as np
from utils.image_processing import apply_mask


class ViTMAE:
    def __init__(self, model_name: str = "facebook/vit-mae-large", device: str = 'cuda'):
        self.device = 'cpu' if not torch.cuda.is_available() and device == 'cuda' else device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTMAEForPreTraining.from_pretrained(model_name)
        self.model.to(device).eval()

        # Model config
        self.patch_size = self.model.config.patch_size
        self.image_size = self.model.config.image_size
        self.num_patches_side = self.image_size // self.patch_size
        self.total_patches = self.num_patches_side ** 2

    def process_image(self, image: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """Converts input to tensor format expected by model."""
        if isinstance(image, torch.Tensor):
            return image.unsqueeze(0) if len(image.shape) == 3 else image

        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((self.image_size, self.image_size))
        inputs = self.processor(image, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def reconstruct(
        self,
        img_tensor: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstructs masked regions of the input image."""
        if not isinstance(img_tensor, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise ValueError("Inputs must be torch.Tensors")

        working_tensor = img_tensor.clone()
        noise = mask.float()

        # Generate reconstruction
        with torch.no_grad():
            outputs = self.model(pixel_values=working_tensor, noise=noise)
            reconstructed = self.model.unpatchify(outputs.logits)

            # Replace unmasked patches with original content
            for b in range(img_tensor.shape[0]):
                for i in range(self.total_patches):
                    if not mask[b, i]:  # If patch should be visible
                        row = (i // self.num_patches_side) * self.patch_size
                        col = (i % self.num_patches_side) * self.patch_size
                        reconstructed[b, :,
                            row:row + self.patch_size,
                            col:col + self.patch_size] = \
                            working_tensor[b, :,
                                row:row + self.patch_size,
                                col:col + self.patch_size]

        # Create masked visualization
        mask_indices = torch.where(mask)[1]
        masked_image = apply_mask(
            mask_indices=mask_indices,
            num_patches_side=self.num_patches_side,
            patch_size=self.patch_size,
            image_tensor=working_tensor,
            fill_value=kwargs.get('fill_value', 0.5),
            fill_color=kwargs.get('mask_color'),
            add_grid=kwargs.get('visualize_grid', False)
        )

        # Denormalize
        mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1).to(self.device)
        denorm = lambda x: torch.clamp(x * std + mean, 0, 1)

        return denorm(reconstructed), denorm(masked_image), denorm(working_tensor)

    @property
    def image_mean(self) -> List[float]:
        return self.processor.image_mean

    @property
    def image_std(self) -> List[float]:
        return self.processor.image_std
