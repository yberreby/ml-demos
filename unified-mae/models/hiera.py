from typing import List
import torch
from transformers import AutoImageProcessor, HieraForPreTraining
from utils.image_processing import apply_mask


class HieraMAE:
    def __init__(
        self,
        model_name: str = "facebook/hiera-huge-224-mae-hf",
        device: str = 'cuda'
    ):
        self.device = 'cpu' if not torch.cuda.is_available() and device == 'cuda' else device

        # Model setup
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = HieraForPreTraining.from_pretrained(model_name)
        self.model.to(device).eval()

        # Key difference from ViT: uses pred_stride instead of patch_size
        self.image_size = self.model.config.image_size[0]
        self.patch_size = self.model.pred_stride
        self.num_patches_side = self.image_size // self.patch_size
        self.total_patches = self.num_patches_side ** 2

    def reconstruct(
        self,
        img_tensor: torch.Tensor,
        mask: torch.Tensor,  # Unused - Hiera handles masking internally
        **kwargs
    ):
        if not isinstance(img_tensor, torch.Tensor):
            raise ValueError("img_tensor must be torch.Tensor")

        # Generate reconstruction using Hiera's internal masking
        with torch.no_grad():
            outputs = self.model(pixel_values=img_tensor)
            bool_masked_pos = outputs.bool_masked_pos
            logits = outputs.logits

            # Reconstruct image
            reconstructed = self._reconstruct_image(img_tensor, logits, bool_masked_pos)

            # Create masked visualization using model's masking pattern
            mask_indices = torch.where(~bool_masked_pos)[1]  # Remove batch dim, invert for viz
            masked = apply_mask(
                mask_indices=mask_indices,
                num_patches_side=self.num_patches_side,
                patch_size=self.patch_size,
                image_tensor=img_tensor,
                fill_value=0.5,
                fill_color=kwargs.get('mask_color'),
                add_grid=kwargs.get('visualize_grid', False)
            )

        # Denormalize outputs
        mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1).to(self.device)
        denorm = lambda x: torch.clamp(x * std + mean, 0, 1)

        return denorm(reconstructed), denorm(masked), denorm(img_tensor)

    def _reconstruct_image(
        self,
        pixel_values: torch.Tensor,
        logits: torch.Tensor,
        bool_masked_pos: torch.Tensor
    ):
        pixel_values = pixel_values.squeeze(0)
        patch_size = self.patch_size
        output_height, output_width = self.model.decoder.tokens_spatial_shape_final

        # Get indices of masked patches and their positions
        masked_indices = torch.arange(output_height * output_width, device=self.device)[~bool_masked_pos.squeeze(0)]
        row_indices = masked_indices // output_width
        col_indices = masked_indices % output_width

        # Get predicted patches and reshape
        predicted_patches = logits[~bool_masked_pos].reshape(
            -1, self.model.config.num_channels, patch_size, patch_size
        )

        # Replace masked patches with predictions
        reconstructed = pixel_values.clone()
        for patch, row, col in zip(predicted_patches, row_indices, col_indices):
            h_start, w_start = row * patch_size, col * patch_size
            reconstructed[:, h_start:h_start + patch_size, w_start:w_start + patch_size] = patch

        return reconstructed.unsqueeze(0)

    @property
    def image_mean(self) -> List[float]:
        return self.processor.image_mean

    @property
    def image_std(self) -> List[float]:
        return self.processor.image_std
