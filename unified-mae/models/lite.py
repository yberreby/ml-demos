import os
import sys
from typing import Tuple, Dict, Any, Optional, List
import torch
from timm.models import create_model
from utils.image_processing import apply_mask


class MAELite:
    def __init__(
        self,
        checkpoint_path: str = "ckpt/mae_tiny_400e.pth.tar",
        device: str = 'cuda',
        mae_root: Optional[str] = None
    ):
        self.device = 'cpu' if not torch.cuda.is_available() and device == 'cuda' else device

        # Add MAE-Lite paths to system path
        if mae_root is None:
            mae_root = os.path.join(os.path.dirname(__file__), '..', '..', 'thirdparty', 'mae-lite')
        for path in [mae_root, os.path.join(mae_root, 'projects', 'mae_lite')]:
            if path not in sys.path:
                sys.path.append(path)

        from models_mae import mae_vit_tiny_patch16

        # Initialize and load model
        self.model = create_model('mae_vit_tiny_patch16', pretrained=False, norm_pix_loss=True)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        msg = self.model.load_state_dict({
            k[len('module.model.'):] if k.startswith('module.model.') else k: v
            for k, v in state_dict.items()
        })
        print(f"Loaded checkpoint: {msg}")
        self.model.to(device).eval()

        # Model config
        self.image_size = 224  # Fixed for MAE-Lite
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.num_patches_side = self.image_size // self.patch_size
        self.total_patches = self.num_patches_side ** 2
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def reconstruct(
        self,
        img_tensor: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not all(isinstance(x, torch.Tensor) for x in [img_tensor, mask]):
            raise ValueError("Inputs must be torch.Tensors")

        # Convert mask to indices and run model
        mask = mask.to(torch.int)
        ids_shuffle = torch.argsort(mask, dim=1)
        mask_ratio = kwargs.get('mask_ratio', 0.75)

        with torch.no_grad():
            _, pred, mask_out, _ = self.model(img_tensor, mask_ratio=mask_ratio, ids_shuffle=ids_shuffle)
            reconstructed = self.model.unpatchify(pred)

            # Replace unmasked patches
            for b in range(img_tensor.shape[0]):
                for i, m in enumerate(mask_out[b]):
                    if not m:  # If patch should be visible
                        row = (i // self.num_patches_side) * self.patch_size
                        col = (i % self.num_patches_side) * self.patch_size
                        reconstructed[b, :, row:row + self.patch_size, col:col + self.patch_size] = \
                            img_tensor[b, :, row:row + self.patch_size, col:col + self.patch_size]

        # Create masked visualization
        mask_indices = torch.where(mask_out)[1]
        masked = apply_mask(
            mask_indices=mask_indices,
            num_patches_side=self.num_patches_side,
            patch_size=self.patch_size,
            image_tensor=img_tensor.clone(),
            fill_value=0.5
        )

        # Denormalize
        mean = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=self.device).view(1, 3, 1, 1)
        denorm = lambda x: torch.clamp(x * std + mean, 0, 1)

        return denorm(reconstructed), denorm(masked), denorm(img_tensor)

    @property
    def image_mean(self) -> List[float]:
        return self.mean

    @property
    def image_std(self) -> List[float]:
        return self.std
