from dataclasses import dataclass
from pathlib import Path
import torch
from typing import Optional, Dict, Any
import typer

from models import HieraMAE, ViTMAE, MAELite
from utils import (
    preprocess_image, create_random_mask, create_dual_square_mask,
    visualize_results
)
from config import MAEConfig


def get_device(requested: str = 'cuda') -> str:
    return 'cpu' if not torch.cuda.is_available() and requested == 'cuda' else requested

class MAEDemo:
    def __init__(
        self,
        model_type: str,
        device: str,
        config: MAEConfig,
        checkpoint_path: Optional[Path] = None
    ):
        model_map = {
            'hiera': lambda: HieraMAE(device=device),
            'vit': lambda: ViTMAE(device=device),
            'lite': lambda: MAELite(checkpoint_path=str(checkpoint_path), device=device)
        }

        if model_type not in model_map:
            raise ValueError(f"Model type must be one of {list(model_map.keys())}")
        if model_type == 'lite' and not checkpoint_path:
            raise ValueError("checkpoint_path required for MAE-Lite")

        self.model = model_map[model_type]()
        self.config = config

    def process_image(self, image_path: Path) -> torch.Tensor:
        return preprocess_image(
            image_path=str(image_path),
            image_size=self.model.image_size,
            mean=self.model.image_mean,
            std=self.model.image_std,
            device=self.model.device
        )

    def create_mask(
        self,
        mask_type: str,
        mask_ratio: float
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        if mask_type == 'random':
            return create_random_mask(
                self.model.total_patches,
                mask_ratio,
                device=self.model.device
            ), None

        elif mask_type == 'dual_square':
            mask, pos1, pos2 = create_dual_square_mask(
                self.model.num_patches_side,
                mask_ratio,
                device=self.model.device
            )
            return mask, (pos1, pos2)

        raise ValueError(f"Unknown mask type: {mask_type}")

    def run(
        self,
        image_path: Path,
        mask_type: str = 'random',
        mask_ratio: float = 0.75,
        viz_opts: Optional[Dict[str, Any]] = None
    ) -> None:
        img_tensor = self.process_image(image_path)
        mask, mask_info = self.create_mask(mask_type, mask_ratio)

        # Run reconstruction
        reconstructed, masked, original = self.model.reconstruct(
            img_tensor,
            mask,
            **{**self.config.VIZ_DEFAULTS, **(viz_opts or {})}
        )

        # Visualize
        titles = [
            "Original",
            f"Masked ({mask_type}, {mask_ratio:.0%})",
            f"Reconstructed ({type(self.model).__name__})"
        ]
        visualize_results((original, masked, reconstructed), titles, **(viz_opts or {}))

        if mask_info:
            print(f"Square positions: {mask_info}")

def main(
    model: str = typer.Option('vit', help="Model type (hiera/vit/lite)"),
    image: Path = typer.Option("data/yohai.png", help="Input image path"),
    mask_type: str = typer.Option('random', help="Masking pattern"),
    mask_ratio: float = typer.Option(0.75, help="Ratio of patches to mask"),
    device: str = typer.Option('cuda', help="Device to use"),
    checkpoint: Optional[Path] = typer.Option(None, help="MAE-Lite checkpoint"),
    no_grid: bool = typer.Option(False, help="Disable grid overlay")
) -> None:
    """Run MAE visualization demo."""
    config = MAEConfig()
    config.validate()

    viz_opts = {**config.VIZ_DEFAULTS, 'grid': not no_grid}

    demo = MAEDemo(
        model_type=model,
        device=get_device(device),
        config=config,
        checkpoint_path=checkpoint
    )

    demo.run(image, mask_type, mask_ratio, viz_opts)

if __name__ == "__main__":
    typer.run(main)
