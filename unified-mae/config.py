from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
import torch
from pathlib import Path

@dataclass(frozen=True)
class MAEConfig:
    IMAGE_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
    IMAGE_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)
    DEFAULT_MASK_RATIO: float = 0.75
    DEFAULT_IMAGE_SIZE: int = 224
    VIZ_DEFAULTS: Dict[str, Any] = field(default_factory=lambda: {
        'grid': True,
        'figure_size': (15, 5),
        'mask_color': [0.5, 0.5, 0.5]
    })

    def validate(self) -> None:
        assert len(self.IMAGE_MEAN) == len(self.IMAGE_STD) == 3
        assert 0 < self.DEFAULT_MASK_RATIO < 1
