
from transformers import AutoImageProcessor, ViTMAEForPreTraining
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class MAEVisualizer:
    def __init__(self, model_name="facebook/vit-mae-large"):
        """Initialize the MAE visualizer with model and processor."""
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTMAEForPreTraining.from_pretrained(model_name)
        self.patch_size = self.model.config.patch_size
        self.image_size = self.model.config.image_size
        self.num_patches_side = self.image_size // self.patch_size
        self.total_patches = self.num_patches_side ** 2

    def create_dual_square_mask(self, mask_ratio=0.75):
        """
        Creates a mask with two non-overlapping squares of visible (unmasked) patches.
        Maintains exactly mask_ratio of masked patches.

        Args:
            mask_ratio: Desired ratio of masked patches (e.g., 0.75)
        """
        assert 0 < mask_ratio < 1, "Mask ratio must be between 0 and 1"

        # Initialize mask (1 = masked, 0 = visible)
        mask = torch.ones(1, self.total_patches)
        num_visible_patches = int(self.total_patches * (1 - mask_ratio))

        # Calculate size for each square
        patches_per_square = num_visible_patches // 2
        square_side = int(np.sqrt(patches_per_square))

        def get_patch_idx(x, y):
            return y * self.num_patches_side + x

        def make_square_visible(top_left_x, top_left_y, remaining_patches):
            """Make a square of patches visible starting from given coordinates."""
            patches_made_visible = 0
            for y in range(top_left_y, min(top_left_y + square_side, self.num_patches_side)):
                for x in range(top_left_x, min(top_left_x + square_side, self.num_patches_side)):
                    if remaining_patches > 0:
                        idx = get_patch_idx(x, y)
                        if mask[0, idx] == 1:  # Only if it's still masked
                            mask[0, idx] = 0
                            patches_made_visible += 1
                            remaining_patches -= 1
            return patches_made_visible

        def get_random_position():
            """Get random position ensuring the square fits within image bounds."""
            max_pos = self.num_patches_side - square_side
            return np.random.randint(0, max_pos + 1), np.random.randint(0, max_pos + 1)

        # Place first square
        x1, y1 = get_random_position()
        remaining = num_visible_patches
        made_visible = make_square_visible(x1, y1, patches_per_square)
        remaining -= made_visible

        # Place second square (ensure no overlap)
        max_attempts = 100
        for _ in range(max_attempts):
            x2, y2 = get_random_position()

            # Check if squares would overlap
            x1_range = range(x1, min(x1 + square_side, self.num_patches_side))
            y1_range = range(y1, min(y1 + square_side, self.num_patches_side))
            x2_range = range(x2, min(x2 + square_side, self.num_patches_side))
            y2_range = range(y2, min(y2 + square_side, self.num_patches_side))

            if not (any(x in x2_range for x in x1_range) and
                   any(y in y2_range for y in y1_range)):
                made_visible = make_square_visible(x2, y2, remaining)
                remaining -= made_visible
                break

        # Add any remaining visible patches adjacent to either square if needed
        if remaining > 0:
            for y in range(self.num_patches_side):
                for x in range(self.num_patches_side):
                    if remaining > 0 and mask[0, get_patch_idx(x, y)] == 1:
                        mask[0, get_patch_idx(x, y)] = 0
                        remaining -= 1

        # Verify exact count
        actual_masked = mask.sum().item()
        expected_masked = int(self.total_patches * mask_ratio)
        assert actual_masked == expected_masked, \
            f"Expected {expected_masked} masked patches, got {actual_masked}"

        return mask, (x1, y1), (x2, y2)

    def process_image(self, image_path):
        """Process image and return tensor."""
        image = Image.open(image_path)
        image = image.resize((self.image_size, self.image_size))
        inputs = self.processor(image, return_tensors="pt")
        return inputs["pixel_values"]

    def reconstruct_image(self, pixel_values, mask):
        """Run MAE reconstruction with given mask."""
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, noise=mask.float(), return_dict=True)
            reconstructed_patches = outputs.logits
            reconstructed_image = self.model.unpatchify(reconstructed_patches)

            # Replace unmasked patches with original
            for i in range(self.total_patches):
                if not mask[0, i]:
                    row = (i // self.num_patches_side) * self.patch_size
                    col = (i % self.num_patches_side) * self.patch_size
                    reconstructed_image[0, :, row:row + self.patch_size,
                                     col:col + self.patch_size] = \
                        pixel_values[0, :, row:row + self.patch_size,
                                   col:col + self.patch_size]

        return reconstructed_image

    def visualize(self, image_path, mask_ratio=0.75):
        """Complete visualization pipeline."""
        pixel_values = self.process_image(image_path)
        mask, (x1, y1), (x2, y2) = self.create_dual_square_mask(mask_ratio)
        reconstructed_image = self.reconstruct_image(pixel_values, mask)

        # Prepare images for visualization
        def denormalize(tensor):
            mean = torch.tensor(self.processor.image_mean).view(3, 1, 1)
            std = torch.tensor(self.processor.image_std).view(3, 1, 1)
            return torch.clamp(tensor * std + mean, 0, 1)

        original_image = denormalize(pixel_values.squeeze())
        reconstructed_image = denormalize(reconstructed_image.squeeze())

        # Create masked visualization
        masked_image = original_image.clone()
        mask_value = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)  # Red for masked regions

        for i in range(self.total_patches):
            if mask[0, i]:
                row = (i // self.num_patches_side) * self.patch_size
                col = (i % self.num_patches_side) * self.patch_size
                masked_image[:, row:row + self.patch_size,
                           col:col + self.patch_size] = mask_value

        # Display results
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for img, title, idx in zip(
            [original_image, masked_image, reconstructed_image],
            ["Original Image",
             f"Dual Square Masked Image ({mask_ratio*100:.1f}% masked)",
             "Reconstructed Image"],
            range(3)
        ):
            axs[idx].imshow(img.permute(1, 2, 0).numpy())
            axs[idx].set_title(title)
            axs[idx].axis("off")

        # Print statistics
        print(f"Square 1 position: ({x1}, {y1})")
        print(f"Square 2 position: ({x2}, {y2})")
        print(f"Square side length: {int(np.sqrt(self.total_patches * (1 - mask_ratio) / 2))} patches")
        print(f"Total patches: {self.total_patches}")
        print(f"Visible patches: {int(self.total_patches * (1 - mask_ratio))}")

        plt.show()

# Usage
visualizer = MAEVisualizer()
visualizer.visualize("data/yohai.png", mask_ratio=0.75)
