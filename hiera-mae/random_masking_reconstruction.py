# Caveat of this code:
# The FAIR pretrained Hiera models appear to have been trained to predict
# locally-normalized patches, rather than the original patches.
# Thus, we keep texture/structure informatioin, but lose color.
# Denormalizing with global statistics is insufficient,
# but we don't have access to local statistics.
# Could probably be recovered with a little bit of fine-tuning.

import torch
from transformers import AutoImageProcessor, HieraForPreTraining
from PIL import Image
import matplotlib.pyplot as plt


def setup_device(device="cuda"):
    return torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")


def load_model_and_processor(model_name, device):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = HieraForPreTraining.from_pretrained(model_name).to(device).eval()
    return processor, model


def preprocess_image(image_path, processor, device):
    return processor(images=Image.open(image_path).convert('RGB').resize((224, 224)), return_tensors="pt").to(device)


def reconstruct_image(inputs, outputs, processor, model, device):
    pixel_values = inputs.pixel_values.squeeze(0) # bye batch
    assert pixel_values.shape == (3, 224, 224)

    # torch.BoolTensor of shape (sequence_length)
    # Tensor indicating which patches are masked (0) and which are not (1).
    # THIS IS IMPORTANT
    bool_masked_pos = outputs.bool_masked_pos[0]
    assert len(bool_masked_pos.shape) == 1, "should be indices"

    logits = outputs.logits.squeeze(0) # bye batch

    # Model and image parameters
    patch_size = model.pred_stride
    output_height, output_width = model.decoder.tokens_spatial_shape_final

    # The indices that ARE hidden / masked.
    # bool_masked_pos = 0 means that the patch is masked.
    masked_indices = torch.arange(output_height * output_width, device=device)[~bool_masked_pos]
    row_indices = masked_indices // output_width
    col_indices = masked_indices % output_width

    masked_image = pixel_values.clone()
    for row, col in zip(row_indices, col_indices):
        h_start, w_start = row * patch_size, col * patch_size
        masked_image[:, h_start:h_start + patch_size, w_start:w_start + patch_size] = 0

    # Decode the masked patches predicted by the model
    predicted_patches = logits[~bool_masked_pos]
    predicted_patches = predicted_patches.reshape(-1, model.config.num_channels, patch_size, patch_size)

    # Reconstruct the image by placing the predicted patches
    reconstructed_image = pixel_values.clone()
    for patch, row, col in zip(predicted_patches, row_indices, col_indices):
        h_start, w_start = row * patch_size, col * patch_size
        reconstructed_image[:, h_start:h_start + patch_size, w_start:w_start + patch_size] = patch

    # Denormalize images for visualization
    mean, std = map(
        lambda x: torch.tensor(x).view(-1, 1, 1).to(device),
        (processor.image_mean, processor.image_std)
    )
    denormalize = lambda x: (x * std + mean).clamp(0, 1)
    return map(denormalize, (inputs.pixel_values, masked_image, reconstructed_image))


def visualize_images(original, masked, reconstructed):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, [original, masked, reconstructed], ["Original", "Masked", "Reconstructed"]):
        img = img.squeeze(0)
        ax.imshow(img.cpu().permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_mae_reconstruction(image_path, model_name="facebook/hiera-huge-224-mae-hf", device="cuda"):
    device = setup_device(device)
    processor, model = load_model_and_processor(model_name, device)
    inputs = preprocess_image(image_path, processor, device)
    with torch.no_grad():
        outputs = model(**inputs)
    original, masked, reconstructed = reconstruct_image(inputs, outputs, processor, model, device)
    visualize_images(original, masked, reconstructed)
    return {'original': original, 'masked': masked, 'reconstructed': reconstructed}


if __name__ == "__main__":
    visualize_mae_reconstruction("data/yohai.png")
