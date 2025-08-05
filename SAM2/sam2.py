from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import torch
import numpy as np
import random


def load_sam(sam2_checkpoint: str, model_cfg: str, device: str):
    sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    return sam2_predictor


def segment_image(image: np.ndarray, sam2_predictor, grounded_boxes):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam2_predictor.set_image(np.array(image))
        sam_masks, sam_scores, sam_logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=grounded_boxes,
            multimask_output=False
        )
    return sam_masks, sam_scores, sam_logits


def visualize_sam_masks(image: np.ndarray, sam_masks, alpha=0.3):
    final_image = image.copy().astype(np.float32)
    overlay_layer = np.zeros_like(image, dtype=np.float32)
    for mask in sam_masks:
        color = [random.randint(0, 255) for _ in range(3)]
        colored_mask = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            colored_mask[:, :, i] = mask * color[i]
        overlay_layer += colored_mask

    final_image = final_image * (1 - alpha) + overlay_layer * alpha
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    return Image.fromarray(final_image).resize((720, 720))