from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import torch
import numpy as np
import random


def load_sam(sam2_checkpoint: str, model_cfg: str, device: str):
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
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


def overlay_mask(image: np.ndarray, mask, color, alpha=0.5):
    colored_mask = np.zeros_like(image)
    for i in range(3):
        colored_mask[:, :, i] = mask * color[i]
    return np.clip(image * (1 - alpha) + colored_mask * alpha, 0, 255).astype(np.uint8)


def visualize_sam_masks(image: np.ndarray, sam_masks, scores=None):
    image = image.astype(np.uint8)
    for mask in sam_masks:
        color = [random.randint(0, 255) for _ in range(3)]
        image = overlay_mask(image, mask.astype(bool), color, alpha=0.5)
    return Image.fromarray(image).resize((720, 720))