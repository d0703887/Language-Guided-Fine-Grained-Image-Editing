import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image


def load_grounding_dino(device: str):
    model_id = "IDEA-Research/grounding-dino-tiny"

    grounding_processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return grounding_processor, grounding_model


def run_grounding_dino(image: Image, dino_prompt: str, grounding_processor, grounding_model, device):
    dino_inputs = grounding_processor(images=image, text=dino_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        raw_dino_outputs = grounding_model(**dino_inputs)
    dino_result = grounding_processor.post_process_grounded_object_detection(
        raw_dino_outputs,
        dino_inputs.input_ids,
        box_threshold=0.35,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    grounded_boxes = dino_result[0]['boxes'].cpu().numpy()
    return grounded_boxes