import gradio as gradio
from PIL import Image
import torch
import numpy as np
from llm.llm import *
from GroundingDINO.grounding_dino import *
from SAM2.sam2 import *
from Inpainting.inpaint import *


def load_models():
    llm = load_llm("llama3.1:latest")
    llm_chain = load_chain(llm, parse_json_output)
    grounding_processor, grounding_model = load_grounding_dino()
    sam2_predictor = load_sam()
    pipe_sd_xl = load_diffusion()
    return llm_chain, grounding_processor, grounding_model, sam2_predictor, pipe_sd_xl


def main(user_prompt: str, img_path: str, llm_chain, grounding_processor, grounding_model, sam2_predictor, pipe_sd_xl, device: str):
    image = Image.open(img_path)
    original_size = image.size
    resized_img = image.resize((1024, 1024))
    dino_input, diffusion_input = parse_user_prompt(user_prompt, llm_chain)
    grounded_boxes = run_grounding_dino(resized_img, dino_input, grounding_processor, grounding_model, device)

    resized_img_np = np.array(resized_img)
    sam_masks, sam_scores, sam_logits = segment_image(resized_img_np, sam2_predictor, grounded_boxes)

    combined_mask = (np.any(sam_masks, axis=0).astype(np.uint8) * 255).squeeze()
    mask_img = Image.fromarray(combined_mask).convert("L")
    mask_img = dilate_mask(mask_img)
    generator = torch.Generator("cuda").manual_seed(920)
    inpainted_img = inpaint(pipe_sd_xl, diffusion_input, resized_img, mask_img, generator, device)
    final_img = inpainted_img.resize(original_size)
    return final_img


