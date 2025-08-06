from transformers import CLIPTextModel
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import cv2
import numpy as np
from PIL import Image


def load_diffusion():
    pipe_sd_xl = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        # Huggingface workspace don't have gpu
        torch_dtype=torch.float16,
        variant="fp16"
    )
    return pipe_sd_xl


def dilate_mask(mask_img: Image.Image, radius: int = 10) -> Image.Image:
    mask_np = np.array(mask_img.convert("L"))
    binary_mask = (mask_np > 127).astype(np.uint8)  # 0, 255 => 0, 1
    kernel = np.ones((radius, radius), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    dilated_mask = Image.fromarray((dilated_mask * 255).astype(np.uint8))
    return dilated_mask


def inpaint(pipe_sd_xl, diffusion_prompt: str, image: Image, mask_image: Image, generator, device):
    pipe_sd_xl = pipe_sd_xl.to(device)
    result = pipe_sd_xl(
        prompt=diffusion_prompt,
        image=image,
        mask_image=mask_image,
        generator=generator,
        num_inference_steps=50,
        guidance_scale=8.5
    ).images[0]
    pipe_sd_xl = pipe_sd_xl.to("cpu")
    return result