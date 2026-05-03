import logging
import random
import warnings
import os
import gradio as gr
import numpy as np
import spaces
import torch
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
from gradio_imageslider import ImageSlider
from PIL import Image
from huggingface_hub import snapshot_download

css = """
#col-container {
    margin: 0 auto;
    max-width: 512px;
}
"""

if torch.cuda.is_available():
    power_device = "GPU"
    device = "cuda"
else:
    power_device = "CPU"
    device = "cpu"


huggingface_token = os.getenv("HUGGINFACE_TOKEN")

model_path = snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev", 
    repo_type="model", 
    ignore_patterns=["*.md", "*..gitattributes"],
    local_dir="FLUX.1-dev",
    token=huggingface_token, # type a new token-id.
)


# Load pipeline
controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=torch.bfloat16
).to(device)
pipe = FluxControlNetPipeline.from_pretrained(
    model_path, controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to(device)

MAX_SEED = 1000000
MAX_PIXEL_BUDGET = 1024 * 1024


def process_input(input_image, upscale_factor, **kwargs):
    w, h = input_image.size
    w_original, h_original = w, h
    aspect_ratio = w / h

    was_resized = False

    if w * h * upscale_factor**2 > MAX_PIXEL_BUDGET:
        # Calculate the maximum allowed input pixels
        max_input_pixels = MAX_PIXEL_BUDGET // (upscale_factor ** 2)
        current_pixels = w * h
        
        if current_pixels > max_input_pixels:
            # Scale down proportionally while maintaining aspect ratio
            scale_factor = (max_input_pixels / current_pixels) ** 0.5
            target_w = int(w * scale_factor)
            target_h = int(h * scale_factor)
            
            # Ensure dimensions are multiples of 8 (model requirement)
            target_w = target_w - target_w % 8
            target_h = target_h - target_h % 8
            
            warnings.warn(
                f"Requested output image is too large ({w * upscale_factor}x{h * upscale_factor}). "
                f"Resizing input to ({target_w}x{target_h}) pixels."
            )
            gr.Info(
                f"Requested output image is too large ({w * upscale_factor}x{h * upscale_factor}). "
                f"Resizing input to ({target_w}x{target_h}) pixels."
            )
            input_image = input_image.resize((target_w, target_h))
            was_resized = True

    # If no resizing was needed above, still ensure dimensions are multiples of 8
    if not was_resized:
        w, h = input_image.size
        w = w - w % 8
        h = h - h % 8
        if w != input_image.size[0] or h != input_image.size[1]:
            input_image = input_image.resize((w, h))

    return input_image, w_original, h_original, was_resized
