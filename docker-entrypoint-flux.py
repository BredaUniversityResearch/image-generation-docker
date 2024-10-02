#!/usr/bin/env python
import argparse
import datetime
import gc
import inspect
import os
import re
import warnings
from typing import Optional

import numpy as np
import torch
import uvicorn
from diffusers import FluxPipeline, FluxControlNetModel, FluxControlNetPipeline
from fastapi import FastAPI
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from rembg import remove

app = FastAPI()

device_ids = [0, 1, 2, 3]

# Round-robin index for device selection
current_device_idx = 0


# instantiate pydantic json payload
class ImageGenerationConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    attention_slicing: bool = Field(
        False, description="Use less memory at the expense of inference speed"
    )
    character: Optional[bool] = Field(
        False,
        description="Use for generating character with no background and specific poses",
    )
    controlnet_conditioning_scale: float = Field(
        0.6, description="Generalization for controlnet"
    )
    device: str = Field(
        "cuda", description="The cpu or cuda device to use to render images"
    )
    height: int = Field(512, description="Image height in pixels")
    image: Optional[str] = Field(
        None, description="The input image to use for image-to-image diffusion"
    )
    image_scale: Optional[float] = Field(
        None, description="How closely the image should follow the original image"
    )
    ip_adapter_image: Optional[str] = Field(
        None, description="The image used as base for ip adapter"
    )
    iters: int = Field(1, description="Number of times to run pipeline")
    max_sequence_length: int = Field(
        256, description="Maximum sequence length. Cannot be over 256 for flux schnell."
    )
    model: str = Field(
        "black-forest-labs/FLUX.1-schnell",
        description="The model used to render images",
    )
    prompt: Optional[str] = Field(
        None, description="The prompt to render into an image"
    )
    samples: int = Field(1, description="Number of images to create per run")
    scale: float = Field(
        0, description="How closely the image should follow the prompt"
    )

    seed: int = Field(42, description="RNG seed for repeatability")
    steps: int = Field(4, description="Number of sampling steps")
    token: Optional[str] = Field(None, description="Huggingface user access token")
    vae_slicing: bool = Field(
        False, description="Use less memory when creating large batches of images"
    )
    vae_tiling: bool = Field(
        False, description="Use less memory when creating ultra-high resolution images"
    )
    width: int = Field(512, description="Image width in pixels")
    dtype: Optional[torch.dtype] = None
    controlnet: Optional[torch.nn.Module] = None
    diffuser: Optional[torch.nn.Module] = None
    revision: Optional[str] = None
    generator: Optional[torch.Generator] = None
    pipeline: Optional[torch.nn.Module] = None


def iso_date_time():
    return datetime.datetime.now().isoformat()


def load_image(path):
    image = Image.open(os.path.join("input", path)).convert("RGB")
    print(f"loaded image from {path}:", iso_date_time(), flush=True)
    return image


def remove_unused_args(p):
    params = inspect.signature(p.pipeline).parameters.keys()
    args = {
        "prompt": p.prompt,
        "height": p.height,
        "width": p.width,
        "num_images_per_prompt": p.samples,
        "num_inference_steps": p.steps,
        "guidance_scale": p.scale,
        "controlnet": p.controlnet,
        "control_image": p.image,
        "control_mode": 4,
        "image_guidance_scale": p.image_scale,
        "controlnet_conditioning_scale": p.controlnet_conditioning_scale,
        "ip_adapter_image": p.ip_adapter_image,
        "generator": p.generator,
        "max_sequence_length": p.max_sequence_length,
    }
    return {p: args[p] for p in params if p in args}


def remove_background(image_location: str):
    """Remove background from the generated image of the character.

    Args:
        image_location (str): Location of the generated image
    """

    # load image
    input = Image.open(image_location)

    # remove background
    output = remove(input)

    # save image
    output.save(image_location)

    return


def flux_pipeline(p):
    p.dtype = torch.float16

    p.diffuser = FluxPipeline

    if p.image is not None:
        p.image = load_image(p.image)

    if p.seed == 0:
        p.seed = torch.random.seed()

    p.generator = torch.Generator(device=p.device).manual_seed(p.seed)

    print("load pipeline start:", iso_date_time(), flush=True)

    with warnings.catch_warnings():
        for c in [UserWarning, FutureWarning]:
            warnings.filterwarnings("ignore", category=c)
        global current_device_idx
        # Select device using round-robin index
        device_id = device_ids[current_device_idx]
        current_device_idx = (current_device_idx + 1) % len(device_ids)

        if p.character == True:
            if p.image is not None:
                p.controlnet = FluxControlNetModel.from_pretrained(
                    "InstantX/FLUX.1-dev-Controlnet-Union",
                    torch_dtype=p.dtype,
                )

                p.diffuser = FluxControlNetPipeline

        pipeline = p.diffuser.from_pretrained(
            p.model,
            controlnet=p.controlnet,
            torch_dtype=p.dtype,
        )

        pipeline.enable_sequential_cpu_offload()

    p.pipeline = pipeline

    print("loaded models after:", iso_date_time(), flush=True)

    return p


def flux_inference(p):
    prefix = (
        re.sub(r"[\\/:*?\"<>|]", "", p.prompt)
        .replace(" ", "_")
        .encode("utf-8")[:170]
        .decode("utf-8", "ignore")
    )

    for j in range(p.iters):
        result = p.pipeline(**remove_unused_args(p))

        for i, img in enumerate(result.images):
            idx = j * p.samples + i + 1
            out = f"{prefix}__steps_{p.steps}__scale_{p.scale:.2f}__seed_{p.seed}__n_{idx}.png"
            img.save(os.path.join("output", out))

    print("completed pipeline:", iso_date_time(), flush=True)
    # if only 1 image is generated return the png image
    if i == 0:
        return FileResponse(os.path.join("output", out), media_type="image/png")
    else:
        return None


@app.post("/")
def image_generation(p: ImageGenerationConfig):
    pipeline = flux_pipeline(p)
    img = flux_inference(pipeline)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return img


if __name__ == "__main__":
    uvicorn.run(
        "docker-entrypoint-flux:app",
        host="0.0.0.0",
        workers=4,
        port=3750,
        log_level="info",
    )
