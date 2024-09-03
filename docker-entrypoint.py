#!/usr/bin/env python
import argparse
import datetime
import inspect
import os
import re
import warnings
from typing import Optional

import numpy as np
import torch
import uvicorn
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    ControlNetModel,
    DiffusionPipeline,
    OnnxStableDiffusionImg2ImgPipeline,
    OnnxStableDiffusionInpaintPipeline,
    OnnxStableDiffusionPipeline,
    schedulers,
)
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
    character: bool = Field(
        False,
        description="Use for generating character with no background and specific poses",
    )
    device: str = Field(
        "cuda", description="The cpu or cuda device to use to render images"
    )
    half: bool = Field(
        False, description="Use float16 (half-sized) tensors instead of float32"
    )
    height: int = Field(512, description="Image height in pixels")
    image: Optional[str] = Field(
        None, description="The input image to use for image-to-image diffusion"
    )
    image_scale: Optional[float] = Field(
        None, description="How closely the image should follow the original image"
    )
    iters: int = Field(1, description="Number of times to run pipeline")
    mask: Optional[str] = Field(
        None, description="The input mask to use for diffusion inpainting"
    )
    model: str = Field(
        "stabilityai/stable-diffusion-xl-base-1.0",
        description="The model used to render images",
    )
    negative_prompt: Optional[str] = Field(
        None, description="The prompt to not render into an image"
    )
    onnx: bool = Field(False, description="Use the onnx runtime for inference")
    prompt: Optional[str] = Field(
        None, description="The prompt to render into an image"
    )
    samples: int = Field(1, description="Number of images to create per run")
    scale: float = Field(
        30, description="How closely the image should follow the prompt"
    )
    scheduler: Optional[str] = Field(
        None, description="Override the scheduler used to denoise the image"
    )
    seed: int = Field(42, description="RNG seed for repeatability")
    skip: bool = Field(False, description="Skip the safety checker")
    steps: int = Field(60, description="Number of sampling steps")
    strength: float = Field(
        0.75, description="Diffusion strength to apply to the input image"
    )
    token: Optional[str] = Field(None, description="Huggingface user access token")
    vae_slicing: bool = Field(
        False, description="Use less memory when creating large batches of images"
    )
    vae_tiling: bool = Field(
        False, description="Use less memory when creating ultra-high resolution images"
    )
    width: int = Field(512, description="Image width in pixels")
    xformers_memory_efficient_attention: bool = Field(
        False, description="Use less memory but require the xformers library"
    )
    dtype: Optional[torch.dtype] = None
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
        "negative_prompt": p.negative_prompt,
        "image": p.image,
        "mask_image": p.mask,
        "height": p.height,
        "width": p.width,
        "num_images_per_prompt": p.samples,
        "num_inference_steps": p.steps,
        "guidance_scale": p.scale,
        "image_guidance_scale": p.image_scale,
        "strength": p.strength,
        "generator": p.generator,
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


def stable_diffusion_pipeline(p):
    p.dtype = torch.float16 if p.half else torch.float32

    if p.onnx:
        p.diffuser = OnnxStableDiffusionPipeline
        p.revision = "onnx"
    else:
        p.diffuser = DiffusionPipeline
        p.revision = "fp16" if p.half else "main"

    autos = argparse.Namespace(
        **{
            "sd": ["StableDiffusionPipeline"],
            "sdxl": ["StableDiffusionXLPipeline"],
        }
    )

    config = DiffusionPipeline.load_config(p.model)
    is_auto_pipeline = config["_class_name"] in [autos.sd, autos.sdxl]

    if is_auto_pipeline:
        p.diffuser = AutoPipelineForText2Image

    if p.image is not None:
        if p.revision == "onnx":
            p.diffuser = OnnxStableDiffusionImg2ImgPipeline
        elif is_auto_pipeline:
            p.diffuser = AutoPipelineForImage2Image
        p.image = load_image(p.image)

    if p.mask is not None:
        if p.revision == "onnx":
            p.diffuser = OnnxStableDiffusionInpaintPipeline
        elif is_auto_pipeline:
            p.diffuser = AutoPipelineForInpainting
        p.mask = load_image(p.mask)

    if p.token is None:
        with open("token.txt") as f:
            p.token = f.read().replace("\n", "")

    if p.seed == 0:
        p.seed = torch.random.seed()

    if p.revision == "onnx":
        p.seed = p.seed >> 32 if p.seed > 2**32 - 1 else p.seed
        p.generator = np.random.RandomState(p.seed)
    else:
        p.generator = torch.Generator(device=p.device).manual_seed(p.seed)

    print("load pipeline start:", iso_date_time(), flush=True)

    with warnings.catch_warnings():
        for c in [UserWarning, FutureWarning]:
            warnings.filterwarnings("ignore", category=c)
        global current_device_idx
        # Select device using round-robin index
        device_id = device_ids[current_device_idx]
        current_device_idx = (current_device_idx + 1) % len(device_ids)

        pipeline = p.diffuser.from_pretrained(
            p.model,
            torch_dtype=p.dtype,
            revision=p.revision,
            use_auth_token=p.token,
        ).to(f"{p.device}:{device_id}")

    if p.scheduler is not None:
        scheduler = getattr(schedulers, p.scheduler)
        pipeline.scheduler = scheduler.from_config(pipeline.scheduler.config)

    if p.skip:
        pipeline.safety_checker = None

    if p.attention_slicing:
        pipeline.enable_attention_slicing()

    if p.xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if p.vae_slicing:
        pipeline.enable_vae_slicing()

    if p.vae_tiling:
        pipeline.enable_vae_tiling()

    p.pipeline = pipeline

    print("loaded models after:", iso_date_time(), flush=True)

    return p


def stable_diffusion_inference(p):
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
            if p.character == True:
                remove_background(image_location=os.path.join("output", out))

    print("completed pipeline:", iso_date_time(), flush=True)
    # if only 1 image is generated return the png image
    if i == 0:
        return FileResponse(os.path.join("output", out), media_type="image/png")
    else:
        return None


@app.post("/")
def image_generation(p: ImageGenerationConfig):
    pipeline = stable_diffusion_pipeline(p)
    img = stable_diffusion_inference(pipeline)
    return img


if __name__ == "__main__":
    uvicorn.run(
        "docker-entrypoint:app",
        host="0.0.0.0",
        workers=4,
        port=3750,
        log_level="info",
    )
