from __future__ import annotations

from typing import Any

from app.config import CACHE_ROOT
from app.core.device import get_torch_device

MODEL_ID = "stabilityai/stable-video-diffusion-img2vid"


def load_model() -> Any:
    import torch
    from diffusers import StableVideoDiffusionPipeline

    device = get_torch_device()
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=CACHE_ROOT,
    )
    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        if hasattr(pipe, "unet") and hasattr(pipe.unet, "enable_forward_chunking"):
            pipe.unet.enable_forward_chunking()
    else:
        pipe.to(device)
    return pipe


def generate(pipe: Any, image_bytes: bytes, num_frames: int, seed: int) -> list[Any]:
    from io import BytesIO

    import torch
    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as source:
        image = source.convert("RGB").resize((1024, 576), Image.Resampling.LANCZOS)
    device = get_torch_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        generated = pipe(
            image,
            num_frames=num_frames,
            decode_chunk_size=2,
            generator=generator,
        ).frames
    if not isinstance(generated, (list, tuple)):
        raise RuntimeError("Stable Video Diffusion returned an unsupported frame batch.")
    frames = list(generated)
    if frames and isinstance(frames[0], (list, tuple)):
        frames = list(frames[0])
    if not frames or not all(isinstance(frame, Image.Image) for frame in frames):
        raise RuntimeError("Stable Video Diffusion returned an invalid frame collection.")
    return frames
