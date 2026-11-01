from __future__ import annotations

from typing import Any

from app.config import CACHE_ROOT
from app.core.device import get_torch_device

MODEL_ID = "segmind/tiny-sd"


def load_model() -> Any:
    import torch
    from diffusers import DiffusionPipeline

    device = get_torch_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, cache_dir=CACHE_ROOT
    )
    pipe.to(device)
    return pipe


def generate(pipe: Any, prompt: str, steps: int, seed: int | None) -> Any:
    import torch

    device = get_torch_device()
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    output = pipe(
        prompt,
        num_inference_steps=steps,
        generator=generator,
    )
    if not output.images or len(output.images) < 1:
        raise RuntimeError("TinySD returned no image.")
    return output.images[0]
