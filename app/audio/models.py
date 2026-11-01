from __future__ import annotations

from typing import Any

from app.config import CACHE_ROOT
from app.core.device import get_torch_device

MODEL_ID = "suno/bark-small"


def load_model() -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoProcessor, BarkModel

    device = get_torch_device()
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_ROOT)
    model = BarkModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        cache_dir=CACHE_ROOT,
    ).to(device)
    model.eval()
    return processor, model, device


def generate(model_data: tuple[Any, Any, Any], prompt: str, preset: str):
    import numpy as np
    import torch

    processor, model, device = model_data
    inputs = processor(
        text=[prompt], voice_preset=preset, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        waveform = model.generate(**inputs)
    audio = waveform.detach().float().cpu().numpy().squeeze()
    if not isinstance(audio, np.ndarray) or audio.size == 0:
        raise RuntimeError("Bark returned an empty waveform.")
    sample_rate = int(model.generation_config.sample_rate)
    return audio, sample_rate
