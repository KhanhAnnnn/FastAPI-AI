from __future__ import annotations

from typing import Any

from app.config import CACHE_ROOT
from app.core.device import get_torch_device

MODEL_ID = "openai/shap-e"


def load_model() -> Any:
    import torch
    from diffusers import ShapEPipeline

    device = get_torch_device()
    pipe = ShapEPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        cache_dir=CACHE_ROOT,
    )
    pipe.to(device)
    return pipe


def generate(pipe: Any, prompt: str, steps: int) -> Any:
    output = pipe(
        prompt,
        guidance_scale=15.0,
        num_inference_steps=steps,
        output_type="mesh",
    )
    meshes = output.images
    if isinstance(meshes, (list, tuple)):
        if not meshes:
            raise RuntimeError("Shap-E returned no meshes.")
        mesh = meshes[0]
    else:
        mesh = meshes
    return {
        "verts": mesh.verts.detach().cpu().numpy(),
        "faces": mesh.faces.detach().cpu().numpy(),
        "vertex_channels": {
            channel: value.detach().cpu().numpy()
            for channel, value in getattr(mesh, "vertex_channels", {}).items()
        }
        if isinstance(getattr(mesh, "vertex_channels", None), dict)
        else {},
    }
