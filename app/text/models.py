from __future__ import annotations

from typing import Any

from app.config import CACHE_ROOT
from app.core.device import get_torch_device

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SYSTEM_PROMPT = (
    "You are FastAPI bot, a helpful assistant responsible for teaching FastAPI. "
    "Always respond in Markdown."
)


def load_model() -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_torch_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_ROOT)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_ROOT,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def generate(
    model_data: tuple[Any, Any, Any],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    tokenizer, model, device = model_data
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    generation_options: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_options["temperature"] = temperature
        generation_options["top_p"] = 0.95
    with torch.inference_mode():
        output_ids = model.generate(input_ids, **generation_options)
    new_tokens = output_ids[0, input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
