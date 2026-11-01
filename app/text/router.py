import json
import logging
import time
from uuid import uuid4
from typing import Annotated

from fastapi import APIRouter, Query

from app.config import OUTPUT_ROOT
from app.core.model_manager import model_manager
from app.text.models import generate, load_model

router = APIRouter(tags=["Text"])
logger = logging.getLogger(__name__)


@router.get("/generate/text")
async def generate_text(
    prompt: Annotated[str, Query(min_length=1, max_length=2_000)],
    max_new_tokens: Annotated[int, Query(ge=1, le=512)] = 128,
    temperature: Annotated[float, Query(ge=0, le=2)] = 0.7,
):
    text, metrics = await model_manager.run(
        "text", load_model, lambda model: generate(model, prompt, max_new_tokens, temperature)
    )
    if not text:
        raise RuntimeError("TinyLlama returned an empty response.")
    serialization_started = time.perf_counter()
    record = {"text": text, "metrics": metrics}
    output_path = OUTPUT_ROOT / "text" / f"{uuid4().hex}.json"
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Output serialized: model_id=TinyLlama/TinyLlama-1.1B-Chat-v1.0 "
        "serialization_seconds=%.3f",
        time.perf_counter() - serialization_started,
    )
    return record
