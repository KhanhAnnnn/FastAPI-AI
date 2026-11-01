from typing import Annotated
import logging
import time
from uuid import uuid4

from fastapi import APIRouter, Query, Response
from app.config import OUTPUT_ROOT
from app.core.metrics import response_metric_headers

from app.core.model_manager import model_manager
from app.image.models import generate, load_model
from app.image.utils import image_to_png

router = APIRouter(tags=["Image"])
logger = logging.getLogger(__name__)


@router.get(
    "/generate/image",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def generate_image(
    prompt: Annotated[str, Query(min_length=1, max_length=2_000)],
    num_inference_steps: Annotated[int, Query(ge=1, le=50)] = 10,
    seed: Annotated[int | None, Query(ge=0, le=2**32 - 1)] = None,
):
    image, metrics = await model_manager.run(
        "image",
        load_model,
        lambda pipe: generate(pipe, prompt, num_inference_steps, seed),
    )
    serialization_started = time.perf_counter()
    png = image_to_png(image)
    serialization_seconds = time.perf_counter() - serialization_started
    (OUTPUT_ROOT / "images" / f"{uuid4().hex}.png").write_bytes(png)
    logger.info(
        "Output serialized: model_id=segmind/tiny-sd serialization_seconds=%.3f "
        "output_bytes=%d resolution=%sx%s",
        serialization_seconds,
        len(png),
        image.width,
        image.height,
    )
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Content-Disposition": 'inline; filename="generated.png"',
            "X-Output-Bytes": str(len(png)),
            "X-Output-Width": str(image.width),
            "X-Output-Height": str(image.height),
            **response_metric_headers(metrics, serialization_seconds),
        },
    )
