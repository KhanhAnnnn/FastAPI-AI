from typing import Annotated
import logging
import time
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from uuid import uuid4

from app.config import MAX_UPLOAD_BYTES
from app.config import OUTPUT_ROOT
from app.core.metrics import response_metric_headers
from app.core.model_manager import model_manager
from app.video.models import generate, load_model
from app.video.utils import frames_to_mp4

router = APIRouter(tags=["Video"])
logger = logging.getLogger(__name__)


@router.post(
    "/generate/video",
    response_class=StreamingResponse,
    responses={200: {"content": {"video/mp4": {}}}},
)
async def generate_video(
    image: UploadFile = File(...),
    num_frames: Annotated[int, Query(ge=14, le=25)] = 25,
    seed: Annotated[int, Query(ge=0, le=2**32 - 1)] = 42,
):
    image_data = await image.read(MAX_UPLOAD_BYTES + 1)
    if not image_data:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")
    if len(image_data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MiB upload limit.")
    try:
        with Image.open(BytesIO(image_data)) as decoded:
            decoded.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="Upload must be a valid image.") from exc
    frames, metrics = await model_manager.run(
        "video",
        load_model,
        lambda pipe: generate(pipe, image_data, num_frames, seed),
    )
    serialization_started = time.perf_counter()
    video = frames_to_mp4(frames)
    serialization_seconds = time.perf_counter() - serialization_started
    if not video:
        raise RuntimeError("PyAV produced an empty MP4.")
    (OUTPUT_ROOT / "videos" / f"{uuid4().hex}.mp4").write_bytes(video)
    logger.info(
        "Output serialized: model_id=stabilityai/stable-video-diffusion-img2vid "
        "serialization_seconds=%.3f output_bytes=%d frames=%d resolution=%sx%s",
        serialization_seconds,
        len(video),
        len(frames),
        frames[0].width,
        frames[0].height,
    )
    return StreamingResponse(
        BytesIO(video),
        media_type="video/mp4",
        headers={
            "Content-Disposition": 'attachment; filename="generated.mp4"',
            "X-Output-Bytes": str(len(video)),
            "X-Frame-Count": str(len(frames)),
            **response_metric_headers(metrics, serialization_seconds),
        },
    )
