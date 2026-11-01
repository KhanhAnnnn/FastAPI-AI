from typing import Annotated, Literal
import logging
import time
from uuid import uuid4

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from io import BytesIO

from app.audio.models import generate, load_model
from app.config import OUTPUT_ROOT
from app.audio.utils import waveform_to_wav
from app.core.metrics import response_metric_headers
from app.core.model_manager import model_manager

router = APIRouter(tags=["Audio"])
VoicePreset = Literal["v2/en_speaker_1", "v2/en_speaker_9"]
logger = logging.getLogger(__name__)


@router.get(
    "/generate/audio",
    response_class=StreamingResponse,
    responses={200: {"content": {"audio/wav": {}}}},
)
async def generate_audio(
    prompt: Annotated[str, Query(min_length=1, max_length=500)],
    preset: VoicePreset = "v2/en_speaker_1",
):
    (waveform, sample_rate), metrics = await model_manager.run(
        "audio", load_model, lambda model: generate(model, prompt, preset)
    )
    serialization_started = time.perf_counter()
    wav_data = waveform_to_wav(waveform, sample_rate)
    serialization_seconds = time.perf_counter() - serialization_started
    output_path = OUTPUT_ROOT / "audio" / f"{uuid4().hex}.wav"
    output_path.write_bytes(wav_data)
    logger.info(
        "Output serialized: model_id=suno/bark-small serialization_seconds=%.3f "
        "output_bytes=%d sample_rate=%d",
        serialization_seconds,
        len(wav_data),
        sample_rate,
    )
    return StreamingResponse(
        BytesIO(wav_data),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="generated.wav"',
            "X-Sample-Rate": str(sample_rate),
            "X-Output-Bytes": str(len(wav_data)),
            **response_metric_headers(metrics, serialization_seconds),
        },
    )
