import re
import logging
import time
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from io import BytesIO
from app.config import OUTPUT_ROOT
from app.core.metrics import response_metric_headers

from app.core.model_manager import model_manager
from app.threed.models import generate, load_model
from app.threed.utils import mesh_to_obj

router = APIRouter(tags=["3D"])
logger = logging.getLogger(__name__)


@router.get(
    "/generate/3d",
    response_class=StreamingResponse,
    responses={200: {"content": {"model/obj": {}}}},
)
async def generate_3d(
    prompt: Annotated[str, Query(min_length=1, max_length=500)],
    num_inference_steps: Annotated[int, Query(ge=1, le=100)] = 25,
):
    mesh, metrics = await model_manager.run(
        "3d", load_model, lambda pipe: generate(pipe, prompt, num_inference_steps)
    )
    serialization_started = time.perf_counter()
    obj_data, vertex_count, face_count = mesh_to_obj(mesh)
    serialization_seconds = time.perf_counter() - serialization_started
    (OUTPUT_ROOT / "models_3d" / f"{uuid4().hex}.obj").write_bytes(obj_data)
    logger.info(
        "Output serialized: model_id=openai/shap-e serialization_seconds=%.3f "
        "output_bytes=%d vertices=%d faces=%d",
        serialization_seconds,
        len(obj_data),
        vertex_count,
        face_count,
    )
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", prompt).strip("_")[:64] or "mesh"
    return StreamingResponse(
        BytesIO(obj_data),
        media_type="model/obj",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.obj"',
            "X-Output-Bytes": str(len(obj_data)),
            "X-Vertex-Count": str(vertex_count),
            "X-Face-Count": str(face_count),
            **response_metric_headers(metrics, serialization_seconds),
        },
    )
