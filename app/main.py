import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.config import CACHE_ROOT
from app.core.device import environment_summary
from app.core.exceptions import ModelBusyError
from app.core.model_manager import model_manager
from app.audio.router import router as audio_router
from app.image.router import router as image_router
from app.text.router import router as text_router
from app.threed.router import router as threed_router
from app.video.router import router as video_router
from app.rag.router import router as rag_router

os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: recover any tasks interrupted by prior shutdown
    import asyncio
    from app.rag.processing_service import get_processing_service
    proc_service = get_processing_service()
    proc_service.set_loop(asyncio.get_running_loop())
    proc_service.recover_interrupted_tasks()

    yield

    # Shutdown: cancel active worker tasks and release vector database locks
    proc_service.shutdown()
    from app.rag.vector_repository import close_vector_repository
    close_vector_repository()


app = FastAPI(
    title="Local Generative AI Lab",
    description="Five independent generative models served locally.",
    version="1.0.0",
    lifespan=lifespan,
)

for router in (text_router, audio_router, image_router, video_router, threed_router, rag_router):
    app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/system/info")
async def system_info():
    return environment_summary()


@app.get("/models/status")
async def models_status():
    return await model_manager.status()


@app.post("/models/{model_name}/unload")
async def unload_model(model_name: Literal["text", "audio", "image", "video", "3d"]):
    unloaded = await model_manager.unload(model_name)
    if not unloaded:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not loaded.")
    return {"status": "unloaded", "model": model_name}


@app.get("/", include_in_schema=False)
async def web_ui():
    ui_path = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    return FileResponse(ui_path)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_request, exc: RuntimeError):
    logger.error("Inference runtime error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ModelBusyError)
async def model_busy_error_handler(_request, exc: ModelBusyError):
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc)},
        headers={"Retry-After": "2"},
    )
