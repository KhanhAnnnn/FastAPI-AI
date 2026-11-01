from __future__ import annotations

import asyncio
import gc
import logging
import time
from collections.abc import Callable
from typing import Any

from app.core.exceptions import ModelBusyError

logger = logging.getLogger(__name__)
MODEL_IDS = {
    "text": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "audio": "suno/bark-small",
    "image": "segmind/tiny-sd",
    "video": "stabilityai/stable-video-diffusion-img2vid",
    "3d": "openai/shap-e",
}


class ModelManager:
    """Serializes model loading/inference and keeps at most one model in memory."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._loaded_name: str | None = None
        self._model: Any = None
        self._loaded_at: float | None = None
        self._load_seconds: float | None = None
        self._active_request: str | None = None
        self._status: dict[str, dict[str, Any]] = {}

    async def run(
        self,
        name: str,
        loader: Callable[[], Any],
        operation: Callable[[Any], Any],
    ) -> tuple[Any, dict[str, Any]]:
        if self._lock.locked():
            raise ModelBusyError("Another model operation is active; retry shortly.")
        await self._lock.acquire()
        try:
            self._active_request = name
            return await asyncio.to_thread(self._run_sync, name, loader, operation)
        finally:
            self._active_request = None
            self._lock.release()

    async def run_exclusive(self, name: str, operation: Callable[[], Any]) -> Any:
        if self._lock.locked():
            raise ModelBusyError("Another model operation is active; retry shortly.")
        await self._lock.acquire()
        self._active_request = name
        task = asyncio.create_task(asyncio.to_thread(self._run_exclusive_sync, operation))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        finally:
            self._active_request = None
            self._lock.release()

    def _run_exclusive_sync(self, operation: Callable[[], Any]) -> Any:
        self._unload_sync()
        return operation()

    def _run_sync(
        self,
        name: str,
        loader: Callable[[], Any],
        operation: Callable[[Any], Any],
    ) -> tuple[Any, dict[str, Any]]:
        import torch
        import psutil

        started = time.perf_counter()
        load_seconds = 0.0
        model = None
        process = psutil.Process()
        rss_before = process.memory_info().rss
        cpu_time_before = sum(process.cpu_times()[:2])
        if self._loaded_name != name:
            self._unload_sync()
            self._status[name] = {
                "state": "loading",
                "model_id": MODEL_IDS[name],
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
            load_started = time.perf_counter()
            try:
                model = loader()
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    model = None
                    self._model = None
                    self._loaded_name = None
                    self._loaded_at = None
                    self._load_seconds = None
                    gc.collect()
                    torch.cuda.empty_cache()
                self._status[name] = {
                    "state": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                logger.exception("Model load failed: %s", name)
                raise
            load_seconds = time.perf_counter() - load_started
            self._model = model
            self._loaded_name = name
            self._loaded_at = time.time()
            self._load_seconds = load_seconds
            self._status[name] = {
                "state": "loaded",
                "model_id": MODEL_IDS[name],
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
            logger.info("Model loaded: name=%s load_seconds=%.3f", name, load_seconds)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            vram_before = torch.cuda.memory_allocated()
        else:
            vram_before = None

        self._status[name] = {
            "state": "running",
            "model_id": MODEL_IDS[name],
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        try:
            result = operation(self._model)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                vram_after = torch.cuda.memory_allocated()
                peak_vram = torch.cuda.max_memory_allocated()
            else:
                vram_after = peak_vram = None
            metrics = {
                "model": name,
                "model_id": MODEL_IDS[name],
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "load_seconds": load_seconds,
                "inference_seconds": time.perf_counter() - started - load_seconds,
                "total_seconds": time.perf_counter() - started,
                "process_rss_before_bytes": rss_before,
                "process_rss_after_bytes": process.memory_info().rss,
                "process_cpu_seconds": sum(process.cpu_times()[:2]) - cpu_time_before,
                "vram_before_bytes": vram_before,
                "vram_after_bytes": vram_after,
                "peak_vram_bytes": peak_vram,
                "gpu_utilization_percent": None,
                "peak_vram_measurement": (
                    "torch.cuda.max_memory_allocated"
                    if peak_vram is not None
                    else "N/A (CUDA unavailable)"
                ),
            }
            self._status[name] = {
                "state": "loaded",
                "model_id": MODEL_IDS[name],
                "device": metrics["device"],
                "last_success": metrics,
            }
            logger.info("Inference completed: %s", metrics)
            return result, metrics
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                model = None
                self._model = None
                self._loaded_name = None
                self._loaded_at = None
                self._load_seconds = None
                gc.collect()
                torch.cuda.empty_cache()
            self._status[name] = {
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            logger.exception("Inference failed: %s", name)
            raise

    def _unload_sync(self) -> None:
        if self._loaded_name is None:
            return
        name = self._loaded_name
        self._model = None
        self._loaded_name = None
        self._loaded_at = None
        self._load_seconds = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self._status[name] = {"state": "unloaded"}
        logger.info("Model unloaded: %s", name)

    async def unload(self, name: str) -> bool:
        async with self._lock:
            if self._loaded_name != name:
                return False
            await asyncio.to_thread(self._unload_sync)
            return True

    async def status(self) -> dict[str, Any]:
        return {
            "loaded_model": self._loaded_name,
            "loaded_at": self._loaded_at,
            "load_seconds": self._load_seconds,
            "active_request": self._active_request,
            "models": {key: value.copy() for key, value in self._status.items()},
        }


model_manager = ModelManager()
