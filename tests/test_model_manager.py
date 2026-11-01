import asyncio
import threading

import pytest
from app.core.exceptions import ModelBusyError
from app.core.model_manager import ModelManager


def test_model_is_loaded_once_for_repeated_calls() -> None:
    manager = ModelManager()
    loads = 0

    def loader():
        nonlocal loads
        loads += 1
        return object()

    async def run_calls():
        first = await manager.run("text", loader, lambda _: "first")
        second = await manager.run("text", loader, lambda _: "second")
        return first, second

    first, second = asyncio.run(run_calls())
    assert first[0] == "first"
    assert second[0] == "second"
    assert loads == 1


def test_overlapping_inference_is_rejected_without_queueing() -> None:
    manager = ModelManager()
    started = threading.Event()
    finish = threading.Event()

    async def run_overlapping():
        first = asyncio.create_task(
            manager.run(
                "text",
                object,
                lambda _: (started.set(), finish.wait(timeout=5)),
            )
        )
        assert await asyncio.to_thread(started.wait, timeout=5)
        with pytest.raises(ModelBusyError):
            await manager.run("audio", object, lambda _: None)
        finish.set()
        await first

    asyncio.run(run_overlapping())
