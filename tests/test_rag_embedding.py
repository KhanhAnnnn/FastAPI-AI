import json
import re
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app
from app.rag import artifacts, embedding_model, embedding_service, storage
from app.rag.embedding_model import EmbeddingModelError, JinaEmbedder


class TensorTokenizer:
    model_max_length = 8192
    is_fast = True

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(self, texts, **_kwargs):
        width = max(len(text.split()) for text in texts) + 2
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, text in enumerate(texts):
            tokens = text.split()
            ids[row, : len(tokens)] = torch.arange(1, len(tokens) + 1)
            mask[row, : len(tokens)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class FiniteTestModel:
    def __call__(self, **inputs):
        ids = inputs["input_ids"].to(torch.float32)
        last_hidden_state = torch.stack((ids, ids + 1, ids + 2), dim=-1)
        return SimpleNamespace(last_hidden_state=last_hidden_state)


class OffsetWordTokenizer(TensorTokenizer):
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {
                "offset_mapping": [
                    match.span() for match in re.finditer(r"\S+", texts)
                ]
            }
        return super().__call__(texts, **kwargs)


class FakeEmbedder:
    tokenizer = OffsetWordTokenizer()
    context_tokens = 8192
    embedding_dimension = 3
    load_seconds = 0.02
    device = "cpu"

    def embed(self, texts, batch_size):
        assert batch_size > 0
        return np.tile(np.asarray([[0.0, 0.6, 0.8]], dtype=np.float32), (len(texts), 1))


def test_embedding_pooling_returns_finite_normalized_vectors() -> None:
    embedder = JinaEmbedder(
        tokenizer=TensorTokenizer(),
        model=FiniteTestModel(),
        context_tokens=32,
        embedding_dimension=3,
        load_seconds=0.0,
    )

    vectors = embedder.embed(["alpha beta", "gamma"], batch_size=1)

    assert vectors.shape == (2, 3)
    assert vectors.dtype.name == "float32"
    assert torch.isfinite(torch.from_numpy(vectors)).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])


@pytest.mark.parametrize("texts", [[], ["  "]])
def test_embedding_rejects_no_chunks_or_empty_chunks(texts: list[str]) -> None:
    embedder = JinaEmbedder(
        tokenizer=TensorTokenizer(),
        model=FiniteTestModel(),
        context_tokens=32,
        embedding_dimension=3,
        load_seconds=0.0,
    )
    with pytest.raises(EmbeddingModelError):
        embedder.embed(texts, batch_size=2)


def test_embedding_rejects_context_overflow() -> None:
    embedder = JinaEmbedder(
        tokenizer=TensorTokenizer(),
        model=FiniteTestModel(),
        context_tokens=3,
        embedding_dimension=3,
        load_seconds=0.0,
    )
    with pytest.raises(EmbeddingModelError, match="context length"):
        embedder.embed(["a b c"], batch_size=1)


def test_embedding_api_checks_document_state_before_remote_code_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "rag"
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", root)
    monkeypatch.setenv("RAG_ALLOW_JINA_REMOTE_CODE", "false")
    client = TestClient(app)

    assert client.post(f"/rag/documents/{uuid4()}/embed").status_code == 404

    document_id = str(uuid4())
    root.mkdir()
    (root / f"{document_id}.pdf").write_bytes(b"PDF source placeholder")
    assert client.post(f"/rag/documents/{document_id}/embed").status_code == 409

    (root / f"{document_id}.extraction.json").write_text(
        json.dumps(
            {
                "document_id": document_id,
                "status": "completed",
                "extraction_id": str(uuid4()),
                "text_filename": "extracted.txt",
                "pages": [
                    {
                        "page_number": 1,
                        "has_text": True,
                        "text_start": 0,
                        "text_end": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "extracted.txt").write_text("text", encoding="utf-8")
    response = client.post(f"/rag/documents/{document_id}/embed")
    assert response.status_code == 503
    assert "RAG_ALLOW_JINA_REMOTE_CODE=true" in response.json()["detail"]
    assert not embedding_model._embedder


def test_embed_api_writes_reusable_artifact_with_test_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "rag"
    artifact_root = root / "artifacts"
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", root)
    monkeypatch.setattr(artifacts, "RAG_ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(embedding_service, "get_embedder", FakeEmbedder)
    monkeypatch.setenv("RAG_ALLOW_JINA_REMOTE_CODE", "true")
    document_id, extraction_id = str(uuid4()), str(uuid4())
    extracted = "First page has several English words.  Tiếng Việt có dấu."
    root.mkdir()
    (root / f"{document_id}.pdf").write_bytes(b"fixture source")
    (root / "extracted.txt").write_text(extracted, encoding="utf-8")
    (root / f"{document_id}.extraction.json").write_text(
        json.dumps(
            {
                "document_id": document_id,
                "original_filename": "fixture.pdf",
                "extraction_id": extraction_id,
                "text_filename": "extracted.txt",
                "status": "completed",
                "pages": [
                    {
                        "page_number": 1,
                        "has_text": True,
                        "text_start": 0,
                        "text_end": len(extracted),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    response = TestClient(app).post(f"/rag/documents/{document_id}/embed")

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["chunk_count"] == 1
    assert result["embedding_dimension"] == 3
    assert result["device"] == "cpu"
    artifact_dir = artifact_root / result["artifact_directory"]
    manifest, chunks, vectors = artifacts.load_artifact(artifact_dir)
    assert manifest["document_id"] == document_id
    assert manifest["extraction_id"] == extraction_id
    assert chunks[0]["page_number"] == 1
    assert "Tiếng Việt" in chunks[0]["cleaned_text"]
    assert vectors.shape == (1, 3)


def test_model_manager_rejects_embedding_when_inference_is_busy() -> None:
    import asyncio
    import threading

    from app.core.exceptions import ModelBusyError
    from app.core.model_manager import ModelManager

    manager = ModelManager()
    started = threading.Event()
    finish = threading.Event()

    async def exercise() -> None:
        task = asyncio.create_task(
            manager.run_exclusive(
                "rag_embedding", lambda: (started.set(), finish.wait(timeout=5))
            )
        )
        assert await asyncio.to_thread(started.wait, timeout=5)
        with pytest.raises(ModelBusyError):
            await manager.run("text", object, lambda _: None)
        finish.set()
        await task

    asyncio.run(exercise())
