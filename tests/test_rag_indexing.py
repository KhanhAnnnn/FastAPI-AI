from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import Distance, VectorParams

from app.main import app
from app.rag import artifacts, storage
from app.rag.repositories.vector_repository import (
    StorageUnavailableError,
    VectorRepository,
    get_vector_repository,
)
from app.rag.services.indexing_service import (
    IndexingConflictError,
    IndexingService,
    get_indexing_service,
)


def _create_mock_artifact(
    artifact_root: Path,
    document_id: str,
    extraction_id: str,
    run_id: str,
    dimension: int = 768,
    chunk_count: int = 2,
    corrupt_manifest: bool = False,
) -> Path:
    destination = artifact_root / f"{document_id}.{extraction_id}.{run_id}"
    destination.mkdir(parents=True, exist_ok=True)

    vectors = np.ones((chunk_count, dimension), dtype=np.float32)
    np.save(destination / "vectors.npy", vectors)

    import hashlib

    def file_sha256(p: Path) -> str:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    lines = []
    for i in range(chunk_count):
        lines.append(
            json.dumps(
                {
                    "chunk_id": f"chunk-{i}",
                    "document_id": document_id,
                    "extraction_id": extraction_id,
                    "source_filename": "test.pdf",
                    "page_number": i + 1,
                    "chunk_index": i,
                    "original_text": f"Original content chunk {i}",
                    "cleaned_text": f"Cleaned content chunk {i}",
                    "token_count": 20,
                    "text_hash": f"hash-{i}",
                }
            )
        )
    chunks_path = destination / "chunks.jsonl"
    chunks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "artifact_version": 1,
        "document_id": document_id,
        "extraction_id": extraction_id,
        "run_id": run_id,
        "model_id": "jinaai/jina-embeddings-v2-base-en",
        "model_revision": "322d4d7e2f35e84137961a65af894fda0385eb7a",
        "code_revision": "f3ec4cf7de7e561007f27c9efc7148b0bd713f81",
        "tokenizer": "jinaai/jina-embeddings-v2-base-en",
        "context_tokens": 8192,
        "chunk_size_tokens": 512,
        "chunk_overlap_tokens": 64,
        "chunk_count": chunk_count,
        "embedding_dimension": dimension,
        "vector_dtype": "float32",
        "device": "cpu",
        "model_load_seconds": 1.0,
        "embedding_seconds": 0.1,
        "source_pdf_sha256": "fake_pdf_sha",
        "extracted_text_sha256": "fake_txt_sha",
        "files": {
            "vectors.npy": {
                "sha256": file_sha256(destination / "vectors.npy"),
                "size_bytes": (destination / "vectors.npy").stat().st_size,
            },
            "chunks.jsonl": {
                "sha256": file_sha256(chunks_path),
                "size_bytes": chunks_path.stat().st_size,
            },
        },
    }

    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest_hash = file_sha256(manifest_path)
    if corrupt_manifest:
        manifest_hash = "0000000000000000000000000000000000000000000000000000000000000000"

    (destination / "manifest.sha256").write_text(
        f"{manifest_hash}  manifest.json\n", encoding="ascii"
    )
    return destination


def test_index_document_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "rag"
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", root)
    monkeypatch.setattr(artifacts, "RAG_ARTIFACT_ROOT", root / "artifacts")
    client = TestClient(app)

    response = client.post(f"/rag/documents/{uuid4()}/index")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_index_document_extraction_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "rag"
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", root)
    monkeypatch.setattr(artifacts, "RAG_ARTIFACT_ROOT", root / "artifacts")
    root.mkdir(parents=True)
    doc_id = str(uuid4())
    (root / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4...")
    client = TestClient(app)

    response = client.post(f"/rag/documents/{doc_id}/index")
    assert response.status_code == 409
    assert "extracted" in response.json()["detail"].lower()


def test_index_document_not_embedded_yet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "rag"
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", root)
    monkeypatch.setattr(artifacts, "RAG_ARTIFACT_ROOT", root / "artifacts")
    root.mkdir(parents=True)
    doc_id = str(uuid4())
    (root / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4...")
    (root / f"{doc_id}.extraction.json").write_text(
        json.dumps({"document_id": doc_id, "status": "completed"}), encoding="utf-8"
    )
    client = TestClient(app)

    response = client.post(f"/rag/documents/{doc_id}/index")
    assert response.status_code == 409
    assert "embedded" in response.json()["detail"].lower()


def test_index_corrupt_artifact_returns_422(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "rag"
    artifact_root = root / "artifacts"
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", root)
    monkeypatch.setattr(artifacts, "RAG_ARTIFACT_ROOT", artifact_root)

    doc_id = str(uuid4())
    ext_id = str(uuid4())
    run_id = str(uuid4())

    _create_mock_artifact(artifact_root, doc_id, ext_id, run_id, corrupt_manifest=True)
    client = TestClient(app)

    response = client.post(f"/rag/documents/{doc_id}/index")
    assert response.status_code == 422
    assert "integrity" in response.json()["detail"].lower() or "validation" in response.json()["detail"].lower()


def test_index_successful_flow_and_idempotency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "rag"
    artifact_root = root / "artifacts"
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", root)
    monkeypatch.setattr(artifacts, "RAG_ARTIFACT_ROOT", artifact_root)

    # Use in-memory repository for isolated test
    test_repo = VectorRepository(mode="memory", collection_name="test_indexing_col")
    test_indexing_service = IndexingService(repository=test_repo)
    monkeypatch.setattr("app.rag.router.get_indexing_service", lambda: test_indexing_service)

    doc_id = str(uuid4())
    ext_id = str(uuid4())
    run_id = str(uuid4())

    _create_mock_artifact(artifact_root, doc_id, ext_id, run_id, chunk_count=3)
    client = TestClient(app)

    # First index request
    resp1 = client.post(
        f"/rag/documents/{doc_id}/index",
        json={"collection_name": "test_indexing_col"},
    )
    assert resp1.status_code == 200, resp1.text
    data1 = resp1.json()
    assert data1["document_id"] == doc_id
    assert data1["status"] == "completed"
    assert data1["indexed_chunks"] == 3
    assert data1["vector_dimension"] == 768
    assert data1["distance"] == "cosine"

    assert test_repo.count_points("test_indexing_col") == 3

    # Idempotent re-index request: MUST NOT duplicate points
    resp2 = client.post(
        f"/rag/documents/{doc_id}/index",
        json={"collection_name": "test_indexing_col"},
    )
    assert resp2.status_code == 200
    assert resp2.json()["indexed_chunks"] == 3
    assert test_repo.count_points("test_indexing_col") == 3


def test_indexing_conflict_raises_409() -> None:
    service = IndexingService(repository=VectorRepository(mode="memory"))
    doc_id = str(uuid4())
    service._active_indexes.add(doc_id)

    with pytest.raises(IndexingConflictError, match="already in progress"):
        service.index_document(doc_id)


def test_indexing_storage_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "rag"
    artifact_root = root / "artifacts"
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", root)
    monkeypatch.setattr(artifacts, "RAG_ARTIFACT_ROOT", artifact_root)

    doc_id = str(uuid4())
    _create_mock_artifact(artifact_root, doc_id, str(uuid4()), str(uuid4()))

    # Mock repository that fails connection
    broken_repo = VectorRepository(mode="unsupported_mode")
    service = IndexingService(repository=broken_repo)
    monkeypatch.setattr("app.rag.router.get_indexing_service", lambda: service)

    client = TestClient(app)
    response = client.post(f"/rag/documents/{doc_id}/index")
    assert response.status_code == 503
