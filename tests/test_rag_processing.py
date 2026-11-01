from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag import storage
from app.rag.processing_schemas import DocumentProcessingStatus
from app.rag.processing_service import (
    ProcessingConflictError,
    get_processing_service,
)
from tests.test_rag_upload import make_pdf


def test_get_status_non_existent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", tmp_path)
    client = TestClient(app)
    unknown_id = uuid4()
    resp = client.get(f"/rag/documents/{unknown_id}/status")
    assert resp.status_code == 404


def test_get_status_uploaded_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", tmp_path)
    client = TestClient(app)
    doc_id = uuid4()
    pdf_path = tmp_path / f"{doc_id}.pdf"
    pdf_path.write_bytes(make_pdf(b"sample content"))

    resp = client.get(f"/rag/documents/{doc_id}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == str(doc_id)
    assert data["status"] == "UPLOADED"


def test_enqueue_processing_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", tmp_path)
    client = TestClient(app)
    doc_id = uuid4()
    pdf_path = tmp_path / f"{doc_id}.pdf"
    pdf_path.write_bytes(make_pdf(b"sample content"))

    service = get_processing_service()
    # monkeypatch worker loop to avoid running real long background tasks in unit test
    async def dummy_worker():
        pass
    monkeypatch.setattr(service, "_ensure_worker_running", lambda: None)

    resp = client.post(f"/rag/documents/{doc_id}/process", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == str(doc_id)
    assert data["status"] == "QUEUED"
    assert data["processing_run_id"] is not None

    # Verify conflict on duplicate enqueue
    resp_conflict = client.post(f"/rag/documents/{doc_id}/process", json={})
    assert resp_conflict.status_code == 409


def test_cancel_processing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", tmp_path)
    client = TestClient(app)
    doc_id = uuid4()
    pdf_path = tmp_path / f"{doc_id}.pdf"
    pdf_path.write_bytes(make_pdf(b"sample content"))

    service = get_processing_service()
    monkeypatch.setattr(service, "_ensure_worker_running", lambda: None)

    client.post(f"/rag/documents/{doc_id}/process", json={})
    cancel_resp = client.post(f"/rag/documents/{doc_id}/cancel")
    assert cancel_resp.status_code == 200
    data = cancel_resp.json()
    assert data["status"] == "CANCELLED"


def test_list_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", tmp_path)
    client = TestClient(app)

    doc1 = uuid4()
    (tmp_path / f"{doc1}.pdf").write_bytes(make_pdf(b"first doc"))
    doc2 = uuid4()
    (tmp_path / f"{doc2}.pdf").write_bytes(make_pdf(b"second doc"))

    resp = client.get("/rag/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert len(docs) == 2
    ids = {d["document_id"] for d in docs}
    assert str(doc1) in ids
    assert str(doc2) in ids


def test_upload_with_auto_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", tmp_path)
    client = TestClient(app)

    service = get_processing_service()
    monkeypatch.setattr(service, "_ensure_worker_running", lambda: None)

    pdf_content = make_pdf(b"autoprocess test content")
    resp = client.post(
        "/rag/documents/upload?auto_process=true",
        files={"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")},
    )
    assert resp.status_code == 201
    doc_id = resp.json()["document_id"]

    status_resp = client.get(f"/rag/documents/{doc_id}/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "QUEUED"


def test_recover_interrupted_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", tmp_path)
    service = get_processing_service()

    doc_id = uuid4()
    st = DocumentProcessingStatus(
        document_id=doc_id,
        status="INDEXING",
        current_stage="indexing",
    )
    st_file = tmp_path / f"{doc_id}.processing.json"
    st_file.write_text(json.dumps(st.model_dump(mode="json")), encoding="utf-8")

    recovered = service.recover_interrupted_tasks()
    assert recovered == 1

    recovered_status = service.get_status(doc_id)
    assert recovered_status.status == "FAILED"
    assert "interrupted by application shutdown" in (recovered_status.error_message or "")
