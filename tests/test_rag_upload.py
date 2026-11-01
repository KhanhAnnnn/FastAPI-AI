import asyncio
import hashlib
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from app.main import app
from app.rag import router as rag_router
from app.rag import storage


def make_pdf(stream_data: bytes = b"minimal test fixture") -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [] /Count 0 >>",
        b"<< /Length "
        + str(len(stream_data)).encode("ascii")
        + b" >>\nstream\n"
        + stream_data
        + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.7\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(b"xref\n0 4\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n")
    pdf.extend(f"{xref_offset}\n%%EOF\n".encode("ascii"))
    return bytes(pdf)


PDF_BYTES = make_pdf()


@pytest.fixture
def upload_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", tmp_path / "rag-uploads")
    return tmp_path / "rag-uploads"


def test_upload_pdf_returns_created_and_stores_exact_bytes(
    upload_root: Path,
) -> None:
    response = TestClient(app).post(
        "/rag/documents/upload",
        files={"file": ("example.pdf", PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 201
    result = response.json()
    assert result == {
        "document_id": result["document_id"],
        "filename": "example.pdf",
        "stored_filename": f'{result["document_id"]}.pdf',
        "size_bytes": len(PDF_BYTES),
        "content_type": "application/pdf",
        "status": "uploaded",
    }
    stored_file = upload_root / result["stored_filename"]
    assert stored_file.is_file()
    assert stored_file.read_bytes() == PDF_BYTES
    assert hashlib.sha256(stored_file.read_bytes()).hexdigest() == hashlib.sha256(
        PDF_BYTES
    ).hexdigest()


@pytest.mark.parametrize(
    ("filename", "contents"),
    [
        ("not-a-pdf.pdf", b"plain text, not a PDF"),
        (
            "invalid-structure.pdf",
            b"%PDF-1.7\nfake document\nstartxref\n0\n%%EOF\n",
        ),
        ("not-a-pdf.txt", PDF_BYTES),
    ],
)
def test_upload_rejects_non_pdf(
    upload_root: Path, filename: str, contents: bytes
) -> None:
    response = TestClient(app).post(
        "/rag/documents/upload",
        files={"file": (filename, contents, "application/pdf")},
    )

    assert response.status_code == 400
    assert not upload_root.exists() or not any(upload_root.iterdir())


def test_upload_rejects_empty_pdf(upload_root: Path) -> None:
    response = TestClient(app).post(
        "/rag/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert not upload_root.exists() or not any(upload_root.iterdir())


def test_upload_enforces_configured_size_limit(
    upload_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage, "RAG_MAX_UPLOAD_BYTES", len(PDF_BYTES) - 1)

    response = TestClient(app).post(
        "/rag/documents/upload",
        files={"file": ("large.pdf", PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 413
    assert not upload_root.exists() or not any(upload_root.iterdir())


def test_uploads_with_same_original_name_get_distinct_files(upload_root: Path) -> None:
    client = TestClient(app)
    first = client.post(
        "/rag/documents/upload",
        files={"file": ("same.pdf", PDF_BYTES, "application/pdf")},
    )
    second_bytes = make_pdf(b"second test fixture")
    second = client.post(
        "/rag/documents/upload",
        files={"file": ("same.pdf", second_bytes, "application/pdf")},
    )

    assert first.status_code == second.status_code == 201
    first_result, second_result = first.json(), second.json()
    assert first_result["stored_filename"] != second_result["stored_filename"]
    assert (upload_root / first_result["stored_filename"]).read_bytes() == PDF_BYTES
    assert (upload_root / second_result["stored_filename"]).read_bytes() == second_bytes


def test_upload_filename_is_sanitized_and_cannot_escape_storage(
    upload_root: Path,
) -> None:
    response = TestClient(app).post(
        "/rag/documents/upload",
        files={"file": ("../../bad name?.pdf", PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 201
    result = response.json()
    assert result["filename"] == "bad_name_.pdf"
    assert Path(result["stored_filename"]).name == result["stored_filename"]
    assert (upload_root / result["stored_filename"]).is_file()
    assert not (upload_root.parent / "bad name?.pdf").exists()


def test_upload_streams_multiple_chunks(
    upload_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage, "RAG_UPLOAD_CHUNK_SIZE", 32 * 1024)
    contents = make_pdf(b"x" * (160 * 1024))

    response = TestClient(app).post(
        "/rag/documents/upload",
        files={"file": ("multi-chunk.pdf", contents, "application/pdf")},
    )

    assert response.status_code == 201
    saved = upload_root / response.json()["stored_filename"]
    assert saved.stat().st_size == len(contents)
    assert hashlib.sha256(saved.read_bytes()).digest() == hashlib.sha256(
        contents
    ).digest()


def test_failed_stream_removes_temporary_file(upload_root: Path) -> None:
    class FailingUpload(UploadFile):
        def __init__(self) -> None:
            super().__init__(file=BytesIO(), filename="interrupted.pdf")
            self.reads = 0

        async def read(self, _size: int) -> bytes:
            self.reads += 1
            if self.reads == 1:
                return PDF_BYTES[:16]
            raise OSError("simulated interrupted upload")

    with pytest.raises(storage.StorageError):
        asyncio.run(storage.store_pdf(FailingUpload()))

    assert upload_root.is_dir()
    assert list(upload_root.iterdir()) == []


def test_storage_error_returns_generic_500_without_leaking_path(
    upload_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_upload(_file: UploadFile) -> None:
        raise storage.StorageError(f"Cannot write to {upload_root}\\secret.pdf")

    monkeypatch.setattr(rag_router, "upload_document", fail_upload)
    response = TestClient(app).post(
        "/rag/documents/upload",
        files={"file": ("valid.pdf", PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Unable to store uploaded PDF."}
    assert str(upload_root) not in response.text


def test_missing_upload_field_returns_422(upload_root: Path) -> None:
    response = TestClient(app).post("/rag/documents/upload")
    assert response.status_code == 422
    assert not upload_root.exists() or not any(upload_root.iterdir())


def test_health_and_existing_generation_routes_remain_available_in_openapi() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    paths = client.get("/openapi.json").json()["paths"]
    assert {
        "/generate/text",
        "/generate/audio",
        "/generate/image",
        "/generate/video",
        "/generate/3d",
    }.issubset(paths)
