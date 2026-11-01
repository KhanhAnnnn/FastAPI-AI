import json
import os
from datetime import UTC, datetime
from pathlib import Path

from fastapi import UploadFile

from app.rag import storage
from app.rag.schemas import UploadedDocument
from app.rag.storage import StorageError


async def upload_document(file: UploadFile) -> UploadedDocument:
    document_id, filename, stored_filename, size_bytes = await storage.store_pdf(file)
    root = storage.RAG_UPLOAD_ROOT.resolve()
    metadata_path = root / f"{document_id}.upload.json"
    temporary_path = root / f".{document_id}.uploading.json"
    metadata = {
        "document_id": document_id,
        "original_filename": filename,
        "stored_filename": stored_filename,
        "size_bytes": size_bytes,
        "content_type": "application/pdf",
        "uploaded_at": datetime.now(UTC).isoformat(),
    }
    try:
        with temporary_path.open("x", encoding="utf-8") as output:
            json.dump(metadata, output, ensure_ascii=False)
        os.link(temporary_path, metadata_path)
    except OSError as exc:
        try:
            (root / stored_filename).unlink(missing_ok=True)
        except OSError:
            raise StorageError("Unable to roll back failed PDF registration.") from exc
        raise StorageError("Unable to store uploaded PDF metadata.") from exc
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError as exc:
            raise StorageError("Unable to clean up upload metadata.") from exc

    return UploadedDocument(
        document_id=document_id,
        filename=filename,
        stored_filename=stored_filename,
        size_bytes=size_bytes,
        content_type="application/pdf",
        status="uploaded",
    )
