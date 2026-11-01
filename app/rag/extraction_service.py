import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from uuid import UUID, uuid4

from app.config import (
    PROJECT_ROOT,
    RAG_EXTRACTION_TIMEOUT_SECONDS,
    RAG_MAX_EXTRACTION_CHARS,
    RAG_MAX_PDF_PAGES,
    RAG_UPLOAD_ROOT,
)
from app.rag.extraction_schemas import ExtractionResult
from app.rag.storage import _is_within_directory

logger = logging.getLogger(__name__)
_document_locks: dict[str, threading.Lock] = {}
_document_locks_guard = threading.Lock()


class DocumentNotFoundError(Exception):
    pass


class InvalidDocumentError(Exception):
    pass


class ExtractionFailedError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _get_lock(document_id: str) -> threading.Lock:
    with _document_locks_guard:
        return _document_locks.setdefault(document_id, threading.Lock())


def _checked_storage_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not _is_within_directory(path, root):
        raise InvalidDocumentError("Document storage path is invalid.")
    return path


def extract_document(document_id_value: str) -> ExtractionResult:
    try:
        document_id = str(UUID(document_id_value))
    except ValueError as exc:
        raise DocumentNotFoundError("Document was not found.") from exc
    if (
        RAG_EXTRACTION_TIMEOUT_SECONDS <= 0
        or RAG_MAX_PDF_PAGES <= 0
        or RAG_MAX_EXTRACTION_CHARS <= 0
    ):
        raise ExtractionFailedError(
            "invalid_limits", "PDF extraction limits must be positive."
        )

    lock = _get_lock(document_id)
    if not lock.acquire(blocking=False):
        raise ExtractionFailedError("already_running", "Extraction is already running.")

    try:
        root = RAG_UPLOAD_ROOT.resolve()
        source_path = _checked_storage_path(root, f"{document_id}.pdf")
        if not source_path.is_file():
            raise DocumentNotFoundError("Document was not found.")

        original_filename: str | None = None
        upload_metadata_path = _checked_storage_path(
            root, f"{document_id}.upload.json"
        )
        if upload_metadata_path.is_file():
            try:
                upload_metadata = json.loads(
                    upload_metadata_path.read_text(encoding="utf-8")
                )
                if not isinstance(upload_metadata, dict):
                    raise InvalidDocumentError("Document metadata is invalid.")
                if upload_metadata.get("document_id") != document_id:
                    raise InvalidDocumentError("Document metadata is invalid.")
                original_filename = upload_metadata.get("original_filename")
            except (OSError, json.JSONDecodeError) as exc:
                raise ExtractionFailedError(
                    "metadata_error", "Document metadata could not be read."
                ) from exc

        extraction_id = uuid4()
        text_filename = f"{document_id}.{extraction_id}.txt"
        metadata_filename = f"{document_id}.extraction.json"
        text_path = _checked_storage_path(root, text_filename)
        metadata_path = _checked_storage_path(root, metadata_filename)
        text_published = False

        try:
            with tempfile.TemporaryDirectory(prefix=".extract-", dir=root) as temp_dir:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "app.rag.pdf_worker",
                        str(source_path),
                        temp_dir,
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=RAG_EXTRACTION_TIMEOUT_SECONDS,
                    check=False,
                )
                if process.returncode:
                    try:
                        error = json.loads(process.stderr.strip().splitlines()[-1])
                    except (IndexError, json.JSONDecodeError):
                        logger.error(
                            "PDF extraction worker failed: return_code=%s stderr=%s",
                            process.returncode,
                            process.stderr[-2_000:],
                        )
                        raise ExtractionFailedError(
                            "worker_error", "PDF extraction failed."
                        )
                    raise ExtractionFailedError(
                        error.get("code", "worker_error"),
                        error.get("message", "PDF extraction failed."),
                    )

                worker_metadata = json.loads(
                    (Path(temp_dir) / "pages.json").read_text(encoding="utf-8")
                )
                os.link(Path(temp_dir) / "extracted.txt", text_path)
                text_published = True
                manifest = {
                    "document_id": document_id,
                    "original_filename": original_filename,
                    "source_filename": source_path.name,
                    "extraction_id": str(extraction_id),
                    "text_filename": text_filename,
                    "status": "completed",
                    **worker_metadata,
                }
                temporary_manifest = Path(temp_dir) / "extraction.json"
                temporary_manifest.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                    newline="",
                )
                os.replace(temporary_manifest, metadata_path)
        except subprocess.TimeoutExpired as exc:
            raise ExtractionFailedError(
                "timeout", "PDF extraction exceeded the configured time limit."
            ) from exc
        except ExtractionFailedError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            if text_published:
                try:
                    text_path.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Unable to remove incomplete extraction output.")
            logger.error("PDF extraction storage failed.", exc_info=exc)
            raise ExtractionFailedError(
                "storage_error", "PDF extraction output could not be stored."
            ) from exc

        return ExtractionResult(
            document_id=document_id,
            original_filename=original_filename,
            status="completed",
            extraction_id=extraction_id,
            text_filename=text_filename,
            metadata_filename=metadata_filename,
            **worker_metadata,
        )
    finally:
        with _document_locks_guard:
            lock.release()
            if _document_locks.get(document_id) is lock:
                _document_locks.pop(document_id)
