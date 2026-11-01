import logging
import os
import re
import unicodedata
from pathlib import Path
from uuid import uuid4

import anyio
from fastapi import UploadFile

from app.config import RAG_MAX_UPLOAD_BYTES, RAG_UPLOAD_CHUNK_SIZE, RAG_UPLOAD_ROOT

logger = logging.getLogger(__name__)

_PDF_HEADER = b"%PDF-"
_PDF_HEADER_WINDOW = 1024
_PDF_TAIL_WINDOW = 64 * 1024
_PDF_XREF_WINDOW = 64 * 1024


class InvalidPDFError(ValueError):
    pass


class UploadTooLargeError(ValueError):
    pass


class StorageError(OSError):
    pass


def _safe_filename(filename: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", filename or "").replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]
    safe_name = re.sub(r"[^\w.-]", "_", basename, flags=re.UNICODE).strip(" ._")
    return safe_name or "document.pdf"


def _is_within_directory(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


async def _has_pdf_structure(path: Path, file_size: int, tail: bytes) -> bool:
    footer_match = re.search(rb"startxref\s+(\d+)\s+%%EOF\s*\Z", tail)
    if footer_match is None:
        return False

    xref_offset = int(footer_match.group(1))
    if xref_offset >= file_size:
        return False

    async with await anyio.open_file(path, "rb") as source:
        await source.seek(xref_offset)
        xref_section = await source.read(_PDF_XREF_WINDOW)

    if xref_section.startswith(b"xref"):
        return b"trailer" in tail and re.search(
            rb"/Root\s+\d+\s+\d+\s+R\b", tail
        ) is not None
    return (
        re.match(rb"\s*\d+\s+\d+\s+obj\b", xref_section) is not None
        and re.search(rb"/Type\s*/XRef\b", xref_section) is not None
        and re.search(rb"/Root\s+\d+\s+\d+\s+R\b", xref_section) is not None
    )


async def store_pdf(file: UploadFile) -> tuple[str, str, str, int]:
    safe_filename = _safe_filename(file.filename)
    if Path(safe_filename).suffix.lower() != ".pdf":
        raise InvalidPDFError("Upload must be a PDF file.")
    if RAG_UPLOAD_CHUNK_SIZE <= 0 or RAG_MAX_UPLOAD_BYTES <= 0:
        raise StorageError("RAG upload settings must be positive integers.")

    root = RAG_UPLOAD_ROOT.resolve()
    document_id = str(uuid4())
    stored_filename = f"{document_id}.pdf"
    destination = (root / stored_filename).resolve()
    temporary_path = (root / f".{document_id}.uploading").resolve()
    if not _is_within_directory(destination, root) or not _is_within_directory(
        temporary_path, root
    ):
        raise StorageError("Resolved upload path is outside the RAG storage directory.")

    size_bytes = 0
    header = bytearray()
    tail = b""
    try:
        root.mkdir(parents=True, exist_ok=True)
        async with await anyio.open_file(temporary_path, "xb") as output:
            while chunk := await file.read(RAG_UPLOAD_CHUNK_SIZE):
                size_bytes += len(chunk)
                if size_bytes > RAG_MAX_UPLOAD_BYTES:
                    raise UploadTooLargeError("PDF exceeds the configured upload limit.")
                if len(header) < _PDF_HEADER_WINDOW:
                    header.extend(chunk[: _PDF_HEADER_WINDOW - len(header)])
                tail = (tail + chunk)[-_PDF_TAIL_WINDOW:]
                await output.write(chunk)

        if not size_bytes:
            raise InvalidPDFError("Uploaded PDF is empty.")
        if _PDF_HEADER not in header or not await _has_pdf_structure(
            temporary_path, size_bytes, tail
        ):
            raise InvalidPDFError("Upload is not a valid PDF file.")

        # A hard link promotes the complete file atomically without replacing a
        # destination if an unexpected UUID collision occurs.
        os.link(temporary_path, destination)
        temporary_path.unlink()
    except (InvalidPDFError, UploadTooLargeError):
        raise
    except OSError as exc:
        raise StorageError("Unable to store uploaded PDF.") from exc
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Unable to remove incomplete RAG upload.")
            raise StorageError("Unable to clean up incomplete upload.")

    return document_id, safe_filename, stored_filename, size_bytes
