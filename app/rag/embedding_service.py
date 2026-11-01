from __future__ import annotations

import json
import hashlib
import logging
import os
import time
from pathlib import Path
from uuid import UUID, uuid4
from typing import Any

import numpy as np

from app.config import (
    RAG_CHUNK_OVERLAP_TOKENS,
    RAG_CHUNK_SIZE_TOKENS,
    RAG_EMBEDDING_BATCH_SIZE,
    RAG_EMBEDDING_CODE_REVISION,
    RAG_EMBEDDING_MODEL_ID,
    RAG_EMBEDDING_MODEL_REVISION,
    RAG_MAX_EXTRACTION_CHARS,
)
from app.core.model_manager import model_manager
from app.rag import artifacts
from app.rag.chunking import TextChunk, chunk_pages
from app.rag.embedding_model import (
    EmbeddingModelError,
    JinaEmbedder,
    RemoteCodeApprovalRequired,
    get_embedder,
)
from app.rag.embedding_schemas import EmbeddedDocument
from app.rag.extraction_service import (
    DocumentNotFoundError,
    InvalidDocumentError,
    _checked_storage_path,
)
from app.rag import storage

logger = logging.getLogger(__name__)


class ExtractionNotReadyError(Exception):
    pass


class EmbeddingFailedError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _validate_document_ready(document_id_value: str) -> str:
    try:
        document_id = str(UUID(document_id_value))
    except ValueError as exc:
        raise DocumentNotFoundError("Document was not found.") from exc
    _load_extraction(storage.RAG_UPLOAD_ROOT.resolve(), document_id)
    return document_id


async def embed_document(document_id: str) -> EmbeddedDocument:
    document_id = _validate_document_ready(document_id)
    if os.getenv("RAG_ALLOW_JINA_REMOTE_CODE", "false").lower() != "true":
        raise RemoteCodeApprovalRequired(
            "Review and explicitly approve the pinned Jina remote-code revisions "
            "before enabling RAG_ALLOW_JINA_REMOTE_CODE=true."
        )
    return await model_manager.run_exclusive(
        "rag_embedding", lambda: _embed_document_sync(document_id)
    )


def _load_extraction(root: Path, document_id: str) -> tuple[dict[str, Any], str]:
    manifest_path = _checked_storage_path(
        root, f"{document_id}.extraction.json"
    )
    source_path = _checked_storage_path(root, f"{document_id}.pdf")
    if not source_path.is_file():
        raise DocumentNotFoundError("Document was not found.")
    if not manifest_path.is_file():
        raise ExtractionNotReadyError("Document has not been extracted.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingFailedError(
            "extraction_metadata_error", "Extraction metadata could not be read."
        ) from exc
    if not isinstance(manifest, dict):
        raise EmbeddingFailedError(
            "extraction_metadata_error", "Extraction metadata could not be read."
        )
    if (
        manifest.get("document_id") != document_id
        or manifest.get("status") != "completed"
    ):
        raise ExtractionNotReadyError("Document has no completed extraction.")
    extraction_id = manifest.get("extraction_id")
    try:
        extraction_id = str(UUID(extraction_id))
    except (ValueError, TypeError) as exc:
        raise EmbeddingFailedError(
            "extraction_metadata_error", "Extraction version is invalid."
        ) from exc
    text_filename = manifest.get("text_filename")
    if not isinstance(text_filename, str) or Path(text_filename).name != text_filename:
        raise InvalidDocumentError("Extraction text path is invalid.")
    text_path = _checked_storage_path(root, text_filename)
    if not text_path.is_file():
        raise EmbeddingFailedError(
            "extraction_output_missing", "Extracted text file is missing."
        )
    if text_path.stat().st_size > RAG_MAX_EXTRACTION_CHARS * 4:
        raise EmbeddingFailedError(
            "extraction_output_too_large", "Extracted text exceeds configured limits."
        )
    try:
        text = text_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EmbeddingFailedError(
            "extraction_output_invalid", "Extracted text is not readable UTF-8."
        ) from exc
    return manifest | {"extraction_id": extraction_id}, text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_document(
    document_id: str,
    manifest: dict[str, Any],
    text: str,
    embedder: JinaEmbedder,
) -> list[TextChunk]:
    special_tokens = int(
        embedder.tokenizer.num_special_tokens_to_add(pair=False)
    )
    chunk_size = RAG_CHUNK_SIZE_TOKENS
    content_context = embedder.context_tokens - special_tokens
    if chunk_size > content_context:
        raise EmbeddingFailedError(
            "invalid_chunk_configuration",
            "Configured chunk size exceeds the embedding model context.",
        )
    try:
        chunks = chunk_pages(
            document_id=document_id,
            extraction_id=manifest["extraction_id"],
            source_filename=manifest.get("original_filename"),
            extracted_text=text,
            pages=manifest["pages"],
            tokenizer=embedder.tokenizer,
            chunk_size_tokens=chunk_size,
            chunk_overlap_tokens=RAG_CHUNK_OVERLAP_TOKENS,
            model_context_tokens=embedder.context_tokens,
            special_tokens=special_tokens,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingFailedError(
            "chunking_error", "Extraction page metadata could not be chunked."
        ) from exc
    if not chunks:
        raise EmbeddingFailedError(
            "no_text", "Completed extraction contains no text to embed."
        )
    return chunks


def _embed_document_sync(document_id_value: str) -> EmbeddedDocument:
    document_id = _validate_document_ready(document_id_value)
    root = storage.RAG_UPLOAD_ROOT.resolve()
    manifest, text = _load_extraction(root, document_id)
    source_pdf = _checked_storage_path(root, f"{document_id}.pdf")
    extracted_text_path = _checked_storage_path(root, manifest["text_filename"])
    embedder = get_embedder()
    chunks = _chunk_document(document_id, manifest, text, embedder)

    embedding_started = time.perf_counter()
    try:
        vectors = embedder.embed(
            [chunk.cleaned_text for chunk in chunks],
            batch_size=RAG_EMBEDDING_BATCH_SIZE,
        )
    except EmbeddingModelError as exc:
        raise EmbeddingFailedError("embedding_error", str(exc)) from exc
    embedding_seconds = time.perf_counter() - embedding_started
    if vectors.shape != (len(chunks), embedder.embedding_dimension):
        raise EmbeddingFailedError(
            "embedding_shape_error", "Embedding output has an unexpected shape."
        )
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise EmbeddingFailedError(
            "embedding_values_error", "Embedding output contains invalid values."
        )

    run_id = str(uuid4())
    try:
        artifact_dir, _manifest = artifacts.write_artifact(
            document_id=document_id,
            extraction_id=manifest["extraction_id"],
            run_id=run_id,
            model_id=RAG_EMBEDDING_MODEL_ID,
            model_revision=RAG_EMBEDDING_MODEL_REVISION,
            code_revision=RAG_EMBEDDING_CODE_REVISION,
            tokenizer_name=RAG_EMBEDDING_MODEL_ID,
            context_tokens=embedder.context_tokens,
            chunk_size_tokens=RAG_CHUNK_SIZE_TOKENS,
            chunk_overlap_tokens=RAG_CHUNK_OVERLAP_TOKENS,
            embedding_dimension=embedder.embedding_dimension,
            device=embedder.device,
            model_load_seconds=embedder.load_seconds,
            embedding_seconds=embedding_seconds,
            source_pdf_sha256=_sha256_file(source_pdf),
            extracted_text_sha256=_sha256_file(extracted_text_path),
            chunks=chunks,
            vectors=vectors,
        )
    except (OSError, ValueError) as exc:
        logger.exception("RAG embedding artifact creation failed.")
        raise EmbeddingFailedError(
            "artifact_error", "Embedding artifacts could not be stored."
        ) from exc

    return EmbeddedDocument(
        document_id=document_id,
        extraction_id=manifest["extraction_id"],
        run_id=run_id,
        status="completed",
        model_id=RAG_EMBEDDING_MODEL_ID,
        model_revision=RAG_EMBEDDING_MODEL_REVISION,
        code_revision=RAG_EMBEDDING_CODE_REVISION,
        chunk_count=len(chunks),
        embedding_dimension=embedder.embedding_dimension,
        device="cpu",
        model_load_seconds=embedder.load_seconds,
        embedding_seconds=embedding_seconds,
        artifact_directory=str(
            artifact_dir.relative_to(artifacts.RAG_ARTIFACT_ROOT.resolve())
        ),
        manifest_filename="manifest.json",
    )
