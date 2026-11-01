from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from app.config import QDRANT_COLLECTION, RAG_UPLOAD_ROOT
from app.rag import storage
from app.rag.embedding_service import embed_document
from app.rag.extraction_service import (
    DocumentNotFoundError as ExtractionDocNotFoundError,
    extract_document,
)
from app.rag.processing_schemas import (
    DocumentProcessingStatus,
    DocumentSummary,
    ProcessingState,
)
from app.rag.services.indexing_service import (
    DocumentNotFoundError as IndexingDocNotFoundError,
    get_indexing_service,
)

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Base exception for pipeline processing."""


class DocumentNotFoundError(ProcessingError):
    """Raised when the document cannot be found."""


class ProcessingConflictError(ProcessingError):
    """Raised when a document is already being processed or queued."""


class ProcessingService:
    """Manages the end-to-end background processing pipeline for PDF documents."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[tuple[str, str | None, UUID]] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._active_documents: set[str] = set()
        self._cancellation_requested: set[str] = set()
        self._indexing_service = get_indexing_service()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind processing service to an explicit asyncio event loop."""
        self._loop = loop
        if self._queue is None:
            self._queue = asyncio.Queue()

    def _get_queue(self) -> asyncio.Queue[tuple[str, str | None, UUID]]:
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    def _status_path(self, document_id: str) -> Path:
        root = storage.RAG_UPLOAD_ROOT.resolve()
        return root / f"{document_id}.processing.json"

    def _save_status(self, status: DocumentProcessingStatus) -> None:
        """Atomically persist status to disk."""
        path = self._status_path(str(status.document_id))
        root = path.parent
        root.mkdir(parents=True, exist_ok=True)
        temp_path = root / f".{status.document_id}.processing.tmp"

        data = status.model_dump(mode="json")
        try:
            temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            temp_path.replace(path)
        except OSError as exc:
            logger.error("Failed to persist processing status for %s: %s", status.document_id, exc)
            temp_path.unlink(missing_ok=True)
            raise

    def get_status(self, document_id_value: str | UUID) -> DocumentProcessingStatus:
        """Retrieve current processing status for a document."""
        try:
            doc_uuid = UUID(str(document_id_value))
            doc_id = str(doc_uuid)
        except (ValueError, TypeError) as exc:
            raise DocumentNotFoundError("Invalid document ID format.") from exc

        path = self._status_path(doc_id)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return DocumentProcessingStatus(**data)
            except Exception as exc:
                logger.warning("Corrupted status file for %s, re-reading upload: %s", doc_id, exc)

        # Fallback: check if uploaded PDF exists
        upload_root = storage.RAG_UPLOAD_ROOT.resolve()
        pdf_path = upload_root / f"{doc_id}.pdf"
        if not pdf_path.is_file():
            raise DocumentNotFoundError(f"Document '{doc_id}' was not found.")

        # Read original filename from upload metadata if present
        upload_meta_path = upload_root / f"{doc_id}.upload.json"
        orig_filename = None
        if upload_meta_path.is_file():
            try:
                meta = json.loads(upload_meta_path.read_text(encoding="utf-8"))
                orig_filename = meta.get("original_filename")
            except Exception:
                pass

        initial_status = DocumentProcessingStatus(
            document_id=doc_uuid,
            status="UPLOADED",
            original_filename=orig_filename,
        )
        self._save_status(initial_status)
        return initial_status

    def list_documents(self) -> list[DocumentSummary]:
        """List all known documents in the system with their status and metadata."""
        upload_root = storage.RAG_UPLOAD_ROOT.resolve()
        if not upload_root.is_dir():
            return []

        summaries: list[DocumentSummary] = []
        for pdf_file in upload_root.glob("*.pdf"):
            doc_id = pdf_file.stem
            try:
                doc_uuid = UUID(doc_id)
            except ValueError:
                continue

            try:
                st = self.get_status(doc_uuid)
                upload_meta_path = upload_root / f"{doc_id}.upload.json"
                size_bytes = pdf_file.stat().st_size
                uploaded_at = None
                orig_name = st.original_filename
                if upload_meta_path.is_file():
                    try:
                        u_meta = json.loads(upload_meta_path.read_text(encoding="utf-8"))
                        orig_name = orig_name or u_meta.get("original_filename")
                        size_bytes = u_meta.get("size_bytes", size_bytes)
                        uploaded_at = u_meta.get("uploaded_at")
                    except Exception:
                        pass

                summaries.append(
                    DocumentSummary(
                        document_id=doc_uuid,
                        original_filename=orig_name or pdf_file.name,
                        status=st.status,
                        size_bytes=size_bytes,
                        uploaded_at=uploaded_at,
                        indexed_chunk_count=st.indexed_chunk_count,
                        failure_stage=st.failure_stage,
                        error_message=st.error_message,
                    )
                )
            except Exception as exc:
                logger.warning("Error generating summary for %s: %s", doc_id, exc)

        return sorted(summaries, key=lambda s: s.uploaded_at or "", reverse=True)

    def enqueue_processing(
        self,
        document_id_value: str | UUID,
        collection_name: str | None = None,
        force_reprocess: bool = False,
    ) -> DocumentProcessingStatus:
        """Enqueue document for background processing."""
        current_status = self.get_status(document_id_value)
        doc_id = str(current_status.document_id)

        with self._lock:
            if doc_id in self._active_documents:
                if force_reprocess:
                    logger.warning("Force reprocess requested for active document %s; clearing active state.", doc_id)
                    self._active_documents.discard(doc_id)
                else:
                    raise ProcessingConflictError(f"Document '{doc_id}' is already queued or processing.")

            if current_status.status == "COMPLETED" and not force_reprocess:
                return current_status

            run_id = uuid4()
            current_status.status = "QUEUED"
            current_status.processing_run_id = run_id
            current_status.current_stage = "queued"
            current_status.started_at = datetime.now(UTC).isoformat()
            current_status.completed_at = None
            current_status.failure_stage = None
            current_status.error_message = None
            current_status.retry_count = (
                current_status.retry_count + 1 if current_status.status == "FAILED" else current_status.retry_count
            )

            self._active_documents.add(doc_id)
            self._cancellation_requested.discard(doc_id)
            self._save_status(current_status)

        try:
            self._ensure_worker_running()
            queue = self._get_queue()
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop is not None and (self._loop is None or running_loop is self._loop):
                queue.put_nowait((doc_id, collection_name, run_id))
            elif self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(queue.put_nowait, (doc_id, collection_name, run_id))
            else:
                queue.put_nowait((doc_id, collection_name, run_id))
        except Exception:
            with self._lock:
                self._active_documents.discard(doc_id)
            raise

        return current_status

    def cancel_processing(self, document_id_value: str | UUID) -> DocumentProcessingStatus:
        """Request cancellation of an active or queued processing task."""
        current_status = self.get_status(document_id_value)
        doc_id = str(current_status.document_id)

        with self._lock:
            self._cancellation_requested.add(doc_id)
            self._active_documents.discard(doc_id)
            if current_status.status in ("QUEUED", "EXTRACTING", "EMBEDDING", "INDEXING"):
                current_status.status = "CANCELLED"
                current_status.error_message = "Processing cancelled by user request."
                current_status.completed_at = datetime.now(UTC).isoformat()
                self._save_status(current_status)
                logger.info("Cancelled processing for document '%s'", doc_id)

        return current_status

    def _ensure_worker_running(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._loop

        if loop is None:
            return

        if self._loop is None and loop.is_running():
            self._loop = loop

        if self._worker_task is None or self._worker_task.done():
            try:
                curr_loop = asyncio.get_running_loop()
            except RuntimeError:
                curr_loop = None

            if curr_loop is loop and loop.is_running():
                self._worker_task = loop.create_task(self._worker_loop())
            elif loop.is_running():
                asyncio.run_coroutine_threadsafe(self._start_worker_if_needed(), loop)

    async def _start_worker_if_needed(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        """Background loop consuming tasks sequentially with single-worker concurrency."""
        logger.info("RAG processing worker loop started.")
        queue = self._get_queue()
        while True:
            try:
                document_id, collection_name, run_id = await queue.get()
                try:
                    await self._process_single_document(document_id, collection_name, run_id)
                except Exception as exc:
                    logger.exception("Unexpected error in worker loop for %s: %s", document_id, exc)
                finally:
                    queue.task_done()
                    with self._lock:
                        self._active_documents.discard(document_id)
            except asyncio.CancelledError:
                logger.info("RAG processing worker loop cancelled.")
                break
            except Exception as exc:
                logger.error("Error in RAG processing worker: %s", exc, exc_info=True)
                await asyncio.sleep(1.0)

    async def _process_single_document(
        self,
        document_id: str,
        collection_name: str | None,
        run_id: UUID,
    ) -> None:
        st = self.get_status(document_id)
        if document_id in self._cancellation_requested:
            logger.info("Skipping cancelled document %s before execution", document_id)
            st.status = "CANCELLED"
            st.completed_at = datetime.now(UTC).isoformat()
            self._save_status(st)
            return

        target_collection = collection_name or QDRANT_COLLECTION

        # Stage 1: Extraction
        st.status = "EXTRACTING"
        st.current_stage = "extraction"
        self._save_status(st)
        logger.info("Starting extraction for document %s", document_id)

        try:
            # extract_document is synchronous, execute in thread pool to not block asyncio event loop
            ext_res = await asyncio.to_thread(extract_document, document_id)
            st.extraction_version = str(ext_res.extraction_id)
            self._save_status(st)
        except Exception as exc:
            logger.error("Extraction failed for %s: %s", document_id, exc)
            st.status = "FAILED"
            st.failure_stage = "extraction"
            st.error_message = str(exc)
            st.completed_at = datetime.now(UTC).isoformat()
            self._save_status(st)
            return

        if document_id in self._cancellation_requested:
            st.status = "CANCELLED"
            st.completed_at = datetime.now(UTC).isoformat()
            self._save_status(st)
            return

        # Check if text was extractable
        if ext_res.pages_with_text == 0:
            logger.warning("Document %s has no extractable text", document_id)
            st.status = "FAILED"
            st.failure_stage = "extraction"
            st.error_message = "No extractable text layer found in document."
            st.completed_at = datetime.now(UTC).isoformat()
            self._save_status(st)
            return

        # Stage 2: Chunking & Embedding
        st.status = "EMBEDDING"
        st.current_stage = "embedding"
        self._save_status(st)
        logger.info("Starting embedding for document %s", document_id)

        try:
            emb_res = await embed_document(document_id)
            st.embedding_artifact_id = emb_res.artifact_directory
            self._save_status(st)
        except Exception as exc:
            logger.error("Embedding failed for %s: %s", document_id, exc)
            st.status = "FAILED"
            st.failure_stage = "embedding"
            st.error_message = str(exc)
            st.completed_at = datetime.now(UTC).isoformat()
            self._save_status(st)
            return

        if document_id in self._cancellation_requested:
            st.status = "CANCELLED"
            st.completed_at = datetime.now(UTC).isoformat()
            self._save_status(st)
            return

        # Stage 3: Qdrant Indexing
        st.status = "INDEXING"
        st.current_stage = "indexing"
        self._save_status(st)
        logger.info("Starting indexing for document %s into %s", document_id, target_collection)

        try:
            idx_res = await asyncio.to_thread(
                self._indexing_service.index_document,
                document_id_value=document_id,
                extraction_id=st.extraction_version,
                collection_name=target_collection,
            )
            st.indexed_chunk_count = idx_res.indexed_chunks
        except Exception as exc:
            logger.error("Indexing failed for %s: %s", document_id, exc)
            st.status = "FAILED"
            st.failure_stage = "indexing"
            st.error_message = str(exc)
            st.completed_at = datetime.now(UTC).isoformat()
            self._save_status(st)
            return

        # Stage 4: Completed
        st.status = "COMPLETED"
        st.current_stage = None
        st.error_message = None
        st.completed_at = datetime.now(UTC).isoformat()
        self._save_status(st)
        logger.info(
            "Document %s fully processed and indexed (%d chunks)",
            document_id,
            st.indexed_chunk_count,
        )

    def recover_interrupted_tasks(self) -> int:
        """Scan upload directory for tasks interrupted by previous server shutdowns."""
        upload_root = storage.RAG_UPLOAD_ROOT.resolve()
        if not upload_root.is_dir():
            return 0

        recovered = 0
        for status_file in upload_root.glob("*.processing.json"):
            try:
                data = json.loads(status_file.read_text(encoding="utf-8"))
                st = DocumentProcessingStatus(**data)
                if st.status in ("QUEUED", "EXTRACTING", "EMBEDDING", "INDEXING"):
                    logger.warning(
                        "Recovering interrupted task for document %s (state was %s)",
                        st.document_id,
                        st.status,
                    )
                    st.status = "FAILED"
                    st.failure_stage = st.current_stage or st.status.lower()
                    st.error_message = "Task interrupted by application shutdown. Safe to retry."
                    st.completed_at = datetime.now(UTC).isoformat()
                    self._save_status(st)
                    recovered += 1
            except Exception as exc:
                logger.warning("Error inspecting status file %s during recovery: %s", status_file, exc)

        if recovered:
            logger.info("Recovered %d interrupted background tasks.", recovered)
        return recovered

    def shutdown(self) -> None:
        """Cancel worker task cleanly on shutdown."""
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()


_global_processing_service: ProcessingService | None = None
_proc_lock = threading.Lock()


def get_processing_service() -> ProcessingService:
    global _global_processing_service
    if _global_processing_service is not None:
        return _global_processing_service
    with _proc_lock:
        if _global_processing_service is None:
            _global_processing_service = ProcessingService()
    return _global_processing_service
