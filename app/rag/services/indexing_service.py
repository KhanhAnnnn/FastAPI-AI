from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from qdrant_client.models import Distance, PointStruct

from app.config import (
    QDRANT_COLLECTION,
    RAG_INDEXING_BATCH_SIZE,
)
from app.rag import artifacts, storage
from app.rag.index_schemas import IndexDocumentResponse
from app.rag.repositories.vector_repository import (
    CollectionConfigMismatchError,
    StorageUnavailableError,
    VectorRepository,
    VectorStorageError,
    get_vector_repository,
)

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Base exception for indexing errors."""


class DocumentNotFoundError(IndexingError):
    """Raised when document is not found."""


class ExtractionNotReadyError(IndexingError):
    """Raised when document extraction has not been performed."""


class ExtractionNotEmbeddedError(IndexingError):
    """Raised when document has not been embedded yet."""


class IndexingConflictError(IndexingError):
    """Raised when an indexing operation is already active for this document."""


class InvalidArtifactError(IndexingError):
    """Raised when artifact files, manifest, or vectors fail validation."""


class CollectionIncompatibleError(IndexingError):
    """Raised when target collection is incompatible with the artifact."""


class PartialIndexingError(IndexingError):
    """Raised when some points failed verification during indexing."""


def generate_point_id(document_id: str, extraction_id: str, chunk_id: str) -> str:
    """Generate a deterministic, standard UUID string point ID from document_id, extraction_id, and chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag://{document_id}/{extraction_id}/{chunk_id}"))


class IndexingService:
    """Service orchestrating artifact verification and batch indexing into Qdrant."""

    def __init__(self, repository: VectorRepository | None = None) -> None:
        self._repo = repository
        self._active_indexes: set[str] = set()
        self._lock = threading.RLock()

    @property
    def repository(self) -> VectorRepository:
        if self._repo is not None:
            return self._repo
        return get_vector_repository()

    def _locate_artifact(
        self,
        document_id: str,
        extraction_id: str | None = None,
    ) -> Path:
        """Find the matching artifact directory for the given document and optional extraction version."""
        root = artifacts.RAG_ARTIFACT_ROOT.resolve()
        candidates: list[Path] = []
        if root.is_dir():
            for entry in root.iterdir():
                if not entry.is_dir():
                    continue
                parts = entry.name.split(".")
                if len(parts) >= 3 and parts[0] == document_id:
                    if extraction_id is None or parts[1] == str(extraction_id):
                        candidates.append(entry)

        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]

        # No artifact found: determine why
        upload_root = storage.RAG_UPLOAD_ROOT.resolve()
        source_pdf = upload_root / f"{document_id}.pdf"
        extraction_manifest = upload_root / f"{document_id}.extraction.json"

        if not source_pdf.is_file():
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")
        if not extraction_manifest.is_file():
            raise ExtractionNotReadyError(f"Document '{document_id}' has not been extracted yet.")
        raise ExtractionNotEmbeddedError(f"Document '{document_id}' has not been embedded yet.")

    def index_document(
        self,
        document_id_value: str,
        extraction_id: str | None = None,
        collection_name: str | None = None,
        batch_size: int = RAG_INDEXING_BATCH_SIZE,
    ) -> IndexDocumentResponse:
        """Verify artifact and index all chunks into Qdrant idempotently."""
        # Validate UUID format
        try:
            document_uuid = UUID(document_id_value)
            document_id = str(document_uuid)
        except (ValueError, TypeError) as exc:
            raise DocumentNotFoundError("Document ID is not a valid UUID.") from exc

        if extraction_id:
            try:
                extraction_id = str(UUID(extraction_id))
            except (ValueError, TypeError) as exc:
                raise InvalidArtifactError("Specified extraction_id is not a valid UUID.") from exc

        # Concurrency control
        with self._lock:
            if document_id in self._active_indexes:
                raise IndexingConflictError(
                    f"Indexing is already in progress for document '{document_id}'."
                )
            self._active_indexes.add(document_id)

        try:
            return self._index_document_internal(
                document_id=document_id,
                document_uuid=document_uuid,
                extraction_id=extraction_id,
                collection_name=collection_name,
                batch_size=batch_size,
            )
        finally:
            with self._lock:
                self._active_indexes.discard(document_id)

    def _index_document_internal(
        self,
        document_id: str,
        document_uuid: UUID,
        extraction_id: str | None,
        collection_name: str | None,
        batch_size: int,
    ) -> IndexDocumentResponse:
        artifact_dir = self._locate_artifact(document_id, extraction_id)

        # Step 3, 4, 5, 6, 7, 8: Load and verify artifact
        try:
            manifest, chunks, vectors = artifacts.load_artifact(artifact_dir)
        except Exception as exc:
            logger.error("Failed to load or verify artifact '%s': %s", artifact_dir, exc)
            raise InvalidArtifactError(f"Artifact integrity validation failed: {exc}") from exc

        expected_dimension = manifest.get("embedding_dimension", 768)
        chunk_count = manifest.get("chunk_count", len(chunks))

        if len(chunks) != chunk_count or vectors.shape[0] != chunk_count:
            raise InvalidArtifactError(
                f"Artifact chunk count mismatch: manifest says {chunk_count}, "
                f"found {len(chunks)} chunks and {vectors.shape[0]} vectors."
            )
        if vectors.shape[1] != expected_dimension:
            raise InvalidArtifactError(
                f"Vector dimension {vectors.shape[1]} does not match manifest {expected_dimension}."
            )
        if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
            raise InvalidArtifactError("Vectors must be finite float32 arrays.")

        # Step 9: Create or verify collection
        target_collection = collection_name or self.repository.default_collection
        model_id = manifest.get("model_id")
        model_rev = manifest.get("model_revision")

        try:
            self.repository.create_collection_if_not_exists(
                collection_name=target_collection,
                dimension=expected_dimension,
                distance=Distance.COSINE,
                metadata={"model_id": model_id, "model_revision": model_rev},
            )
        except CollectionConfigMismatchError as exc:
            logger.error("Target collection '%s' is incompatible: %s", target_collection, exc)
            raise CollectionIncompatibleError(str(exc)) from exc
        except StorageUnavailableError as exc:
            raise exc

        # Step 10: Build point structures with stable UUIDs and full payload
        points: list[PointStruct] = []
        point_ids: list[str] = []
        ext_version = str(manifest.get("extraction_id") or extraction_id or "")

        for idx, chunk in enumerate(chunks):
            c_id = str(chunk.get("chunk_id", f"chunk-{idx}"))
            c_ext = str(chunk.get("extraction_id") or ext_version)
            pid = generate_point_id(document_id=document_id, extraction_id=c_ext, chunk_id=c_id)
            point_ids.append(pid)

            payload = {
                "document_id": document_id,
                "extraction_version": c_ext,
                "chunk_id": c_id,
                "chunk_index": int(chunk.get("chunk_index", idx)),
                "page_number": int(chunk.get("page_number", 1)),
                "source_filename": str(chunk.get("source_filename", "")),
                "original_text": str(chunk.get("original_text", "")),
                "cleaned_text": str(chunk.get("cleaned_text", "")),
                "token_count": int(chunk.get("token_count", 0)),
                "text_hash": str(chunk.get("text_hash", "")),
                "embedding_model": str(model_id or ""),
                "embedding_revision": str(model_rev or ""),
            }

            vec = vectors[idx].tolist()
            points.append(PointStruct(id=pid, vector=vec, payload=payload))

        # Upsert in batches
        actual_batch_size = max(1, batch_size)
        try:
            for start in range(0, len(points), actual_batch_size):
                batch = points[start : start + actual_batch_size]
                self.repository.upsert_batch(points=batch, collection_name=target_collection)
        except (VectorStorageError, StorageUnavailableError) as exc:
            logger.error("Error during batch upsert for document '%s': %s", document_id, exc)
            raise

        # Step 11: Verify points are saved (Idempotency and completeness check)
        try:
            verified = self.repository.verify_points_exist(
                point_ids=point_ids,
                collection_name=target_collection,
            )
            if not verified:
                raise PartialIndexingError(
                    f"Partial indexing detected: not all {len(point_ids)} points could be verified in Qdrant."
                )
        except (VectorStorageError, StorageUnavailableError) as exc:
            raise PartialIndexingError(f"Failed to verify point existence: {exc}") from exc

        logger.info(
            "Successfully indexed document '%s': chunks=%d, collection='%s', dimension=%d",
            document_id,
            len(points),
            target_collection,
            expected_dimension,
        )

        return IndexDocumentResponse(
            document_id=document_uuid,
            collection=target_collection,
            status="completed",
            indexed_chunks=len(points),
            vector_dimension=expected_dimension,
            distance="cosine",
        )


_global_indexing_service: IndexingService | None = None
_service_lock = threading.Lock()


def get_indexing_service() -> IndexingService:
    global _global_indexing_service
    if _global_indexing_service is not None:
        return _global_indexing_service
    with _service_lock:
        if _global_indexing_service is None:
            _global_indexing_service = IndexingService()
    return _global_indexing_service
