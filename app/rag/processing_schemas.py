from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

ProcessingState = Literal[
    "UPLOADED",
    "QUEUED",
    "EXTRACTING",
    "EMBEDDING",
    "INDEXING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
]


class ProcessDocumentRequest(BaseModel):
    collection_name: str | None = Field(
        default=None,
        description="Target Qdrant collection name. Defaults to configured collection.",
    )
    force_reprocess: bool = Field(
        default=False,
        description="If True, re-runs processing even if previously completed or failed.",
    )


class DocumentProcessingStatus(BaseModel):
    document_id: UUID
    processing_run_id: UUID | None = None
    status: ProcessingState
    current_stage: str | None = None
    extraction_version: str | None = None
    embedding_artifact_id: str | None = None
    indexed_chunk_count: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    failure_stage: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    original_filename: str | None = None


class DocumentSummary(BaseModel):
    document_id: UUID
    original_filename: str | None = None
    status: ProcessingState
    size_bytes: int = 0
    uploaded_at: str | None = None
    indexed_chunk_count: int = 0
    failure_stage: str | None = None
    error_message: str | None = None
