from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class IndexDocumentRequest(BaseModel):
    extraction_id: str | None = Field(
        default=None,
        description="Optional extraction version UUID. If omitted, the latest completed artifact is indexed.",
    )
    collection_name: str | None = Field(
        default=None,
        description="Optional target collection name. Defaults to configured collection.",
    )


class IndexDocumentResponse(BaseModel):
    document_id: UUID
    collection: str
    status: Literal["completed", "failed"]
    indexed_chunks: int
    vector_dimension: int
    distance: str
