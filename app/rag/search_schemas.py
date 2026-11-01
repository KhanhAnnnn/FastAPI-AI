from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.config import MAX_PROMPT_LENGTH, RAG_RETRIEVAL_TOP_K, RAG_SCORE_THRESHOLD


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PROMPT_LENGTH,
        description="Search query string.",
    )
    top_k: int = Field(
        default=RAG_RETRIEVAL_TOP_K,
        ge=1,
        le=100,
        description="Maximum number of relevant chunks to retrieve.",
    )
    score_threshold: float = Field(
        default=RAG_SCORE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score threshold (0.0 to 1.0).",
    )
    document_id: UUID | None = Field(
        default=None,
        description="Optional document UUID to filter retrieval.",
    )
    document_ids: list[UUID] | None = Field(
        default=None,
        description="Optional list of document UUIDs to filter retrieval.",
    )

    @field_validator("query")
    @classmethod
    def validate_non_whitespace(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Query string cannot be empty or pure whitespace.")
        return trimmed


class SearchResultItem(BaseModel):
    document_id: UUID
    page_number: int
    chunk_id: str
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    total_results: int


class IndexStatusResponse(BaseModel):
    collection_name: str
    collection_exists: bool
    vector_dimension: int | None = None
    distance: str | None = None
    point_count: int = 0
    storage_mode: str = "local"
    status: str
