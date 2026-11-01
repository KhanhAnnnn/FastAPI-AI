from __future__ import annotations

from uuid import UUID
from pydantic import BaseModel, Field, field_validator

from app.config import MAX_PROMPT_LENGTH, RAG_RETRIEVAL_TOP_K


class SourceItem(BaseModel):
    document_id: UUID
    page_number: int
    chunk_id: str
    score: float
    text: str


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_PROMPT_LENGTH,
        description="User question or query for RAG retrieval and answering.",
    )
    document_id: UUID | None = Field(
        default=None,
        description="Optional single document UUID filter.",
    )
    document_ids: list[UUID] | None = Field(
        default=None,
        description="Optional list of document UUIDs to filter retrieval.",
    )
    top_k: int = Field(
        default=RAG_RETRIEVAL_TOP_K,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve.",
    )
    score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (0.0 to 1.0) for context inclusion.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional override for the RAG system prompt.",
    )
    max_new_tokens: int = Field(
        default=256,
        ge=1,
        le=512,
        description="Maximum tokens for TinyLlama generation.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower values produce more grounded, deterministic responses.",
    )

    @field_validator("query")
    @classmethod
    def validate_non_whitespace(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Query string cannot be empty or pure whitespace.")
        return trimmed


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceItem]
    retrieval_count: int
    model_used: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    latency_ms: dict[str, float] = Field(default_factory=dict)
