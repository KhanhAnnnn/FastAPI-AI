from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class ExtractedPage(BaseModel):
    page_number: int
    has_text: bool
    extraction_status: Literal["text", "empty", "no_text_layer"]
    character_count: int
    text_start: int | None
    text_end: int | None


class ExtractionResult(BaseModel):
    document_id: UUID
    original_filename: str | None
    total_pages: int
    pages_with_text: int
    pages_without_text: int
    status: Literal["completed"]
    extraction_id: UUID
    text_filename: str
    metadata_filename: str
    pages: list[ExtractedPage]
