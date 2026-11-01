from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class UploadedDocument(BaseModel):
    document_id: UUID
    filename: str
    stored_filename: str
    size_bytes: int
    content_type: Literal["application/pdf"]
    status: Literal["uploaded"]


from app.rag.chat_schemas import (  # noqa: E402
    ChatRequest,
    ChatResponse,
    SourceItem,
)
from app.rag.index_schemas import IndexDocumentRequest, IndexDocumentResponse  # noqa: E402
from app.rag.processing_schemas import (  # noqa: E402
    DocumentProcessingStatus,
    DocumentSummary,
    ProcessDocumentRequest,
)
from app.rag.search_schemas import (  # noqa: E402
    IndexStatusResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

__all__ = [
    "UploadedDocument",
    "IndexDocumentRequest",
    "IndexDocumentResponse",
    "ProcessDocumentRequest",
    "DocumentProcessingStatus",
    "DocumentSummary",
    "SearchRequest",
    "SearchResultItem",
    "SearchResponse",
    "IndexStatusResponse",
    "ChatRequest",
    "ChatResponse",
    "SourceItem",
]
