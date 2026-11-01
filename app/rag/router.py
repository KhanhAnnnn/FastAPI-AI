from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile, status

from app.core.exceptions import ModelBusyError
from app.rag.chat_schemas import ChatRequest, ChatResponse
from app.rag.chat_service import get_chat_service
from app.rag.embedding_model import (
    EmbeddingModelError,
    RemoteCodeApprovalRequired,
)
from app.rag.embedding_schemas import EmbeddedDocument
from app.rag.embedding_service import (
    DocumentNotFoundError as EmbeddingDocumentNotFoundError,
    EmbeddingFailedError,
    ExtractionNotReadyError,
    embed_document,
)
from app.rag.extraction_schemas import ExtractionResult
from app.rag.extraction_service import (
    DocumentNotFoundError,
    ExtractionFailedError,
    InvalidDocumentError,
    extract_document,
)
from app.rag.index_schemas import IndexDocumentRequest, IndexDocumentResponse
from app.rag.indexing_service import (
    CollectionIncompatibleError,
    DocumentNotFoundError as IndexDocumentNotFoundError,
    ExtractionNotEmbeddedError,
    ExtractionNotReadyError as IndexExtractionNotReadyError,
    IndexingConflictError,
    InvalidArtifactError,
    PartialIndexingError,
    get_indexing_service,
)
from app.rag.processing_schemas import (
    DocumentProcessingStatus,
    DocumentSummary,
    ProcessDocumentRequest,
)
from app.rag.processing_service import (
    DocumentNotFoundError as ProcessingDocumentNotFoundError,
    ProcessingConflictError,
    get_processing_service,
)
from app.rag.repositories.vector_repository import (
    StorageUnavailableError,
    VectorStorageError,
    get_vector_repository,
)
from app.rag.retrieval_service import (
    InvalidQueryError,
    get_retrieval_service,
)
from app.rag.schemas import UploadedDocument
from app.rag.search_schemas import (
    IndexStatusResponse,
    SearchRequest,
    SearchResponse,
)
from app.rag.storage import InvalidPDFError, StorageError, UploadTooLargeError
from app.rag.upload_service import upload_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])
documents_router = APIRouter(prefix="/documents", tags=["RAG documents"])


# ==========================================
# Document Upload, Extraction, Embedding
# ==========================================


@documents_router.get(
    "",
    response_model=list[DocumentSummary],
    status_code=status.HTTP_200_OK,
)
def list_documents_endpoint() -> list[DocumentSummary]:
    service = get_processing_service()
    return service.list_documents()


@documents_router.post(
    "/upload",
    response_model=UploadedDocument,
    status_code=status.HTTP_201_CREATED,
)
async def upload_pdf(
    file: Annotated[UploadFile, File(...)],
    auto_process: bool = Query(False, description="Queue pipeline processing immediately after upload"),
) -> UploadedDocument:
    try:
        uploaded = await upload_document(file)
        if auto_process:
            get_processing_service().enqueue_processing(uploaded.document_id)
        return uploaded
    except InvalidPDFError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except StorageError as exc:
        logger.error("RAG PDF storage failed.", exc_info=exc)
        raise HTTPException(
            status_code=500,
            detail="Unable to store uploaded PDF.",
        ) from exc
    finally:
        await file.close()


@documents_router.post("/{document_id}/extract", response_model=ExtractionResult)
def extract_pdf(document_id: str) -> ExtractionResult:
    try:
        return extract_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ExtractionFailedError as exc:
        status_code = {
            "already_running": 409,
            "invalid_pdf": 400,
            "encrypted_pdf": 422,
            "page_extraction_error": 422,
            "page_limit": 413,
            "character_limit": 413,
            "timeout": 408,
            "metadata_error": 500,
            "invalid_limits": 500,
            "storage_error": 500,
            "worker_error": 500,
        }.get(exc.code, 500)
        if status_code >= 500:
            logger.error("RAG PDF extraction failed: code=%s", exc.code)
            detail = "PDF extraction failed."
        else:
            detail = str(exc)
        raise HTTPException(status_code=status_code, detail=detail) from exc


@documents_router.post("/{document_id}/embed", response_model=EmbeddedDocument)
async def embed_pdf_document(document_id: str) -> EmbeddedDocument:
    try:
        return await embed_document(document_id)
    except RemoteCodeApprovalRequired as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelBusyError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "2"},
        ) from exc
    except EmbeddingDocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExtractionNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EmbeddingFailedError, EmbeddingModelError) as exc:
        code = getattr(exc, "code", "embedding_error")
        status_code = {
            "invalid_chunk_configuration": 500,
            "chunking_error": 422,
            "no_text": 422,
            "embedding_error": 502,
            "embedding_shape_error": 502,
            "embedding_values_error": 502,
            "artifact_error": 500,
            "extraction_metadata_error": 500,
            "extraction_output_missing": 500,
            "extraction_output_too_large": 413,
            "extraction_output_invalid": 500,
        }.get(code, 502)
        logger.error("RAG embedding failed: code=%s", code)
        detail = str(exc) if status_code in (413, 422) else "Embedding failed."
        raise HTTPException(status_code=status_code, detail=detail) from exc


# ==========================================
# Background Processing Endpoints (Stage 5)
# ==========================================


@documents_router.post(
    "/{document_id}/process",
    response_model=DocumentProcessingStatus,
    status_code=status.HTTP_200_OK,
)
async def process_document_endpoint(
    document_id: str,
    request: ProcessDocumentRequest = Body(default_factory=ProcessDocumentRequest),
) -> DocumentProcessingStatus:
    service = get_processing_service()
    try:
        return service.enqueue_processing(
            document_id_value=document_id,
            collection_name=request.collection_name,
            force_reprocess=request.force_reprocess,
        )
    except ProcessingDocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProcessingConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error enqueuing document processing: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to enqueue document processing.") from exc


@documents_router.get(
    "/{document_id}/status",
    response_model=DocumentProcessingStatus,
    status_code=status.HTTP_200_OK,
)
def get_document_status_endpoint(document_id: str) -> DocumentProcessingStatus:
    service = get_processing_service()
    try:
        return service.get_status(document_id)
    except ProcessingDocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error retrieving document status: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve document status.") from exc


@documents_router.post(
    "/{document_id}/cancel",
    response_model=DocumentProcessingStatus,
    status_code=status.HTTP_200_OK,
)
async def cancel_document_processing_endpoint(document_id: str) -> DocumentProcessingStatus:
    service = get_processing_service()
    try:
        return service.cancel_processing(document_id)
    except ProcessingDocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error cancelling document processing: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to cancel document processing.") from exc


# ==========================================
# Document Indexing Endpoint (Stage 4)
# ==========================================


@documents_router.post(
    "/{document_id}/index",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_200_OK,
)
def index_document_endpoint(
    document_id: str,
    request: IndexDocumentRequest = Body(default_factory=IndexDocumentRequest),
) -> IndexDocumentResponse:
    service = get_indexing_service()
    try:
        return service.index_document(
            document_id_value=document_id,
            extraction_id=str(request.extraction_id) if request.extraction_id else None,
            collection_name=request.collection_name,
        )
    except IndexDocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IndexingConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (IndexExtractionNotReadyError, ExtractionNotEmbeddedError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (InvalidArtifactError, CollectionIncompatibleError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except StorageUnavailableError as exc:
        logger.error("Vector storage is unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Vector storage is unavailable.") from exc
    except PartialIndexingError as exc:
        logger.error("Partial indexing detected: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except VectorStorageError as exc:
        logger.error("Vector storage error during indexing: %s", exc)
        raise HTTPException(status_code=502, detail="Vector storage operation failed.") from exc
    except Exception as exc:
        logger.exception("Unexpected error during document indexing: %s", exc)
        raise HTTPException(status_code=500, detail="Document indexing failed.") from exc


# ==========================================
# Semantic Retrieval Endpoint (Stage 4)
# ==========================================


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
)
async def search_endpoint(request: SearchRequest) -> SearchResponse:
    service = get_retrieval_service()
    try:
        return await service.search(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            document_id=request.document_id,
            document_ids=request.document_ids,
        )
    except RemoteCodeApprovalRequired as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelBusyError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "2"},
        ) from exc
    except InvalidQueryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmbeddingModelError as exc:
        logger.error("Embedding model error during search: %s", exc)
        raise HTTPException(status_code=502, detail="Embedding model inference failed.") from exc
    except StorageUnavailableError as exc:
        logger.error("Vector storage unavailable during search: %s", exc)
        raise HTTPException(status_code=503, detail="Vector storage is unavailable.") from exc
    except VectorStorageError as exc:
        logger.error("Vector storage error during search: %s", exc)
        raise HTTPException(status_code=502, detail="Vector retrieval failed.") from exc
    except Exception as exc:
        logger.exception("Unexpected error during semantic search: %s", exc)
        raise HTTPException(status_code=500, detail="Search failed.") from exc


# ==========================================
# RAG Chat Endpoint (Stage 6)
# ==========================================


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    chat_svc = get_chat_service()
    try:
        return await chat_svc.chat(request)
    except RemoteCodeApprovalRequired as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelBusyError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "2"},
        ) from exc
    except InvalidQueryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmbeddingModelError as exc:
        logger.error("Embedding model error during chat: %s", exc)
        raise HTTPException(status_code=502, detail="Embedding model inference failed.") from exc
    except StorageUnavailableError as exc:
        logger.error("Vector storage unavailable during chat: %s", exc)
        raise HTTPException(status_code=503, detail="Vector storage is unavailable.") from exc
    except VectorStorageError as exc:
        logger.error("Vector storage error during chat: %s", exc)
        raise HTTPException(status_code=502, detail="Vector retrieval failed.") from exc
    except Exception as exc:
        logger.exception("Unexpected error during RAG chat: %s", exc)
        raise HTTPException(status_code=500, detail="RAG chat generation failed.") from exc


# ==========================================
# Index Status Endpoint (Stage 4)
# ==========================================


@router.get(
    "/index/status",
    response_model=IndexStatusResponse,
    status_code=status.HTTP_200_OK,
)
def index_status_endpoint(
    collection_name: str | None = None,
) -> IndexStatusResponse:
    repo = get_vector_repository()
    try:
        info = repo.get_collection_info(collection_name=collection_name)
        return IndexStatusResponse(**info)
    except StorageUnavailableError as exc:
        logger.error("Vector storage unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Vector storage is unavailable.") from exc
    except Exception as exc:
        logger.exception("Failed to query index status: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to retrieve index status.") from exc


# Mount document sub-router
router.include_router(documents_router)
