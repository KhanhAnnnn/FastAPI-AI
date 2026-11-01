from __future__ import annotations

import logging
import os
import re
import threading
import time
import unicodedata
from uuid import UUID

import numpy as np

from app.config import (
    MAX_PROMPT_LENGTH,
    QDRANT_COLLECTION,
    RAG_RETRIEVAL_TOP_K,
    RAG_SCORE_THRESHOLD,
)
from app.core.exceptions import ModelBusyError
from app.core.model_manager import model_manager
from app.rag.embedding_model import (
    EmbeddingModelError,
    JinaEmbedder,
    RemoteCodeApprovalRequired,
    get_embedder,
)
from app.rag.repositories.vector_repository import (
    StorageUnavailableError,
    VectorRepository,
    VectorStorageError,
    get_vector_repository,
)
from app.rag.search_schemas import SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)

STOP_WORDS: set[str] = {
    # Vietnamese common particles & functional stop words
    "là", "và", "có", "trong", "của", "các", "những", "được", "cho", "với", "về",
    "khi", "này", "đó", "thì", "ở", "tại", "ra", "lại", "đã", "sẽ", "đang", "từ",
    "gì", "nào", "sao", "bao", "nhiêu", "ai", "đâu", "hay", "hoặc", "nếu",
    "một", "như", "theo", "đến", "bởi", "do", "vì", "để", "làm", "rồi", "vào",
    "cũng", "đều", "chỉ", "qua", "trên", "dưới", "sau", "trước",
    # English common particles & stop words
    "the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "for", "of",
    "with", "as", "by", "or", "so", "be", "this", "that", "it", "from", "are",
    "was", "were", "what", "how", "when", "where", "who", "why", "can", "could",
    "would", "should", "do", "does", "did", "have", "has", "had",
}


def extract_query_entities(query: str) -> list[str]:
    """Extract query entities, numbers, dates, mixed-case words, and capitalized terms without domain bias."""
    q_norm = unicodedata.normalize("NFC", query)
    tokens = re.findall(r"\b[\w\./\-]+\b", q_norm)
    entities: list[str] = []
    for i, tok in enumerate(tokens):
        # Digits or dates (e.g. 2025, 292, 18/11/2025, 300)
        if re.search(r"\d", tok):
            entities.append(tok.lower())
        # CamelCase / MixedCase (e.g. NovaDesk, FastAPI, PostgreSQL, Qdrant)
        elif any(c.isupper() for c in tok[1:]):
            entities.append(tok.lower())
        # Acronyms (e.g. AI, PDF, API, SQL)
        elif tok.isupper() and len(tok) >= 2:
            entities.append(tok.lower())
        # Capitalized proper nouns not at sentence start
        elif tok[0].isupper() and i > 0 and tok.lower() not in STOP_WORDS and len(tok) > 2:
            entities.append(tok.lower())
    return list(dict.fromkeys(entities))


def compute_lexical_overlap(query: str, text: str) -> float:
    """Calculate normalized semantic relevance using entity coverage, rare-term weights, and phrases.
    
    Guarantees:
    - Bounded in [0.0, 1.0].
    - No position-dependent penalty or bias against sections lacking arbitrary prefixes.
    - Preserves multi-entity and entity-role relationship scoring.
    """
    q_norm = unicodedata.normalize("NFC", query).lower()
    t_norm = unicodedata.normalize("NFC", text).lower()

    words = re.findall(r"\w+", q_norm)
    key_words = [w for w in words if w not in STOP_WORDS and (len(w) > 1 or w.isdigit())]
    if not key_words:
        return 0.5

    unique_keys = list(dict.fromkeys(key_words))
    entities = extract_query_entities(query)

    # Specificity weighting: query entities and digits carry higher informational value
    weights: dict[str, float] = {}
    for w in unique_keys:
        if w in entities:
            weights[w] = 3.0
        elif w.isdigit():
            weights[w] = 2.5
        elif len(w) >= 6:
            weights[w] = 1.5
        else:
            weights[w] = 1.0

    total_weight = sum(weights.values())
    matched_weight = sum(w_val for w, w_val in weights.items() if w in t_norm)
    weighted_unigram = matched_weight / total_weight if total_weight > 0 else 0.0

    # Bigram phrase matching across adjacent content words
    bigrams = [
        f"{words[i]} {words[i+1]}"
        for i in range(len(words) - 1)
        if words[i] not in STOP_WORDS and words[i+1] not in STOP_WORDS
    ]
    bigram_ratio = 0.0
    if bigrams:
        bg_matches = sum(1 for bg in set(bigrams) if bg in t_norm)
        bigram_ratio = bg_matches / len(set(bigrams))

    # Entity / coverage ratio
    if entities:
        ent_matches = sum(1 for e in entities if e in t_norm)
        coverage_ratio = ent_matches / len(entities)
    else:
        coverage_ratio = sum(1 for w in unique_keys if w in t_norm) / len(unique_keys)

    score = 0.40 * weighted_unigram + 0.35 * bigram_ratio + 0.25 * coverage_ratio
    return max(0.0, min(1.0, round(score, 4)))


class RetrievalError(Exception):
    """Base exception for semantic retrieval operations."""


class InvalidQueryError(RetrievalError):
    """Raised when the search query is invalid."""


class RetrievalService:
    """Service handling semantic retrieval via real Jina embeddings and Qdrant search."""

    def __init__(
        self,
        repository: VectorRepository | None = None,
        embedder_provider: Any = None,
    ) -> None:
        self._repo = repository
        self._embedder_provider = embedder_provider or get_embedder
        self.last_retrieval_latency: float | None = None

    @property
    def repository(self) -> VectorRepository:
        if self._repo is not None:
            return self._repo
        return get_vector_repository()

    async def search(
        self,
        query: str,
        top_k: int = RAG_RETRIEVAL_TOP_K,
        score_threshold: float = RAG_SCORE_THRESHOLD,
        document_id: str | UUID | None = None,
        document_ids: list[str | UUID] | None = None,
        collection_name: str | None = None,
    ) -> SearchResponse:
        """Execute semantic search against indexed document chunks."""
        # Step 1 & 2: Validate prompt
        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError("Query string must not be empty.")
        clean_query = query.strip()
        if len(clean_query) > MAX_PROMPT_LENGTH:
            raise InvalidQueryError(f"Query exceeds maximum allowed length of {MAX_PROMPT_LENGTH} characters.")

        if top_k < 1 or top_k > 100:
            raise InvalidQueryError("top_k must be between 1 and 100.")
        if score_threshold < 0.0 or score_threshold > 1.0:
            raise InvalidQueryError("score_threshold must be between 0.0 and 1.0.")

        doc_filter_str: str | None = None
        if document_id is not None:
            try:
                doc_filter_str = str(UUID(str(document_id)))
            except (ValueError, TypeError) as exc:
                raise InvalidQueryError(f"Invalid document_id '{document_id}': must be a valid UUID.") from exc

        doc_ids_filter: list[str] | None = None
        if document_ids is not None:
            doc_ids_filter = []
            for d in document_ids:
                try:
                    doc_ids_filter.append(str(UUID(str(d))))
                except (ValueError, TypeError) as exc:
                    raise InvalidQueryError(f"Invalid document_id in document_ids list '{d}': must be a valid UUID.") from exc

        # Step 3, 4, 11: Generate real embedding using pinned Jina model on CPU
        # Remote code gate check
        if os.getenv("RAG_ALLOW_JINA_REMOTE_CODE", "false").lower() != "true":
            raise RemoteCodeApprovalRequired(
                "Jina model code execution is disabled. Review and explicitly approve "
                "the pinned Jina remote-code revisions before enabling "
                "RAG_ALLOW_JINA_REMOTE_CODE=true."
            )

        target_collection = collection_name or self.repository.default_collection

        # If collection does not exist, return empty results immediately
        if not self.repository.collection_exists(target_collection):
            return SearchResponse(
                query=clean_query,
                results=[],
                total_results=0,
            )

        started = time.perf_counter()

        def _compute_query_vector() -> np.ndarray:
            embedder: JinaEmbedder = self._embedder_provider()
            vectors = embedder.embed([clean_query], batch_size=1)
            return vectors[0]

        # Use exclusive execution so embedding does not conflict with generative models
        try:
            query_vector = await model_manager.run_exclusive(
                "rag_retrieval", _compute_query_vector
            )
        except (RemoteCodeApprovalRequired, ModelBusyError, EmbeddingModelError):
            raise
        except Exception as exc:
            logger.error("Failed to generate query embedding: %s", exc, exc_info=True)
            raise EmbeddingModelError(f"Query embedding generation failed: {exc}") from exc

        # Step 5, 6, 7, 8: Call Qdrant similarity search with candidate pool limit
        candidate_limit = min(100, max(top_k * 5, 25))
        try:
            points = self.repository.search_vectors(
                collection_name=target_collection,
                query_vector=query_vector,
                limit=candidate_limit,
                score_threshold=score_threshold,
                document_id=doc_filter_str,
                document_ids=doc_ids_filter,
            )
        except (VectorStorageError, StorageUnavailableError):
            raise
        except Exception as exc:
            logger.error("Similarity search failed: %s", exc, exc_info=True)
            raise VectorStorageError(f"Similarity search failed: {exc}") from exc

        # Step 9: Hybrid reranking combining cosine similarity and lexical keyword overlap
        scored_candidates = []
        for pt in points:
            payload = pt.payload or {}
            text_content = (
                payload.get("cleaned_text")
                or payload.get("original_text")
                or payload.get("text")
                or ""
            )
            cos_score = float(pt.score)
            lex_overlap = compute_lexical_overlap(clean_query, text_content)
            hybrid_score = 0.5 * cos_score + 0.5 * lex_overlap
            scored_candidates.append((hybrid_score, cos_score, pt, text_content))

        # Sort primarily by hybrid_score descending, secondarily by cos_score descending
        scored_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = scored_candidates[:top_k]

        # Step 10: Record retrieval latency
        elapsed = time.perf_counter() - started
        self.last_retrieval_latency = elapsed
        logger.info(
            "Semantic search executed in %.4fs: query='%s', top_k=%d, score_threshold=%.2f, candidates=%d, returned=%d",
            elapsed,
            clean_query[:50],
            top_k,
            score_threshold,
            len(points),
            len(selected),
        )

        # Format results without raw vectors
        results: list[SearchResultItem] = []
        for hybrid_score, cos_score, pt, text_content in selected:
            payload = pt.payload or {}
            raw_doc_id = payload.get("document_id")
            try:
                item_doc_uuid = UUID(str(raw_doc_id))
            except (ValueError, TypeError):
                item_doc_uuid = UUID("00000000-0000-0000-0000-000000000000")

            results.append(
                SearchResultItem(
                    document_id=item_doc_uuid,
                    page_number=int(payload.get("page_number", 1)),
                    chunk_id=str(payload.get("chunk_id", str(pt.id))),
                    score=round(hybrid_score, 4),
                    text=str(text_content),
                )
            )

        # Step 11: Pure retrieval only; no TinyLlama generation
        return SearchResponse(
            query=clean_query,
            results=results,
            total_results=len(results),
        )


_global_retrieval_service: RetrievalService | None = None
_retrieval_lock = threading.Lock()


def get_retrieval_service() -> RetrievalService:
    global _global_retrieval_service
    if _global_retrieval_service is not None:
        return _global_retrieval_service
    with _retrieval_lock:
        if _global_retrieval_service is None:
            _global_retrieval_service = RetrievalService()
    return _global_retrieval_service
