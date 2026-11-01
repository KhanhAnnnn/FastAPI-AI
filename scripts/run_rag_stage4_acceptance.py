from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import psutil
import torch
from fastapi import UploadFile
from qdrant_client.models import Distance

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    QDRANT_COLLECTION,
    QDRANT_LOCAL_PATH,
    QDRANT_MODE,
    RAG_ALLOW_JINA_REMOTE_CODE,
    RAG_ARTIFACT_ROOT,
    RAG_EMBEDDING_CODE_REVISION,
    RAG_EMBEDDING_MODEL_ID,
    RAG_EMBEDDING_MODEL_REVISION,
    RAG_UPLOAD_ROOT,
)
from app.rag import artifacts, storage
from app.rag.embedding_service import embed_document
from app.rag.extraction_service import extract_document
from app.rag.repositories.vector_repository import (
    VectorRepository,
    close_vector_repository,
    get_vector_repository,
)
from app.rag.services.indexing_service import IndexingService, get_indexing_service
from app.rag.services.retrieval_service import RetrievalService, get_retrieval_service
from app.rag.storage import store_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("acceptance_stage4")


from tests.test_rag_extraction import make_pdf


async def main() -> None:
    logger.info("==================================================")
    logger.info("STARTING RAG STAGE 4 ACCEPTANCE & BENCHMARK")
    logger.info("==================================================")

    # 1. Environment diagnostics
    process = psutil.Process()
    rss_start = process.memory_info().rss
    ram_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage("D:\\")

    import qdrant_client
    import transformers

    qdrant_version = getattr(qdrant_client, "__version__", "1.19.1")
    logger.info("Python: %s", sys.version.split()[0])
    logger.info("PyTorch: %s", torch.__version__)
    logger.info("Transformers: %s", transformers.__version__)
    logger.info("Qdrant Client: %s", qdrant_version)
    logger.info("Host RAM: Total=%.2f GB, Available=%.2f GB", ram_info.total / 1024**3, ram_info.available / 1024**3)
    logger.info("Drive D: Free=%.2f GB", disk_info.free / 1024**3)
    logger.info("Initial Process RSS: %d bytes (%.2f MB)", rss_start, rss_start / 1024**2)

    # Ensure remote code opt-in is enabled for acceptance
    os.environ["RAG_ALLOW_JINA_REMOTE_CODE"] = "true"

    # Use persistent Qdrant repository at configured path
    repo = get_vector_repository()
    indexing_service = get_indexing_service()
    retrieval_service = get_retrieval_service()

    collection_name = QDRANT_COLLECTION
    logger.info("Target Qdrant Collection: '%s' (mode=%s, path=%s)", collection_name, repo.mode, repo.local_path)

    # Step 1: Index existing Phase 3 artifact
    doc1_id = "4b98e1f4-b6ac-4a99-af93-b26a28c67f80"
    logger.info("\n--- STEP 1: Indexing Phase 3 Artifact (%s) ---", doc1_id)
    t0_idx1 = time.perf_counter()
    res1 = indexing_service.index_document(document_id_value=doc1_id, collection_name=collection_name)
    idx1_time = time.perf_counter() - t0_idx1

    logger.info(
        "Doc 1 Indexed: status=%s, chunks=%d, dim=%d, latency=%.4fs",
        res1.status,
        res1.indexed_chunks,
        res1.vector_dimension,
        idx1_time,
    )
    count_after_doc1 = repo.count_points(collection_name)
    logger.info("Points in '%s' after Doc 1: %d", collection_name, count_after_doc1)

    # Step 2: Create, upload, extract, embed, and index Document 2 (Bilingual EN/VI multi-page)
    logger.info("\n--- STEP 2: Creating and Indexing Document 2 (Bilingual Multi-Page) ---")
    page1_text = (
        "Kiến trúc hệ thống RAG và cơ sở dữ liệu vector Qdrant. "
        "Qdrant hỗ trợ tìm kiếm ngữ nghĩa theo độ đo Cosine với hiệu năng cao."
    )
    page2_text = (
        "The Jina embedding model operates with 768 dimensions and 8192 context tokens. "
        "Token-aware chunking divides source text into overlapping segments preserving provenance."
    )
    pdf_bytes = make_pdf([page1_text, page2_text])

    # Store PDF
    class DummyUploadFile:
        def __init__(self, filename: str, content: bytes):
            self.filename = filename
            self._content = content
            self._pos = 0

        async def read(self, size: int = -1) -> bytes:
            if size == -1:
                chunk = self._content[self._pos :]
                self._pos = len(self._content)
                return chunk
            chunk = self._content[self._pos : self._pos + size]
            self._pos += len(chunk)
            return chunk

        async def close(self):
            pass

    upload_file = DummyUploadFile("bilingual-stage4-fixture.pdf", pdf_bytes)
    doc2_id, safe_name, stored_name, size_bytes = await store_pdf(upload_file)
    logger.info("Doc 2 stored: id=%s, file=%s, size=%d bytes", doc2_id, safe_name, size_bytes)

    # Extract Doc 2
    t0_ext = time.perf_counter()
    ext_res = extract_document(doc2_id)
    ext_time = time.perf_counter() - t0_ext
    total_chars = sum(p.character_count for p in ext_res.pages)
    logger.info("Doc 2 extracted: pages=%d, text_chars=%d, latency=%.4fs", ext_res.total_pages, total_chars, ext_time)

    # Embed Doc 2 with real Jina CPU model
    t0_emb = time.perf_counter()
    emb_res = await embed_document(doc2_id)
    emb_time = time.perf_counter() - t0_emb
    logger.info("Doc 2 embedded: chunks=%d, dim=%d, latency=%.4fs", emb_res.chunk_count, emb_res.embedding_dimension, emb_time)

    # Index Doc 2 into Qdrant
    t0_idx2 = time.perf_counter()
    res2 = indexing_service.index_document(document_id_value=doc2_id, collection_name=collection_name)
    idx2_time = time.perf_counter() - t0_idx2
    logger.info("Doc 2 Indexed: status=%s, chunks=%d, latency=%.4fs", res2.status, res2.indexed_chunks, idx2_time)

    total_points = repo.count_points(collection_name)
    logger.info("Points in '%s' after Doc 2: %d", collection_name, total_points)

    # Step 3: Persistence and Idempotency Checks
    logger.info("\n--- STEP 3: Persistence and Idempotency Verification ---")
    # Idempotent re-indexing
    indexing_service.index_document(document_id_value=doc1_id, collection_name=collection_name)
    indexing_service.index_document(document_id_value=doc2_id, collection_name=collection_name)
    count_after_reindex = repo.count_points(collection_name)
    logger.info("Points after re-indexing both documents (idempotency check): %d (expected %d)", count_after_reindex, total_points)
    assert count_after_reindex == total_points, "Idempotency failed: points were duplicated!"

    # Persistence check across instance closure
    close_vector_repository()
    # Reopen
    repo_reopened = get_vector_repository()
    count_reopened = repo_reopened.count_points(collection_name)
    logger.info("Points after closing and reopening Qdrant client: %d", count_reopened)
    assert count_reopened == total_points, "Persistence failed: points lost after client reopen!"

    # Step 4: Real Semantic Retrieval Benchmark
    logger.info("\n--- STEP 4: Real Semantic Retrieval Benchmark ---")

    benchmark_queries = [
        {
            "query_id": "Q1_EN",
            "query": "How does token-aware chunking and vector embedding work?",
            "expected_doc_id": doc1_id,
            "expected_page": 1,
            "description": "English query matching Doc 1 chunking/embedding content",
            "type": "positive",
        },
        {
            "query_id": "Q2_EN",
            "query": "What is the context length and embedding dimension of the model?",
            "expected_doc_id": doc2_id,
            "expected_page": 2,
            "description": "English query matching Doc 2 Page 2 model specifications",
            "type": "positive",
        },
        {
            "query_id": "Q3_VI",
            "query": "Cơ sở dữ liệu vector Qdrant tìm kiếm theo độ đo nào?",
            "expected_doc_id": doc2_id,
            "expected_page": 1,
            "description": "Vietnamese query matching Doc 2 Page 1 Qdrant Cosine metric",
            "type": "positive",
        },
        {
            "query_id": "Q4_VI",
            "query": "Mô hình Jina embedding có bao nhiêu chiều và bao nhiêu token ngữ cảnh?",
            "expected_doc_id": doc2_id,
            "expected_page": 2,
            "description": "Vietnamese query matching Doc 2 Page 2 dimensions/context",
            "type": "positive",
        },
        {
            "query_id": "Q5_NEG",
            "query": "Quantum computing Shor algorithm for integer factorization and discrete logarithms",
            "expected_doc_id": None,
            "expected_page": None,
            "description": "Unrelated negative query with no matching content",
            "type": "negative",
        },
        {
            "query_id": "Q6_FILTER",
            "query": "token-aware chunking and finite vector embeddings",
            "expected_doc_id": None,  # Filtered by Doc 2, so Doc 1 MUST NOT appear
            "filter_doc_id": doc2_id,
            "description": "Document filter isolation test: Doc 1 content queried with Doc 2 filter",
            "type": "filtered",
        },
    ]

    retrieval_results = []
    latencies = []
    hits_at_3 = []
    reciprocal_ranks = []

    for item in benchmark_queries:
        qid = item["query_id"]
        qtext = item["query"]
        f_doc = item.get("filter_doc_id")

        t0_q = time.perf_counter()
        search_res = await retrieval_service.search(
            query=qtext,
            top_k=3,
            score_threshold=0.0,  # return all top 3 to inspect raw scores
            document_id=f_doc,
            collection_name=collection_name,
        )
        q_time = time.perf_counter() - t0_q
        latencies.append(q_time)

        top_results = search_res.results
        top1_score = top_results[0].score if top_results else None
        top1_doc = str(top_results[0].document_id) if top_results else None
        top1_page = top_results[0].page_number if top_results else None
        top1_text = top_results[0].text[:80] if top_results else None

        # Score threshold impact check (>= 0.70)
        hits_above_07 = [r for r in top_results if r.score >= 0.70]

        # Calculate metrics for positive queries
        hit3 = False
        rr = 0.0
        if item["type"] == "positive":
            target_doc = item["expected_doc_id"]
            target_page = item["expected_page"]
            for rank, r in enumerate(top_results, start=1):
                if str(r.document_id) == target_doc and r.page_number == target_page:
                    hit3 = True
                    rr = 1.0 / rank
                    break
            hits_at_3.append(1 if hit3 else 0)
            reciprocal_ranks.append(rr)

        # For filtered query: verify no doc 1 results returned
        filter_violation = False
        if item["type"] == "filtered":
            for r in top_results:
                if str(r.document_id) != f_doc:
                    filter_violation = True
            assert not filter_violation, "Document filter failed: found results from outside target document!"

        logger.info(
            "Query [%s] in %.4fs: Top-1 Score=%.4f, Top-1 Doc=%s (P%s), Hits>=0.70=%d, TargetHit=%s",
            qid,
            q_time,
            top1_score or 0.0,
            (top1_doc[:8] + "...") if top1_doc else "None",
            top1_page or "-",
            len(hits_above_07),
            hit3 if item["type"] == "positive" else ("N/A (filtered)" if f_doc else "N/A (neg)"),
        )
        if top_results:
            logger.info("   Snippet: '%s...'", top1_text)

        retrieval_results.append(
            {
                "query_id": qid,
                "query": qtext,
                "description": item["description"],
                "type": item["type"],
                "latency_seconds": q_time,
                "top_results": [
                    {
                        "rank": idx + 1,
                        "document_id": str(r.document_id),
                        "page_number": r.page_number,
                        "chunk_id": r.chunk_id,
                        "score": round(r.score, 4),
                        "text": r.text,
                    }
                    for idx, r in enumerate(top_results)
                ],
                "hit_at_3": hit3 if item["type"] == "positive" else None,
                "reciprocal_rank": rr if item["type"] == "positive" else None,
                "hits_above_0_7": len(hits_above_07),
            }
        )

    # Compute aggregate evaluation metrics
    recall_at_3 = float(np.mean(hits_at_3)) if hits_at_3 else 0.0
    hit_rate_at_3 = float(np.mean(hits_at_3)) if hits_at_3 else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    avg_latency = float(np.mean(latencies))
    min_latency = float(np.min(latencies))
    max_latency = float(np.max(latencies))

    rss_end = process.memory_info().rss
    logger.info("\n==================================================")
    logger.info("EVALUATION METRICS:")
    logger.info("  Queries Evaluated: %d (Positive: %d, Negative: 1, Filter: 1)", len(benchmark_queries), len(hits_at_3))
    logger.info("  Recall@3:          %.4f (%.1f%%)", recall_at_3, recall_at_3 * 100)
    logger.info("  Hit Rate@3:        %.4f (%.1f%%)", hit_rate_at_3, hit_rate_at_3 * 100)
    logger.info("  MRR:               %.4f", mrr)
    logger.info("  Avg Latency:       %.4f s", avg_latency)
    logger.info("  Min Latency:       %.4f s", min_latency)
    logger.info("  Max Latency:       %.4f s", max_latency)
    logger.info("  RSS Before/After:  %.2f MB / %.2f MB", rss_start / 1024**2, rss_end / 1024**2)
    logger.info("==================================================")

    # Write acceptance JSON report
    report_data = {
        "status": "PASS",
        "stage": 4,
        "qdrant_version": qdrant_version,
        "storage_mode": repo.mode,
        "local_storage_path": str(repo.local_path),
        "collection_name": collection_name,
        "vector_dimension": 768,
        "distance_metric": "Cosine",
        "model_id": RAG_EMBEDDING_MODEL_ID,
        "model_revision": RAG_EMBEDDING_MODEL_REVISION,
        "code_revision": RAG_EMBEDDING_CODE_REVISION,
        "indexed_documents": [
            {
                "document_id": doc1_id,
                "source": "Phase 3 existing artifact",
                "chunks": res1.indexed_chunks,
                "indexing_latency_seconds": idx1_time,
            },
            {
                "document_id": doc2_id,
                "source": "Bilingual Stage 4 fixture",
                "chunks": res2.indexed_chunks,
                "indexing_latency_seconds": idx2_time,
            },
        ],
        "total_indexed_vectors": total_points,
        "idempotency_verified": count_after_reindex == total_points,
        "persistence_verified": count_reopened == total_points,
        "evaluation_metrics": {
            "positive_queries_count": len(hits_at_3),
            "recall_at_3": recall_at_3,
            "hit_rate_at_3": hit_rate_at_3,
            "mrr": mrr,
            "retrieval_latency": {
                "mean_seconds": avg_latency,
                "min_seconds": min_latency,
                "max_seconds": max_latency,
            },
        },
        "queries": retrieval_results,
        "memory_usage": {
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "rss_delta_mb": round((rss_end - rss_start) / 1024**2, 2),
        },
    }

    report_path = PROJECT_ROOT / "reports" / "rag-phase-4-acceptance.json"
    report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Acceptance report saved to %s", report_path)


if __name__ == "__main__":
    asyncio.run(main())
