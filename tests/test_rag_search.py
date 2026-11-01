from __future__ import annotations

import uuid
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client.models import Distance, PointStruct

from app.core.exceptions import ModelBusyError
from app.main import app
from app.rag.repositories.vector_repository import (
    VectorRepository,
)
from app.rag.services.retrieval_service import (
    RetrievalService,
    get_retrieval_service,
)


class MockJinaEmbedder:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [1.0, 0.0]

    def embed(self, texts: list[str], batch_size: int = 1) -> np.ndarray:
        return np.tile(np.asarray([self.vector], dtype=np.float32), (len(texts), 1))


def test_search_validation_rejects_invalid_inputs() -> None:
    client = TestClient(app)

    # Empty query
    assert client.post("/rag/search", json={"query": ""}).status_code == 422

    # Pure whitespace query
    assert client.post("/rag/search", json={"query": "   \n\t  "}).status_code == 422

    # Query too long (> 2000 chars)
    long_query = "x" * 2001
    assert client.post("/rag/search", json={"query": long_query}).status_code == 422

    # top_k < 1 or > 100
    assert client.post("/rag/search", json={"query": "hello", "top_k": 0}).status_code == 422
    assert client.post("/rag/search", json={"query": "hello", "top_k": 101}).status_code == 422

    # score_threshold < 0 or > 1
    assert client.post("/rag/search", json={"query": "hello", "score_threshold": -0.1}).status_code == 422
    assert client.post("/rag/search", json={"query": "hello", "score_threshold": 1.1}).status_code == 422

    # Invalid document_id format
    assert client.post("/rag/search", json={"query": "hello", "document_id": "not-a-uuid"}).status_code == 422


def test_search_remote_code_gate_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ALLOW_JINA_REMOTE_CODE", "false")
    # Point repo to a collection that exists
    repo = VectorRepository(mode="memory", collection_name="knowledgebase")
    repo.create_collection_if_not_exists("knowledgebase", dimension=2)
    service = RetrievalService(repository=repo)
    monkeypatch.setattr("app.rag.router.get_retrieval_service", lambda: service)

    client = TestClient(app)
    response = client.post("/rag/search", json={"query": "test query"})
    assert response.status_code == 503
    assert "remote-code" in response.json()["detail"].lower()


def test_search_non_existent_collection_returns_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ALLOW_JINA_REMOTE_CODE", "true")
    empty_repo = VectorRepository(mode="memory", collection_name="does_not_exist")
    service = RetrievalService(repository=empty_repo)
    monkeypatch.setattr("app.rag.router.get_retrieval_service", lambda: service)

    client = TestClient(app)
    response = client.post("/rag/search", json={"query": "test query"})
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == []
    assert data["total_results"] == 0


def test_search_successful_retrieval_and_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ALLOW_JINA_REMOTE_CODE", "true")

    # Setup memory repo with 2 documents
    repo = VectorRepository(mode="memory", collection_name="test_search_col")
    repo.create_collection_if_not_exists("test_search_col", dimension=2, distance=Distance.COSINE)

    doc_a = str(uuid4())
    doc_b = str(uuid4())

    # Point 1: doc A, aligned with [1.0, 0.0] -> score 1.0
    p1 = PointStruct(
        id=str(uuid4()),
        vector=[1.0, 0.0],
        payload={
            "document_id": doc_a,
            "page_number": 2,
            "chunk_id": "chunk-a-1",
            "cleaned_text": "Content about artificial intelligence in doc A.",
        },
    )
    # Point 2: doc A, slightly off [0.8, 0.6] -> score 0.8
    p2 = PointStruct(
        id=str(uuid4()),
        vector=[0.8, 0.6],
        payload={
            "document_id": doc_a,
            "page_number": 3,
            "chunk_id": "chunk-a-2",
            "cleaned_text": "Further notes on machine learning in doc A.",
        },
    )
    # Point 3: doc B, aligned with [1.0, 0.0] -> score 1.0
    p3 = PointStruct(
        id=str(uuid4()),
        vector=[1.0, 0.0],
        payload={
            "document_id": doc_b,
            "page_number": 1,
            "chunk_id": "chunk-b-1",
            "cleaned_text": "Content from doc B.",
        },
    )

    repo.upsert_batch([p1, p2, p3], collection_name="test_search_col")

    # Mock embedder that returns [1.0, 0.0]
    embedder = MockJinaEmbedder([1.0, 0.0])
    service = RetrievalService(repository=repo, embedder_provider=lambda: embedder)
    service.repository.default_collection = "test_search_col"
    monkeypatch.setattr("app.rag.router.get_retrieval_service", lambda: service)

    client = TestClient(app)

    # Search without filter, score_threshold 0.7
    resp = client.post("/rag/search", json={"query": "artificial intelligence", "top_k": 5, "score_threshold": 0.7})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["query"] == "artificial intelligence"
    assert data["total_results"] == 3
    # Check descending order of scores
    scores = [item["score"] for item in data["results"]]
    assert scores == sorted(scores, reverse=True)
    # Verify no raw vectors in response
    for item in data["results"]:
        assert "vector" not in item
        assert "document_id" in item
        assert "page_number" in item
        assert "text" in item

    # Search with score_threshold 0.9 (should exclude p2)
    resp_high_thresh = client.post(
        "/rag/search",
        json={"query": "artificial intelligence", "top_k": 5, "score_threshold": 0.9},
    )
    assert resp_high_thresh.status_code == 200
    assert resp_high_thresh.json()["total_results"] == 2

    # Search filtered by doc_a (should exclude p3)
    resp_filtered = client.post(
        "/rag/search",
        json={"query": "artificial intelligence", "top_k": 5, "score_threshold": 0.7, "document_id": doc_a},
    )
    assert resp_filtered.status_code == 200
    filtered_data = resp_filtered.json()
    assert filtered_data["total_results"] == 2
    for item in filtered_data["results"]:
        assert item["document_id"] == doc_a


def test_index_status_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = VectorRepository(mode="memory", collection_name="status_col")
    repo.create_collection_if_not_exists("status_col", dimension=768, distance=Distance.COSINE)
    repo.upsert_batch(
        [PointStruct(id=str(uuid4()), vector=[0.1] * 768, payload={"document_id": "test"})],
        collection_name="status_col",
    )
    monkeypatch.setattr("app.rag.router.get_vector_repository", lambda: repo)

    client = TestClient(app)
    response = client.get("/rag/index/status", params={"collection_name": "status_col"})
    assert response.status_code == 200
    data = response.json()
    assert data["collection_name"] == "status_col"
    assert data["collection_exists"] is True
    assert data["vector_dimension"] == 768
    assert data["distance"] == "Cosine"
    assert data["point_count"] == 1
    assert data["status"] == "ready"
