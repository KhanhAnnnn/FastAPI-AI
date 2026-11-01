from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.rag.repositories.vector_repository import (
    CollectionConfigMismatchError,
    StorageUnavailableError,
    VectorRepository,
    VectorStorageError,
)
from app.rag.services.indexing_service import generate_point_id


def test_repository_memory_mode_lifecycle() -> None:
    repo = VectorRepository(mode="memory", collection_name="test_col")
    assert not repo.collection_exists("test_col")

    created = repo.create_collection_if_not_exists(
        collection_name="test_col",
        dimension=4,
        distance=Distance.COSINE,
        metadata={"model_id": "test_model", "model_revision": "rev1"},
    )
    assert created
    assert repo.collection_exists("test_col")

    # Creating again is idempotent and returns False
    created_again = repo.create_collection_if_not_exists(
        collection_name="test_col",
        dimension=4,
        distance=Distance.COSINE,
        metadata={"model_id": "test_model", "model_revision": "rev1"},
    )
    assert not created_again
    repo.close()


def test_repository_invalid_mode_raises() -> None:
    repo = VectorRepository(mode="unsupported_mode")
    with pytest.raises(StorageUnavailableError, match="Unsupported QDRANT_MODE"):
        repo.get_client()


def test_repository_dimension_mismatch_raises() -> None:
    repo = VectorRepository(mode="memory", collection_name="mismatch_col")
    repo.create_collection_if_not_exists("mismatch_col", dimension=4, distance=Distance.COSINE)

    with pytest.raises(CollectionConfigMismatchError, match="vector dimension mismatch"):
        repo.verify_collection_compatibility("mismatch_col", expected_dimension=8)
    repo.close()


def test_repository_distance_mismatch_raises() -> None:
    repo = VectorRepository(mode="memory", collection_name="dist_col")
    repo.create_collection_if_not_exists("dist_col", dimension=4, distance=Distance.DOT)

    with pytest.raises(CollectionConfigMismatchError, match="distance metric mismatch"):
        repo.verify_collection_compatibility("dist_col", expected_dimension=4, expected_distance="Cosine")
    repo.close()


def test_repository_model_metadata_mismatch_raises() -> None:
    repo = VectorRepository(mode="memory", collection_name="meta_col")
    repo.create_collection_if_not_exists(
        "meta_col",
        dimension=4,
        distance=Distance.COSINE,
        metadata={"model_id": "model_A", "model_revision": "rev1"},
    )

    with pytest.raises(CollectionConfigMismatchError, match="model mismatch"):
        repo.verify_collection_compatibility(
            "meta_col",
            expected_dimension=4,
            expected_model="model_B",
        )
    repo.close()


def test_generate_point_id_is_deterministic_valid_uuid() -> None:
    doc_id = str(uuid.uuid4())
    ext_id = str(uuid.uuid4())
    chunk_id = "chunk-42"

    id1 = generate_point_id(doc_id, ext_id, chunk_id)
    id2 = generate_point_id(doc_id, ext_id, chunk_id)

    assert id1 == id2
    # Verify standard UUID format
    parsed = uuid.UUID(id1)
    assert str(parsed) == id1

    different_chunk = generate_point_id(doc_id, ext_id, "chunk-43")
    assert different_chunk != id1


def test_repository_upsert_query_filter_and_count() -> None:
    repo = VectorRepository(mode="memory", collection_name="query_col")
    repo.create_collection_if_not_exists("query_col", dimension=2, distance=Distance.COSINE)

    doc_a = str(uuid.uuid4())
    doc_b = str(uuid.uuid4())

    p1 = PointStruct(
        id=str(uuid.uuid4()),
        vector=[1.0, 0.0],
        payload={"document_id": doc_a, "text": "Doc A vector", "page_number": 1},
    )
    p2 = PointStruct(
        id=str(uuid.uuid4()),
        vector=[0.0, 1.0],
        payload={"document_id": doc_b, "text": "Doc B vector", "page_number": 2},
    )

    repo.upsert_batch([p1, p2], collection_name="query_col")

    # Count total and filtered
    assert repo.count_points("query_col") == 2
    assert repo.count_points("query_col", document_id=doc_a) == 1
    assert repo.count_points("query_col", document_id=doc_b) == 1
    assert repo.count_points("query_col", document_id=str(uuid.uuid4())) == 0

    # Query with exact match for Doc A
    hits = repo.search_vectors([1.0, 0.0], limit=5, collection_name="query_col")
    assert len(hits) == 2
    assert hits[0].payload["document_id"] == doc_a
    assert hits[0].score > 0.99

    # Query with document filter
    filtered_hits = repo.search_vectors(
        [1.0, 0.0], limit=5, document_id=doc_b, collection_name="query_col"
    )
    assert len(filtered_hits) == 1
    assert filtered_hits[0].payload["document_id"] == doc_b

    # Score threshold filtering
    high_threshold_hits = repo.search_vectors(
        [1.0, 0.0], limit=5, score_threshold=0.5, collection_name="query_col"
    )
    assert len(high_threshold_hits) == 1
    assert high_threshold_hits[0].payload["document_id"] == doc_a

    # Verification of points existence
    assert repo.verify_points_exist([str(p1.id), str(p2.id)], "query_col")
    assert not repo.verify_points_exist([str(p1.id), str(uuid.uuid4())], "query_col")

    repo.close()


def test_repository_get_collection_info() -> None:
    repo = VectorRepository(mode="memory", collection_name="info_col")

    # Before creation
    uninit_info = repo.get_collection_info("info_col")
    assert not uninit_info["collection_exists"]
    assert uninit_info["status"] == "not_initialized"

    repo.create_collection_if_not_exists("info_col", dimension=4, distance=Distance.COSINE)
    empty_info = repo.get_collection_info("info_col")
    assert empty_info["collection_exists"]
    assert empty_info["vector_dimension"] == 4
    assert empty_info["point_count"] == 0
    assert empty_info["status"] == "empty"

    repo.upsert_batch(
        [PointStruct(id=str(uuid.uuid4()), vector=[0.1, 0.2, 0.3, 0.4], payload={"document_id": "test"})],
        collection_name="info_col",
    )
    ready_info = repo.get_collection_info("info_col")
    assert ready_info["point_count"] == 1
    assert ready_info["status"] == "ready"
    repo.close()
