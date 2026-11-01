from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from qdrant_client.models import Distance, PointStruct

from app.rag import artifacts, storage
from app.rag.repositories.vector_repository import (
    CollectionConfigMismatchError,
    VectorRepository,
)
from app.rag.services.indexing_service import (
    CollectionIncompatibleError,
    IndexingService,
    PartialIndexingError,
)
from app.rag.services.retrieval_service import RetrievalService
from tests.test_rag_indexing import _create_mock_artifact


def test_qdrant_local_persistent_mode_lifecycle_and_reopen(tmp_path: Path) -> None:
    """Test real Qdrant local persistent mode, saving vectors to disk, closing, and reopening."""
    db_path = tmp_path / "qdrant_data"

    # Step 1: Open repository and create collection
    repo1 = VectorRepository(mode="local", collection_name="persistent_kb", local_path=db_path)
    created = repo1.create_collection_if_not_exists("persistent_kb", dimension=4, distance=Distance.COSINE)
    assert created
    assert repo1.collection_exists("persistent_kb")

    # Step 2: Upsert real vectors
    p1 = PointStruct(
        id=str(uuid4()),
        vector=[0.5, 0.5, 0.5, 0.5],
        payload={"document_id": "doc1", "text": "persistent text"},
    )
    repo1.upsert_batch([p1], collection_name="persistent_kb")
    assert repo1.count_points("persistent_kb") == 1

    # Close repo1 to release lock
    repo1.close()

    # Step 3: Reopen repository on same directory
    repo2 = VectorRepository(mode="local", collection_name="persistent_kb", local_path=db_path)
    assert repo2.collection_exists("persistent_kb")
    assert repo2.count_points("persistent_kb") == 1

    # Query reopened database
    hits = repo2.search_vectors([0.5, 0.5, 0.5, 0.5], limit=2, collection_name="persistent_kb")
    assert len(hits) == 1
    assert hits[0].payload["document_id"] == "doc1"
    assert hits[0].payload["text"] == "persistent text"

    repo2.close()


def test_indexing_and_retrieval_integration_across_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test full integration: create artifacts for 2 documents, index them, verify isolation and persistence."""
    root = tmp_path / "rag"
    artifact_root = root / "artifacts"
    db_path = tmp_path / "qdrant_isolated"
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", root)
    monkeypatch.setattr(artifacts, "RAG_ARTIFACT_ROOT", artifact_root)

    repo = VectorRepository(mode="local", collection_name="integration_kb", local_path=db_path)
    indexing_service = IndexingService(repository=repo)

    doc_a = str(uuid4())
    doc_b = str(uuid4())

    _create_mock_artifact(artifact_root, doc_a, str(uuid4()), str(uuid4()), dimension=768, chunk_count=2)
    _create_mock_artifact(artifact_root, doc_b, str(uuid4()), str(uuid4()), dimension=768, chunk_count=3)

    # Index doc A
    res_a = indexing_service.index_document(doc_a, collection_name="integration_kb")
    assert res_a.status_code if hasattr(res_a, "status_code") else res_a.status == "completed"
    assert res_a.indexed_chunks == 2
    assert repo.count_points("integration_kb") == 2
    assert repo.count_points("integration_kb", document_id=doc_a) == 2

    # Re-index doc A: Idempotency check (count must stay 2)
    res_a_repeat = indexing_service.index_document(doc_a, collection_name="integration_kb")
    assert res_a_repeat.indexed_chunks == 2
    assert repo.count_points("integration_kb") == 2

    # Index doc B: Data from doc A must not be lost
    res_b = indexing_service.index_document(doc_b, collection_name="integration_kb")
    assert res_b.indexed_chunks == 3
    assert repo.count_points("integration_kb") == 5  # 2 from doc A + 3 from doc B
    assert repo.count_points("integration_kb", document_id=doc_a) == 2
    assert repo.count_points("integration_kb", document_id=doc_b) == 3

    # Retrieval isolation test: Filter by doc_a must not return doc_b chunks
    mock_vec = [1.0] + [0.0] * 767
    hits_a = repo.search_vectors(mock_vec, limit=10, document_id=doc_a, collection_name="integration_kb")
    assert len(hits_a) == 2
    for h in hits_a:
        assert h.payload["document_id"] == doc_a

    hits_b = repo.search_vectors(mock_vec, limit=10, document_id=doc_b, collection_name="integration_kb")
    assert len(hits_b) == 3
    for h in hits_b:
        assert h.payload["document_id"] == doc_b

    repo.close()


def test_incompatible_collection_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that indexing into a pre-existing collection with wrong dimension fails safely."""
    root = tmp_path / "rag"
    artifact_root = root / "artifacts"
    db_path = tmp_path / "qdrant_incompat"
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", root)
    monkeypatch.setattr(artifacts, "RAG_ARTIFACT_ROOT", artifact_root)

    repo = VectorRepository(mode="local", collection_name="incompat_kb", local_path=db_path)
    # Pre-create collection with 512 dimensions (artifact has 768)
    repo.create_collection_if_not_exists("incompat_kb", dimension=512, distance=Distance.COSINE)

    indexing_service = IndexingService(repository=repo)
    doc_id = str(uuid4())
    _create_mock_artifact(artifact_root, doc_id, str(uuid4()), str(uuid4()), dimension=768, chunk_count=1)

    with pytest.raises(CollectionIncompatibleError, match="dimension mismatch"):
        indexing_service.index_document(doc_id, collection_name="incompat_kb")

    # Verify that existing collection was NOT dropped or deleted
    assert repo.collection_exists("incompat_kb")
    info = repo.get_collection_info("incompat_kb")
    assert info["vector_dimension"] == 512

    repo.close()


def test_partial_indexing_verification_and_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test partial indexing detection when verification fails, and idempotent recovery on retry."""
    root = tmp_path / "rag"
    artifact_root = root / "artifacts"
    db_path = tmp_path / "qdrant_partial"
    monkeypatch.setattr(storage, "RAG_UPLOAD_ROOT", root)
    monkeypatch.setattr(artifacts, "RAG_ARTIFACT_ROOT", artifact_root)

    repo = VectorRepository(mode="local", collection_name="partial_kb", local_path=db_path)
    doc_id = str(uuid4())
    _create_mock_artifact(artifact_root, doc_id, str(uuid4()), str(uuid4()), dimension=768, chunk_count=4)

    indexing_service = IndexingService(repository=repo)

    # Simulate a partial verification failure on first run
    original_verify = repo.verify_points_exist
    fail_once = True

    def mock_verify_points_exist(point_ids, collection_name=None):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            return False  # Verification failure
        return original_verify(point_ids, collection_name)

    monkeypatch.setattr(repo, "verify_points_exist", mock_verify_points_exist)

    with pytest.raises(PartialIndexingError, match="Partial indexing detected"):
        indexing_service.index_document(doc_id, collection_name="partial_kb")

    # Retry indexing (recovery): succeeds cleanly
    result = indexing_service.index_document(doc_id, collection_name="partial_kb")
    assert result.status == "completed"
    assert result.indexed_chunks == 4
    assert repo.count_points("partial_kb") == 4

    repo.close()
