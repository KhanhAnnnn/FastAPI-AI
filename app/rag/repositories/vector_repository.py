from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from app.config import (
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_LOCAL_PATH,
    QDRANT_MODE,
    QDRANT_PORT,
)

logger = logging.getLogger(__name__)


class VectorRepositoryError(Exception):
    """Base exception for vector repository operations."""


class StorageUnavailableError(VectorRepositoryError):
    """Raised when vector storage backend is unreachable or cannot be initialized."""


class CollectionConfigMismatchError(VectorRepositoryError):
    """Raised when an existing collection configuration is incompatible with requirements."""


class VectorStorageError(VectorRepositoryError):
    """Raised when an error occurs during vector read/write operations."""


class VectorRepository:
    """Repository managing Qdrant vector database connections, collection lifecycle, and indexing."""

    def __init__(
        self,
        mode: str = QDRANT_MODE,
        collection_name: str = QDRANT_COLLECTION,
        local_path: Path | str | None = None,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        client: QdrantClient | None = None,
    ) -> None:
        self.mode = mode.lower()
        self.default_collection = collection_name
        self.host = host
        self.port = port
        self.local_path = (
            Path(local_path).resolve()
            if local_path is not None
            else QDRANT_LOCAL_PATH
        )
        self._client: QdrantClient | None = client
        self._lock = threading.RLock()

    def get_client(self) -> QdrantClient:
        """Lazily initialize and return the QdrantClient instance under lock."""
        if self._client is not None:
            return self._client

        with self._lock:
            if self._client is not None:
                return self._client

            try:
                if self.mode == "memory":
                    self._client = QdrantClient(location=":memory:")
                elif self.mode == "local":
                    self.local_path.mkdir(parents=True, exist_ok=True)
                    self._client = QdrantClient(path=str(self.local_path))
                elif self.mode == "server":
                    self._client = QdrantClient(host=self.host, port=self.port)
                else:
                    raise StorageUnavailableError(
                        f"Unsupported QDRANT_MODE '{self.mode}'. Supported modes: 'local', 'server', 'memory'."
                    )
                logger.info(
                    "Connected to Qdrant: mode=%s, target=%s",
                    self.mode,
                    self.local_path if self.mode == "local" else f"{self.host}:{self.port}",
                )
                return self._client
            except Exception as exc:
                logger.error("Failed to initialize Qdrant client: %s", exc, exc_info=True)
                raise StorageUnavailableError(
                    f"Vector storage backend ({self.mode}) is unavailable: {exc}"
                ) from exc

    def collection_exists(self, collection_name: str | None = None) -> bool:
        """Check whether the given collection exists."""
        target = collection_name or self.default_collection
        client = self.get_client()
        try:
            return bool(client.collection_exists(collection_name=target))
        except Exception as exc:
            logger.error("Failed to check if collection '%s' exists: %s", target, exc)
            raise StorageUnavailableError(f"Failed to check collection status: {exc}") from exc

    def create_collection_if_not_exists(
        self,
        collection_name: str | None = None,
        dimension: int = 768,
        distance: Distance = Distance.COSINE,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Create the collection if it does not already exist. Returns True if created, False if already existed."""
        target = collection_name or self.default_collection
        with self._lock:
            if self.collection_exists(target):
                self.verify_collection_compatibility(
                    collection_name=target,
                    expected_dimension=dimension,
                    expected_distance=distance.value if isinstance(distance, Distance) else str(distance),
                    expected_model=metadata.get("model_id") if metadata else None,
                    expected_revision=metadata.get("model_revision") if metadata else None,
                )
                return False

            client = self.get_client()
            try:
                client.create_collection(
                    collection_name=target,
                    vectors_config=VectorParams(size=dimension, distance=distance),
                    metadata=metadata,
                )
                logger.info(
                    "Created Qdrant collection '%s' with dimension=%d, distance=%s",
                    target,
                    dimension,
                    distance,
                )
                return True
            except Exception as exc:
                logger.error("Failed to create Qdrant collection '%s': %s", target, exc)
                raise StorageUnavailableError(f"Failed to create collection '{target}': {exc}") from exc

    def verify_collection_compatibility(
        self,
        collection_name: str | None = None,
        expected_dimension: int = 768,
        expected_distance: str = "Cosine",
        expected_model: str | None = None,
        expected_revision: str | None = None,
    ) -> None:
        """Verify that an existing collection matches expected vector configuration and model."""
        target = collection_name or self.default_collection
        client = self.get_client()
        try:
            info = client.get_collection(collection_name=target)
        except Exception as exc:
            raise StorageUnavailableError(f"Unable to inspect collection '{target}': {exc}") from exc

        vectors_config = info.config.params.vectors
        actual_dimension: int | None = None
        actual_distance: str | None = None

        if isinstance(vectors_config, VectorParams):
            actual_dimension = vectors_config.size
            actual_distance = str(vectors_config.distance.value)
        elif isinstance(vectors_config, dict):
            actual_dimension = vectors_config.get("size")
            actual_distance = str(vectors_config.get("distance"))

        if actual_dimension != expected_dimension:
            raise CollectionConfigMismatchError(
                f"Collection '{target}' vector dimension mismatch: expected {expected_dimension}, "
                f"found {actual_dimension}."
            )

        expected_dist_clean = expected_distance.strip().lower()
        actual_dist_clean = (actual_distance or "").strip().lower()
        if expected_dist_clean != actual_dist_clean:
            raise CollectionConfigMismatchError(
                f"Collection '{target}' distance metric mismatch: expected {expected_distance}, "
                f"found {actual_distance}."
            )

        # Verify embedding model compatibility if stored in metadata or points
        config_metadata = getattr(info.config, "metadata", None) or {}
        recorded_model = config_metadata.get("model_id")
        recorded_rev = config_metadata.get("model_revision")

        if expected_model and recorded_model and expected_model != recorded_model:
            raise CollectionConfigMismatchError(
                f"Collection '{target}' model mismatch: expected '{expected_model}', "
                f"recorded '{recorded_model}'."
            )
        if expected_revision and recorded_rev and expected_revision != recorded_rev:
            raise CollectionConfigMismatchError(
                f"Collection '{target}' model revision mismatch: expected '{expected_revision}', "
                f"recorded '{recorded_rev}'."
            )

        # Also probe one point payload if points exist
        if expected_model and (info.points_count or 0) > 0:
            try:
                scroll_res, _ = client.scroll(
                    collection_name=target,
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
                if scroll_res and scroll_res[0].payload:
                    point_model = scroll_res[0].payload.get("embedding_model")
                    point_rev = scroll_res[0].payload.get("embedding_revision")
                    if point_model and point_model != expected_model:
                        raise CollectionConfigMismatchError(
                            f"Collection '{target}' contains points embedded with '{point_model}', "
                            f"incompatible with target model '{expected_model}'."
                        )
                    if expected_revision and point_rev and point_rev != expected_revision:
                        raise CollectionConfigMismatchError(
                            f"Collection '{target}' contains points with revision '{point_rev}', "
                            f"incompatible with target revision '{expected_revision}'."
                        )
            except CollectionConfigMismatchError:
                raise
            except Exception as exc:
                logger.warning("Could not probe collection points for model verification: %s", exc)

    def upsert_batch(
        self,
        points: list[PointStruct],
        collection_name: str | None = None,
    ) -> None:
        """Upsert a batch of points into the target collection under lock."""
        if not points:
            return
        target = collection_name or self.default_collection
        client = self.get_client()
        with self._lock:
            try:
                client.upsert(collection_name=target, points=points)
            except Exception as exc:
                logger.error("Failed to upsert points into '%s': %s", target, exc)
                raise VectorStorageError(f"Failed to upsert points into collection '{target}': {exc}") from exc

    def search_vectors(
        self,
        query_vector: list[float] | np.ndarray,
        limit: int = 3,
        score_threshold: float | None = None,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        collection_name: str | None = None,
    ) -> list[ScoredPoint]:
        """Perform similarity search with cosine distance, optional score threshold, and optional document filter."""
        target = collection_name or self.default_collection
        if not self.collection_exists(target):
            return []

        if isinstance(query_vector, np.ndarray):
            vector = query_vector.tolist()
        else:
            vector = list(query_vector)

        query_filter: Filter | None = None
        if document_id:
            query_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=str(document_id)))]
            )
        elif document_ids:
            clean_ids = [str(d) for d in document_ids if d]
            if len(clean_ids) == 1:
                query_filter = Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=clean_ids[0]))]
                )
            elif len(clean_ids) > 1:
                query_filter = Filter(
                    must=[FieldCondition(key="document_id", match=MatchAny(any=clean_ids))]
                )

        client = self.get_client()
        try:
            response = client.query_points(
                collection_name=target,
                query=vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )
            return list(response.points)
        except Exception as exc:
            logger.error("Query failed on collection '%s': %s", target, exc)
            raise VectorStorageError(f"Query failed on collection '{target}': {exc}") from exc

    def count_points(
        self,
        collection_name: str | None = None,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> int:
        """Return the exact count of points in the collection, optionally filtered by document_id(s)."""
        target = collection_name or self.default_collection
        if not self.collection_exists(target):
            return 0

        count_filter: Filter | None = None
        if document_id:
            count_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=str(document_id)))]
            )
        elif document_ids:
            clean_ids = [str(d) for d in document_ids if d]
            if len(clean_ids) == 1:
                count_filter = Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=clean_ids[0]))]
                )
            elif len(clean_ids) > 1:
                count_filter = Filter(
                    must=[FieldCondition(key="document_id", match=MatchAny(any=clean_ids))]
                )

        client = self.get_client()
        try:
            return int(client.count(collection_name=target, count_filter=count_filter, exact=True).count)
        except Exception as exc:
            logger.error("Failed to count points in collection '%s': %s", target, exc)
            raise VectorStorageError(f"Failed to count points in collection '{target}': {exc}") from exc

    def verify_points_exist(
        self,
        point_ids: list[str],
        collection_name: str | None = None,
    ) -> bool:
        """Verify that all given point IDs exist in the collection."""
        if not point_ids:
            return True
        target = collection_name or self.default_collection
        client = self.get_client()
        try:
            retrieved = client.retrieve(
                collection_name=target,
                ids=point_ids,
                with_payload=False,
                with_vectors=False,
            )
            return len(retrieved) == len(point_ids)
        except Exception as exc:
            logger.error("Failed to verify point existence in '%s': %s", target, exc)
            raise VectorStorageError(f"Failed to verify points in collection '{target}': {exc}") from exc

    def get_collection_info(self, collection_name: str | None = None) -> dict[str, Any]:
        """Return diagnostics and status dictionary for the collection."""
        target = collection_name or self.default_collection
        if not self.collection_exists(target):
            return {
                "collection_name": target,
                "collection_exists": False,
                "vector_dimension": None,
                "distance": None,
                "point_count": 0,
                "storage_mode": self.mode,
                "status": "not_initialized",
            }

        client = self.get_client()
        try:
            info = client.get_collection(collection_name=target)
            count = int(client.count(collection_name=target, exact=True).count)
            vectors_cfg = info.config.params.vectors
            dimension: int | None = None
            distance_name: str | None = None
            if isinstance(vectors_cfg, VectorParams):
                dimension = vectors_cfg.size
                distance_name = vectors_cfg.distance.value
            elif isinstance(vectors_cfg, dict):
                dimension = vectors_cfg.get("size")
                distance_name = str(vectors_cfg.get("distance"))

            status_str = "ready" if count > 0 else "empty"
            return {
                "collection_name": target,
                "collection_exists": True,
                "vector_dimension": dimension,
                "distance": distance_name,
                "point_count": count,
                "storage_mode": self.mode,
                "status": status_str,
            }
        except Exception as exc:
            logger.error("Failed to retrieve collection info for '%s': %s", target, exc)
            raise StorageUnavailableError(f"Failed to retrieve collection info: {exc}") from exc

    def close(self) -> None:
        """Safely close client and release underlying locks."""
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception as exc:
                    logger.warning("Error while closing QdrantClient: %s", exc)
                finally:
                    self._client = None


_global_repo: VectorRepository | None = None
_global_repo_lock = threading.Lock()


def get_vector_repository() -> VectorRepository:
    """Singleton provider for the application vector repository."""
    global _global_repo
    if _global_repo is not None:
        return _global_repo
    with _global_repo_lock:
        if _global_repo is None:
            _global_repo = VectorRepository()
    return _global_repo


def close_vector_repository() -> None:
    """Close the global singleton vector repository."""
    global _global_repo
    with _global_repo_lock:
        if _global_repo is not None:
            _global_repo.close()
            _global_repo = None
