from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from app.config import RAG_ARTIFACT_ROOT
from app.rag.chunking import TextChunk


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_artifact(
    *,
    document_id: str,
    extraction_id: str,
    run_id: str,
    model_id: str,
    model_revision: str,
    code_revision: str,
    tokenizer_name: str,
    context_tokens: int,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    embedding_dimension: int,
    device: str,
    model_load_seconds: float,
    embedding_seconds: float,
    source_pdf_sha256: str,
    extracted_text_sha256: str,
    chunks: list[TextChunk],
    vectors: np.ndarray,
) -> tuple[Path, dict[str, Any]]:
    UUID(document_id)
    UUID(extraction_id)
    UUID(run_id)
    if vectors.ndim != 2 or vectors.shape != (
        len(chunks),
        embedding_dimension,
    ):
        raise ValueError("Embedding matrix shape does not match chunk metadata.")
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("Embedding vectors must be finite float32 values.")

    root = RAG_ARTIFACT_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifact_name = f"{document_id}.{extraction_id}.{run_id}"
    destination = root / artifact_name
    with tempfile.TemporaryDirectory(prefix=".artifact-", dir=root) as temp_name:
        temporary = Path(temp_name)
        vector_path = temporary / "vectors.npy"
        metadata_path = temporary / "chunks.jsonl"
        np.save(vector_path, vectors, allow_pickle=False)
        with metadata_path.open("x", encoding="utf-8", newline="\n") as output:
            for chunk in chunks:
                output.write(
                    json.dumps(chunk.__dict__, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )

        manifest: dict[str, Any] = {
            "artifact_version": 1,
            "document_id": document_id,
            "extraction_id": extraction_id,
            "run_id": run_id,
            "model_id": model_id,
            "model_revision": model_revision,
            "code_revision": code_revision,
            "tokenizer": tokenizer_name,
            "context_tokens": context_tokens,
            "chunk_size_tokens": chunk_size_tokens,
            "chunk_overlap_tokens": chunk_overlap_tokens,
            "chunk_count": len(chunks),
            "embedding_dimension": embedding_dimension,
            "vector_dtype": "float32",
            "device": device,
            "model_load_seconds": model_load_seconds,
            "embedding_seconds": embedding_seconds,
            "source_pdf_sha256": source_pdf_sha256,
            "extracted_text_sha256": extracted_text_sha256,
            "files": {
                "vectors.npy": {
                    "sha256": _sha256(vector_path),
                    "size_bytes": vector_path.stat().st_size,
                },
                "chunks.jsonl": {
                    "sha256": _sha256(metadata_path),
                    "size_bytes": metadata_path.stat().st_size,
                },
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="",
        )
        manifest_hash_path = temporary / "manifest.sha256"
        manifest_hash_path.write_text(
            f"{_sha256(manifest_path)}  manifest.json\n",
            encoding="ascii",
            newline="\n",
        )
        os.rename(temporary, destination)
    return destination, manifest


def load_artifact(artifact_dir: Path) -> tuple[dict[str, Any], list[dict], np.ndarray]:
    root = RAG_ARTIFACT_ROOT.resolve()
    resolved = artifact_dir.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        raise ValueError("Artifact path is outside the configured artifact directory.")
    manifest_path = resolved / "manifest.json"
    manifest_hash_path = resolved / "manifest.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_hash = manifest_hash_path.read_text(encoding="ascii").split()[0]
    if _sha256(manifest_path) != expected_manifest_hash:
        raise ValueError("Artifact manifest integrity check failed.")
    for name, facts in manifest["files"].items():
        path = resolved / name
        if not path.is_file() or _sha256(path) != facts["sha256"]:
            raise ValueError(f"Artifact integrity check failed for {name}.")
    metadata_path = resolved / "chunks.jsonl"
    chunks = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    vectors = np.load(resolved / "vectors.npy", allow_pickle=False)
    if vectors.shape != (
        manifest["chunk_count"],
        manifest["embedding_dimension"],
    ):
        raise ValueError("Artifact vector and manifest dimensions do not match.")
    if not np.isfinite(vectors).all():
        raise ValueError("Artifact vectors contain NaN or Inf values.")
    return manifest, chunks, vectors
