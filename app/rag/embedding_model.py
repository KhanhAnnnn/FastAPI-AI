from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from app.config import (
    RAG_EMBEDDING_CODE_REVISION,
    RAG_EMBEDDING_MODEL_ID,
    RAG_EMBEDDING_MODEL_REVISION,
)

logger = logging.getLogger(__name__)
_embedder: JinaEmbedder | None = None
_embedder_lock = threading.Lock()


class RemoteCodeApprovalRequired(RuntimeError):
    pass


class EmbeddingModelError(RuntimeError):
    pass


@dataclass
class JinaEmbedder:
    tokenizer: Any
    model: Any
    context_tokens: int
    embedding_dimension: int
    load_seconds: float
    device: str = "cpu"

    def embed(self, texts: list[str], batch_size: int) -> np.ndarray:
        if batch_size < 1:
            raise EmbeddingModelError("Embedding batch size must be positive.")
        if any(not text.strip() for text in texts):
            raise EmbeddingModelError("Cannot embed an empty chunk.")
        vectors: list[np.ndarray] = []
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            if input_ids.shape[1] > self.context_tokens:
                raise EmbeddingModelError(
                    "Tokenized chunk exceeds the embedding model context length."
                )
            model_inputs = {key: value.to("cpu") for key, value in encoded.items()}
            with torch.inference_mode():
                result = self.model(**model_inputs)
                token_embeddings = result.last_hidden_state
                attention_mask = model_inputs["attention_mask"].unsqueeze(-1)
                mask = attention_mask.expand(token_embeddings.size()).to(
                    token_embeddings.dtype
                )
                pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp(min=1e-9)
                pooled_norms = torch.linalg.vector_norm(pooled, ord=2, dim=1)
                if (
                    not torch.isfinite(pooled_norms).all().item()
                    or torch.any(pooled_norms <= 1e-12).item()
                ):
                    raise EmbeddingModelError(
                        "Embedding model returned an empty or non-finite vector."
                    )
                normalized = functional.normalize(pooled, p=2, dim=1)
            if normalized.ndim != 2 or normalized.shape[1] != self.embedding_dimension:
                raise EmbeddingModelError(
                    "Embedding model returned an unexpected vector dimension."
                )
            if not torch.isfinite(normalized).all().item():
                raise EmbeddingModelError("Embedding model returned NaN or Inf values.")
            batch_vectors = normalized.detach().cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(batch_vectors).all():
                raise EmbeddingModelError("Embedding vectors contain NaN or Inf values.")
            vectors.extend(batch_vectors)
        if not vectors:
            raise EmbeddingModelError("No vectors were generated.")
        return np.stack(vectors)


def get_embedder() -> JinaEmbedder:
    global _embedder
    if os.getenv("RAG_ALLOW_JINA_REMOTE_CODE", "false").lower() != "true":
        raise RemoteCodeApprovalRequired(
            "Jina model code execution is disabled. Review and explicitly approve "
            "the pinned Jina remote-code revisions before enabling "
            "RAG_ALLOW_JINA_REMOTE_CODE=true."
        )
    if _embedder is not None:
        return _embedder

    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        started = time.perf_counter()
        try:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                RAG_EMBEDDING_MODEL_ID,
                revision=RAG_EMBEDDING_MODEL_REVISION,
                trust_remote_code=False,
                use_fast=True,
            )
            if not tokenizer.is_fast:
                raise EmbeddingModelError(
                    "Jina embedding tokenizer must support offset mappings."
                )
            model = AutoModel.from_pretrained(
                RAG_EMBEDDING_MODEL_ID,
                revision=RAG_EMBEDDING_MODEL_REVISION,
                code_revision=RAG_EMBEDDING_CODE_REVISION,
                trust_remote_code=True,
            )
            model.to("cpu")
            model.eval()
            context_tokens = int(model.config.max_position_embeddings)
            embedding_dimension = int(model.config.hidden_size)
            if context_tokens < 1 or embedding_dimension < 1:
                raise EmbeddingModelError("Jina model configuration is invalid.")
        except Exception as exc:
            logger.exception("Jina embedding model load failed.")
            raise EmbeddingModelError(
                "Jina embedding model could not be loaded."
            ) from exc

        _embedder = JinaEmbedder(
            tokenizer=tokenizer,
            model=model,
            context_tokens=context_tokens,
            embedding_dimension=embedding_dimension,
            load_seconds=time.perf_counter() - started,
        )
        logger.info(
            "Jina embedding model loaded: model_id=%s revision=%s "
            "code_revision=%s device=cpu context_tokens=%d dimension=%d load_seconds=%.3f",
            RAG_EMBEDDING_MODEL_ID,
            RAG_EMBEDDING_MODEL_REVISION,
            RAG_EMBEDDING_CODE_REVISION,
            context_tokens,
            embedding_dimension,
            _embedder.load_seconds,
        )
        return _embedder
