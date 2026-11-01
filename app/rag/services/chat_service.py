from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from typing import Any
from uuid import UUID

from app.core.exceptions import ModelBusyError
from app.core.model_manager import model_manager
from app.rag.chat_schemas import ChatRequest, ChatResponse, SourceItem
from app.rag.embedding_model import EmbeddingModelError, RemoteCodeApprovalRequired
from app.rag.repositories.vector_repository import (
    StorageUnavailableError,
    VectorStorageError,
)
from app.rag.retrieval_service import (
    InvalidQueryError,
    RetrievalService,
    compute_lexical_overlap,
    extract_query_entities,
    get_retrieval_service,
)
from app.text.models import MODEL_ID, load_model

logger = logging.getLogger(__name__)

DEFAULT_ENGLISH_RAG_SYSTEM_PROMPT = (
    "You are a helpful and strictly factual assistant. Answer the user's question based strictly and ONLY on the provided context below.\n"
    "Mandatory rules:\n"
    "1. Only use facts, figures, and dates explicitly present in the provided context.\n"
    "2. Do NOT hallucinate, speculate, or fabricate any facts or numbers outside the context.\n"
    "3. If the context does not contain enough information or states that the information is unpublished, state clearly that the documents do not have enough information to answer.\n"
    "4. Answer concisely and directly to the question.\n\n"
    "Context:\n{context}"
)

DEFAULT_VIETNAMESE_RAG_SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi đáp tài liệu. Hãy trả lời câu hỏi bằng cách dẫn lại câu văn chứa câu trả lời trong văn bản.\n"
    "Nếu văn bản không có hoặc chưa công bố thông tin, hãy trả lời: 'Tài liệu không có đủ thông tin để xác định.'\n\n"
    "Văn bản:\n{context}"
)

DEFAULT_RAG_SYSTEM_PROMPT = DEFAULT_ENGLISH_RAG_SYSTEM_PROMPT

VIETNAMESE_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)


def _is_vietnamese(text: str) -> bool:
    if VIETNAMESE_RE.search(text):
        return True
    lower = text.lower()
    return any(w in lower for w in ["hoc vien", "tai lieu", "bao nhieu", "la gi", "trung tam", "thap hon", "muc tieu"])


def segment_into_cohesive_blocks(text: str) -> list[str]:
    """Segment chunk into cohesive logical blocks (sections or paragraphs) without splitting tables or sentences."""
    normalized = unicodedata.normalize("NFC", text).strip()
    raw_blocks = re.split(r"(?:\n\s*\n+|\n(?=(?:#{1,6}\s|\d+\.|\d+\))\s+[A-ZĐa-z0-9]))", normalized)
    blocks: list[str] = []
    pending_header = ""
    for b in raw_blocks:
        b_clean = b.strip()
        if not b_clean:
            continue
        # If this is a very short header (e.g. document title or section tag < 40 chars), attach to next block
        if len(b_clean) < 40 and not blocks and not pending_header:
            pending_header = b_clean
            continue
        if pending_header:
            blocks.append(f"{pending_header}\n{b_clean}")
            pending_header = ""
        else:
            blocks.append(b_clean)
    if pending_header:
        blocks.append(pending_header)
    return blocks if blocks else [text.strip()]


def extract_focused_context(chunk_text: str, query: str, max_chars: int = 1200) -> str:
    """Extract context preserving entity-relation integrity, logging selection decisions and actual token budget."""
    blocks = segment_into_cohesive_blocks(chunk_text)
    entities = extract_query_entities(query)

    scored_blocks: list[tuple[int, float, bool, str]] = []
    for idx, block in enumerate(blocks):
        score = compute_lexical_overlap(query, block)
        b_norm = block.lower()
        has_entity = any(e in b_norm for e in entities) if entities else False
        scored_blocks.append((idx, score, has_entity, block))

    # Sort descending by relevance score, secondary by entity presence
    ranked = sorted(scored_blocks, key=lambda x: (x[1], 1 if x[2] else 0), reverse=True)

    selected_indices: set[int] = set()
    accumulated_chars = 0

    for idx, score, has_entity, block in ranked:
        block_len = len(block)
        if score >= 0.25 or has_entity or not selected_indices:
            if accumulated_chars + block_len <= max_chars or not selected_indices:
                selected_indices.add(idx)
                accumulated_chars += block_len
                logger.info(
                    "Context Selection: Block %d (score=%.4f, len=%d, entity=%s) -> SELECTED. Accumulated chars: %d",
                    idx, score, block_len, has_entity, accumulated_chars,
                )
            else:
                logger.info(
                    "Context Selection: Block %d (score=%.4f, len=%d, entity=%s) -> DROPPED (Budget exceeded: %d + %d > %d)",
                    idx, score, block_len, has_entity, accumulated_chars, block_len, max_chars,
                )
        else:
            logger.info(
                "Context Selection: Block %d (score=%.4f, len=%d, entity=%s) -> DROPPED (Score below threshold and no entity)",
                idx, score, block_len, has_entity,
            )

    # Reassemble selected blocks in original document order to preserve narrative structure and entity relationships
    chosen_blocks = [blocks[i] for i in sorted(selected_indices)]
    assembled_context = "\n\n".join(chosen_blocks)
    estimated_tokens = len(assembled_context) // 3
    logger.info(
        "Context Construction complete: blocks=%d/%d, chars=%d, estimated_tokens=%d, max_budget=%d",
        len(chosen_blocks), len(blocks), len(assembled_context), estimated_tokens, max_chars,
    )
    return assembled_context


def _generate_rag_answer(
    model_data: tuple[Any, Any, Any],
    system_prompt: str,
    query: str,
    max_new_tokens: int,
    temperature: float,
    assistant_prefix: str = "",
) -> str:
    import torch

    tokenizer, model, device = model_data
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if assistant_prefix:
        prompt = prompt + assistant_prefix

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    # Hard token budget guard to strictly prevent exceeding 2048 context window
    if input_ids.shape[-1] > 1750:
        input_ids = input_ids[:, -1750:]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -1750:]

    generation_options: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
        "repetition_penalty": 1.15,
    }
    if temperature > 0:
        generation_options["temperature"] = temperature
        generation_options["top_p"] = 0.95

    with torch.inference_mode():
        output_ids = model.generate(input_ids, attention_mask=attention_mask, **generation_options)

    new_tokens = output_ids[0, input_ids.shape[-1] :]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if assistant_prefix:
        # If TinyLlama echoed the user's question before the answer: e.g. "Câu hỏi...?" Bạn sẽ trả lời: ...
        # Strip the echoed question so the real answer is surfaced cleanly
        cleaned_gen = re.sub(
            r'^\s*[^"”\n\r?]*\?\s*["”]\s*(?:Bạn sẽ trả lời:|Hãy trả lời:|Trả lời:|Bình thường,)?\s*',
            '',
            generated_text,
            flags=re.IGNORECASE,
        ).strip()
        if cleaned_gen:
            generated_text = cleaned_gen

        if assistant_prefix.endswith('"'):
            # If generated text contains a closed quote, extract up to that closing quote to prevent trailing hallucinations
            quote_match = re.search(r'^(.*?[^"\\\s])["”]', generated_text, flags=re.DOTALL)
            if quote_match:
                quoted_body = quote_match.group(1).strip()
                return f'{assistant_prefix}{quoted_body}"'
            if generated_text.startswith('"') or generated_text.startswith('“'):
                full_text = f"{assistant_prefix[:-1]}{generated_text}".strip()
            else:
                full_text = f"{assistant_prefix}{generated_text}".strip()
            if not full_text.endswith('"'):
                full_text = f'{full_text}"'
            return full_text

        return f"{assistant_prefix}{generated_text}".strip()
    return generated_text


class ChatService:
    """End-to-end RAG chat service combining semantic retrieval with grounded TinyLlama answering."""

    def __init__(self, retrieval_service: RetrievalService | None = None) -> None:
        self._retrieval_service = retrieval_service

    @property
    def retrieval_service(self) -> RetrievalService:
        if self._retrieval_service is not None:
            return self._retrieval_service
        return get_retrieval_service()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        total_start = time.perf_counter()
        is_vn = _is_vietnamese(request.query)

        # Step 1: Semantic Retrieval (with hybrid reranking)
        retrieval_start = time.perf_counter()
        search_res = await self.retrieval_service.search(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            document_id=request.document_id,
            document_ids=request.document_ids,
        )
        retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # Step 2: Grounding Gate & Evidence Verification
        # Check if any candidate chunks have sufficient lexical keyword overlap (evidence grounding)
        relevant_results = [
            r for r in search_res.results
            if compute_lexical_overlap(request.query, r.text) >= 0.25
        ]

        if not search_res.results or not relevant_results:
            if is_vn:
                refusal_msg = "Không tìm thấy đủ thông tin trong tài liệu đã cung cấp để trả lời câu hỏi này."
            else:
                refusal_msg = "I could not find sufficient information in the provided documents to answer this question."

            total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
            return ChatResponse(
                query=request.query,
                answer=refusal_msg,
                sources=[],
                retrieval_count=0,
                model_used=MODEL_ID,
                latency_ms={
                    "retrieval_ms": round(retrieval_elapsed_ms, 2),
                    "generation_ms": 0.0,
                    "total_ms": round(total_elapsed_ms, 2),
                },
            )

        # Sort relevant results descending by score
        relevant_results.sort(key=lambda r: r.score, reverse=True)

        # Step 3: Assemble context within token budget (~1000 tokens / ~1500 chars)
        # Drop low-confidence candidates if there is a clear top candidate (gap > 0.08)
        top_score = relevant_results[0].score
        confident_results = [
            r for r in relevant_results
            if r.score >= top_score - 0.08
        ]

        context_blocks: list[str] = []
        used_results: list[SearchResultItem] = []
        char_count = 0
        max_context_chars = 1500

        for r in confident_results:
            focused_text = extract_focused_context(r.text, request.query)
            block = f"[Document: {r.document_id}, Page: {r.page_number}]\n{focused_text}"
            if char_count + len(block) > max_context_chars and context_blocks:
                break
            context_blocks.append(block)
            used_results.append(r)
            char_count += len(block)

        context_str = "\n\n".join(context_blocks)

        # Format sources using ONLY chunks that actually formed the context
        sources: list[SourceItem] = [
            SourceItem(
                document_id=r.document_id,
                page_number=r.page_number,
                chunk_id=r.chunk_id,
                score=round(r.score, 4),
                text=r.text,
            )
            for r in used_results
        ]

        # Step 4: Construct system prompt & assistant prefill
        if request.system_prompt:
            system_prompt = f"{request.system_prompt}\n\nContext:\n{context_str}"
            assistant_prefix = ""
        else:
            if is_vn:
                system_prompt = DEFAULT_VIETNAMESE_RAG_SYSTEM_PROMPT.format(context=context_str)
                assistant_prefix = 'Theo văn bản: "'
            else:
                system_prompt = DEFAULT_ENGLISH_RAG_SYSTEM_PROMPT.format(context=context_str)
                assistant_prefix = "According to the documents: "

        # Step 5: Execute generation using TinyLlama under ModelManager exclusive lock
        gen_start = time.perf_counter()
        try:
            answer, _gen_metrics = await model_manager.run(
                "text",
                load_model,
                lambda model_data: _generate_rag_answer(
                    model_data=model_data,
                    system_prompt=system_prompt,
                    query=request.query,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    assistant_prefix=assistant_prefix,
                ),
            )
        except Exception as exc:
            logger.error("Error during TinyLlama RAG generation: %s", exc, exc_info=True)
            raise

        gen_elapsed_ms = (time.perf_counter() - gen_start) * 1000.0
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        if not answer:
            answer = "I was unable to generate an answer from the retrieved document context."

        return ChatResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            retrieval_count=len(sources),
            model_used=MODEL_ID,
            latency_ms={
                "retrieval_ms": round(retrieval_elapsed_ms, 2),
                "generation_ms": round(gen_elapsed_ms, 2),
                "total_ms": round(total_elapsed_ms, 2),
            },
        )


_global_chat_service: ChatService | None = None
_chat_lock = threading.Lock()


def get_chat_service() -> ChatService:
    global _global_chat_service
    if _global_chat_service is not None:
        return _global_chat_service
    with _chat_lock:
        if _global_chat_service is None:
            _global_chat_service = ChatService()
    return _global_chat_service
