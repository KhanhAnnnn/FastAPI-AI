from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID, uuid5

from app.config import RAG_CHUNK_OVERLAP_TOKENS, RAG_CHUNK_SIZE_TOKENS

_CHUNK_NAMESPACE = UUID("f67bac22-6e61-4fa7-9f37-77c4c4d74c70")
_SENTENCE_BOUNDARY = re.compile(r"[.!?。！？](?=\s|$)")


class ChunkingError(ValueError):
    pass


class OffsetTokenizer:
    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> Mapping[str, list]:
        raise NotImplementedError


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    extraction_id: str
    source_filename: str | None
    page_number: int
    chunk_index: int
    original_text: str
    cleaned_text: str
    token_count: int
    text_hash: str


@dataclass(frozen=True)
class _CleanedText:
    text: str
    source_spans: tuple[tuple[int, int], ...]


def clean_page_text(text: str) -> _CleanedText:
    cleaned: list[str] = []
    source_spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        if not text[cursor].isspace():
            cleaned.append(text[cursor])
            source_spans.append((cursor, cursor + 1))
            cursor += 1
            continue

        start = cursor
        has_line_break = False
        while cursor < len(text) and text[cursor].isspace():
            has_line_break = has_line_break or text[cursor] in "\r\n"
            cursor += 1
        if cleaned and cursor < len(text):
            cleaned.append("\n" if has_line_break else " ")
            source_spans.append((start, cursor))

    if cleaned and cleaned[-1] in {" ", "\n"}:
        cleaned.pop()
        source_spans.pop()
    return _CleanedText("".join(cleaned), tuple(source_spans))


def _token_offsets(tokenizer: OffsetTokenizer, text: str) -> list[tuple[int, int]]:
    try:
        result = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        offsets = result["offset_mapping"]
        if offsets and isinstance(offsets[0], list):
            offsets = offsets[0]
        valid = [
            (int(start), int(end))
            for start, end in offsets
            if 0 <= int(start) < int(end) <= len(text)
        ]
    except Exception as exc:
        raise ChunkingError("The embedding tokenizer did not return usable offsets.") from exc
    if not valid:
        raise ChunkingError("Tokenizer returned no content tokens for non-empty text.")
    return valid


def chunk_pages(
    *,
    document_id: str,
    extraction_id: str,
    source_filename: str | None,
    extracted_text: str,
    pages: list[dict],
    tokenizer: OffsetTokenizer,
    chunk_size_tokens: int = RAG_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = RAG_CHUNK_OVERLAP_TOKENS,
    model_context_tokens: int,
    special_tokens: int,
) -> list[TextChunk]:
    try:
        UUID(document_id)
        UUID(extraction_id)
    except ValueError as exc:
        raise ChunkingError("Document and extraction identifiers must be UUIDs.") from exc
    if chunk_size_tokens < 1:
        raise ChunkingError("Chunk size must be at least one token.")
    if chunk_overlap_tokens < 0 or chunk_overlap_tokens >= chunk_size_tokens:
        raise ChunkingError("Chunk overlap must be non-negative and smaller than chunk size.")
    if model_context_tokens <= 0 or special_tokens < 0:
        raise ChunkingError("Model context and special-token counts are invalid.")
    if chunk_size_tokens + special_tokens > model_context_tokens:
        raise ChunkingError("Chunk size exceeds the embedding model context length.")

    chunks: list[TextChunk] = []
    chunk_index = 0
    expected_start = 0
    for page in pages:
        page_number = page.get("page_number")
        text_start = page.get("text_start")
        text_end = page.get("text_end")
        if not isinstance(page_number, int) or page_number < 1:
            raise ChunkingError("Extraction page metadata is invalid.")
        if text_start is None and text_end is None:
            continue
        if (
            not isinstance(text_start, int)
            or not isinstance(text_end, int)
            or text_start < expected_start
            or text_end < text_start
            or text_end > len(extracted_text)
        ):
            raise ChunkingError("Extraction text offsets are invalid.")
        expected_start = text_end
        raw_page = extracted_text[text_start:text_end]
        cleaned = clean_page_text(raw_page)
        if not cleaned.text:
            continue
        offsets = _token_offsets(tokenizer, cleaned.text)
        sentence_boundaries = {
            match.end() for match in _SENTENCE_BOUNDARY.finditer(cleaned.text)
        }
        token_start = 0
        while token_start < len(offsets):
            hard_end = min(token_start + chunk_size_tokens, len(offsets))
            token_end = hard_end
            if hard_end < len(offsets):
                minimum_end = token_start + max(1, (chunk_size_tokens * 3 + 4) // 5)
                for candidate in range(hard_end, minimum_end - 1, -1):
                    boundary = offsets[candidate - 1][1]
                    if boundary in sentence_boundaries:
                        token_end = candidate
                        break

            char_start = offsets[token_start][0]
            char_end = offsets[token_end - 1][1]
            cleaned_chunk = cleaned.text[char_start:char_end]
            if not cleaned_chunk.strip():
                raise ChunkingError("Chunking produced an empty text range.")
            source_start = cleaned.source_spans[char_start][0]
            source_end = cleaned.source_spans[char_end - 1][1]
            original_chunk = raw_page[source_start:source_end]
            chunk_hash = hashlib.sha256(cleaned_chunk.encode("utf-8")).hexdigest()
            stable_key = (
                f"{document_id}:{extraction_id}:{page_number}:{chunk_index}:{chunk_hash}"
            )
            chunks.append(
                TextChunk(
                    chunk_id=str(uuid5(_CHUNK_NAMESPACE, stable_key)),
                    document_id=document_id,
                    extraction_id=extraction_id,
                    source_filename=source_filename,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    original_text=original_chunk,
                    cleaned_text=cleaned_chunk,
                    token_count=token_end - token_start,
                    text_hash=chunk_hash,
                )
            )
            chunk_index += 1
            if token_end == len(offsets):
                break
            next_start = max(token_start + 1, token_end - chunk_overlap_tokens)
            if next_start <= token_start:
                raise ChunkingError("Chunker failed to advance.")
            token_start = next_start
    return chunks
