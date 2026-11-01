import hashlib
import re
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from app.rag import artifacts
from app.rag.chunking import ChunkingError, chunk_pages
from app.rag.chunking import clean_page_text


class WordOffsetTokenizer:
    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict[str, list]:
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "offset_mapping": [
                match.span() for match in re.finditer(r"\S+", text, re.UNICODE)
            ]
        }


@pytest.fixture
def page_text() -> str:
    return "English paragraph with punctuation.  \n\nSecond paragraph has more words."


def make_chunks(
    text: str,
    *,
    size: int = 8,
    overlap: int = 2,
    pages: list[dict] | None = None,
    tokenizer: WordOffsetTokenizer | None = None,
):
    doc_id = str(uuid4())
    extraction_id = str(uuid4())
    return chunk_pages(
        document_id=doc_id,
        extraction_id=extraction_id,
        source_filename="sample.pdf",
        extracted_text=text,
        pages=pages
        if pages is not None
        else [
            {
                "page_number": 1,
                "has_text": True,
                "text_start": 0,
                "text_end": len(text),
            }
        ],
        tokenizer=tokenizer or WordOffsetTokenizer(),
        chunk_size_tokens=size,
        chunk_overlap_tokens=overlap,
        model_context_tokens=8192,
        special_tokens=2,
    )


def test_chunking_cleans_whitespace_and_keeps_original_source(page_text: str) -> None:
    chunks = make_chunks(page_text, size=20, overlap=2)

    assert len(chunks) == 1
    assert chunks[0].cleaned_text == (
        "English paragraph with punctuation.\nSecond paragraph has more words."
    )
    assert chunks[0].original_text == page_text
    assert chunks[0].page_number == 1
    assert chunks[0].source_filename == "sample.pdf"


def test_chunks_multi_page_documents_without_losing_page_attribution() -> None:
    text = "Page one has a sentence. Page two is separate."
    first_end = len("Page one has a sentence.")
    second_start = first_end + 2
    chunks = make_chunks(
        text,
        size=4,
        overlap=1,
        pages=[
            {
                "page_number": 1,
                "has_text": True,
                "text_start": 0,
                "text_end": first_end,
            },
            {
                "page_number": 2,
                "has_text": True,
                "text_start": second_start,
                "text_end": len(text),
            },
        ],
    )

    assert chunks
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(chunk.document_id == chunks[0].document_id for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_vietnamese_unicode_is_preserved() -> None:
    text = "Tiếng Việt có dấu: Trường học, Nguyễn Ái Quốc."
    chunks = make_chunks(text, size=32)

    assert len(chunks) == 1
    assert chunks[0].cleaned_text == text
    assert chunks[0].original_text == text
    assert chunks[0].text_hash == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_long_text_chunks_respect_size_and_overlap_and_advance() -> None:
    words = [f"token{i}" for i in range(1_200)]
    text = " ".join(words)
    chunks = make_chunks(text, size=64, overlap=8)

    assert len(chunks) > 1
    assert all(0 < chunk.token_count <= 64 for chunk in chunks)
    assert all(chunk.cleaned_text.strip() for chunk in chunks)
    assert all(
        left.cleaned_text.split()[-8:] == right.cleaned_text.split()[:8]
        for left, right in zip(chunks, chunks[1:])
    )
    observed = [token for chunk in chunks for token in chunk.cleaned_text.split()]
    for token in words:
        assert token in observed
    assert all(left.chunk_index < right.chunk_index for left, right in zip(chunks, chunks[1:]))
    assert max(len(chunk.cleaned_text) for chunk in chunks) < len(text)


def test_chunker_prefers_a_sentence_boundary_when_it_fits() -> None:
    chunks = make_chunks(
        "one two three four five six seven. eight nine ten eleven twelve thirteen",
        size=10,
        overlap=2,
    )

    assert len(chunks) == 2
    assert chunks[0].cleaned_text.endswith(".")
    assert chunks[0].token_count == 7


def test_short_and_empty_pages_do_not_create_empty_chunks() -> None:
    text = "short page"
    chunks = make_chunks(
        text,
        pages=[
            {"page_number": 1, "has_text": False, "text_start": None, "text_end": None},
            {
                "page_number": 2,
                "has_text": True,
                "text_start": 0,
                "text_end": len(text),
            },
        ],
    )
    assert len(chunks) == 1
    assert chunks[0].page_number == 2
    assert chunks[0].cleaned_text == text


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (4, 4), (4, 5), (4, -1)],
)
def test_invalid_chunk_size_and_overlap_are_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ChunkingError):
        make_chunks("one two three", size=size, overlap=overlap)


def test_chunk_configuration_cannot_exceed_model_context() -> None:
    with pytest.raises(ChunkingError, match="context length"):
        chunk_pages(
            document_id=str(uuid4()),
            extraction_id=str(uuid4()),
            source_filename=None,
            extracted_text="text",
            pages=[
                {"page_number": 1, "text_start": 0, "text_end": 4}
            ],
            tokenizer=WordOffsetTokenizer(),
            chunk_size_tokens=8,
            chunk_overlap_tokens=2,
            model_context_tokens=8,
            special_tokens=2,
        )


def test_artifact_round_trip_and_hash_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page_text: str
) -> None:
    root = tmp_path / "artifacts"
    monkeypatch.setattr(artifacts, "RAG_ARTIFACT_ROOT", root)
    chunks = make_chunks(page_text, size=32)
    vectors = np.asarray([[0.25, -0.75, 0.5]], dtype=np.float32)
    document_id, extraction_id, run_id = (
        chunks[0].document_id,
        chunks[0].extraction_id,
        str(uuid4()),
    )

    artifact_dir, manifest = artifacts.write_artifact(
        document_id=document_id,
        extraction_id=extraction_id,
        run_id=run_id,
        model_id="test/model",
        model_revision="test-revision",
        code_revision="test-code-revision",
        tokenizer_name="test/tokenizer",
        context_tokens=100,
        chunk_size_tokens=32,
        chunk_overlap_tokens=2,
        embedding_dimension=3,
        device="cpu",
        model_load_seconds=0.1,
        embedding_seconds=0.2,
        source_pdf_sha256="source-pdf-sha",
        extracted_text_sha256="extracted-text-sha",
        chunks=chunks,
        vectors=vectors,
    )

    loaded_manifest, loaded_chunks, loaded_vectors = artifacts.load_artifact(artifact_dir)
    assert manifest["files"]["vectors.npy"]["sha256"] == hashlib.sha256(
        (artifact_dir / "vectors.npy").read_bytes()
    ).hexdigest()
    assert loaded_manifest["model_revision"] == "test-revision"
    assert loaded_chunks[0]["page_number"] == chunks[0].page_number
    np.testing.assert_array_equal(loaded_vectors, vectors)


def test_corrupted_artifact_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page_text: str
) -> None:
    root = tmp_path / "artifacts"
    monkeypatch.setattr(artifacts, "RAG_ARTIFACT_ROOT", root)
    chunks = make_chunks(page_text, size=32)
    artifact_dir, _ = artifacts.write_artifact(
        document_id=chunks[0].document_id,
        extraction_id=chunks[0].extraction_id,
        run_id=str(uuid4()),
        model_id="test/model",
        model_revision="test-revision",
        code_revision="test-code-revision",
        tokenizer_name="test/tokenizer",
        context_tokens=100,
        chunk_size_tokens=32,
        chunk_overlap_tokens=2,
        embedding_dimension=2,
        device="cpu",
        model_load_seconds=0.1,
        embedding_seconds=0.2,
        source_pdf_sha256="source-pdf-sha",
        extracted_text_sha256="extracted-text-sha",
        chunks=chunks,
        vectors=np.asarray([[1, 2]], dtype=np.float32),
    )
    with (artifact_dir / "vectors.npy").open("ab") as output:
        output.write(b"tamper")

    with pytest.raises(ValueError, match="integrity"):
        artifacts.load_artifact(artifact_dir)
