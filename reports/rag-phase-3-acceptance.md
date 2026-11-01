# RAG Phase 3 Acceptance — Text Chunking & Vector Embedding

## Overall status: PASS

The chunking, embedding integration, API, local artifact round-trip, full
automated test suite, and a real Jina CPU inference through the embed endpoint
were verified. Real inference was run only after explicit approval to download
and execute the pinned remote-code revision.

## Scope

Implemented only RAG Phase 3. No Qdrant, retrieval, background indexing,
OCR, or TinyLlama integration was added. No existing Generative AI model
inference implementation was changed.

## Environment and dependency findings

- Python: `3.13.2`
- PyTorch: `2.7.1+cu126`
- Transformers: `4.57.3`
- Tokenizers: `0.22.2`
- NumPy: `2.2.6`
- CUDA availability probe: `True` (read-only probe; no model inference)
- GPU: NVIDIA Quadro P4000, 8192 MiB total, 5437 MiB free at measurement
- Host RAM: 63.83 GiB total, 45.24 GiB free at measurement
- Drive D: 56,512,782,336 bytes free at measurement
- No Phase 3 package was installed and no dependency manifest or lockfile was
  changed.
- The model and pinned custom code were downloaded and loaded after approval.
- Hugging Face emitted a Windows symlink warning: cache files were stored
  without symlinks under `D:\AI_CACHE\huggingface`, using additional disk
  space. This did not prevent loading.

The model ID is `jinaai/jina-embeddings-v2-base-en`, pinned to model revision
`322d4d7e2f35e84137961a65af894fda0385eb7a` and custom-code revision
`f3ec4cf7de7e561007f27c9efc7148b0bd713f81`. The model card identifies the
model as English-only and requires `trust_remote_code=True`. The loader uses
CPU. The runtime gate remains disabled by default; the explicit opt-in was set
only in the isolated acceptance process. The tested embedding dimension and
context are 768 and 8192 tokens.

The installed CUDA runtime is available, but embedding was deliberately run on
CPU to avoid competing with the existing generative models. CUDA inference and
peak VRAM were not measured. Process RSS was recorded before and after the
request; this is not a peak-memory measurement.

## Implementation summary

- Token-aware chunks are created from per-page extraction metadata, using the
  pinned model tokenizer's fast-tokenizer offset mapping at runtime.
- Defaults: 512 content tokens per chunk, 64-token overlap, batch size 4.
  Chunking preserves page attribution, original and cleaned text, Unicode,
  token count, extraction ID, and SHA-256 of cleaned text. Sentence boundaries
  are preferred when available without exceeding the configured size.
- Runtime checks enforce chunk/context limits, non-empty text, embedding
  dimensions, finite vectors, and non-zero vector norms.
- Model loading is lazy and cached. The existing model manager serializes
  embedding against the existing model work.
- Artifacts are written to a unique versioned directory as `vectors.npy`
  (`float32`), `chunks.jsonl`, `manifest.json`, and `manifest.sha256`.
  The manifest records model/tokenizer revisions, configuration, chunk
  metadata file hashes, and SHA-256 values for the source PDF and extracted
  text. Existing artifacts are not overwritten.
- `POST /rag/documents/{document_id}/embed` only accepts completed extraction
  records, takes no caller-supplied path, and does not return vectors. With
  the default remote-code gate off, it returns HTTP 503 before model loading.
  Missing documents return 404; extraction not ready returns 409; busy model
  work returns 429 with `Retry-After`.

## Files created

- `app/rag/chunking.py`
- `app/rag/artifacts.py`
- `app/rag/embedding_model.py`
- `app/rag/embedding_service.py`
- `app/rag/embedding_schemas.py`
- `tests/test_rag_chunking.py`
- `tests/test_rag_embedding.py`
- `reports/rag-phase-3-acceptance.md`

## Files updated

- `app/config.py`
- `app/core/model_manager.py`
- `app/rag/router.py`
- `.env.example`
- `README.md`

No dependency files were changed. Existing model inference code was not
modified.

## Automated test evidence

Command:

```powershell
.\.venv-genai\Scripts\python.exe -m pytest -q
```

Raw output:

```text
.............................................................            [100%]
============================== warnings summary ===============================
.venv-genai\Lib\site-packages\fastapi\testclient.py:1
  D:\FastAPI-AI\.venv-genai\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
61 passed, 1 warning in 7.68s
```

The passing tests cover chunk boundaries/overlap, long and empty pages,
multi-page attribution, Vietnamese Unicode, context validation, fake-model
pooling and vector validation, API gating/state checks, serialization,
artifact round-trip and integrity rejection, plus the existing test suite.
Mock-based tests are supplemented by the real-model acceptance below.

## Real-model acceptance evidence

Fixture source: locally generated, controlled English PDF using the PDF fixture
builder in `tests/test_rag_extraction.py`; no external or user document was
used. The fixture was parsed by the real Phase 2 extraction worker, then sent
through `POST /rag/documents/{document_id}/embed` with the real pinned model.

Observed result:

| Measurement | Result |
|---|---|
| HTTP status | 200 |
| PDF pages / pages with text | 1 / 1 |
| Extracted character count | 242 |
| Jina tokenizer content token count | 47 |
| Chunk size / overlap | 512 / 64 tokens |
| Chunk count | 1 |
| Model context / embedding dimension | 8192 / 768 |
| Device | CPU |
| Model load time | 24.220319900021423 seconds |
| Embedding latency | 0.1268870000494644 seconds |
| Vector shape / dtype | 1 x 768 / float32 |
| Vector validation | finite and non-zero |
| Process RSS before / after | 472,276,992 / 4,411,109,376 bytes |
| Peak RAM / peak VRAM | Not measured / not applicable to CPU inference |
| Real vectors returned by the API | No |

The API-created artifact was loaded again with `artifacts.load_artifact`; its
vector shape, chunk count, finiteness, manifest hashes, and source hashes were
verified. The retained artifact directory is:

```text
D:\FastAPI-AI\uploads\rag\artifacts\4b98e1f4-b6ac-4a99-af93-b26a28c67f80.0a8aa7c4-0959-40d9-a788-d08dad7a3f54.62af56ff-840b-4dbe-aada-73023b70cac3
```

SHA-256:

```text
source PDF:      db36b6ec66c78862dfc8c93d45af5d16f6b2d6a8de346582ba72e7f122099e27
extracted text:  91d9946e29b854f17bd0eadd33fca332c6e5f4c082ed9cf8616c3fcde8f97fa5
vectors.npy:     f467e66929785c2a8f49fbe56ef29ab3b5e6604cad6cf7ebdcb8a59a844b75ce
chunks.jsonl:    610cf29fbcf18712ecb3aa6df18987268402f4cc18abb2847650c373fb460359
manifest.json:   0073284c826a4305593236f8a06fb55a683da5b65763bbfb412feef1d3a59ac5
```

Raw runtime evidence:

```text
Jina embedding model loaded: model_id=jinaai/jina-embeddings-v2-base-en revision=322d4d7e2f35e84137961a65af894fda0385eb7a code_revision=f3ec4cf7de7e561007f27c9efc7148b0bd713f81 device=cpu context_tokens=8192 dimension=768 load_seconds=24.220
POST /rag/documents/4b98e1f4-b6ac-4a99-af93-b26a28c67f80/embed "HTTP/1.1 200 OK"
```

The acceptance script also verified `HTTP 200`, one extracted page, one chunk,
the hashes above, a `(1, 768)` finite `float32` vector, successful artifact
reload, and that the API response did not contain vectors.

## Limitations and acceptance result

- Jina is an English model; Vietnamese text is preserved by chunking, but
  semantic embedding quality for Vietnamese is not established.
- Scanned pages without a text layer produce no text to chunk; OCR is out of
  scope.
- Peak RAM and VRAM were not measured. The observed after-request RSS was about
  4.11 GB; inference ran on CPU.
- Current status: **PASS** for the implemented Phase 3 scope and this
  single-document real embedding acceptance. The report does not claim broad
  multilingual quality or universal PDF support.

## Proposed next work

No Phase 4 work is included or authorized by this report.
