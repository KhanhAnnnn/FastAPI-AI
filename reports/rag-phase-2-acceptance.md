# RAG Stage 2 — PDF Text Extraction

## Acceptance result

**PASS for the implemented extraction contract and tested fixtures.**

The actual FastAPI endpoint extracted text with `pypdf` from generated, valid
PDF files, preserved Vietnamese Unicode, returned correct page counts and
statuses, and wrote UTF-8 text plus JSON page metadata. Malformed and encrypted
PDFs, unknown IDs, extraction limits, and worker timeouts return explicit
errors. The upload endpoint and legacy route registration tests also pass.

This is not a claim that `pypdf` can extract every PDF. There is no OCR. Pages
from which pypdf returns no text are reported as `no_text_layer` when they
contain page content; this does not assert that such a page is scanned. Blank
pages are reported as `empty`. Password-protected PDFs are unsupported because
the endpoint does not accept or store passwords.

No embedding, vector store, background indexing, retrieval, or model inference
was implemented or run.

## Dependency and environment

- Project interpreter: Python 3.13.2, `.venv-genai`.
- Dependency added: `pypdf==6.20.0`.
- `pypdf` was absent before this phase; `pip index versions pypdf` listed
  6.20.0 as the current release, and installation in the actual Python 3.13.2
  project environment succeeded.
- Disk check before install: D: had 52.64 GiB free.
- Only `pypdf` was installed. No global Python, PyTorch, CUDA, driver, or
  virtual-environment changes were made.
- `requirements.txt` and `requirements-lock.txt` now pin the dependency.
- Dependency consistency check: `pip check` reported `No broken requirements
  found.`

## API contract

`POST /rag/documents/{document_id}/extract`

- `document_id` must be a UUID for a PDF under the configured RAG upload root.
  No user-supplied path is opened.
- Extraction uses pypdf with `strict=True` in a separate Python worker process.
- Success returns `200` with document ID, source filename when known, total
  pages, pages with/without text, extraction ID, versioned `.txt` filename,
  metadata filename, and per-page status/count/text offsets. Extracted document
  content is not returned in the response.
- Status values:
  - `text`: pypdf extracted non-whitespace text.
  - `empty`: page has no content stream or its decoded content stream is empty.
  - `no_text_layer`: page has content but pypdf extracted no text. This does not
    distinguish a scan from drawings or other non-text content.
- Per-page text offsets use Python Unicode-character offsets into the UTF-8
  text file after decoding, not byte offsets.
- Errors: `400` malformed PDF or invalid storage path; `404` unknown/malformed
  document ID; `408` worker timeout; `409` same-document extraction already in
  progress; `413` page/character limit exceeded; `422` encrypted PDF or page
  extraction failure; `500` metadata, configuration, worker, or output-storage
  failure.
- Defaults: inherited 50 MiB upload ceiling, 500 PDF pages, 5,000,000 extracted
  characters, and a 60-second worker timeout. Extraction settings can be
  changed with `RAG_MAX_PDF_PAGES`, `RAG_MAX_EXTRACTION_CHARS`, and
  `RAG_EXTRACTION_TIMEOUT_SECONDS`.
- Each run gets a new `<document UUID>.<extraction UUID>.txt` file. The
  `<document UUID>.extraction.json` manifest atomically points to the latest
  completed run. Earlier extraction files are retained.
- Upload registration now stores a sidecar with the sanitized original
  filename. For PDFs uploaded before this sidecar was introduced, the API
  returns `original_filename: null`; it still extracts from the UUID-named PDF.

## PDF fixtures and actual extraction results

Fixtures in `D:\FastAPI-AI\tests\test_rag_extraction.py` are constructed as
actual PDF bytes with object tables, page trees, streams, and Unicode ToUnicode
maps. The test suite exercises:

| Fixture | Origin and purpose | Result |
|---|---|---|
| One-page PDF with text | Generated in test code; validates a real text layer | PASS |
| Three-page PDF | Generated in test code; validates page count and page text offsets | PASS |
| Vietnamese PDF text | Generated in test code with a ToUnicode map; validates UTF-8 text | PASS |
| Empty page | Empty content stream; verifies `empty` classification | PASS |
| Page with vector content but no text | Generated drawing stream; verifies `no_text_layer` without claiming scan | PASS |
| Malformed PDF bytes | Deliberately truncated/invalid test input | HTTP 400 |
| Encrypted PDF | Valid fixture encrypted with pypdf test writer | HTTP 422 |
| Page/character limit fixtures | Generated valid PDFs, worker limits set for test | HTTP 413 |
| Timeout fixture | Generated valid PDF, worker timeout shortened for test | HTTP 408 |

An additional endpoint acceptance run used a generated 4-page PDF containing
two text pages, one empty page, and one vector-only page. Raw measured results:

```text
HTTP status: 200
Source PDF size: 3355 bytes
Total pages: 4
Pages with text: 2
Pages without text: 2
Page statuses: text, text, empty, no_text_layer
Extracted UTF-8 characters: 81
Metadata page records match API page records: true
Source PDF SHA-256: a678ad53c03492acbdddb9d1bd3ad45f81c88194a71adddf59749ab22f93469a
Extracted TXT SHA-256: ebe5d71ca40f97ffbb5503a148aab2bd28118f2ecf5da6ec161575b88a69c322
```

The source PDF was stored in a temporary acceptance directory; no existing
uploaded document or model data was modified.

## Files created or changed

- `D:\FastAPI-AI\app\config.py`
- `D:\FastAPI-AI\app\rag\router.py`
- `D:\FastAPI-AI\app\rag\upload_service.py`
- `D:\FastAPI-AI\app\rag\extraction_schemas.py`
- `D:\FastAPI-AI\app\rag\extraction_service.py`
- `D:\FastAPI-AI\app\rag\pdf_worker.py`
- `D:\FastAPI-AI\tests\test_rag_extraction.py`
- `D:\FastAPI-AI\requirements.txt`
- `D:\FastAPI-AI\requirements-lock.txt`
- `D:\FastAPI-AI\.env.example`
- `D:\FastAPI-AI\README.md`
- `D:\FastAPI-AI\reports\rag-phase-2-acceptance.md`

The existing upload tests in `D:\FastAPI-AI\tests\test_rag_upload.py` were run
as regression tests and were not modified for this phase.

## Test command and raw output

Command, using the existing project environment:

```powershell
.\.venv-genai\Scripts\python.exe -m pytest -q
```

Raw output:

```text
.........................................                                [100%]
============================== warnings summary ===============================
.venv-genai\Lib\site-packages\fastapi\testclient.py:1
  D:\FastAPI-AI\.venv-genai\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
41 passed, 1 warning in 7.37s
```

The warning is from the pre-existing FastAPI/Starlette test-client integration;
no unrelated package was installed to suppress it.

Dependency validation:

```text
requirements-lock.txt:37:pypdf==6.20.0
No broken requirements found.
```

## Remaining limitations

- pypdf text extraction is not OCR and does not extract text from image-only
  pages.
- Extraction is limited by compressed PDF upload size, page count, extracted
  character count, and worker wall-clock timeout. A separate operating-system
  memory quota is not configured for the worker.
- Existing uploads without a Stage 1 upload-metadata sidecar cannot recover
  their original client filename; their source remains safely addressable by
  document UUID.
- The per-document in-process lock relies on the project's single Uvicorn
  worker configuration. Do not run multiple Uvicorn workers against this
  storage implementation if cross-worker same-document exclusion is required.
- This stage does not persist extracted text into a database or vector store.

## Out of scope

No text chunking, Jina embeddings, Qdrant, background jobs, retrieval, OCR, or
TinyLlama RAG integration was added. Work beyond PDF extraction requires an
explicit subsequent request.
