# RAG Stage 1 — PDF Upload & File Storage

## Result

**Upload and storage tests: PASS.** The application accepts a PDF upload at
`POST /rag/documents/upload`, streams it into a temporary file in bounded chunks,
checks the configured byte limit, validates basic PDF structure, and atomically
publishes it under a UUID-based filename.

**Full phase acceptance: BLOCKED.** This environment has no PDF parsing library
installed. Validation checks the `.pdf` extension, PDF header, `startxref`,
cross-reference table or stream marker, trailer/root marker where applicable,
and `%%EOF`; it does not parse every indirect object or prove that a PDF can be
rendered. The existing virtual environment and dependency set were not changed,
so full parser-based validation remains unverified.

No model was loaded, downloaded, or run. No inference implementation was
modified.

## Files created or changed

- `D:\FastAPI-AI\app\config.py`
- `D:\FastAPI-AI\app\main.py`
- `D:\FastAPI-AI\app\rag\__init__.py`
- `D:\FastAPI-AI\app\rag\router.py`
- `D:\FastAPI-AI\app\rag\schemas.py`
- `D:\FastAPI-AI\app\rag\storage.py`
- `D:\FastAPI-AI\app\rag\upload_service.py`
- `D:\FastAPI-AI\tests\test_rag_upload.py`
- `D:\FastAPI-AI\.env.example`
- `D:\FastAPI-AI\.gitignore`
- `D:\FastAPI-AI\README.md`
- `D:\FastAPI-AI\reports\rag-phase-1-acceptance.md`

No `AGENTS.md` was found under the project directory.

## API contract implemented

`POST /rag/documents/upload`

- Request: `multipart/form-data`, required field `file`.
- Success: HTTP `201`, JSON response:

  ```json
  {
    "document_id": "uuid",
    "filename": "example.pdf",
    "stored_filename": "uuid.pdf",
    "size_bytes": 123456,
    "content_type": "application/pdf",
    "status": "uploaded"
  }
  ```

- `400`: empty, unsupported, or structurally invalid PDF.
- `413`: uploaded file exceeds `RAG_MAX_UPLOAD_BYTES`.
- `422`: missing `file` field or invalid multipart request.
- `500`: storage I/O failure; response does not include the storage path.
- Default storage directory: `D:\FastAPI-AI\uploads\rag`.
- Default per-file maximum: 52,428,800 bytes (50 MiB).
- Default read/write chunk size: 1,048,576 bytes (1 MiB).
- Configuration: `RAG_MAX_UPLOAD_BYTES` and `RAG_UPLOAD_CHUNK_SIZE`.
- Stored files use `<UUID>.pdf`; original filenames are sanitized for response
  display only and are not used in storage paths. Temporary files are removed
  on rejected or interrupted uploads.

## Required verification

| Check | Result |
|---|---|
| Valid PDF upload returns HTTP 201 | PASS |
| Saved file exists on disk and has the exact uploaded bytes | PASS |
| SHA-256 of upload and saved file match | PASS |
| Non-PDF and invalid PDF structure rejected | PASS |
| Empty upload rejected | PASS |
| Configured size limit returns HTTP 413 | PASS |
| Repeated original filenames receive distinct stored names | PASS |
| Traversal and unsafe filename are sanitized | PASS |
| Multi-chunk upload preserves bytes | PASS |
| Interrupted stream returns storage failure | PASS |
| No temporary file remains after interrupted stream | PASS |
| `/health` and the five existing generation routes remain registered | PASS |
| Inference outputs from existing model endpoints | NOT RUN (explicitly out of scope) |
| Complete PDF semantic parsing/rendering | BLOCKED (no parser installed) |
| Model inference behavior unchanged | No inference code was edited; no inference run |

The integration tests use a dedicated temporary directory, not the production
upload directory.

## SHA-256 evidence

The integration test compared the complete source fixture bytes against the
saved file and compared their SHA-256 digests. The fixture was 323 bytes.

```text
Upload SHA-256: 583d0d4a563f1221b9620f02954230c9873162d565c2a7c7f7e3e93a2b8cb252
Saved file SHA-256: 583d0d4a563f1221b9620f02954230c9873162d565c2a7c7f7e3e93a2b8cb252
```

## Test command and raw output

Command, run using the existing `.venv-genai`:

```powershell
.\.venv-genai\Scripts\python.exe -m pytest -q
```

Raw output:

```text
........................                                                 [100%]
============================== warnings summary ===============================
.venv-genai\Lib\site-packages\fastapi\testclient.py:1
  D:\FastAPI-AI\.venv-genai\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
24 passed, 1 warning in 3.81s
```

No dependency was installed to silence the pre-existing Starlette/httpx
deprecation warning.

## Remaining limitations

- PDF validation is structural marker validation, not complete parsing or
  rendering validation.
- Documents are stored on disk only; there is no metadata database, listing,
  deletion, authentication, or deduplication.
- The application-level size cap is enforced while copying the multipart
  `UploadFile`; the framework may spool the incoming multipart body before this
  endpoint processes it.
- Inference endpoint behavior was not exercised, to avoid running model
  inference.

## Proposed Stage 2 work (not implemented)

1. Add parser-backed PDF extraction with clear behavior for malformed, encrypted,
   and image-only PDFs.
2. Define document metadata and lifecycle persistence.
3. Add deterministic text chunking and source/page metadata.
4. Select and configure an embedding model and vector store (including Qdrant
   only after explicit design approval).
5. Add retrieval/query endpoints and evaluate retrieval quality.
6. Integrate retrieved context with TinyLlama only after the user explicitly
   requests Stage 2.
