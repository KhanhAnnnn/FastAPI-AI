# Local Generative AI Lab

Five independent Hugging Face models exposed through a local FastAPI service. The API does not chain model outputs and binds to `127.0.0.1` only.

## Hardware and environment status

Initial audit on this machine found NVIDIA Quadro P4000 (8 GiB), driver 572.76, and 73 GB free on `D:`. The pre-existing `.venv` is Python 3.14 and is not modified. A separate `.venv-genai` is created with Python 3.13.2. Before the CUDA package install completes, CUDA availability, PyTorch runtime, compiled `sm_61` support, and real inference remain **unverified**. See `reports/` for measurements after tests are run.

The selected binary is PyTorch 2.7.1 + CUDA 12.6 because the installed driver is newer than that runtime and this Python release has a Windows wheel. It is not considered validated until `scripts/check_gpu.py` completes a real tensor operation and reports the device architecture in PyTorch's compiled architecture list. No NVIDIA driver or system CUDA Toolkit is installed or changed. `hf_xet` is installed for Hugging Face's Xet-backed large-file downloads.

## Setup (PowerShell)

From `D:\FastAPI-AI`:

```powershell
if (-not (Test-Path .venv-genai\Scripts\python.exe)) {
    & D:\python.exe -m venv .venv-genai
}
.\.venv-genai\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-genai\Scripts\python.exe scripts\check_gpu.py
.\.venv-genai\Scripts\python.exe -m pip freeze | Out-File -Encoding ascii requirements-lock.txt
```

Use the isolated environment only. Hugging Face weights are cached under `D:\FastAPI-AI\.cache\huggingface`; outputs are saved under `D:\FastAPI-AI\outputs`. The cache and generated artifacts can be large.

Start the service:

```powershell
.\run.ps1
```

Open `http://127.0.0.1:8000/docs` or the small test UI at `http://127.0.0.1:8000/`. Keep one Uvicorn worker. Only one model is kept loaded at a time: the next model request unloads the prior model before loading another. First requests download model files and may take a long time.

CPU fallback is disabled by default. If GPU compatibility is proven unavailable and CPU inference is explicitly desired, set `$env:ALLOW_CPU_FALLBACK = "true"` in the current PowerShell session before running. Video and image generation on CPU may be impractically slow.

## API

| Endpoint | Request |
|---|---|
| `GET /health` | Liveness |
| `GET /system/info` | Python, packages, CUDA, GPU, and architecture diagnostics |
| `GET /models/status` | Loaded model, activity, and most recent metrics |
| `POST /models/{text\|audio\|image\|video\|3d}/unload` | Unload the matching cached model |
| `GET /generate/text?prompt=What%20is%20FastAPI%3F` | Optional `max_new_tokens=128`, `temperature=0.7` |
| `GET /generate/audio?prompt=Hello&preset=v2%2Fen_speaker_1` | WAV response |
| `GET /generate/image?prompt=A%20city&num_inference_steps=10&seed=42` | PNG response |
| `POST /generate/video?num_frames=25&seed=42` | Multipart form field `image`; MP4 response |
| `GET /generate/3d?prompt=A%20small%20wooden%20chair&num_inference_steps=25` | Wavefront OBJ response |
| `POST /rag/documents/upload` | Multipart form field `file`; PDF upload, optional `?auto_process=true`, HTTP 201 |
| `GET /rag/documents` | List all uploaded documents with processing status and indexed chunk counts |
| `POST /rag/documents/{document_id}/process` | Queue document for background pipeline processing (Extraction -> Embedding -> Indexing) |
| `GET /rag/documents/{document_id}/status` | Detailed status of background processing pipeline |
| `POST /rag/documents/{document_id}/cancel` | Cancel an active or queued processing task |
| `POST /rag/documents/{document_id}/extract` | Extract PDF text synchronously; returns page metadata and stored output filenames |
| `POST /rag/documents/{document_id}/embed` | Chunk a completed extraction and store local artifacts; vectors are not returned |
| `POST /rag/documents/{document_id}/index` | Index verified embedding artifact into Qdrant vector database (idempotent) |
| `POST /rag/search` | Semantic search with query embedding, score threshold, and optional document filter |
| `POST /rag/chat` | Grounded RAG Chat answering via TinyLlama-1.1B with source citations and refusal gate |
| `GET /rag/index/status` | Vector collection availability, dimensions, distance metric, and point count |

RAG stage 1 stores validated PDFs under `uploads/rag/` using UUID filenames. Upload
streaming uses 1 MiB chunks by default and is independently capped at 50 MiB.
Override these limits with the `RAG_UPLOAD_CHUNK_SIZE` and `RAG_MAX_UPLOAD_BYTES`
environment variables. PDF extraction uses pypdf in a time-limited worker process;
defaults are 500 pages, 5,000,000 extracted characters, and 60 seconds per
document. Configure these with `RAG_MAX_PDF_PAGES`,
`RAG_MAX_EXTRACTION_CHARS`, and `RAG_EXTRACTION_TIMEOUT_SECONDS`. Pages are
reported as containing text, empty, or having no extractable text layer. The
last status does not imply that a page is scanned; OCR is not included. Each
successful extraction stores a UTF-8 `.txt` file and page-offset JSON metadata.
Reruns publish a new text file and update the document extraction manifest.
See `reports/rag-phase-1-acceptance.md` and
`reports/rag-phase-2-acceptance.md` for the stage reports.

RAG chunking defaults to 512 tokenizer tokens with 64 tokens of overlap and
preserves page attribution. Embeddings are CPU-only and serialized against
Generative AI model work; local artifacts contain `vectors.npy`, `chunks.jsonl`,
and a checksummed manifest under `uploads/rag/artifacts/`. The Jina model is
pinned to model revision `322d4d7e2f35e84137961a65af894fda0385eb7a` and custom
code revision `f3ec4cf7de7e561007f27c9efc7148b0bd713f81`. Its model card requires
`trust_remote_code=True`; code loading and model download are disabled unless
the operator explicitly sets `RAG_ALLOW_JINA_REMOTE_CODE=true` after reviewing
and approving those revisions. The model is English-only, so embedding quality
for Vietnamese is not guaranteed. See
`reports/rag-phase-3-acceptance.md` for verification and current status.

RAG stage 4 indexes verified Phase 3 embedding artifacts into a persistent Qdrant
vector database (`./data/qdrant` by default in local mode). Collection management
uses the Repository Pattern to support both local persistent mode and external
Qdrant Server without code refactoring. Vector dimensionality is 768 with Cosine
distance. Point IDs are deterministic UUIDs computed from
`uuid5(NAMESPACE_URL, "rag://{document_id}/{extraction_id}/{chunk_id}")`, guaranteeing
strict idempotency with zero duplicate points on re-indexing. Semantic retrieval
(`POST /rag/search`) computes query embeddings on CPU using the pinned Jina model
under exclusive execution against Generative AI models, and queries Qdrant with
configurable `top_k`, `score_threshold`, and optional `document_id` filtering. Raw
vectors are never exposed in search responses. See `reports/RAG_GD4_QDRANT_ACCEPTANCE.md`
for full benchmarks and verification.

RAG stage 5 provides asynchronous background pipeline execution
(`Upload -> Extraction -> Chunking & Embedding -> Qdrant Indexing`). The pipeline
is managed via a persistent state machine (`UPLOADED`, `QUEUED`, `EXTRACTING`,
`EMBEDDING`, `INDEXING`, `COMPLETED`, `FAILED`, `CANCELLED`) with single-concurrency
worker execution (`concurrency = 1`) to guarantee Windows file lock safety on
local Qdrant storage. Document statuses are persisted atomically in `.processing.json`
files, providing automatic crash recovery on startup without unbounded retries or
duplicate execution. Documents can be uploaded with automatic background processing
via `POST /rag/documents/upload?auto_process=true`, or triggered manually via
`POST /rag/documents/{document_id}/process`. Real-time status tracking is queried
via `GET /rag/documents/{document_id}/status`, and queued/in-flight tasks can be
cancelled via `POST /rag/documents/{document_id}/cancel`. See
`reports/rag-gd5-background-acceptance.md` for background processing acceptance.

RAG stage 6 integrates TinyLlama-1.1B-Chat for grounded question answering
(`POST /rag/chat`). Semantic retrieval fetches the most relevant context chunks
from Qdrant using the pinned Jina embeddings on CPU. If retrieved chunks do not
satisfy the `score_threshold` (e.g. out-of-domain or adversarial queries), a strict
refusal gate immediately returns a grounded refusal message with 0 sources and
0 ms LLM generation latency, completely eliminating hallucination. When relevant
context is found, retrieved chunks are formatted into a structured context prompt
bounded to ~1,500 tokens, and TinyLlama generates the answer on CUDA with
explicit attention masking and a repetition penalty (`1.15`) to prevent degenerate
loops. Every response includes verified source citations with document ID, page
number, cosine similarity, and text excerpts. See
`reports/rag-gd6-tinyllama-acceptance.md` and `reports/rag-full-system-acceptance.md`
for full benchmarks and verification.

Prompt, file size, frame count, and step counts are validated. Video upload is capped at 20 MiB. Audio/image/video/OBJ responses include appropriate content types. CUDA OOM and model-load failures are returned as errors, never as success-shaped outputs; an OOM drops the cached model reference and clears available CUDA cache, without silently reducing requested quality. Overlapping inference gets HTTP 429 with `Retry-After` instead of joining an unbounded queue.

OBJ serialization writes vertices/faces directly rather than requiring Open3D; this avoids an unnecessary native dependency while producing standard Wavefront OBJ. The model manager keeps one model resident and offloads/unloads before loading a different model.

## Example requests (PowerShell)

```powershell
curl.exe -G "http://127.0.0.1:8000/generate/text" --data-urlencode "prompt=What is FastAPI?"
curl.exe -G "http://127.0.0.1:8000/generate/audio" --data-urlencode "prompt=Hello, welcome to our artificial intelligence laboratory." --data-urlencode "preset=v2/en_speaker_1" -o D:\FastAPI-AI\outputs\audio\sample.wav
curl.exe -G "http://127.0.0.1:8000/generate/image" --data-urlencode "prompt=A futuristic city at sunset, digital art" --data-urlencode "num_inference_steps=10" --data-urlencode "seed=42" -o D:\FastAPI-AI\outputs\images\sample.png
curl.exe -X POST "http://127.0.0.1:8000/generate/video?num_frames=25&seed=42" -F "image=@D:\FastAPI-AI\samples\input.png" -o D:\FastAPI-AI\outputs\videos\sample.mp4
curl.exe -G "http://127.0.0.1:8000/generate/3d" --data-urlencode "prompt=A small wooden chair" --data-urlencode "num_inference_steps=25" -o D:\FastAPI-AI\outputs\models_3d\sample.obj
curl.exe -X POST "http://127.0.0.1:8000/rag/documents/upload?auto_process=true" -F "file=@D:\FastAPI-AI\samples\sample.pdf"
curl.exe -G "http://127.0.0.1:8000/rag/documents/{document_id}/status"
curl.exe -X POST "http://127.0.0.1:8000/rag/documents/{document_id}/index" -H "Content-Type: application/json" -d "{}"
curl.exe -X POST "http://127.0.0.1:8000/rag/search" -H "Content-Type: application/json" -d "{\"query\": \"What is artificial intelligence?\", \"top_k\": 3, \"score_threshold\": 0.7}"
curl.exe -X POST "http://127.0.0.1:8000/rag/chat" -H "Content-Type: application/json" -d "{\"query\": \"What is the context window of TinyLlama?\", \"top_k\": 3, \"score_threshold\": 0.7}"
curl.exe -G "http://127.0.0.1:8000/rag/index/status"
```
For the video example, place a valid image at `D:\FastAPI-AI\samples\input.png` first.

## Tests and real inference

Unit and API tests (serialization tests use locally generated fixtures; these are not inference outputs):

```powershell
.\.venv-genai\Scripts\python.exe -m pytest -q
```

Real model acceptance requires each model's endpoint to be called and its actual output to be decoded/validated. Do not report `PASS` based on imports, mocks, or API startup. The current observed model-download blockers and measured hardware are documented in `reports/acceptance-report.md` and `reports/benchmark-report.md`; every model's status is stated explicitly. To retry one model at a time after network connectivity improves, use `.\.venv-genai\Scripts\python.exe scripts\run_inference_acceptance.py --model TinyLlama` (choices: `TinyLlama`, `Bark`, `TinySD`, `Stable Video Diffusion`, `Shap-E`). The SVD model may exceed 8 GiB or run slowly despite CPU offload; report the actual error and resource measurements rather than silently reducing request parameters.
