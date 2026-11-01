from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CACHE_ROOT = PROJECT_ROOT / ".cache" / "huggingface"
MAX_PROMPT_LENGTH = 2_000
MAX_AUDIO_TEXT_LENGTH = 500
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
RAG_MAX_UPLOAD_BYTES = int(os.getenv("RAG_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
RAG_UPLOAD_CHUNK_SIZE = int(os.getenv("RAG_UPLOAD_CHUNK_SIZE", str(1024 * 1024)))
RAG_UPLOAD_ROOT = PROJECT_ROOT / "uploads" / "rag"
RAG_MAX_PDF_PAGES = int(os.getenv("RAG_MAX_PDF_PAGES", "500"))
RAG_MAX_EXTRACTION_CHARS = int(os.getenv("RAG_MAX_EXTRACTION_CHARS", "5000000"))
RAG_EXTRACTION_TIMEOUT_SECONDS = int(os.getenv("RAG_EXTRACTION_TIMEOUT_SECONDS", "60"))
RAG_CHUNK_SIZE_TOKENS = int(os.getenv("RAG_CHUNK_SIZE_TOKENS", "512"))
RAG_CHUNK_OVERLAP_TOKENS = int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", "64"))
RAG_EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "4"))
RAG_ALLOW_JINA_REMOTE_CODE = os.getenv(
    "RAG_ALLOW_JINA_REMOTE_CODE", "false"
).lower() == "true"
RAG_ARTIFACT_ROOT = RAG_UPLOAD_ROOT / "artifacts"
RAG_EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v2-base-en"
RAG_EMBEDDING_MODEL_REVISION = "322d4d7e2f35e84137961a65af894fda0385eb7a"
RAG_EMBEDDING_CODE_REVISION = "f3ec4cf7de7e561007f27c9efc7148b0bd713f81"
QDRANT_MODE = os.getenv("QDRANT_MODE", "local").lower()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledgebase")
_raw_qdrant_path = os.getenv("QDRANT_LOCAL_PATH", "./data/qdrant")
QDRANT_LOCAL_PATH = (
    Path(_raw_qdrant_path).resolve()
    if Path(_raw_qdrant_path).is_absolute()
    else (PROJECT_ROOT / _raw_qdrant_path).resolve()
)
QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
RAG_RETRIEVAL_TOP_K = int(os.getenv("RAG_RETRIEVAL_TOP_K", "3"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.7"))
RAG_INDEXING_BATCH_SIZE = int(os.getenv("RAG_INDEXING_BATCH_SIZE", "50"))
CPU_FALLBACK_ENABLED = os.getenv("ALLOW_CPU_FALLBACK", "false").lower() == "true"
os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "20")

for directory in (
    CACHE_ROOT,
    OUTPUT_ROOT / "text",
    OUTPUT_ROOT / "audio",
    OUTPUT_ROOT / "images",
    OUTPUT_ROOT / "videos",
    OUTPUT_ROOT / "models_3d",
):
    directory.mkdir(parents=True, exist_ok=True)
