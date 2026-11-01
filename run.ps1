$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv-genai\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Project venv not found. Create it and install requirements first."
}
$env:HF_HOME = Join-Path $ProjectRoot ".cache\huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:ALLOW_CPU_FALLBACK = if ($env:ALLOW_CPU_FALLBACK) { $env:ALLOW_CPU_FALLBACK } else { "false" }
$env:RAG_ALLOW_JINA_REMOTE_CODE = if ($env:RAG_ALLOW_JINA_REMOTE_CODE) { $env:RAG_ALLOW_JINA_REMOTE_CODE } else { "true" }
& $Python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
