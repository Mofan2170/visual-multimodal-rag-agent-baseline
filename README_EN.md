# Visual Multimodal RAG Agent

**Language / 语言:** [中文](README.md) | [English](README_EN.md)

A generalized visual multimodal RAG workbench that runs locally. Configure any OpenAI-compatible Chat API, select your own YOLO `.pt` model, upload rule documents and images, and answer questions with traceable YOLO detections and retrieved evidence.

Current release: `v0.2.0`

> YOLO weights are not included. Use a model you trained or obtained from a trusted source, such as `best.pt`. PyTorch `.pt` files may execute code during deserialization, so only load trusted files locally.

## Features

- FastAPI backend and a single-page web workbench
- Runtime API Key, Base URL, Chat Model, and Embedding Model configuration
- Runtime API keys stay in backend process memory and are never written to `.env`
- Local YOLO path selection and trusted `.pt` upload
- Detection labels, confidence scores, bounding boxes, label counts, and model classes
- `.txt`, `.md`, and `.pdf` rule ingestion with chunking and content deduplication
- Milvus Lite retrieval with a Local JSON fallback
- Separate Milvus collections per embedding dimension with automatic local-record synchronization
- Reusable image analysis through `image_id`, avoiding duplicate uploads and inference
- Safely rendered Markdown answers with visual evidence, citations, sources, scores, and uncertainty notes

## What's New in v0.2.0

- Runtime configuration for any OpenAI-compatible Chat API, kept only in process memory
- Dynamic YOLO `.pt` switching through a local path or trusted file upload
- Generalized prompts and RAG logic with no hard-coded helmet scenario
- Document deduplication, encoding-corruption rejection, Milvus Lite fixes, and Local JSON fallback
- Reusable detections, class counts, citation excerpts, and structured Markdown answers
- Limits for upload size, PDF pages, extracted text, image pixels, and question length
- Localhost-only access by default, same-origin checks, browser security headers, and CDN integrity checks
- CI coverage for Ruff, unit tests, frontend syntax, dependency consistency, and vulnerability auditing

## Quick Start

```powershell
cd <PROJECT_ROOT>
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Skip virtual-environment creation if `.venv` already exists. Open:

- Web workbench: `http://127.0.0.1:8000`
- API documentation: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

Copying `.env.example` to `.env` is optional and provides startup defaults. The API and model can also be configured directly in the web page.

## API Configuration

`.env` can provide startup defaults, or you can enter temporary values in the web runtime form:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-api-provider-base-url
CHAT_MODEL=your-chat-model
EMBEDDING_MODEL=local
```

The project does not require a specific provider. Any service compatible with OpenAI Chat Completions can be used, for example:

```env
# DeepSeek example
OPENAI_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-chat

# OpenAI example
OPENAI_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4o-mini
```

`EMBEDDING_MODEL=local` is recommended by default because some Chat API providers do not expose an embeddings endpoint. When a remote embedding model is configured, the app calls `<BASE_URL>/embeddings`; failures are reported and use the local fallback.

The “API values entered” status means only that the backend holds the configuration. The key, URL, and model are validated on the first question request.

## YOLO Model

The web page supports two loading methods:

- Enter a local path such as `C:\path\to\your\best.pt`
- Upload a trusted `.pt` file into the local `data/models/` runtime directory

Different models can serve construction safety, traffic monitoring, shelf inspection, industrial quality control, and other scenarios. YOLO classes define what the app can see, while uploaded rules define how detections are interpreted. The code is not tied to helmet detection.

## Demo Flow

1. Enter API settings, or leave the API key empty to use the local fallback answer.
2. Select your YOLO `.pt` model and confirm that its classes appear.
3. Upload a rule document such as `samples\safety_rules.txt`.
4. Upload an image, run detection, and inspect boxes and label counts.
5. Enter a question and run the workflow.
6. Review the answer, visual evidence, citations, sources, and warnings.

Uploading identical document content skips duplicate vector ingestion. Switching the YOLO model invalidates previous image-analysis cache entries, so the image must be analyzed again.

## Local Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m compileall -q app
.\.venv\Scripts\ruff.exe check app tests
node --check web\app.js
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip_audit -r requirements.txt
```

Tests cover document processing, unique chunk IDs, multi-chunk Milvus retrieval, embedding-dimension switching, and document deduplication. GitHub Actions configuration is provided in `.github/workflows/ci.yml`.

## Data and Limitations

- Documents, images, models, and vectors are stored under `data/` and ignored by Git.
- Default upload limits are 20 MB for documents, 20 MB for images, and 500 MB for models. They are configurable in `.env`.
- Default content limits are 2 million document characters, 500 PDF pages, 40 million image pixels, and 8,000 question characters.
- Runtime configuration and image-analysis cache live in one backend process and reset to `.env` defaults after restart.
- The built-in hashing embedding is suitable for a baseline demo, not production-grade semantic retrieval.

## Security Notes

- `ALLOW_REMOTE_ACCESS=false` is the default, so only local requests are accepted. Start Uvicorn on `127.0.0.1`.
- This release has no user authentication, authorization, or tenant isolation and must not be exposed directly to the public internet.
- `.env`, uploads, vector data, and `.pt` weights are ignored by Git; a secret scan is still recommended before each release.
- A `.pt` weight can contain executable deserialization payloads. Load only models you trained or explicitly trust.
- Base URLs must use HTTP/HTTPS and cannot contain embedded credentials, query strings, or fragments.
- Markdown is sanitized with DOMPurify before insertion; third-party scripts are version-pinned and protected with SRI.

## Stop the Backend

Press `Ctrl+C` in the terminal running Uvicorn, or use PowerShell:

```powershell
$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    $conn | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
        Stop-Process -Id $_
    }
}
```

## Roadmap

- `v0.1`: Visual detection, RAG retrieval, API answering, and web demo baseline.
- `v0.2.0`: General runtime configuration, YOLO switching, deduplication, retrieval fixes, security hardening, and generic rule-based RAG.
- `v0.3`: LangGraph orchestration, Neo4j knowledge graph, conversation management, and a broader evaluation suite.
