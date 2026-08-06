# Visual Multimodal RAG Agent Baseline

**Language / 语言:** [中文](README.md) | [English](README_EN.md)

This is a baseline implementation of a visual multimodal RAG agent. It uploads knowledge documents, analyzes images with YOLO object detection, retrieves document evidence, and generates traceable answers from visual results plus retrieved context.

> Note: YOLO weight files are not included in this repository. You need to provide your own trained model, such as `best.pt`, and configure `YOLO_MODEL` in `.env`.

## Features

- FastAPI backend
- YOLO image object detection with labels, confidence scores, and bounding boxes
- Document upload, text extraction, and chunking
- Milvus Lite vector search with local JSON fallback
- OpenAI-compatible Chat API, ready for DeepSeek-style usage
- Simple web demo for document upload, image upload, and question answering
- Answers include visual detections and retrieved document citations

## Project Structure

```text
app/
  main.py              FastAPI entrypoint
  config.py            Environment and path settings
  schemas.py           API schemas
  services/
    vision.py          YOLO detection
    documents.py       Document loading and chunking
    retriever.py       Milvus Lite / local vector retrieval
    llm.py             Chat API and local embedding fallback
    rag.py             RAG answer flow
web/                   Demo page
samples/               Example safety rules
data/                  Local runtime data, ignored by git
```

## Quick Start

```powershell
cd <PROJECT_ROOT>
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Configure your API key and YOLO model path in `.env`:

```env
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-chat
EMBEDDING_MODEL=local

YOLO_MODEL=C:\path\to\your\best.pt
YOLO_CONFIDENCE=0.25
```

Start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the web demo:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Demo Flow

1. Upload `samples\safety_rules.txt`.
2. Upload a test image.
3. Run image detection and inspect labels, confidence scores, and bounding boxes.
4. Ask a question such as: `Is there any safety issue in this image? What is the evidence?`
5. Review the RAG answer, detections, and retrieved citations.

## YOLO Model Requirement

The repository does not ship with model weights. Prepare or train your own YOLO model and configure it in `.env`:

```env
YOLO_MODEL=C:\path\to\your\best.pt
```

For helmet safety detection, recommended classes include:

```text
head
helmet
person
```

The generic `yolov8n.pt` model only supports COCO classes and usually cannot detect safety helmets accurately.

## Useful Commands

Check backend health:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

Stop the backend on port 8000:

```powershell
$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    $conn | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
        Stop-Process -Id $_
    }
}
```

Inspect YOLO model classes:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; m=YOLO(r'C:\path\to\your\best.pt'); print(m.names)"
```

## Roadmap

- `v0.1`: Baseline with visual detection, RAG retrieval, API answering, and web demo.
- `v0.2`: Better prompting, UI refinements, error handling, and evaluation scripts.
- `v0.3`: LangGraph orchestration, Neo4j knowledge graph, and richer traceability.
