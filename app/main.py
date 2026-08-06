from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.schemas import AskResponse, DocumentUploadResponse, HealthResponse, ImageAnalysisResponse
from app.services.documents import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    DocumentProcessingError,
    extract_text,
    safe_filename,
    split_text,
)
from app.services.llm import LLMClient
from app.services.rag import RAGService
from app.services.retriever import RetrieverService
from app.services.vision import VisionService


settings = get_settings()
llm_client = LLMClient(settings)
vision_service = VisionService(settings)
retriever_service = RetrieverService(llm_client, settings)
rag_service = RAGService(llm_client, retriever_service, vision_service, settings)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(settings.web_dir)), name="static")
app.mount("/uploads", StaticFiles(directory=str(settings.uploads_dir)), name="uploads")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(settings.web_dir / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        store_mode=retriever_service.store_mode,
        model_configured=llm_client.is_configured,
    )


@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported document type. Allowed: {allowed}")

    saved_path = await _save_upload(file, settings.document_upload_dir)

    try:
        text = extract_text(saved_path)
        chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in the uploaded document.")

    result = await retriever_service.add_chunks(saved_path.name, chunks)
    return DocumentUploadResponse(
        document_id=result.document_id,
        filename=saved_path.name,
        chunks=result.chunks,
        characters=result.characters,
        store_mode=result.store_mode,
        warnings=result.warnings,
    )


@app.post("/api/images/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(image: UploadFile = File(...)) -> ImageAnalysisResponse:
    saved_path = await _save_upload(image, settings.image_upload_dir)
    return vision_service.analyze(saved_path, saved_path.stem, saved_path.name)


@app.post("/api/ask", response_model=AskResponse)
async def ask(
    question: str = Form(...),
    image: UploadFile | None = File(None),
    top_k: int = Form(default=5),
) -> AskResponse:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    image_path: Path | None = None
    if image is not None and image.filename:
        image_path = await _save_upload(image, settings.image_upload_dir)

    bounded_top_k = max(1, min(top_k, 10))
    return await rag_service.answer(
        question=question.strip(),
        image_path=image_path,
        image_id=image_path.stem if image_path else None,
        filename=image_path.name if image_path else None,
        top_k=bounded_top_k,
    )


async def _save_upload(file: UploadFile, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    original = safe_filename(file.filename, "upload")
    suffix = Path(original).suffix
    stem = Path(original).stem or "upload"
    filename = f"{uuid.uuid4().hex}_{stem}{suffix}"
    path = directory / filename

    with path.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    return path
