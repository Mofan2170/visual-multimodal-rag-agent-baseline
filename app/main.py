from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import get_settings
from app.schemas import (
    AskResponse,
    DocumentUploadResponse,
    HealthResponse,
    ImageAnalysisResponse,
    ModelSelectRequest,
    ModelSelectResponse,
    ModelUploadResponse,
    RuntimeConfigRequest,
    RuntimeConfigResponse,
    RuntimeStatusResponse,
)
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    retriever_service.close()


app = FastAPI(title=settings.app_name, version="0.2.0", lifespan=lifespan)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.allow_remote_access else list(settings.allowed_hosts),
)
app.mount("/static", StaticFiles(directory=str(settings.web_dir)), name="static")

SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@app.middleware("http")
async def secure_local_requests(request: Request, call_next) -> Response:
    if not settings.allow_remote_access and not _is_loopback_client(request):
        return _secure_response(
            JSONResponse(
                status_code=403,
                content={"detail": "Remote access is disabled. Use this workbench from localhost."},
            ),
            request.url.path,
        )

    if request.url.path.startswith("/api/") and request.method in {
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
    }:
        origin = request.headers.get("origin")
        if origin and not _is_same_origin(origin, request.headers.get("host", "")):
            return _secure_response(
                JSONResponse(
                    status_code=403, content={"detail": "Cross-origin API request rejected."}
                ),
                request.url.path,
            )

    response = await call_next(request)
    return _secure_response(response, request.url.path)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(settings.web_dir / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    model_info = await run_in_threadpool(vision_service.ensure_model_info)
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        store_mode=retriever_service.store_mode,
        model_configured=llm_client.is_configured,
        yolo_model_loaded=model_info.loaded,
        yolo_model_name=model_info.model_name,
        yolo_classes=model_info.classes,
    )


@app.get("/api/runtime/status", response_model=RuntimeStatusResponse)
async def runtime_status() -> RuntimeStatusResponse:
    return await _runtime_status()


@app.post("/api/runtime/config", response_model=RuntimeConfigResponse)
async def configure_runtime(request: RuntimeConfigRequest) -> RuntimeConfigResponse:
    try:
        return llm_client.configure(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/select", response_model=ModelSelectResponse)
async def select_model(request: ModelSelectRequest) -> ModelSelectResponse:
    try:
        info = await run_in_threadpool(vision_service.reload_model, request.model_path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelSelectResponse(yolo_model=info)


@app.post("/api/models/upload", response_model=ModelUploadResponse)
async def upload_model(file: Annotated[UploadFile, File()]) -> ModelUploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".pt":
        raise HTTPException(
            status_code=400, detail="Only trusted .pt YOLO model files are supported."
        )

    saved_path = await _save_upload(
        file,
        settings.model_upload_dir,
        settings.max_model_upload_mb * 1024 * 1024,
    )
    try:
        info = await run_in_threadpool(vision_service.reload_model, str(saved_path))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ModelUploadResponse(
        filename=saved_path.name,
        model_path=str(saved_path),
        yolo_model=info,
        warnings=["Only upload .pt files that you trained yourself or trust locally."],
    )


@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: Annotated[UploadFile, File()]) -> DocumentUploadResponse:
    source = safe_filename(file.filename, "document")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise HTTPException(
            status_code=400, detail=f"Unsupported document type. Allowed: {allowed}"
        )

    saved_path = await _save_upload(
        file,
        settings.document_upload_dir,
        settings.max_document_upload_mb * 1024 * 1024,
    )

    try:
        text = extract_text(
            saved_path,
            max_characters=settings.max_document_characters,
            max_pdf_pages=settings.max_pdf_pages,
        )
        chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
    except DocumentProcessingError as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not chunks:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail="No readable text found in the uploaded document."
        )

    result = await retriever_service.add_chunks(source, chunks)
    if result.deduplicated:
        saved_path.unlink(missing_ok=True)
    return DocumentUploadResponse(
        document_id=result.document_id,
        filename=result.source,
        chunks=result.chunks,
        characters=result.characters,
        store_mode=result.store_mode,
        deduplicated=result.deduplicated,
        warnings=result.warnings,
    )


@app.post("/api/images/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(image: Annotated[UploadFile, File()]) -> ImageAnalysisResponse:
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported image type. Allowed: {allowed}")
    saved_path = await _save_upload(
        image,
        settings.image_upload_dir,
        settings.max_image_upload_mb * 1024 * 1024,
    )
    _validate_image(saved_path)
    return await run_in_threadpool(
        vision_service.analyze,
        saved_path,
        saved_path.stem,
        saved_path.name,
    )


@app.post("/api/ask", response_model=AskResponse)
async def ask(
    question: Annotated[
        str,
        Form(min_length=1, max_length=settings.max_question_characters),
    ],
    image: Annotated[UploadFile | None, File()] = None,
    image_id: Annotated[str | None, Form(max_length=128)] = None,
    top_k: Annotated[int | None, Form(ge=1, le=10)] = None,
) -> AskResponse:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    image_path: Path | None = None
    if image is not None and image.filename:
        suffix = Path(image.filename).suffix.lower()
        if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
            raise HTTPException(
                status_code=400, detail=f"Unsupported image type. Allowed: {allowed}"
            )
        image_path = await _save_upload(
            image,
            settings.image_upload_dir,
            settings.max_image_upload_mb * 1024 * 1024,
        )
        _validate_image(image_path)

    requested_top_k = top_k if top_k is not None else settings.retrieval_top_k
    bounded_top_k = max(1, min(requested_top_k, 10))
    return await rag_service.answer(
        question=question.strip(),
        image_path=image_path,
        image_id=image_path.stem if image_path else (image_id.strip() if image_id else None),
        filename=image_path.name if image_path else None,
        top_k=bounded_top_k,
    )


async def _save_upload(file: UploadFile, directory: Path, max_bytes: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    original = safe_filename(file.filename, "upload")
    suffix = Path(original).suffix
    stem = Path(original).stem or "upload"
    filename = f"{uuid.uuid4().hex}_{stem}{suffix}"
    path = directory / filename

    total_bytes = 0
    try:
        with path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413, detail="Uploaded file exceeds the configured size limit."
                    )
                buffer.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise

    return path


def _validate_image(path: Path) -> None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > settings.max_image_pixels:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Image dimensions exceed the configured limit of "
                        f"{settings.max_image_pixels} pixels."
                    ),
                )
            image.verify()
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc


async def _runtime_status() -> RuntimeStatusResponse:
    llm_config = llm_client.runtime_config()
    warnings: list[str] = []
    if not llm_config.model_configured:
        warnings.append("API key is not configured; chat answers will use local fallback.")
    model_info = await run_in_threadpool(vision_service.ensure_model_info)
    if not model_info.loaded and model_info.warning:
        warnings.append(model_info.warning)
    return RuntimeStatusResponse(
        model_configured=llm_config.model_configured,
        base_url=llm_config.base_url,
        chat_model=llm_config.chat_model,
        embedding_model=llm_config.embedding_model,
        store_mode=retriever_service.store_mode,
        yolo_model=model_info,
        warnings=warnings,
    )


def _is_loopback_client(request: Request) -> bool:
    if request.client is None or request.client.host == "testclient":
        return True
    host = request.client.host.split("%", 1)[0]
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _is_same_origin(origin: str, host: str) -> bool:
    parsed = urlsplit(origin)
    return parsed.scheme in {"http", "https"} and parsed.netloc.lower() == host.lower()


def _secure_response(response: Response, path: str) -> Response:
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://unpkg.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' blob: data:; "
        "connect-src 'self'; object-src 'none'; base-uri 'self'; "
        "frame-ancestors 'none'; form-action 'self'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response
