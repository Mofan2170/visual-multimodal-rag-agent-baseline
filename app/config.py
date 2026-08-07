from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool = False) -> bool:
    fallback = "true" if default else "false"
    return _env(name, fallback).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    app_name: str = "Visual Multimodal RAG Workbench"

    openai_api_key: str = _env("OPENAI_API_KEY", "")
    openai_base_url: str = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    chat_model: str = _env("CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = _env("EMBEDDING_MODEL", "text-embedding-3-small")
    request_timeout_seconds: float = float(_env("REQUEST_TIMEOUT_SECONDS", "60"))

    yolo_model: str = _env("YOLO_MODEL", "yolov8n.pt")
    yolo_confidence: float = float(_env("YOLO_CONFIDENCE", "0.25"))

    chunk_size: int = int(_env("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "150"))
    retrieval_top_k: int = int(_env("RETRIEVAL_TOP_K", "5"))

    max_document_upload_mb: int = int(_env("MAX_DOCUMENT_UPLOAD_MB", "20"))
    max_image_upload_mb: int = int(_env("MAX_IMAGE_UPLOAD_MB", "20"))
    max_model_upload_mb: int = int(_env("MAX_MODEL_UPLOAD_MB", "500"))
    max_document_characters: int = int(_env("MAX_DOCUMENT_CHARACTERS", "2000000"))
    max_pdf_pages: int = int(_env("MAX_PDF_PAGES", "500"))
    max_image_pixels: int = int(_env("MAX_IMAGE_PIXELS", "40000000"))
    max_question_characters: int = int(_env("MAX_QUESTION_CHARACTERS", "8000"))

    allow_remote_access: bool = _env_bool("ALLOW_REMOTE_ACCESS", False)
    allowed_hosts: tuple[str, ...] = tuple(
        host.strip()
        for host in _env("ALLOWED_HOSTS", "127.0.0.1,localhost,::1,testserver").split(",")
        if host.strip()
    )

    milvus_collection: str = _env("MILVUS_COLLECTION", "visual_multimodal_rag_chunks")

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def document_upload_dir(self) -> Path:
        return self.uploads_dir / "documents"

    @property
    def image_upload_dir(self) -> Path:
        return self.uploads_dir / "images"

    @property
    def milvus_dir(self) -> Path:
        return self.data_dir / "milvus"

    @property
    def milvus_db_path(self) -> Path:
        return self.milvus_dir / "visual_multimodal_rag.db"

    @property
    def local_store_dir(self) -> Path:
        return self.data_dir / "local_store"

    @property
    def local_store_path(self) -> Path:
        return self.local_store_dir / "chunks.json"

    @property
    def model_upload_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def web_dir(self) -> Path:
        return self.root_dir / "web"

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.uploads_dir,
            self.document_upload_dir,
            self.image_upload_dir,
            self.milvus_dir,
            self.local_store_dir,
            self.model_upload_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
