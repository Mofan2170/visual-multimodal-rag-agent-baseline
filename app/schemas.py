from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] = Field(
        min_length=4,
        max_length=4,
        description="Bounding box as [x1, y1, x2, y2].",
    )


class ImageAnalysisResponse(BaseModel):
    image_id: str
    filename: str
    width: int | None = None
    height: int | None = None
    detections: list[Detection]
    detection_counts: dict[str, int] = Field(default_factory=dict)
    summary: str
    yolo_model_name: str | None = None
    yolo_classes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    characters: int
    store_mode: str
    deduplicated: bool = False
    warnings: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    question: str
    answer: str
    image_id: str | None = None
    detections: list[Detection] = Field(default_factory=list)
    detection_counts: dict[str, int] = Field(default_factory=dict)
    visual_summary: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    store_mode: str
    yolo_model_name: str | None = None
    yolo_classes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RuntimeConfigRequest(BaseModel):
    api_key: str | None = Field(default=None, max_length=4096)
    base_url: str | None = Field(default=None, max_length=2048)
    chat_model: str | None = Field(default=None, max_length=256)
    embedding_model: str | None = Field(default=None, max_length=256)


class RuntimeConfigResponse(BaseModel):
    model_configured: bool
    base_url: str
    chat_model: str
    embedding_model: str
    warnings: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    model_path: str | None = None
    model_name: str | None = None
    loaded: bool = False
    classes: list[str] = Field(default_factory=list)
    warning: str | None = None


class RuntimeStatusResponse(BaseModel):
    model_configured: bool
    base_url: str
    chat_model: str
    embedding_model: str
    store_mode: str
    yolo_model: ModelInfo
    warnings: list[str] = Field(default_factory=list)


class ModelSelectRequest(BaseModel):
    model_path: str = Field(min_length=1, max_length=4096)


class ModelSelectResponse(BaseModel):
    yolo_model: ModelInfo
    warnings: list[str] = Field(default_factory=list)


class ModelUploadResponse(BaseModel):
    filename: str
    model_path: str
    yolo_model: ModelInfo
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app: str
    store_mode: str
    model_configured: bool
    yolo_model_loaded: bool = False
    yolo_model_name: str | None = None
    yolo_classes: list[str] = Field(default_factory=list)
