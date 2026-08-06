from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] = Field(description="Bounding box as [x1, y1, x2, y2].")


class ImageAnalysisResponse(BaseModel):
    image_id: str
    filename: str
    width: int | None = None
    height: int | None = None
    detections: list[Detection]
    summary: str
    warnings: list[str] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    characters: int
    store_mode: str
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
    detections: list[Detection] = Field(default_factory=list)
    visual_summary: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    store_mode: str
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    app: str
    store_mode: str
    model_configured: bool
