from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.schemas import Detection, ImageAnalysisResponse


class VisionService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any | None = None
        self._model_warning: str | None = None

    def analyze(self, image_path: Path, image_id: str, filename: str) -> ImageAnalysisResponse:
        width, height = _image_size(image_path)
        warnings: list[str] = []
        detections: list[Detection] = []

        try:
            model = self._load_model()
            if self._model_warning:
                warnings.append(self._model_warning)
            if model is not None:
                result = model(str(image_path), conf=self.settings.yolo_confidence, verbose=False)[0]
                names = getattr(result, "names", {}) or {}
                for box in result.boxes:
                    xyxy = [float(value) for value in box.xyxy[0].tolist()]
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    label = names.get(cls_id, str(cls_id))
                    detections.append(Detection(label=label, confidence=confidence, bbox=xyxy))
        except Exception as exc:
            warnings.append(f"YOLO detection failed: {exc}")

        return ImageAnalysisResponse(
            image_id=image_id,
            filename=filename,
            width=width,
            height=height,
            detections=detections,
            summary=summarize_detections(detections),
            warnings=warnings,
        )

    def _load_model(self) -> Any | None:
        if self._model is not None or self._model_warning is not None:
            return self._model

        try:
            from ultralytics import YOLO

            self._model = YOLO(self.settings.yolo_model)
        except Exception as exc:
            self._model_warning = f"Could not load YOLO model {self.settings.yolo_model!r}: {exc}"
            self._model = None
        return self._model


def summarize_detections(detections: list[Detection]) -> str:
    if not detections:
        return "未检测到明确目标。"

    counts = Counter(detection.label for detection in detections)
    parts = [f"{label} {count} 个" for label, count in counts.most_common()]
    strongest = max(detections, key=lambda item: item.confidence)
    return (
        f"检测到 {len(detections)} 个目标：{', '.join(parts)}。"
        f"最高置信度目标为 {strongest.label}，置信度 {strongest.confidence:.2f}。"
    )


def _image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None
