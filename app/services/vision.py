from __future__ import annotations

from collections import Counter, OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any

from app.config import Settings, get_settings
from app.schemas import Detection, ImageAnalysisResponse, ModelInfo


class VisionService:
    CACHE_SIZE = 32

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.current_model_path = self.settings.yolo_model
        self._model: Any | None = None
        self._model_warning: str | None = None
        self._model_classes: list[str] = []
        self._analysis_cache: OrderedDict[str, ImageAnalysisResponse] = OrderedDict()
        self._lock = RLock()

    def analyze(self, image_path: Path, image_id: str, filename: str) -> ImageAnalysisResponse:
        width, height = _image_size(image_path)
        warnings: list[str] = []
        detections: list[Detection] = []

        with self._lock:
            try:
                model = self._load_model()
                if self._model_warning:
                    warnings.append(self._model_warning)
                if model is not None:
                    result = model(
                        str(image_path), conf=self.settings.yolo_confidence, verbose=False
                    )[0]
                    names = getattr(result, "names", {}) or {}
                    self._model_classes = _names_to_list(names)
                    for box in result.boxes:
                        xyxy = [float(value) for value in box.xyxy[0].tolist()]
                        cls_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        label = _class_name(names, cls_id)
                        detections.append(Detection(label=label, confidence=confidence, bbox=xyxy))
            except Exception as exc:
                warnings.append(f"YOLO detection failed: {exc}")

        counts = detection_counts(detections)
        analysis = ImageAnalysisResponse(
            image_id=image_id,
            filename=filename,
            width=width,
            height=height,
            detections=detections,
            detection_counts=counts,
            summary=summarize_detections(detections),
            yolo_model_name=self.model_info().model_name,
            yolo_classes=self._model_classes,
            warnings=warnings,
        )
        with self._lock:
            self._analysis_cache[image_id] = analysis
            self._analysis_cache.move_to_end(image_id)
            while len(self._analysis_cache) > self.CACHE_SIZE:
                self._analysis_cache.popitem(last=False)
        return analysis

    def get_analysis(self, image_id: str) -> ImageAnalysisResponse | None:
        with self._lock:
            analysis = self._analysis_cache.get(image_id)
            if analysis is None:
                return None
            self._analysis_cache.move_to_end(image_id)
            return analysis.model_copy(deep=True)

    def reload_model(self, model_path: str) -> ModelInfo:
        path = Path(model_path).expanduser().resolve(strict=False)
        if path.suffix.lower() != ".pt":
            raise ValueError("Only .pt YOLO model files are supported.")
        if not path.is_file():
            raise FileNotFoundError(f"YOLO model file does not exist: {path}")

        with self._lock:
            old_model = self._model
            old_path = self.current_model_path
            old_warning = self._model_warning
            old_classes = self._model_classes

            self.current_model_path = str(path)
            self._model = None
            self._model_warning = None
            self._model_classes = []

            model = self._load_model()
            if model is None:
                message = self._model_warning or f"Could not load YOLO model: {path}"
                self.current_model_path = old_path
                self._model = old_model
                self._model_warning = old_warning
                self._model_classes = old_classes
                raise RuntimeError(message)
            self._analysis_cache.clear()
            return self.model_info()

    def model_info(self) -> ModelInfo:
        with self._lock:
            path = Path(self.current_model_path) if self.current_model_path else None
            return ModelInfo(
                model_path=str(path) if path else None,
                model_name=path.name if path else None,
                loaded=self._model is not None,
                classes=self._model_classes,
                warning=self._model_warning,
            )

    def ensure_model_info(self) -> ModelInfo:
        with self._lock:
            self._load_model()
            return self.model_info()

    def _load_model(self) -> Any | None:
        if self._model is not None or self._model_warning is not None:
            return self._model

        try:
            from ultralytics import YOLO

            self._model = YOLO(self.current_model_path)
            names = getattr(self._model, "names", {}) or {}
            self._model_classes = _names_to_list(names)
        except Exception as exc:
            self._model_warning = f"Could not load YOLO model {self.current_model_path!r}: {exc}"
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


def detection_counts(detections: list[Detection]) -> dict[str, int]:
    return dict(Counter(detection.label for detection in detections))


def _names_to_list(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None
