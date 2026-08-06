from __future__ import annotations

from pathlib import Path

from app.config import Settings, get_settings
from app.schemas import AskResponse
from app.services.llm import LLMClient
from app.services.retriever import RetrieverService
from app.services.vision import VisionService


class RAGService:
    def __init__(
        self,
        llm: LLMClient,
        retriever: RetrieverService,
        vision: VisionService,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.retriever = retriever
        self.vision = vision

    async def answer(
        self,
        question: str,
        image_path: Path | None = None,
        image_id: str | None = None,
        filename: str | None = None,
        top_k: int | None = None,
    ) -> AskResponse:
        warnings: list[str] = []
        detections = []
        visual_summary = None

        if image_path is not None:
            analysis = self.vision.analyze(image_path, image_id or image_path.stem, filename or image_path.name)
            detections = analysis.detections
            visual_summary = analysis.summary
            warnings.extend(analysis.warnings)

        retrieval_query = question
        if visual_summary:
            retrieval_query = f"{question}\n\n视觉检测摘要：{visual_summary}"

        search = await self.retriever.search(retrieval_query, top_k or self.settings.retrieval_top_k)
        warnings.extend(search.warnings)

        answer, llm_warnings = await self.llm.generate_answer(
            question=question,
            detections=detections,
            visual_summary=visual_summary,
            citations=search.citations,
        )
        warnings.extend(llm_warnings)

        return AskResponse(
            question=question,
            answer=answer,
            detections=detections,
            visual_summary=visual_summary,
            citations=search.citations,
            store_mode=search.store_mode,
            warnings=warnings,
        )
