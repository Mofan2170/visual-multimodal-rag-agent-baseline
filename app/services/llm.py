from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.config import Settings, get_settings
from app.schemas import Citation, Detection, RuntimeConfigRequest, RuntimeConfigResponse

LOCAL_EMBEDDING_DIMENSION = 384
LOCAL_EMBEDDING_NAMES = {"local", "fallback", "hash"}


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    warnings: list[str]


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.api_key = self.settings.openai_api_key
        self.base_url = _normalize_base_url(self.settings.openai_base_url)
        self.chat_model = self.settings.chat_model
        self.embedding_model = self.settings.embedding_model

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip())

    def configure(self, request: RuntimeConfigRequest) -> RuntimeConfigResponse:
        next_api_key = self.api_key if request.api_key is None else request.api_key.strip()
        next_base_url = self.base_url
        if request.base_url is not None and request.base_url.strip():
            next_base_url = _normalize_base_url(request.base_url)
        next_chat_model = self.chat_model
        if request.chat_model is not None and request.chat_model.strip():
            next_chat_model = request.chat_model.strip()
        next_embedding_model = self.embedding_model
        if request.embedding_model is not None and request.embedding_model.strip():
            next_embedding_model = request.embedding_model.strip()

        self.api_key = next_api_key
        self.base_url = next_base_url
        self.chat_model = next_chat_model
        self.embedding_model = next_embedding_model

        warnings: list[str] = []
        if not self.is_configured:
            warnings.append("API key is empty; chat answers will use the local fallback.")
        if self.embedding_model.strip().lower() not in LOCAL_EMBEDDING_NAMES:
            warnings.append(
                "Embedding model is not local. Re-upload documents if vector dimensions change."
            )
        return self.runtime_config(warnings)

    def runtime_config(self, warnings: list[str] | None = None) -> RuntimeConfigResponse:
        return RuntimeConfigResponse(
            model_configured=self.is_configured,
            base_url=self.base_url,
            chat_model=self.chat_model,
            embedding_model=self.embedding_model,
            warnings=warnings or [],
        )

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult([], [])

        embedding_model = self.embedding_model.strip().lower()
        if not self.is_configured or embedding_model in LOCAL_EMBEDDING_NAMES:
            return EmbeddingResult([_local_embedding(text) for text in texts], [])

        url = f"{self.base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.embedding_model, "input": texts}

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            data = response.json()
            embeddings = sorted(data["data"], key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in embeddings]
            _validate_embeddings(vectors, len(texts))
            return EmbeddingResult(vectors, [])
        except Exception as exc:
            warning = f"Embedding request failed; local embedding fallback was used. Detail: {exc}"
            return EmbeddingResult([_local_embedding(text) for text in texts], [warning])

    async def generate_answer(
        self,
        question: str,
        detections: list[Detection],
        visual_summary: str | None,
        citations: list[Citation],
    ) -> tuple[str, list[str]]:
        if not self.is_configured:
            return _fallback_answer(question, detections, visual_summary, citations), [
                "API key is not configured; returned a local fallback answer."
            ]

        messages = _build_messages(question, detections, visual_summary, citations)
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": 0.2,
        }

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip(), []
        except Exception as exc:
            warning = f"Chat model request failed; returned local fallback answer. Detail: {exc}"
            return _fallback_answer(question, detections, visual_summary, citations), [warning]


def _build_messages(
    question: str,
    detections: list[Detection],
    visual_summary: str | None,
    citations: list[Citation],
) -> list[dict[str, str]]:
    detection_payload = [d.model_dump() for d in detections]
    citation_payload = [c.model_dump(exclude={"metadata"}) for c in citations]

    system = (
        "你是一个通用视觉多模态 RAG 智能体。你不绑定任何固定场景。"
        "请根据用户问题、YOLO 检测结果、用户上传的规则/知识文档证据来回答。"
        "如果证据不足，请明确说明不确定，不要编造没有出现在图像检测或文档中的事实。"
        "回答优先使用用户提问的语言；如果无法判断语言，使用中文。"
    )
    user = f"""
用户问题：
{question}

视觉摘要：
{visual_summary or "未提供图片，或未检测到可用视觉结果。"}

YOLO 检测结果 JSON：
{json.dumps(detection_payload, ensure_ascii=False, indent=2)}

检索到的规则/知识文档证据 JSON：
{json.dumps(citation_payload, ensure_ascii=False, indent=2)}

请使用 Markdown 标题和列表输出，不要输出 HTML：
1. 结论
2. 视觉依据
3. 文档/规则依据
4. 不确定性
5. 建议
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _fallback_answer(
    question: str,
    detections: list[Detection],
    visual_summary: str | None,
    citations: list[Citation],
) -> str:
    label_counts: dict[str, int] = {}
    for detection in detections:
        label_counts[detection.label] = label_counts.get(detection.label, 0) + 1

    if detections and citations:
        conclusion = "已获得视觉检测结果和规则文档证据，但当前未配置可用 API，只能给出保守摘要。"
    elif detections:
        conclusion = "已获得视觉检测结果，但缺少可引用的规则文档证据。"
    elif citations:
        conclusion = "已检索到规则文档证据，但没有可用视觉检测结果。"
    else:
        conclusion = "证据不足，无法给出可靠判断。"

    lines = [
        "> 当前未配置可用的大模型 API，以下为本地 fallback 证据摘要。",
        "",
        "## 结论",
        "",
        conclusion,
        "",
        f"**问题：** {question}",
        "",
        "## 视觉依据",
        "",
    ]

    if visual_summary:
        lines.append(f"- {visual_summary}")
    elif detections:
        detected_labels = ", ".join(
            f"{label} {count} 个" for label, count in sorted(label_counts.items())
        )
        lines.append(f"- 检测到的目标包括 {detected_labels}。")
    else:
        lines.append("- 没有图片输入，或未检测到明确目标。")

    lines.extend(["", "## 文档/规则依据", ""])
    if citations:
        for index, citation in enumerate(citations[:3], start=1):
            text = citation.text.replace("\n", " ").strip()
            if len(text) > 180:
                text = text[:180] + "..."
            lines.append(f"{index}. **{citation.source}** / `{citation.chunk_id}`：{text}")
    else:
        lines.append("- 当前没有检索到相关片段。")

    lines.extend(
        [
            "",
            "## 建议",
            "",
            "- 请在页面运行配置中填写 OpenAI-compatible Chat API，"
            "以获得更完整的业务判断和自然语言解释。",
        ]
    )
    return "\n".join(lines)


def _local_embedding(text: str, dimension: int = LOCAL_EMBEDDING_DIMENSION) -> list[float]:
    vector = [0.0] * dimension
    tokens = _tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower())


def _validate_embeddings(vectors: list[list[float]], expected_count: int) -> None:
    if len(vectors) != expected_count:
        raise ValueError(
            f"Embedding response returned {len(vectors)} vectors for {expected_count} texts."
        )
    dimensions = {len(vector) for vector in vectors}
    if not dimensions or 0 in dimensions or len(dimensions) != 1:
        raise ValueError("Embedding response contains empty or inconsistent vector dimensions.")


def _normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Base URL must be an absolute HTTP or HTTPS URL.")
    if parsed.username or parsed.password:
        raise ValueError("Base URL must not contain embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("Base URL must not contain a query string or fragment.")
    return normalized
