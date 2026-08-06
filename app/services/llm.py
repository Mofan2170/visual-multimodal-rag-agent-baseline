from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.schemas import Citation, Detection


LOCAL_EMBEDDING_DIMENSION = 384


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.openai_api_key)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embedding_model = self.settings.embedding_model.strip().lower()
        if not self.is_configured or embedding_model in {"local", "fallback", "hash"}:
            return [_local_embedding(text) for text in texts]

        url = f"{self.settings.openai_base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.settings.embedding_model, "input": texts}

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            data = response.json()
            embeddings = sorted(data["data"], key=lambda item: item.get("index", 0))
            return [item["embedding"] for item in embeddings]
        except Exception:
            return [_local_embedding(text) for text in texts]

    async def generate_answer(
        self,
        question: str,
        detections: list[Detection],
        visual_summary: str | None,
        citations: list[Citation],
    ) -> tuple[str, list[str]]:
        if not self.is_configured:
            return _fallback_answer(question, detections, visual_summary, citations), [
                "OPENAI_API_KEY is not configured; returned a local baseline answer."
            ]

        messages = _build_messages(question, detections, visual_summary, citations)
        url = f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.settings.chat_model,
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
        "你是一个视觉多模态 RAG 智能体。回答必须基于给定的视觉检测结果和检索证据。"
        "如果证据不足，请明确说明不确定，不要编造。回答用中文，结构清晰，并给出依据。"
    )
    user = f"""
用户问题：
{question}

视觉摘要：
{visual_summary or "未提供图片或未检测到可用视觉结果。"}

视觉检测结果 JSON：
{json.dumps(detection_payload, ensure_ascii=False, indent=2)}

检索证据 JSON：
{json.dumps(citation_payload, ensure_ascii=False, indent=2)}

请输出：
1. 结论
2. 视觉依据
3. 文档依据
4. 不确定性或下一步建议
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _fallback_answer(
    question: str,
    detections: list[Detection],
    visual_summary: str | None,
    citations: list[Citation],
) -> str:
    labels = [d.label.lower() for d in detections]
    label_counts: dict[str, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    evidence_text = "\n".join(citation.text for citation in citations).lower()
    helmet_required = "安全帽" in evidence_text or "helmet" in evidence_text
    has_person = "person" in labels
    has_helmet = any(label in {"helmet", "hardhat", "hard hat", "safety helmet"} for label in labels)
    possible_helmet_mislabel = any(label in {"frisbee", "sports ball", "bowl"} for label in labels)

    if has_person and helmet_required and not has_helmet:
        conclusion = "可能存在安全风险：检测到了人员，但没有检测到明确的安全帽类别。"
    elif detections and citations:
        conclusion = "已找到视觉检测结果和相关文档依据，可以基于这些证据进行初步判断。"
    elif detections:
        conclusion = "已有视觉检测结果，但缺少可引用的规则文档，结论需要保守处理。"
    else:
        conclusion = "证据不足，无法给出可靠判断。"

    lines = [
        "这是本地 baseline 回答：当前未配置可用的大模型 API，因此只根据检测结果、简单规则和检索片段做保守判断。",
        "",
        f"问题：{question}",
        "",
        f"结论：{conclusion}",
    ]

    if visual_summary:
        lines.extend(["", f"视觉依据：{visual_summary}"])
    elif detections:
        detected_labels = ", ".join(f"{label} {count} 个" for label, count in sorted(label_counts.items()))
        lines.extend(["", f"视觉依据：检测到的目标包括 {detected_labels}。"])
    else:
        lines.extend(["", "视觉依据：没有图片输入，或未检测到明确目标。"])

    if has_person and helmet_required and not has_helmet:
        lines.append("风险规则：文档要求进入施工现场必须佩戴安全帽；当前检测结果包含 person，但没有 helmet/hardhat 类目标。")
    if possible_helmet_mislabel:
        lines.append("模型限制：当前使用的是 YOLOv8n 通用 COCO 模型，不包含专业安全帽类别，黄色安全帽可能被误识别为 frisbee 等类别，需要人工复核或更换 PPE 专用检测模型。")

    if citations:
        lines.append("")
        lines.append("文档依据：")
        for index, citation in enumerate(citations[:3], start=1):
            text = citation.text.replace("\n", " ").strip()
            if len(text) > 180:
                text = text[:180] + "..."
            lines.append(f"{index}. {citation.source} / {citation.chunk_id}: {text}")
    else:
        lines.extend(["", "文档依据：当前没有检索到相关文档片段。"])

    lines.extend(["", "建议：配置大模型 API 后可获得更自然的解释；如果要准确识别安全帽，后续应换成安全帽/PPE 专用 YOLO 权重或训练自定义数据集。"])

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
