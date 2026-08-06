from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.schemas import Citation
from app.services.llm import LLMClient


@dataclass
class AddResult:
    document_id: str
    chunks: int
    characters: int
    store_mode: str
    warnings: list[str]


@dataclass
class SearchResult:
    citations: list[Citation]
    store_mode: str
    warnings: list[str]


class RetrieverService:
    def __init__(self, llm: LLMClient, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.local_store = LocalJsonVectorStore(self.settings.local_store_path)
        self._milvus: MilvusLiteIndex | None = None
        self._milvus_error: str | None = None

    @property
    def store_mode(self) -> str:
        if self._milvus is not None and self._milvus.ready:
            return "milvus-lite+local-json"
        if self._milvus_error:
            return "local-json"
        return "local-json"

    async def add_chunks(self, source: str, chunks: list[str]) -> AddResult:
        warnings: list[str] = []
        document_id = uuid.uuid4().hex
        vectors = await self.llm.embed_texts(chunks)
        now = int(time.time())
        records: list[dict[str, Any]] = []

        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_id = f"{document_id}-{index:04d}"
            records.append(
                {
                    "id": chunk_id,
                    "numeric_id": _numeric_id(chunk_id),
                    "document_id": document_id,
                    "chunk_index": index,
                    "source": source,
                    "text": chunk,
                    "vector": vector,
                    "created_at": now,
                }
            )

        self.local_store.add(records)

        if records:
            milvus = self._ensure_milvus(len(records[0]["vector"]))
            if milvus is not None:
                try:
                    milvus.upsert(records)
                except Exception as exc:
                    self._milvus_error = f"Milvus Lite upsert failed: {exc}"
                    warnings.append(self._milvus_error)
            elif self._milvus_error:
                warnings.append(self._milvus_error)

        return AddResult(
            document_id=document_id,
            chunks=len(records),
            characters=sum(len(chunk) for chunk in chunks),
            store_mode=self.store_mode,
            warnings=warnings,
        )

    async def search(self, query: str, top_k: int = 5) -> SearchResult:
        warnings: list[str] = []
        vectors = await self.llm.embed_texts([query])
        if not vectors:
            return SearchResult([], self.store_mode, warnings)

        query_vector = vectors[0]
        if self._milvus is None and self._milvus_error is None:
            self._ensure_milvus(len(query_vector))

        if self._milvus is not None and self._milvus.ready:
            try:
                citations = self._milvus.search(query_vector, top_k)
                return SearchResult(citations, self.store_mode, warnings)
            except Exception as exc:
                self._milvus_error = f"Milvus Lite search failed; using local JSON fallback. Detail: {exc}"
                warnings.append(self._milvus_error)

        if self._milvus_error and self._milvus_error not in warnings:
            warnings.append(self._milvus_error)
        citations = self.local_store.search(query_vector, top_k)
        return SearchResult(citations, "local-json", warnings)

    def _ensure_milvus(self, dimension: int) -> "MilvusLiteIndex | None":
        if self._milvus is not None:
            return self._milvus
        if self._milvus_error is not None:
            return None
        try:
            self._milvus = MilvusLiteIndex(self.settings, dimension)
            return self._milvus
        except Exception as exc:
            self._milvus_error = f"Milvus Lite is unavailable; using local JSON fallback. Detail: {exc}"
            self._milvus = None
            return None


class LocalJsonVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, records: list[dict[str, Any]]) -> None:
        existing = self._load()
        existing.extend(records)
        self._save(existing)

    def search(self, query_vector: list[float], top_k: int) -> list[Citation]:
        records = self._load()
        scored: list[tuple[float, dict[str, Any]]] = []
        for record in records:
            vector = record.get("vector") or []
            if len(vector) != len(query_vector):
                continue
            scored.append((_cosine_similarity(query_vector, vector), record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [_record_to_citation(record, score) for score, record in scored[:top_k]]

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save(self, records: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


class MilvusLiteIndex:
    def __init__(self, settings: Settings, dimension: int) -> None:
        from pymilvus import MilvusClient

        self.settings = settings
        self.client = MilvusClient(uri=str(settings.milvus_db_path))
        self.collection_name = settings.milvus_collection
        self.ready = True
        self._loaded = False

        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=dimension,
                metric_type="COSINE",
                consistency_level="Strong",
            )

    def upsert(self, records: list[dict[str, Any]]) -> None:
        data = [
            {
                "id": record["numeric_id"],
                "vector": record["vector"],
                "chunk_id": record["id"],
                "document_id": record["document_id"],
                "chunk_index": record["chunk_index"],
                "source": record["source"],
                "text": record["text"],
                "created_at": record["created_at"],
            }
            for record in records
        ]
        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)
            self._loaded = False

    def search(self, query_vector: list[float], top_k: int) -> list[Citation]:
        self.load()
        result = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["chunk_id", "document_id", "chunk_index", "source", "text", "created_at"],
        )
        hits = result[0] if result else []
        citations: list[Citation] = []
        for hit in hits:
            entity = hit.get("entity", {})
            citations.append(
                Citation(
                    chunk_id=str(entity.get("chunk_id", hit.get("id", ""))),
                    source=str(entity.get("source", "")),
                    text=str(entity.get("text", "")),
                    score=float(hit.get("distance", 0.0)),
                    metadata={
                        "document_id": entity.get("document_id"),
                        "chunk_index": entity.get("chunk_index"),
                        "created_at": entity.get("created_at"),
                    },
                )
        )
        return citations

    def load(self) -> None:
        if self._loaded:
            return
        self.client.load_collection(collection_name=self.collection_name)
        self._loaded = True


def _record_to_citation(record: dict[str, Any], score: float) -> Citation:
    return Citation(
        chunk_id=str(record.get("id", "")),
        source=str(record.get("source", "")),
        text=str(record.get("text", "")),
        score=float(score),
        metadata={
            "document_id": record.get("document_id"),
            "chunk_index": record.get("chunk_index"),
            "created_at": record.get("created_at"),
        },
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _numeric_id(value: str) -> int:
    return uuid.UUID(value[:32]).int % ((1 << 63) - 1) if len(value) >= 32 else abs(hash(value))
