from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from app.config import Settings, get_settings
from app.schemas import Citation
from app.services.documents import text_looks_corrupted
from app.services.llm import LLMClient

MILVUS_INDEX_VERSION = "v4"
logger = logging.getLogger(__name__)


@dataclass
class AddResult:
    document_id: str
    source: str
    chunks: int
    characters: int
    store_mode: str
    deduplicated: bool
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
        self._milvus_dimension: int | None = None

    @property
    def store_mode(self) -> str:
        if self._milvus is not None and self._milvus.ready:
            return "milvus-lite+local-json"
        return "local-json"

    async def add_chunks(self, source: str, chunks: list[str]) -> AddResult:
        warnings: list[str] = []
        content_hash = _document_hash(chunks)
        existing = self.local_store.find_document(content_hash)
        if existing:
            vector = existing[0].get("vector") or []
            if vector:
                self._ensure_milvus(len(vector))
            if self._milvus_error:
                warnings.append(self._milvus_error)
            return AddResult(
                document_id=str(existing[0].get("document_id", "")),
                source=str(existing[0].get("source", source)),
                chunks=len(existing),
                characters=sum(len(str(record.get("text", ""))) for record in existing),
                store_mode=self.store_mode,
                deduplicated=True,
                warnings=[
                    "Document content is already indexed; duplicate upload was skipped.",
                    *warnings,
                ],
            )

        document_id = uuid.uuid4().hex
        embedding = await self.llm.embed_texts(chunks)
        vectors = embedding.vectors
        warnings.extend(embedding.warnings)
        now = int(time.time())
        records: list[dict[str, Any]] = []

        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
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
                    "content_hash": content_hash,
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
                    warning = f"Milvus Lite upsert failed; using local JSON fallback. Detail: {exc}"
                    self._disable_milvus(warning)
                    warnings.append(warning)
            elif self._milvus_error:
                warnings.append(self._milvus_error)

        return AddResult(
            document_id=document_id,
            source=source,
            chunks=len(records),
            characters=sum(len(chunk) for chunk in chunks),
            store_mode=self.store_mode,
            deduplicated=False,
            warnings=warnings,
        )

    async def search(self, query: str, top_k: int = 5) -> SearchResult:
        warnings: list[str] = []
        embedding = await self.llm.embed_texts([query])
        warnings.extend(embedding.warnings)
        if not embedding.vectors:
            return SearchResult([], self.store_mode, warnings)

        query_vector = embedding.vectors[0]
        self._ensure_milvus(len(query_vector))

        if self._milvus is not None and self._milvus.ready:
            try:
                citations = self._milvus.search(query_vector, top_k)
                return SearchResult(citations, self.store_mode, warnings)
            except Exception as exc:
                warning = f"Milvus Lite search failed; using local JSON fallback. Detail: {exc}"
                self._disable_milvus(warning)
                warnings.append(warning)

        if self._milvus_error and self._milvus_error not in warnings:
            warnings.append(self._milvus_error)
        citations = self.local_store.search(query_vector, top_k)
        return SearchResult(citations, "local-json", warnings)

    def _ensure_milvus(self, dimension: int) -> MilvusLiteIndex | None:
        if self._milvus_dimension != dimension:
            self._close_milvus()
            self._milvus_error = None
            self._milvus_dimension = dimension
        if self._milvus is not None and self._milvus.dimension == dimension:
            return self._milvus
        if self._milvus_error is not None:
            return None
        try:
            self._milvus = MilvusLiteIndex(self.settings, dimension)
            matching_records = self.local_store.records_for_dimension(dimension)
            if matching_records:
                self._milvus.upsert(matching_records)
            return self._milvus
        except Exception as exc:
            self._milvus_error = (
                f"Milvus Lite is unavailable; using local JSON fallback. Detail: {exc}"
            )
            self._milvus = None
            return None

    def _disable_milvus(self, warning: str) -> None:
        self._milvus_error = warning
        self._close_milvus()

    def _close_milvus(self) -> None:
        if self._milvus is not None:
            self._milvus.close()
        self._milvus = None

    def close(self) -> None:
        self._close_milvus()


class LocalJsonVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def add(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            existing = self._load()
            existing.extend(records)
            self._save(existing)

    def find_document(self, content_hash: str) -> list[dict[str, Any]]:
        with self._lock:
            records = self._load()
        direct = [record for record in records if record.get("content_hash") == content_hash]
        if direct:
            return sorted(direct, key=lambda record: int(record.get("chunk_index", 0)))

        documents: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            document_id = str(record.get("document_id", ""))
            documents.setdefault(document_id, []).append(record)
        for document_records in documents.values():
            ordered = sorted(document_records, key=lambda record: int(record.get("chunk_index", 0)))
            if _document_hash([str(record.get("text", "")) for record in ordered]) == content_hash:
                return ordered
        return []

    def records_for_dimension(self, dimension: int) -> list[dict[str, Any]]:
        with self._lock:
            return [
                record
                for record in self._load()
                if len(record.get("vector") or []) == dimension
                and not text_looks_corrupted(str(record.get("text", "")))
            ]

    def search(self, query_vector: list[float], top_k: int) -> list[Citation]:
        with self._lock:
            records = self._load()
        scored: list[tuple[float, dict[str, Any]]] = []
        for record in records:
            if text_looks_corrupted(str(record.get("text", ""))):
                continue
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
            records = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Local vector store is unreadable: {self.path}") from exc
        if not isinstance(records, list):
            raise RuntimeError(f"Local vector store must contain a JSON list: {self.path}")
        return records

    def _save(self, records: list[dict[str, Any]]) -> None:
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)


class MilvusLiteIndex:
    def __init__(self, settings: Settings, dimension: int) -> None:
        from pymilvus import MilvusClient

        self.settings = settings
        self.client = MilvusClient(uri=str(settings.milvus_db_path))
        self.dimension = dimension
        self.collection_name = f"{settings.milvus_collection}_{MILVUS_INDEX_VERSION}_d{dimension}"
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
                "id": _numeric_id(str(record["id"])),
                "vector": record["vector"],
                "chunk_id": record["id"],
                "document_id": record["document_id"],
                "chunk_index": record["chunk_index"],
                "source": record["source"],
                "text": record["text"],
                "content_hash": record.get("content_hash", ""),
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
            output_fields=[
                "chunk_id",
                "document_id",
                "chunk_index",
                "source",
                "text",
                "created_at",
            ],
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

    def close(self) -> None:
        try:
            self.client.close()
        except Exception as exc:
            logger.debug("Milvus client shutdown failed: %s", exc)


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
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _numeric_id(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _document_hash(chunks: list[str]) -> str:
    content = "\n\n".join(chunk.strip() for chunk in chunks)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
