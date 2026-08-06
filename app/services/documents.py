from __future__ import annotations

import re
from pathlib import Path


SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".pdf"}


class DocumentProcessingError(ValueError):
    pass


def safe_filename(filename: str | None, fallback: str = "upload") -> str:
    name = Path(filename or fallback).name
    cleaned = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", name).strip("._")
    return cleaned or fallback


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise DocumentProcessingError(f"Unsupported document type {suffix!r}. Allowed: {allowed}")

    if suffix in {".txt", ".md"}:
        return _read_text_file(path)

    return _read_pdf(path)


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []

    chunk_size = max(200, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))
    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()

        if end < len(normalized):
            break_at = max(chunk.rfind("\n\n"), chunk.rfind("\n"), chunk.rfind("。"), chunk.rfind("."))
            if break_at > chunk_size * 0.5:
                end = start + break_at + 1
                chunk = normalized[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break
        start = max(0, end - overlap)

    return chunks


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentProcessingError("Could not decode text file as UTF-8 or GB18030.")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentProcessingError("PDF support requires pypdf. Run: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {page_number}]\n{text.strip()}")
    return "\n\n".join(pages)
