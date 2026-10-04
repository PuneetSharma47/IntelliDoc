"""Document ingestion: file on disk -> :class:`~docintel.schema.Document`.

Responsibilities kept deliberately narrow: hashing, page reading through the
chosen OCR engine, geometry indexing. No field logic lives here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import layout
from .config import Settings
from .ocr import OCREngineError, resolve_engine
from .schema import Document, SourceKind

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class IngestionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _doc_id(path: Path, digest: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in path.stem)[:40]
    return f"{stem}-{digest[:8]}"


def ingest_file(path: str | Path, settings: Settings) -> tuple[Document, str]:
    """Read ``path`` into a Document. Returns ``(document, engine_note)``."""
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise IngestionError(f"unsupported file type: {path.suffix}")

    engine, note = resolve_engine(settings.ocr, path)
    try:
        pages = engine.read_pages(path)
    except OCREngineError as exc:
        raise IngestionError(str(exc)) from exc

    if not any(p.tokens for p in pages):
        raise IngestionError(f"no text recovered from {path.name} (empty or corrupt scan)")

    digest = _sha256(path)
    doc = Document(
        doc_id=_doc_id(path, digest),
        filename=path.name,
        pages=pages,
        source=getattr(engine, "source", SourceKind.PDF_TEXT),
        engine=engine.name,
        sha256=digest,
    )
    return layout.index_document(doc), note
