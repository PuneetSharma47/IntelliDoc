"""FastAPI application: upload, review, correct, search.

The API is a thin shell over :class:`docintel.pipeline.Pipeline` and
:class:`docintel.store.Store`. Everything it can do is also reachable from the CLI,
which is what makes the evaluation numbers meaningful -- there is one code path.

Run locally with::

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from docintel.config import Settings, load_settings
from docintel.ingest import SUPPORTED_SUFFIXES, IngestionError
from docintel.pipeline import Pipeline
from docintel.render import confidence_color, render_page
from docintel.schema import FieldName
from docintel.search import Searcher
from docintel.store import Store

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


class CorrectionRequest(BaseModel):
    field: str
    value: Any
    reviewer: str = "reviewer"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    settings.store.db_path = str(_resolve(settings.store.db_path))
    uploads = _resolve("data/uploads")
    uploads.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="IntelliDoc API",
        version="0.9.0",
        description=(
            "Layout-aware extraction for invoices and receipts with field confidence, "
            "arithmetic validation and a human review queue."
        ),
    )
    store = Store(settings.store.db_path)
    pipeline = Pipeline(settings)
    searcher = Searcher(store, top_k=settings.store.search_top_k, snippet_chars=settings.store.snippet_chars)

    app.state.store = store
    app.state.pipeline = pipeline
    app.state.settings = settings

    # -- pages -------------------------------------------------------------- #

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "extractor": settings.extraction.extractor,
            "settings_fingerprint": settings.fingerprint(),
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # -- ingestion ---------------------------------------------------------- #

    @app.post("/api/documents")
    async def upload(file: UploadFile = File(...)) -> JSONResponse:
        suffix = Path(file.filename or "upload").suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(415, f"unsupported file type: {suffix}")
        target = uploads / Path(file.filename or "upload.pdf").name
        with target.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        try:
            outcome = pipeline.process_path(target)
        except IngestionError as exc:
            raise HTTPException(422, str(exc)) from exc
        store.save_result(outcome.result, settings_fp=settings.fingerprint())
        payload = pipeline.explain(outcome.result)
        payload["engine_note"] = outcome.engine_note
        return JSONResponse(jsonable_encoder(payload))

    @app.post("/api/seed")
    async def seed() -> dict:
        """Process the bundled demo documents so the console is never empty."""
        demo_dir = _resolve("data/demo")
        if not demo_dir.exists():
            raise HTTPException(404, "no demo documents bundled")
        added = 0
        for path in sorted(demo_dir.glob("*.pdf")):
            try:
                outcome = pipeline.process_path(path)
            except IngestionError:
                continue
            target = uploads / path.name
            if not target.exists():
                shutil.copyfile(path, target)
            store.save_result(outcome.result, settings_fp=settings.fingerprint())
            added += 1
        return {"count": added, "stats": store.stats()}

    @app.get("/api/documents")
    def list_documents(status: str | None = None, limit: int = Query(50, le=200)) -> dict:
        return {"documents": store.list_documents(status=status, limit=limit), "stats": store.stats()}

    @app.get("/api/documents/{doc_id}")
    def get_document(doc_id: str) -> dict:
        data = store.get_document(doc_id)
        if data is None:
            raise HTTPException(404, "document not found")
        data["source_url"] = f"/api/documents/{doc_id}/page/0.png"
        return data

    @app.get("/api/documents/{doc_id}/page/{page_no}.png")
    def page_image(doc_id: str, page_no: int) -> Response:
        doc = store.get_document(doc_id)
        if doc is None:
            raise HTTPException(404, "document not found")
        path = uploads / doc["filename"]
        if not path.exists():
            raise HTTPException(404, "source file no longer available")
        boxes = []
        colors = []
        for field in doc["fields"]:
            if field["bbox"] and field["page"] == page_no:
                boxes.append(tuple(field["bbox"]))
                colors.append(confidence_color(field["confidence"] or 0.0))
        try:
            png = render_page(path, page_no, boxes, dpi=120, colors=colors)
        except (IndexError, RuntimeError) as exc:
            raise HTTPException(404, str(exc)) from exc
        return Response(content=png, media_type="image/png")

    # -- review ------------------------------------------------------------- #

    @app.post("/api/documents/{doc_id}/correct")
    def correct(doc_id: str, body: CorrectionRequest) -> dict:
        valid = {f.value for f in FieldName}
        if body.field not in valid:
            raise HTTPException(422, f"unknown field {body.field!r}")
        try:
            store.apply_correction(doc_id, body.field, body.value, reviewer=body.reviewer)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"document": store.get_document(doc_id), "stats": store.stats()}

    @app.post("/api/documents/{doc_id}/approve")
    def approve(doc_id: str, reviewer: str = "reviewer") -> dict:
        if store.get_document(doc_id) is None:
            raise HTTPException(404, "document not found")
        store.approve_document(doc_id, reviewer=reviewer)
        return {"document": store.get_document(doc_id), "stats": store.stats()}

    # -- library ------------------------------------------------------------ #

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=2)) -> dict:
        hits = searcher.search(q)
        return {"query": q, "hits": [hit.to_dict() for hit in hits], "backend": "fts5" if store.has_fts else "like"}

    @app.get("/api/stats")
    def stats() -> dict:
        return {"stats": store.stats(), "corrections": store.export_corrections(limit=20)}

    @app.get("/api/eval/latest")
    def latest_eval() -> dict:
        path = _resolve(settings.benchmarks_dir) / "latest.json"
        if not path.exists():
            return {"available": False, "hint": "run: python scripts/run_eval.py"}
        return {"available": True, "report": json.loads(path.read_text(encoding="utf-8"))}

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


app = create_app()
