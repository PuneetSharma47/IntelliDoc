"""SQLite persistence: documents, fields, issues, review queue, corrections.

Design notes
------------
* One file, no server: a reviewer can be shown the whole history of a document
  with ``sqlite3 data/docintel.db``.
* Field rows keep **both** ``value`` (what the pipeline said) and
  ``corrected_value`` (what a human said). That pair is the training signal for
  the next iteration, and it is only usable if it is never overwritten.
* Page text is indexed with SQLite's FTS5 so search results can point at a page
  number instead of an opaque document.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import DocStatus, ExtractionResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    sha256 TEXT,
    engine TEXT,
    source TEXT,
    doc_type TEXT,
    doc_type_confidence REAL,
    status TEXT,
    extractor TEXT,
    page_count INTEGER,
    token_count INTEGER,
    latency_ms REAL,
    settings_fp TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    doc_id TEXT NOT NULL,
    page_no INTEGER NOT NULL,
    width REAL,
    height REAL,
    text TEXT,
    PRIMARY KEY (doc_id, page_no)
);

CREATE TABLE IF NOT EXISTS fields (
    doc_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value_json TEXT,
    value_text TEXT,
    raw_text TEXT,
    confidence REAL,
    page INTEGER,
    bbox_json TEXT,
    source TEXT,
    rule TEXT,
    alternatives_json TEXT,
    confidence_parts_json TEXT,
    needs_review INTEGER DEFAULT 0,
    corrected_value_json TEXT,
    corrected_by TEXT,
    corrected_at TEXT,
    PRIMARY KEY (doc_id, name)
);

CREATE TABLE IF NOT EXISTS issues (
    doc_id TEXT NOT NULL,
    code TEXT,
    message TEXT,
    severity TEXT,
    fields_json TEXT
);

CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    field TEXT,
    reason TEXT,
    priority REAL,
    detail TEXT,
    status TEXT DEFAULT 'open',
    created_at TEXT,
    resolved_at TEXT,
    resolved_by TEXT
);

CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    field TEXT,
    old_value_json TEXT,
    new_value_json TEXT,
    reviewer TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT,
    extractor TEXT,
    split TEXT,
    settings_fp TEXT,
    metrics_json TEXT,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_fields_doc ON fields(doc_id);
CREATE INDEX IF NOT EXISTS idx_queue_doc ON review_queue(doc_id, status);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS page_fts USING fts5(
    doc_id UNINDEXED,
    page_no UNINDEXED,
    text,
    tokenize = 'unicode61'
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


@dataclass
class StoredField:
    name: str
    value: Any
    corrected: Any
    confidence: float
    page: int
    rule: str
    source: str
    needs_review: bool


class Store:
    """Thin, explicit data-access layer. No ORM, no magic."""

    def __init__(self, db_path: str | Path = "data/docintel.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.has_fts = True
        self.conn.executescript(SCHEMA)
        try:
            self.conn.executescript(FTS_SCHEMA)
        except sqlite3.OperationalError:
            # SQLite built without FTS5: search degrades to LIKE, still works
            self.has_fts = False
        self.conn.commit()

    # -- writes ------------------------------------------------------------- #

    def save_result(self, result: ExtractionResult, settings_fp: str = "") -> str:
        """Upsert a full extraction result. Re-processing a document replaces it."""
        doc = result.document
        cur = self.conn.cursor()
        cur.execute("DELETE FROM pages WHERE doc_id = ?", (doc.doc_id,))
        cur.execute("DELETE FROM fields WHERE doc_id = ?", (doc.doc_id,))
        cur.execute("DELETE FROM issues WHERE doc_id = ?", (doc.doc_id,))
        cur.execute("DELETE FROM review_queue WHERE doc_id = ?", (doc.doc_id,))
        if self.has_fts:
            cur.execute("DELETE FROM page_fts WHERE doc_id = ?", (doc.doc_id,))

        cur.execute(
            """INSERT OR REPLACE INTO documents
               (doc_id, filename, sha256, engine, source, doc_type, doc_type_confidence,
                status, extractor, page_count, token_count, latency_ms, settings_fp, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                doc.doc_id,
                doc.filename,
                doc.sha256,
                doc.engine,
                doc.source.value,
                result.doc_type.value,
                result.doc_type_confidence,
                result.status.value,
                result.extractor,
                len(doc.pages),
                doc.n_tokens,
                result.latency_ms,
                settings_fp,
                _now(),
            ),
        )

        for page in doc.pages:
            cur.execute(
                "INSERT OR REPLACE INTO pages (doc_id, page_no, width, height, text) VALUES (?,?,?,?,?)",
                (doc.doc_id, page.number, page.width, page.height, page.text),
            )
            if self.has_fts:
                cur.execute(
                    "INSERT INTO page_fts (doc_id, page_no, text) VALUES (?,?,?)",
                    (doc.doc_id, page.number, page.text),
                )

        flagged = {task.field for task in result.review_tasks if task.field is not None}
        for name, fv in result.fields.items():
            cur.execute(
                """INSERT OR REPLACE INTO fields
                   (doc_id, name, value_json, value_text, raw_text, confidence, page, bbox_json,
                    source, rule, alternatives_json, confidence_parts_json, needs_review)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    doc.doc_id,
                    name.value,
                    _json(fv.value),
                    str(fv.value),
                    fv.provenance.raw_text,
                    fv.confidence,
                    fv.provenance.page,
                    _json(list(fv.provenance.bbox.as_tuple())) if fv.provenance.bbox else None,
                    fv.provenance.source,
                    fv.provenance.rule,
                    _json(fv.alternatives),
                    _json(fv.confidence_parts),
                    1 if name in flagged else 0,
                ),
            )

        for issue in result.issues:
            cur.execute(
                "INSERT INTO issues (doc_id, code, message, severity, fields_json) VALUES (?,?,?,?,?)",
                (doc.doc_id, issue.code, issue.message, issue.severity.value, _json([f.value for f in issue.fields])),
            )

        for task in result.review_tasks:
            cur.execute(
                """INSERT INTO review_queue (doc_id, field, reason, priority, detail, status, created_at)
                   VALUES (?,?,?,?,?,'open',?)""",
                (
                    doc.doc_id,
                    task.field.value if task.field else None,
                    task.reason.value,
                    task.priority,
                    task.detail,
                    _now(),
                ),
            )
        self.conn.commit()
        return doc.doc_id

    def apply_correction(self, doc_id: str, field: str, new_value: Any, reviewer: str = "reviewer") -> None:
        """Record a human correction and keep the machine value for retraining."""
        cur = self.conn.cursor()
        row = cur.execute("SELECT value_json FROM fields WHERE doc_id = ? AND name = ?", (doc_id, field)).fetchone()
        if row is None:
            raise KeyError(f"no field {field!r} on {doc_id}")
        old_value = json.loads(row["value_json"]) if row["value_json"] else None
        cur.execute(
            "UPDATE fields SET corrected_value_json = ?, corrected_by = ?, corrected_at = ?, needs_review = 0 "
            "WHERE doc_id = ? AND name = ?",
            (_json(new_value), reviewer, _now(), doc_id, field),
        )
        cur.execute(
            "INSERT INTO corrections (doc_id, field, old_value_json, new_value_json, reviewer, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (doc_id, field, _json(old_value), _json(new_value), reviewer, _now()),
        )
        cur.execute(
            "UPDATE review_queue SET status = 'resolved', resolved_at = ?, resolved_by = ? "
            "WHERE doc_id = ? AND field = ? AND status = 'open'",
            (_now(), reviewer, doc_id, field),
        )
        remaining = cur.execute(
            "SELECT COUNT(*) AS n FROM review_queue WHERE doc_id = ? AND status = 'open'", (doc_id,)
        ).fetchone()["n"]
        if remaining == 0:
            cur.execute("UPDATE documents SET status = ? WHERE doc_id = ?", (DocStatus.AUTO_APPROVED.value, doc_id))
        self.conn.commit()

    def approve_document(self, doc_id: str, reviewer: str = "reviewer") -> None:
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE review_queue SET status = 'approved', resolved_at = ?, resolved_by = ? "
            "WHERE doc_id = ? AND status = 'open'",
            (_now(), reviewer, doc_id),
        )
        cur.execute("UPDATE documents SET status = ? WHERE doc_id = ?", (DocStatus.AUTO_APPROVED.value, doc_id))
        self.conn.commit()

    def record_run(self, kind: str, extractor: str, split: str, settings_fp: str, metrics: dict) -> None:
        self.conn.execute(
            "INSERT INTO runs (kind, extractor, split, settings_fp, metrics_json, created_at) VALUES (?,?,?,?,?,?)",
            (kind, extractor, split, settings_fp, _json(metrics), _now()),
        )
        self.conn.commit()

    # -- reads -------------------------------------------------------------- #

    def list_documents(self, status: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
        query = "SELECT * FROM documents"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC, doc_id LIMIT ? OFFSET ?"
        params += [limit, offset]
        return [dict(row) for row in self.conn.execute(query, params)]

    def get_document(self, doc_id: str) -> dict | None:
        doc = self.conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if doc is None:
            return None
        fields = [
            {
                "name": row["name"],
                "value": json.loads(row["value_json"]) if row["value_json"] else None,
                "corrected": json.loads(row["corrected_value_json"]) if row["corrected_value_json"] else None,
                "raw_text": row["raw_text"],
                "confidence": row["confidence"],
                "page": row["page"],
                "bbox": json.loads(row["bbox_json"]) if row["bbox_json"] else None,
                "source": row["source"],
                "rule": row["rule"],
                "alternatives": json.loads(row["alternatives_json"]) if row["alternatives_json"] else [],
                "confidence_parts": json.loads(row["confidence_parts_json"]) if row["confidence_parts_json"] else {},
                "needs_review": bool(row["needs_review"]),
            }
            for row in self.conn.execute(
                "SELECT * FROM fields WHERE doc_id = ? ORDER BY name", (doc_id,)
            )
        ]
        issues = [dict(row) for row in self.conn.execute("SELECT * FROM issues WHERE doc_id = ?", (doc_id,))]
        tasks = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM review_queue WHERE doc_id = ? ORDER BY status, priority DESC", (doc_id,)
            )
        ]
        pages = [
            dict(row)
            for row in self.conn.execute(
                "SELECT page_no, width, height, text FROM pages WHERE doc_id = ? ORDER BY page_no", (doc_id,)
            )
        ]
        out = dict(doc)
        out["fields"] = fields
        out["issues"] = issues
        out["review_tasks"] = tasks
        out["pages"] = pages
        return out

    def stats(self) -> dict:
        cur = self.conn.cursor()
        docs = cur.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
        by_status = {
            row["status"]: row["n"]
            for row in cur.execute("SELECT status, COUNT(*) n FROM documents GROUP BY status")
        }
        open_tasks = cur.execute("SELECT COUNT(*) n FROM review_queue WHERE status='open'").fetchone()["n"]
        corrections = cur.execute("SELECT COUNT(*) n FROM corrections").fetchone()["n"]
        fields = cur.execute("SELECT COUNT(*) n FROM fields").fetchone()["n"]
        avg_conf = cur.execute("SELECT AVG(confidence) c FROM fields").fetchone()["c"] or 0.0
        return {
            "documents": docs,
            "by_status": by_status,
            "open_review_tasks": open_tasks,
            "corrections": corrections,
            "fields": fields,
            "mean_field_confidence": round(float(avg_conf), 4),
        }

    def export_corrections(self, limit: int = 1000) -> list[dict]:
        """Human corrections, in the shape a retraining set needs."""
        rows = self.conn.execute(
            """SELECT doc_id, field, old_value_json, new_value_json, reviewer, created_at
               FROM corrections ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        return [
            {
                "doc_id": row["doc_id"],
                "field": row["field"],
                "predicted": json.loads(row["old_value_json"]) if row["old_value_json"] else None,
                "corrected": json.loads(row["new_value_json"]) if row["new_value_json"] else None,
                "reviewer": row["reviewer"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def page_text(self, doc_id: str, page_no: int) -> str:
        row = self.conn.execute(
            "SELECT text FROM pages WHERE doc_id = ? AND page_no = ?", (doc_id, page_no)
        ).fetchone()
        return row["text"] if row else ""

    def all_page_texts(self) -> Iterable[tuple[str, int, str]]:
        for row in self.conn.execute("SELECT doc_id, page_no, text FROM pages ORDER BY doc_id, page_no"):
            yield row["doc_id"], row["page_no"], row["text"]

    def close(self) -> None:
        self.conn.close()
