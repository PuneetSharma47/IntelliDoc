"""Document library search with page-level citations.

Answers come back as *page hits* rather than document hits: "where in this file
does it say HDFC" is the question a reviewer actually asks, and the page number is
what makes the answer checkable. Ranking uses SQLite FTS5 BM25 when available and
falls back to a LIKE scan otherwise, so the feature never breaks on a SQLite build
without FTS5.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from .store import Store

TOKEN_RE = re.compile(r"[A-Za-z0-9@/_.,:-]+")


@dataclass
class SearchHit:
    doc_id: str
    filename: str
    page: int
    snippet: str
    score: float
    matched_terms: list[str]

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page": self.page,
            "snippet": self.snippet,
            "score": round(self.score, 4),
            "matched_terms": self.matched_terms,
        }


def _sanitize(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    FTS5 treats a long list of characters as operators; a user typing
    ``INV-1048 (paid)`` must not blow up the query. Terms are quoted, which keeps
    hyphens and slashes literal.
    """
    terms = TOKEN_RE.findall(query)
    terms = [t for t in terms if len(t) > 1 or t.isalnum()]
    return " OR ".join(f'"{t}"' for t in terms[:12])


def _snippet(text: str, terms: list[str], width: int = 220) -> str:
    lowered = text.lower()
    position = -1
    for term in terms:
        position = lowered.find(term.lower())
        if position >= 0:
            break
    if position < 0:
        return text[:width].replace("\n", " ").strip()
    start = max(0, position - width // 3)
    end = min(len(text), start + width)
    prefix = "…" if start else ""
    suffix = "…" if end < len(text) else ""
    return (prefix + text[start:end].replace("\n", " ").strip() + suffix)


class Searcher:
    def __init__(self, store: Store, top_k: int = 10, snippet_chars: int = 220) -> None:
        self.store = store
        self.top_k = top_k
        self.snippet_chars = snippet_chars

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        terms = TOKEN_RE.findall(query)
        if not terms:
            return []
        top_k = top_k or self.top_k
        if self.store.has_fts:
            return self._search_fts(query, terms, top_k)
        return self._search_like(terms, top_k)

    def _search_fts(self, query: str, terms: list[str], top_k: int) -> list[SearchHit]:
        match = _sanitize(query)
        if not match:
            return []
        sql = """
            SELECT f.doc_id, f.page_no, f.text, d.filename, bm25(page_fts) AS score
            FROM page_fts f JOIN documents d ON d.doc_id = f.doc_id
            WHERE page_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """
        try:
            rows = self.store.conn.execute(sql, (match, top_k)).fetchall()
        except sqlite3.OperationalError:
            return self._search_like(terms, top_k)
        hits = []
        for row in rows:
            matched = [t for t in terms if t.lower() in row["text"].lower()]
            hits.append(
                SearchHit(
                    doc_id=row["doc_id"],
                    filename=row["filename"],
                    page=int(row["page_no"]),
                    snippet=_snippet(row["text"], terms, self.snippet_chars),
                    score=-float(row["score"]),  # bm25 is ascending: lower is better
                    matched_terms=matched,
                )
            )
        return hits

    def _search_like(self, terms: list[str], top_k: int) -> list[SearchHit]:
        hits: list[SearchHit] = []
        pattern = "%" + terms[0] + "%"
        rows = self.store.conn.execute(
            """SELECT p.doc_id, p.page_no, p.text, d.filename
               FROM pages p JOIN documents d ON d.doc_id = p.doc_id
               WHERE p.text LIKE ? LIMIT 200""",
            (pattern,),
        ).fetchall()
        for row in rows:
            text = row["text"]
            matched = [t for t in terms if t.lower() in text.lower()]
            score = len(matched) + text.lower().count(terms[0].lower()) * 0.01
            hits.append(
                SearchHit(
                    doc_id=row["doc_id"],
                    filename=row["filename"],
                    page=int(row["page_no"]),
                    snippet=_snippet(text, terms, self.snippet_chars),
                    score=score,
                    matched_terms=matched,
                )
            )
        hits.sort(key=lambda h: -h.score)
        return hits[:top_k]
