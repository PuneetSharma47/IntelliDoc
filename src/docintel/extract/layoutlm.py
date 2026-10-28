"""LayoutLMv3 adapter (optional, GPU recommended).

Provides the same interface as the rule and linear-tagger extractors, so the
evaluation harness can compare them on identical inputs. Three modes of use:

1. **Inference from a fine-tuned checkpoint** -- :class:`LayoutLMv3Extractor`
   loads a directory produced by ``scripts/train_layoutlmv3.py`` and tags tokens.
2. **Training data export** -- :func:`training_records` turns the labelled corpus
   into the word-level BIO records LayoutLMv3 expects, with boxes normalised to
   the 0-1000 grid the model was pretrained on.
3. **Graceful absence** -- if torch/transformers is not installed or the
   checkpoint directory is missing, :meth:`LayoutLMv3Extractor.available` is
   ``False`` and the evaluation reports the model as *not run* rather than
   silently substituting another extractor's numbers.

The adapter deliberately does not download a model at import time: inference must
work offline in CI, and a 500 MB download is not something an import should do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..normalize import NormalizationError, parse_date, parse_money
from ..schema import BBox, Document, FieldName, Provenance, Token
from .base import Candidate, ExtractionContext
from .learned import LABEL_TO_FIELD, LABELS, weak_labels
from .rules import GSTIN_LOOSE_RE

MAX_LEN = 512
STRIDE = 128


def available(model_dir: str | Path) -> bool:
    path = Path(model_dir)
    if not path.exists() or not (path / "config.json").exists():
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def missing_reason(model_dir: str | Path) -> str:
    path = Path(model_dir)
    if not path.exists():
        return f"checkpoint directory not found: {path}"
    if not (path / "config.json").exists():
        return f"no config.json in {path}"
    try:
        import torch  # noqa: F401
    except ImportError:
        return "torch not installed (pip install 'intellidoc[ml]')"
    try:
        import transformers  # noqa: F401
    except ImportError:
        return "transformers not installed (pip install 'intellidoc[ml]')"
    return "available"


def normalize_boxes(tokens: list[Token], page_width: float, page_height: float) -> list[list[int]]:
    """Scale boxes to the 0-1000 grid LayoutLM expects."""
    boxes = []
    for tok in tokens:
        boxes.append(
            [
                max(0, min(1000, int(round(tok.bbox.x0 / page_width * 1000)))),
                max(0, min(1000, int(round(tok.bbox.y0 / page_height * 1000)))),
                max(0, min(1000, int(round(tok.bbox.x1 / page_width * 1000)))),
                max(0, min(1000, int(round(tok.bbox.y1 / page_height * 1000)))),
            ]
        )
    return boxes


@dataclass
class WordChunk:
    tokens: list[Token]
    boxes: list[list[int]]


def chunk_document(doc: Document) -> list[WordChunk]:
    """Split a document into overlapping 512-word windows, page by page."""
    chunks: list[WordChunk] = []
    for page in doc.pages:
        tokens = [t for t in page.tokens if t.text.strip()]
        if not tokens:
            continue
        boxes = normalize_boxes(tokens, page.width, page.height)
        start = 0
        while start < len(tokens):
            end = min(start + MAX_LEN, len(tokens))
            chunks.append(WordChunk(tokens=tokens[start:end], boxes=boxes[start:end]))
            if end == len(tokens):
                break
            start = end - STRIDE
    return chunks


def training_records(docs: list[tuple[Document, dict]]) -> list[dict[str, Any]]:
    """Build ``{tokens, bboxes, labels}`` records for fine-tuning."""
    records: list[dict[str, Any]] = []
    for doc, labels in docs:
        word_tags = weak_labels(doc, labels)
        cursor = 0
        for page in doc.pages:
            tokens = [t for t in page.tokens if t.text.strip()]
            if not tokens:
                continue
            tags = word_tags[cursor: cursor + len(tokens)]
            cursor += len(tokens)
            records.append(
                {
                    "id": f"{doc.doc_id}-p{page.number}",
                    "tokens": [t.text for t in tokens],
                    "bboxes": normalize_boxes(tokens, page.width, page.height),
                    "labels": tags,
                }
            )
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


class LayoutLMv3Extractor:
    """Token-classification extractor backed by a fine-tuned LayoutLMv3."""

    name = "layoutlmv3"

    def __init__(self, model_dir: str | Path, base_model: str = "microsoft/layoutlmv3-base") -> None:
        self.model_dir = Path(model_dir)
        self.base_model = base_model
        self._model = None
        self._tokenizer = None

    def available(self) -> bool:
        return available(self.model_dir)

    def _load(self) -> None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, apply_ocr=False)
        self._model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self._model.eval()

    def predict_tags(self, doc: Document) -> list[tuple[Token, str, float]]:
        import torch

        if self._model is None:
            self._load()
        assert self._tokenizer is not None and self._model is not None
        out: list[tuple[Token, str, float]] = []
        for chunk in chunk_document(doc):
            encoded = self._tokenizer(
                chunk.tokens,
                boxes=chunk.boxes,
                truncation=True,
                max_length=MAX_LEN,
                padding="max_length",
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self._model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1)
            word_ids = encoded.word_ids(0)
            seen: set[int] = set()
            for position, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx in seen:
                    continue
                seen.add(word_idx)
                best = int(torch.argmax(probs[position]))
                label = self._model.config.id2label.get(best, "O")
                if label not in LABELS:
                    label = "O"
                out.append((chunk.tokens[word_idx], label, float(probs[position][best])))
        return out

    def extract(self, ctx: ExtractionContext) -> list[Candidate]:
        tagged = self.predict_tags(ctx.document)
        candidates: list[Candidate] = []
        best: dict[FieldName, tuple[float, Token, str]] = {}
        for tok, tag, prob in tagged:
            field = LABEL_TO_FIELD.get(tag)
            if field is None:
                continue
            current = best.get(field)
            if current is None or prob > current[0]:
                best[field] = (prob, tok, tag)
        for field, (prob, tok, _tag) in best.items():
            value = self._typed_value(field, tok)
            if value is None:
                continue
            candidates.append(
                Candidate(
                    field=field,
                    value=value,
                    raw=tok.text,
                    score=0.5 + 0.48 * min(1.0, prob),
                    provenance=Provenance(
                        page=tok.page,
                        bbox=BBox(**tok.bbox.model_dump()),
                        source="layoutlmv3",
                        raw_text=tok.text,
                        rule="token_classification",
                    ),
                )
            )
        return candidates

    def _typed_value(self, field: FieldName, tok: Token):
        try:
            if field in (FieldName.SUBTOTAL, FieldName.TAX_AMOUNT, FieldName.TOTAL_AMOUNT):
                return parse_money(tok.text).amount
            if field in (FieldName.ISSUE_DATE, FieldName.DUE_DATE):
                return parse_date(tok.text)
        except NormalizationError:
            return None
        if field is FieldName.INVOICE_NUMBER:
            return tok.text.strip(".,;:")
        if field is FieldName.VENDOR_GSTIN:
            found = GSTIN_LOOSE_RE.findall(tok.text.upper())
            return found[0] if found else None
        if field is FieldName.VENDOR_NAME:
            return tok.text
        return None
