"""A learned, layout-aware token tagger (no GPU required).

This exists to answer a fair question: is the hand-written rule set actually
earning its keep, or would a small trained model do better? It is a multinomial
logistic regression over per-token features -- normalised position, gap to
neighbours, header/footer band flags, currency/date/number shape, plus a hashed
bag-of-words of the token text. Labels come from *weak supervision*: the synthetic
label file tells us the correct value, and we match it back to a token using the
same label anchors a human would use. That is a legitimate bootstrap for a
prototype and a known limitation (the tagger inherits the rules' notion of where
to label), documented in docs/DATASET.md.

At inference each token gets a class distribution; the per-field winner becomes a
:cclass:`~docintel.extract.base.Candidate` whose score is a monotone map of the
class probability, so downstream confidence code needs no special case.

For the transformer comparison (LayoutLMv3) see :mod:`docintel.extract.layoutlm`.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from ..layout import FEATURE_NAMES, document_lines, token_features
from ..normalize import (
    NormalizationError,
    looks_like_date,
    looks_like_money,
    parse_date,
    parse_money,
)
from ..schema import BBox, Document, FieldName, Provenance, Token
from .base import Candidate, ExtractionContext
from .rules import COMPILED, GSTIN_LOOSE_RE

LABELS = [
    "O",
    "B_ID",
    "B_ISSUE",
    "B_DUE",
    "B_VENDOR",
    "I_VENDOR",
    "B_BUYER",
    "I_BUYER",
    "B_GSTIN",
    "B_SUBTOTAL",
    "B_TAX",
    "B_TOTAL",
]
LABEL_TO_FIELD = {
    "B_ID": FieldName.INVOICE_NUMBER,
    "B_ISSUE": FieldName.ISSUE_DATE,
    "B_DUE": FieldName.DUE_DATE,
    "B_VENDOR": FieldName.VENDOR_NAME,
    "B_BUYER": FieldName.BUYER_NAME,
    "B_GSTIN": FieldName.VENDOR_GSTIN,
    "B_SUBTOTAL": FieldName.SUBTOTAL,
    "B_TAX": FieldName.TAX_AMOUNT,
    "B_TOTAL": FieldName.TOTAL_AMOUNT,
}
FIELD_TO_LABEL = {v: k for k, v in LABEL_TO_FIELD.items()}

HASH_DIM = 256
N_LAYOUT = len(FEATURE_NAMES)
#: hash(token text) + hash(left neighbour) + line keyword flags
N_CONTEXT = HASH_DIM + HASH_DIM
LINE_KEYWORDS = ("invoice", "date", "due", "total", "subtotal", "tax", "gst", "bill")
N_FEATURES = N_LAYOUT + N_CONTEXT + len(LINE_KEYWORDS)


def _hash_bucket(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) % HASH_DIM


def _token_text_key(text: str) -> str:
    """Collapse digits so ``INV-1048`` and ``INV-2201`` share a feature."""
    out = []
    for ch in text.lower():
        if ch.isdigit():
            if not out or out[-1] != "9":
                out.append("9")
        else:
            out.append(ch)
    return "".join(out)[:24]


def featurize(doc: Document) -> tuple[list[Token], np.ndarray]:
    """Build the design matrix for one document.

    Features are geometry (from :mod:`docintel.layout`), the hashed token text, the
    hashed *left neighbour* text and a few line-level keyword flags. The neighbour
    feature matters: "Invoice" vs "Due" is the only difference between an issue
    date and a due date, and a strictly per-token model cannot see it.
    """
    lines_by_token = {}
    for line in document_lines(doc):
        for tok in line.tokens:
            lines_by_token[id(tok)] = line
    tokens: list[Token] = []
    rows: list[np.ndarray] = []
    for page in doc.pages:
        by_line: dict[int, list[Token]] = {}
        for tok in page.tokens:
            by_line.setdefault(tok.line_id, []).append(tok)
        for line_tokens in by_line.values():
            line_tokens.sort(key=lambda t: t.bbox.x0)
        for tok in page.tokens:
            vec = np.zeros(N_FEATURES, dtype=np.float32)
            vec[:N_LAYOUT] = token_features(tok, page, lines_by_token)
            vec[N_LAYOUT + _hash_bucket(_token_text_key(tok.text))] += 1.0
            siblings = by_line.get(tok.line_id, [])
            index = siblings.index(tok) if tok in siblings else 0
            if index > 0:
                vec[N_LAYOUT + HASH_DIM + _hash_bucket(_token_text_key(siblings[index - 1].text))] += 1.0
            line_text = " ".join(t.text for t in siblings).lower()
            base = N_LAYOUT + 2 * HASH_DIM
            for offset, keyword in enumerate(LINE_KEYWORDS):
                if keyword in line_text:
                    vec[base + offset] = 1.0
            tokens.append(tok)
            rows.append(vec)
    if not rows:
        return [], np.zeros((0, N_FEATURES), dtype=np.float32)
    return tokens, np.vstack(rows)


# --------------------------------------------------------------------------- #
# Weak supervision
# --------------------------------------------------------------------------- #


def _anchor_rows(doc: Document):
    from .rules import build_rows

    return build_rows(doc)


def _nearest_anchor_y(rows, field_key: str) -> list[float]:
    pattern = COMPILED[field_key]
    return [row.y for row in rows if pattern.search(row.text)]


def _span_tokens(row, name: str) -> list[Token]:
    """Tokens of ``row`` that spell out ``name`` (prefix-matched, in order)."""
    labelled: list[Token] = []
    acc = ""
    for tok in row.line.tokens:
        acc = f"{acc} {tok.text}".strip()
        if acc.lower() in name.lower() or name.lower().startswith(acc.lower()):
            labelled.append(tok)
        else:
            break
    return labelled


def weak_labels(doc: Document, labels: dict) -> list[str]:
    """Map ground-truth values back to tokens to create BIO-style labels."""
    from ..normalize import canonical_number

    rows = _anchor_rows(doc)
    targets: dict[str, list[tuple[float, Token]]] = {label: [] for label in LABELS}

    def register(label: str, token_values: list[Token], anchors: list[float]) -> None:
        for tok in token_values:
            dist = min((abs(tok.bbox.cy - y) for y in anchors), default=1e6)
            targets[label].append((dist, tok))

    # invoice number
    gt_id = canonical_number(str(labels.get("invoice_number") or ""))
    if gt_id:
        hits = [t for t in doc.tokens if canonical_number(t.text) and canonical_number(t.text) in gt_id]
        register("B_ID", hits, _nearest_anchor_y(rows, "invoice_number"))

    # dates
    for key, label in (("issue_date", "B_ISSUE"), ("due_date", "B_DUE")):
        raw = labels.get(key)
        if not raw:
            continue
        try:
            target = parse_date(str(raw))
        except NormalizationError:
            continue
        hits = []
        for tok in doc.tokens:
            if not looks_like_date(tok.text):
                continue
            try:
                if parse_date(tok.text) == target:
                    hits.append(tok)
            except NormalizationError:
                continue
        register(label, hits, _nearest_anchor_y(rows, "issue_date" if key == "issue_date" else "due_date"))

    # vendor name -> token span
    vendor = str(labels.get("vendor_name") or "").strip()
    if vendor:
        vendor_rows = [row for row in rows if vendor.lower() in row.text.lower()]
        if vendor_rows:
            for i, tok in enumerate(_span_tokens(vendor_rows[0], vendor)):
                targets["B_VENDOR" if i == 0 else "I_VENDOR"].append((0.0, tok))

    # buyer name -> token span (anchored on the buyer block header)
    buyer = str(labels.get("buyer_name") or "").strip()
    if buyer:
        buyer_rows = [row for row in rows if buyer.lower() in row.text.lower()]
        if buyer_rows:
            for i, tok in enumerate(_span_tokens(buyer_rows[0], buyer)):
                targets["B_BUYER" if i == 0 else "I_BUYER"].append((0.0, tok))

    # gstin
    gt_gstin = str(labels.get("vendor_gstin") or "").upper()
    if gt_gstin:
        hits = [t for t in doc.tokens if gt_gstin in t.text.upper().replace("-", "")]
        if not hits:
            hits = [t for t in doc.tokens for g in GSTIN_LOOSE_RE.findall(t.text.upper()) if g == gt_gstin]
        register("B_GSTIN", hits, _nearest_anchor_y(rows, "gstin"))

    # money fields
    for key, label, anchor_key in (
        ("subtotal", "B_SUBTOTAL", "subtotal"),
        ("total_amount", "B_TOTAL", "total_amount"),
        ("tax_amount", "B_TAX", "tax"),
    ):
        value = labels.get(key)
        if value is None:
            continue
        anchors = _nearest_anchor_y(rows, anchor_key)
        hits = []
        for tok in doc.tokens:
            if not looks_like_money(tok.text):
                continue
            try:
                amount = parse_money(tok.text).amount
            except NormalizationError:
                continue
            if abs(amount - float(value)) <= 0.011:
                hits.append(tok)
        if key == "tax_amount" and len(hits) > 1:
            # CGST/SGST components each hold half; keep the strongest anchor only
            pass
        register(label, hits, anchors)

    # Resolve conflicts: one token, one label, nearest anchor wins.
    best: dict[int, tuple[float, str]] = {}
    for label, entries in targets.items():
        if label == "O":
            continue
        for dist, tok in entries:
            cur = best.get(id(tok))
            if cur is None or dist < cur[0]:
                best[id(tok)] = (dist, label)

    out: dict[int, str] = {}
    for page in doc.pages:
        for tok in page.tokens:
            out[id(tok)] = best.get(id(tok), (0.0, "O"))[1]
    ordered: list[str] = []
    for page in doc.pages:
        for tok in page.tokens:
            ordered.append(out[id(tok)])
    # An I-VENDOR without a preceding B-VENDOR becomes B-VENDOR
    for i, tag in enumerate(ordered):
        for base in ("VENDOR", "BUYER"):
            if tag == f"I_{base}" and (i == 0 or ordered[i - 1] not in (f"B_{base}", f"I_{base}")):
                ordered[i] = f"B_{base}"
    return ordered


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class LearnedTagger:
    """Multinomial logistic regression over layout + text-hash features."""

    name = "learned_layout"

    def __init__(self, model=None, dayfirst: bool = True) -> None:
        self.model = model
        self.dayfirst = dayfirst

    # -- training ----------------------------------------------------------- #

    def fit(self, examples: list[tuple[Document, dict]]) -> LearnedTagger:
        from sklearn.linear_model import LogisticRegression

        X_parts: list[np.ndarray] = []
        y_parts: list[str] = []
        for doc, labels in examples:
            tokens, X = featurize(doc)
            if not tokens:
                continue
            tags = weak_labels(doc, labels)
            if len(tags) != len(tokens):
                continue
            X_parts.append(X)
            y_parts.extend(tags)
        if not X_parts:
            raise ValueError("no training documents")
        X = np.vstack(X_parts)
        y = np.array(y_parts)
        self.model = LogisticRegression(
            max_iter=800,
            C=1.0,
            class_weight="balanced",
            multi_class="multinomial",
            n_jobs=None,
        )
        self.model.fit(X, y)
        return self

    # -- persistence -------------------------------------------------------- #

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": FEATURE_NAMES, "labels": LABELS}, path)
        return path

    @classmethod
    def load(cls, path: str | Path, dayfirst: bool = True) -> LearnedTagger:
        import joblib

        blob = joblib.load(path)
        return cls(model=blob["model"], dayfirst=dayfirst)

    def available(self, path: str | Path) -> bool:
        return Path(path).exists()

    # -- inference ---------------------------------------------------------- #

    def predict_tags(self, doc: Document) -> list[tuple[Token, str, float]]:
        if self.model is None:
            raise RuntimeError("learned tagger has no fitted model")
        tokens, X = featurize(doc)
        if not tokens:
            return []
        proba = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        out: list[tuple[Token, str, float]] = []
        for i, tok in enumerate(tokens):
            idx = int(np.argmax(proba[i]))
            out.append((tok, classes[idx], float(proba[i][idx])))
        return out

    def extract(self, ctx: ExtractionContext) -> list[Candidate]:
        tagged = self.predict_tags(ctx.document)
        if not tagged:
            return []

        candidates: list[Candidate] = []
        # span fields: party names are runs of B-/I- tags on one line
        for prefix, field in (("VENDOR", FieldName.VENDOR_NAME), ("BUYER", FieldName.BUYER_NAME)):
            for tokens, prob in self._runs(tagged, prefix):
                name = " ".join(t.text for t in tokens).strip(" :-")
                if len(name) < 3:
                    continue
                candidates.append(_candidate(field, name, name, tokens, prob, "token_tagger_span"))

        picks: dict[FieldName, list[tuple[float, Token]]] = {}
        for tok, tag, prob in tagged:
            field = LABEL_TO_FIELD.get(tag)
            if field is None or field in (FieldName.VENDOR_NAME, FieldName.BUYER_NAME):
                continue
            picks.setdefault(field, []).append((prob, tok))

        for field, items in picks.items():
            if field is FieldName.TAX_AMOUNT:
                # CGST and SGST are separate tokens; a token tagger emits one label
                # per token, so tax components are summed at decode time. This is a
                # decoding constraint, not a hand-written anchor rule.
                components: dict[tuple[int, int], tuple[float, Token]] = {}
                for prob, tok in items:
                    if prob < 0.40:
                        continue
                    key = (tok.page, tok.line_id)
                    if key not in components or prob > components[key][0]:
                        components[key] = (prob, tok)
                sums: list[float] = []
                used: list[Token] = []
                for _prob, tok in components.values():
                    try:
                        sums.append(parse_money(tok.text).amount)
                        used.append(tok)
                    except NormalizationError:
                        continue
                if not sums:
                    continue
                if len(sums) == 1:
                    candidates.append(_candidate(field, sums[0], used[0].text, used, components[(used[0].page, used[0].line_id)][0], "token_tagger"))
                else:
                    candidates.append(
                        _candidate(field, round(sum(sums), 2), " + ".join(t.text for t in used), used, min(p for p, _ in components.values()), "token_tagger_sum")
                    )
                continue

            prob, tok = max(items, key=lambda item: item[0])
            value = self._typed_value(field, tok, tagged)
            if value is None:
                continue
            candidates.append(_candidate(field, value, tok.text, [tok], prob, "token_tagger"))
        return candidates

    def _runs(self, tagged, prefix: str) -> list[tuple[list[Token], float]]:
        runs: list[tuple[list[Token], float]] = []
        current: list[Token] = []
        probs: list[float] = []
        for tok, tag, prob in tagged:
            if tag in (f"B_{prefix}", f"I_{prefix}"):
                if current and token_line(tok) != token_line(current[-1]):
                    runs.append((current, min(probs)))
                    current, probs = [], []
                current.append(tok)
                probs.append(prob)
            elif current:
                runs.append((current, min(probs)))
                current, probs = [], []
        if current:
            runs.append((current, min(probs)))
        return runs

    def _typed_value(self, field: FieldName, tok: Token, tagged):
        """Convert the tagged token to a typed value, joining split tokens.

        Dates and amounts are often split across tokens ("22" "Feb" "2026"), so a
        failed parse on the single token retries with the following tokens on the
        same line before giving up.
        """
        line_tokens = sorted(
            [t for t, _tag, _prob in tagged if t.page == tok.page and t.line_id == tok.line_id],
            key=lambda t: t.bbox.x0,
        )
        try:
            index = line_tokens.index(tok)
        except ValueError:
            index = 0
        neighbours = [tok]
        for offset in range(1, 3):
            for candidate in (index + offset, index - offset):
                if 0 <= candidate < len(line_tokens):
                    neighbours.append(line_tokens[candidate])

        if field in (FieldName.SUBTOTAL, FieldName.TAX_AMOUNT, FieldName.TOTAL_AMOUNT):
            for token in neighbours:
                try:
                    return parse_money(token.text).amount
                except NormalizationError:
                    continue
            return None
        if field in (FieldName.ISSUE_DATE, FieldName.DUE_DATE):
            windows = [
                [tok],
                line_tokens[index: index + 2],
                line_tokens[max(0, index - 1): index + 1],
                line_tokens[index: index + 3],
                line_tokens[max(0, index - 2): index + 1],
            ]
            for window in windows:
                if not window:
                    continue
                joined = " ".join(t.text for t in window)
                try:
                    return parse_date(joined, dayfirst=self.dayfirst)
                except NormalizationError:
                    continue
            return None
        if field is FieldName.INVOICE_NUMBER:
            text = tok.text.strip(".,;:")
            if len(text) < 3:
                return None
            return text + _suffix_id(tok, tagged)
        if field is FieldName.VENDOR_GSTIN:
            found = GSTIN_LOOSE_RE.findall(tok.text.upper())
            if found:
                return found[0]
            if len(tok.text) == 15:
                return tok.text.upper()
            return None
        return None


def token_line(tok: Token) -> int:
    return (tok.page, tok.line_id)


def _suffix_id(tok: Token, tagged) -> str:
    """Reattach a split identifier: ``INV`` token followed by ``1048``."""
    if not tok.text.endswith(("-", "/")):
        return ""
    for other, _tag, _prob in tagged:
        if other is tok:
            continue
        if other.page == tok.page and other.line_id == tok.line_id and other.bbox.x0 > tok.bbox.x1:
            return other.text
    return ""


def _candidate(field: FieldName, value, raw: str, tokens: list[Token], prob: float, rule: str) -> Candidate:
    box: BBox | None = None
    for tok in tokens:
        box = tok.bbox if box is None else box.union(tok.bbox)
    # map class probability into the same score range the rules use, so a single
    # threshold policy applies to both (documented calibration, not a claim).
    score = 0.55 + 0.42 * max(0.0, min(1.0, prob))
    return Candidate(
        field=field,
        value=value,
        raw=raw,
        score=score,
        provenance=Provenance(
            page=tokens[0].page if tokens else 0,
            bbox=box,
            source="learned_layout",
            raw_text=raw,
            rule=rule,
        ),
    )
