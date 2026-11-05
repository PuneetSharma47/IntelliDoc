"""Word-level corruption used to simulate OCR noise on digital test documents.

Real Tesseract output is not available on every machine, and more importantly it
is not *reproducible*: the same scan can yield different tokens between versions.
To measure how the extractors degrade as character error rate (CER) rises, we
corrupt the text layer directly with a seeded, character-level noise model:
substitutions of confusable glyphs (``0/O``, ``1/l``, ``5/S``, ``8/B``, ``2/Z``),
deletions and merged tokens.

This isolates the *robustness* question ("how does accuracy fall as text quality
falls?") from the *OCR engine* question, which is measured separately when a
Tesseract binary is present.
"""

from __future__ import annotations

import random

from .schema import BBox, Document, Page, Token

CONFUSABLE = {
    "0": "OQD",
    "1": "7lI",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "Gb",
    "7": "1",
    "8": "B",
    "9": "g",
    "O": "0Q",
    "I": "1l",
    "l": "1I",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
    "A": "4",
}


def corrupt_text(text: str, cer: float, rng: random.Random) -> str:
    """Apply noise at roughly ``cer`` character error rate."""
    if cer <= 0 or not text:
        return text
    out: list[str] = []
    for ch in text:
        draw = rng.random()
        if draw < cer * 0.65:  # substitution
            pool = CONFUSABLE.get(ch)
            if pool:
                out.append(rng.choice(pool))
            else:
                out.append(ch)
        elif draw < cer:  # deletion
            continue
        else:
            out.append(ch)
    return "".join(out)


def degrade_document(doc: Document, cer: float, seed: int = 7, conf_floor: float = 0.55) -> Document:
    """Return a copy of ``doc`` whose tokens carry simulated OCR errors."""
    rng = random.Random(f"{seed}-{doc.doc_id}-{cer}")
    pages: list[Page] = []
    for page in doc.pages:
        tokens: list[Token] = []
        for tok in page.tokens:
            text = corrupt_text(tok.text, cer, rng)
            if not text.strip():
                continue
            tokens.append(
                Token(
                    text=text,
                    bbox=BBox(**tok.bbox.model_dump()),
                    page=tok.page,
                    ocr_confidence=max(conf_floor, 1.0 - cer * 2.2),
                    line_id=tok.line_id,
                    token_id=tok.token_id,
                )
            )
        pages.append(Page(number=page.number, width=page.width, height=page.height, tokens=tokens))
    clone = doc.model_copy(deep=True)
    clone.pages = pages
    return clone
