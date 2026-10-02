"""Value normalisation: money, dates, GSTIN, currencies.

Extraction returns *strings as printed*; every downstream consumer (arithmetic
validation, the database, the API) needs *typed values as meant*. That gap is
where most invoice pipelines quietly break, so the parsers here are explicit
about what they accept and raise :class:`NormalizationError` rather than
returning a plausible-looking wrong number.

Indian number grouping (``1,18,000.00``) is supported by construction: groups are
stripped and re-validated against both the Indian (2,2,3) and Western (3,3,3)
conventions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

CURRENCY_SYMBOLS = {
    "₹": "INR",
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
}

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_CURRENCY_STRIP = re.compile(
    r"(?i)(₹|rs\.?|inr|usd|eur|gbp|\$|€|£)\s*", re.UNICODE
)
_MONEY_CORE = re.compile(r"^[+-]?\d{1,3}(?:[,.\s]\d{2,3})*(?:[.,]\d{1,2})?$|^[+-]?\d+(?:[.,]\d{1,2})?$")

_DATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), "ymd"),
    (re.compile(r"^(\d{2})[/.\-](\d{2})[/.\-](\d{4})$"), "dmy"),
    (re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2})$"), "dmy2"),
    (re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?[\s\-]+([A-Za-z]{3,9})[\s\-,]+(\d{2,4})$"), "dMy"),
    (re.compile(r"^([A-Za-z]{3,9})[\s\-]+(\d{1,2})(?:st|nd|rd|th)?[\s,\-]+(\d{2,4})$"), "Mdy"),
]

GSTIN_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_RE = re.compile(r"^[0-3][0-9][A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]$")
GSTIN_LOOSE_RE = re.compile(r"\b([0-3][0-9][A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z])\b")

#: Common Indian GST slabs, used by the tax-rate sanity check.
GST_SLABS = (0.0, 0.25, 3.0, 5.0, 12.0, 18.0, 28.0)


class NormalizationError(ValueError):
    """Raised when a raw string cannot be turned into a typed value."""


@dataclass(frozen=True)
class Money:
    amount: float
    currency: str
    raw: str


def detect_currency(text: str) -> str | None:
    lowered = text.lower()
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in lowered:
            return code
    return None


def _strip_currency_junk(text: str) -> str:
    """Remove currency markers and leading/trailing junk glyphs.

    PDF extraction of a rupee sign from a font without that glyph yields things
    like ``·7,978.00`` or ``?7,978.00``; OCR yields ``~1,180.00``. The digits are
    still unambiguous, so anchor the match on the digit run instead of rejecting
    the token.
    """
    core = _CURRENCY_STRIP.sub("", text).strip()
    core = core.replace("(", "").replace(")", "")
    start = next((i for i, ch in enumerate(core) if ch.isdigit()), None)
    if start is None:
        return ""
    end = len(core)
    while end > start and not (core[end - 1].isdigit() or core[end - 1] in ".,"):
        end -= 1
    return core[start:end]


def looks_like_money(text: str) -> bool:
    if "%" in text:
        return False  # a rate, never an amount
    core = _strip_currency_junk(text)
    if not core or not any(ch.isdigit() for ch in core):
        return False
    return bool(_MONEY_CORE.match(core))


def _strip_grouping(digits: str) -> str:
    """Remove thousands separators and pick the decimal separator.

    Ambiguity ``1,180`` / ``1.180`` is resolved by the number of trailing digits:
    three trailing digits after a lone separator is a thousands group, two is a
    decimal fraction.
    """
    if "." in digits and "," in digits:
        if digits.rfind(",") > digits.rfind("."):
            return digits.replace(".", "").replace(",", ".")
        return digits.replace(",", "")
    if "," in digits:
        parts = digits.split(",")
        # A lone two-digit group after a short leading group ("1,18") is a decimal
        # comma; anything else is a thousands separator.
        if len(parts) == 2 and len(parts[-1]) == 2 and len(parts[0]) <= 3:
            return "".join(parts[:-1]) + "." + parts[-1]
        return "".join(parts)
    if digits.count(".") > 1:
        head, _, tail = digits.rpartition(".")
        return head.replace(".", "") + "." + tail
    return digits


def parse_money(raw: str, default_currency: str = "INR") -> Money:
    """Parse a printed money string into ``(amount, currency)``.

    >>> parse_money("₹1,18,000.00").amount
    118000.0
    >>> parse_money("Rs. 11,800/-").amount
    11800.0
    """
    text = raw.strip().replace("\u00a0", " ")
    negative = text.startswith("(") and text.endswith(")") or text.strip().startswith("-")
    currency = detect_currency(text) or default_currency
    core = _strip_currency_junk(text.replace("/-", "").replace("−", "-"))
    if not core or not any(ch.isdigit() for ch in core):
        raise NormalizationError(f"no numeric content in {raw!r}")
    sign = -1.0 if core.startswith("-") or negative else 1.0
    core = core.lstrip("+-")
    try:
        amount = float(_strip_grouping(core)) * sign
    except ValueError as exc:  # pragma: no cover - guarded above
        raise NormalizationError(f"cannot parse money {raw!r}") from exc
    return Money(amount=round(amount, 2), currency=currency, raw=raw)


def looks_like_date(text: str) -> bool:
    try:
        parse_date(text)
        return True
    except NormalizationError:
        return False


def _fix_year(year: int) -> int:
    if year < 100:
        return 2000 + year if year <= 69 else 1900 + year
    return year


def parse_date(raw: str, dayfirst: bool = True) -> date:
    """Parse the date formats that actually appear on Indian invoices.

    ``dayfirst`` decides 05/03/2026; Indian and UK documents put the day first,
    US documents do not. The pipeline exposes it as configuration instead of
    guessing per document, and the evaluation harness reports the cost of
    guessing wrong.
    """
    text = raw.strip().rstrip(",.").replace("\u00a0", " ")
    text = re.sub(r"^(dated|date|dt\.?)[:\s]+", "", text, flags=re.I).strip()
    for pattern, kind in _DATE_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        groups = match.groups()
        try:
            if kind == "ymd":
                return date(int(groups[0]), int(groups[1]), int(groups[2]))
            if kind == "dmy":
                d, m, y = int(groups[0]), int(groups[1]), int(groups[2])
                if not dayfirst and d <= 12:
                    d, m = m, d
                return date(y, m, d)
            if kind == "dmy2":
                d, m, y = int(groups[0]), int(groups[1]), _fix_year(int(groups[2]))
                if not dayfirst and d <= 12:
                    d, m = m, d
                return date(y, m, d)
            if kind == "dMy":
                d, mon, y = int(groups[0]), groups[1].lower()[:4].rstrip("."), int(groups[2])
                month = _MONTHS.get(mon[:3])
                if month is None:
                    continue
                return date(_fix_year(y), month, d)
            if kind == "Mdy":
                mon, d, y = groups[0].lower()[:4].rstrip("."), int(groups[1]), int(groups[2])
                month = _MONTHS.get(mon[:3])
                if month is None:
                    continue
                return date(_fix_year(y), month, d)
        except ValueError:
            continue  # e.g. 31/02/2026 -- fall through to the error below
    raise NormalizationError(f"unrecognised date {raw!r}")


def gstin_checksum(gstin: str) -> str:
    """Return the 15th character implied by the first 14 characters.

    Implements the check digit defined in the GSTIN specification (mod-36 with
    alternating weights 1/2), which catches the transposition errors OCR makes.
    """
    if len(gstin) < 14:
        raise NormalizationError("GSTIN too short")
    total = 0
    for i, ch in enumerate(gstin[:14]):
        if ch not in GSTIN_CHARSET:
            raise NormalizationError(f"invalid GSTIN character {ch!r}")
        value = GSTIN_CHARSET.index(ch) * (2 if i % 2 else 1)
        total += value // 36 + value % 36
    return GSTIN_CHARSET[(36 - total % 36) % 36]


def valid_gstin(gstin: str) -> bool:
    gstin = gstin.strip().upper()
    if not GSTIN_RE.match(gstin):
        return False
    try:
        return gstin_checksum(gstin) == gstin[14]
    except NormalizationError:
        return False


def find_gstins(text: str) -> list[str]:
    return GSTIN_LOOSE_RE.findall(text.upper())


def canonical_number(raw: str) -> str:
    """Uppercase and strip separators from identifiers such as INV-1048."""
    return re.sub(r"\s+", "", raw).upper()


def parse_datetime(raw: str) -> datetime:
    return datetime.combine(parse_date(raw), datetime.min.time())


def round_money(value: float) -> float:
    return round(float(value) + 1e-9, 2)


def money_equal(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(round_money(a) - round_money(b)) <= tol
