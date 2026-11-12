from datetime import date

import pytest

from docintel.normalize import (
    NormalizationError,
    gstin_checksum,
    looks_like_date,
    looks_like_money,
    parse_date,
    parse_money,
    valid_gstin,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("₹1,18,000.00", 118000.0),
        ("Rs. 11,800/-", 11800.0),
        ("INR 1,180", 1180.0),
        ("1,18,000", 118000.0),
        ("118000.50", 118000.5),
        ("·7,978.00", 7978.0),          # rupee glyph lost by the PDF font
        ("(1,180.00)", -1180.0),        # credit notation
        ("-250.75", -250.75),
    ],
)
def test_parse_money(raw, expected):
    assert parse_money(raw).amount == pytest.approx(expected, abs=0.001)


def test_parse_money_currency_detection():
    assert parse_money("$1,200.00").currency == "USD"
    assert parse_money("₹900").currency == "INR"


@pytest.mark.parametrize("raw", ["", "abc", "Rs.", "--"])
def test_parse_money_rejects_garbage(raw):
    with pytest.raises(NormalizationError):
        parse_money(raw)


def test_looks_like_money_shape_checks():
    assert looks_like_money("1,180.00")
    assert not looks_like_money("14%")      # a rate, not an amount
    assert not looks_like_money("INV")
    assert not looks_like_money("")


def test_identifier_guard_lives_in_the_extractor():
    """normalize() is deliberately permissive; the identifier guard is policy.

    ``GST-8809`` parses as a number, so rejecting it belongs to the extractor
    (see ``docintel.extract.rules._is_amount_token``), not the value parser.
    """
    from docintel.extract.rules import _is_amount_token

    assert looks_like_money("GST-8809")
    assert not _is_amount_token("GST-8809")
    assert _is_amount_token("Rs. 1,180.00")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("05/03/2026", date(2026, 3, 5)),
        ("05-03-2026", date(2026, 3, 5)),
        ("05 Mar 2026", date(2026, 3, 5)),
        ("2026-03-05", date(2026, 3, 5)),
        ("5 Mar 26", date(2026, 3, 5)),
        ("Mar 5, 2026", date(2026, 3, 5)),
        ("Dated: 05/03/2026", date(2026, 3, 5)),
    ],
)
def test_parse_date_formats(raw, expected):
    assert parse_date(raw) == expected


def test_parse_date_dayfirst_switch():
    assert parse_date("05/03/2026", dayfirst=True) == date(2026, 3, 5)
    assert parse_date("05/03/2026", dayfirst=False) == date(2026, 5, 3)


def test_parse_date_rejects_impossible():
    with pytest.raises(NormalizationError):
        parse_date("31/02/2026")
    assert not looks_like_date("not a date")


def test_gstin_checksum_roundtrip():
    gstin = "27AAPFU0939F1ZV"
    assert gstin_checksum(gstin[:14]) == gstin[14]
    assert valid_gstin(gstin)
    assert not valid_gstin("27AAPFU0939F1ZX")  # wrong check digit
    assert not valid_gstin("27AAPFU0939F12V")  # wrong length
