# Failure cases

Documented, reproducible failures — the part of the evaluation that a per-field F1
table cannot show. Each entry states the symptom, the cause, what the system does
*instead of silently accepting a wrong value*, and the fix direction.

Reproduce any of them with:

```bash
docintel ingest <file> --review --extractor <extractor>
```

---

## 1. Unfamiliar wording for the totals block (holdout layout)

**Symptom.** On the `ledger` layout, `subtotal` and `total_amount` are missed on all 15
test documents; `due_date` is missed on all 9 documents that have one. `tax_amount` is
still correct.

**Cause.** The anchors are vocabulary, not semantics. The holdout prints *"Value of
Supply"*, *"Net Receivable"* and *"Pay by"*; none of those phrases is in `LABELS`.
`Output Tax CGST` survives because it still contains `CGST`.

**What the pipeline does instead.** `check_money_completeness` raises
`MONEY_BLOCK_INCOMPLETE` ("read total_amount but not subtotal"), `check_required`
raises a blocking `MISSING_FIELD`, and every one of those documents is routed to
review. Auto-approval rate on the holdout slice is exactly 0.000.

**Fix direction.** Add the synonyms (`value of supply`, `net receivable`, `pay by`) and
re-run the holdout to see the gap close. They are intentionally absent here so the
holdout keeps measuring generalisation rather than testing my ability to copy strings
from the test set.

---

## 2. The baseline's "first number after 'total'" bug

**Symptom.** `rules_flat` reports `total_amount = 4.0` for invoices whose line list is
followed by `Total Items: 4`.

**Cause.** The baseline has no geometry, so "the next number after the word total" is
ambiguous: the item count, the subtotal, the tax total and the grand total are all
followed by numbers.

**Impact.** Document exact match 0.000 for the baseline, even though individual fields
look acceptable in isolation. This is why the headline metric in this project is
document-level exact match next to field F1.

**Fix.** Already fixed in `rules_layout` (explicit `TOTAL_EXCLUDE` for
`total items / total qty / total tax`, plus a bottom-of-page prior). The baseline is
kept as-is on purpose: it is the control condition.

---

## 3. Buyer block printed above the seller block

**Symptom.** `vendor_gstin` is the buyer's GSTIN.

**Cause.** `_gstins()` takes the first GSTIN on the page and treats it as the seller's.
When a template puts the consignee block above the seller block, "first" is wrong.

**What the pipeline does instead.** The value is still produced, at confidence 0.90, so
this is a **silent** error unless another rule catches it. On Indian templates the
state code prefix in the GSTIN plus the seller's registered address usually disambiguate,
which is precisely the rule that is missing.

**Fix direction.** Use the `GSTIN` label's row position relative to the `Seller`/`Bill
to` headers (the label-based role assignment I use for buyers) instead of document
order, and add a cross-check against the seller's state code.

This is the most serious known silent failure and is listed first in the roadmap.

---

## 4. Dense templates defeat column splitting

**Symptom.** With small fonts, a label and its amount can end up in the same column
block, so `Subtotal` is paired with the wrong number in the row.

**Cause.** `split_chunks()` breaks a row at gaps wider than `0.85 ×` the line height.
When artwork or a tight grid brings the columns closer than that, the row is treated as
one block and the value lookup degrades to "first money token after the label".

**What the pipeline does instead.** The score drops to `row_rightmost`/`label+far_chunk`
(0.70–0.86), often below the 0.85 field threshold, so the field is routed to review.

**Fix direction.** Use a document-level column model (histogram of token x-positions)
rather than a per-row gap threshold, which is also what makes multi-page totals tables
tractable.

---

## 5. European number formatting

**Symptom.** `1.180,00` becomes `1.18` or `1180000.0`.

**Cause.** `_strip_grouping()` resolves `.`/`,` ambiguity using the number of trailing
digits. `1.180,00` (comma as decimal) is handled, but `1.180` alone is genuinely
ambiguous between 1180 and 1.18.

**What the pipeline does instead.** The arithmetic check usually catches it: a wrong
subtotal will not sum to the printed total, which produces a blocking
`ARITHMETIC_MISMATCH`.

**Fix direction.** Infer the document's locale from the currency symbol and the majority
format, then apply it consistently — and keep a `locale` field in the provenance.

---

## 6. Day-month versus month-day ambiguity

**Symptom.** `05/03/2026` is read as 5 March or 3 May depending on `dayfirst`.

**Cause.** The document itself is ambiguous. The pipeline uses
`extraction.dayfirst: true` (Indian/UK convention) as a config value rather than a
per-document guess.

**What the pipeline does instead.** `check_dates` catches the downstream damage when it
is visible (`DATE_ORDER` when a due date lands before the issue date, `DATE_IN_FUTURE`
for a future issue date) and routes those fields to review.

**Fix direction.** Validate candidate parses against the monospaced digit width and
against other dates on the page; the evaluation harness already reports the cost of
guessing wrong (add a `--dayfirst false` run).

---

## 7. Totals block split across pages

**Symptom.** A 2-page invoice whose totals are at the top of page 2 reports the subtotal
of page 1, or misses all three money fields.

**Cause.** Row-based anchors do not carry state across page boundaries, and
`_next_row_aligned()` deliberately refuses to cross pages.

**What the pipeline does instead.** `MONEY_BLOCK_INCOMPLETE` or `ARITHMETIC_MISMATCH`
fires and the document goes to review. In the test split this case does not occur (the
generator only puts terms and conditions on page 2), so it is a known gap rather than a
measured one.

**Fix direction.** Treat the money block as a single region spanning pages (labels +
"continued" detection), not as independent per-page anchors.

---

## 8. Auto-approval leaks under OCR noise

**Symptom.** At a simulated character error rate of 2%, 31.7% of documents are still
auto-approved but only **47.4%** of those are fully correct (`docs/EVALUATION.md` §3).

**Cause.** Field confidence falls (OCR confidence is part of it), but not everywhere:
short numeric tokens with one corrupted digit still parse, still sit next to their
label, and still clear 0.85. The arithmetic check catches many, not all.

**What the pipeline does instead.** Not enough. This is the weakest behaviour in the
project and the reason the next item on the roadmap is a document-level OCR quality
gate (`LOW_OCR_QUALITY` currently only raises a warning and does not block
auto-approval).

**Fix direction.** Make `LOW_OCR_QUALITY` a routing condition: if mean OCR confidence
< 0.85, force review regardless of field confidence.

---

## 9. GSTIN check digit versus OCR confusables

**Symptom.** A GSTIN that was correct on paper fails the mod-36 checksum after OCR turns
`0` into `O` or `8` into `B`.

**Cause.** The checksum is exact; OCR is not.

**What the pipeline does instead.** `GSTIN_CHECKSUM` is a **warning** by default
(`validation.require_gstin_checksum: false`), because a blocking failure on a
scannable-character confusion would flood the queue. Set it to `true` once OCR quality
is high enough for the check to be meaningful.

**Fix direction.** Attempt single-character repairs against the checksum (the search
space is 15 × 35) and report which repair was applied, instead of failing the field.

---

## 10. The learned tagger confuses line-item amounts with the subtotal

**Symptom.** `subtotal` precision/recall 1.000 on dev but 0.767 on test for `learned`;
inspect and you find line-item amounts tagged `B_SUBTOTAL`.

**Cause.** Line-item amount tokens share the geometry (money column, same band) and the
digit-shape features of the subtotal. Context features (`left neighbour`, line keywords)
help but do not encode "this token is inside the item table".

**What the pipeline does instead.** The wrong value is produced at moderate probability,
so the field usually lands below the 0.85 threshold and goes to review. Document exact
match for `learned` is 0.378 versus 1.000 for the rules.

**Fix direction.** Add table-structure features (column id, row index inside a contiguous
numeric block, distance to the nearest horizontal rule) — or stop pretending a linear
tagger should do this and use the LayoutLMv3 path, which is what the adapter exists for.

---

## 11. Duplicate invoices are configured but not detected

**Symptom.** Uploading the same invoice twice produces two approved documents.

**Cause.** `validation.duplicate_invoice_window_days` exists in the config, and
`documents.sha256` is stored, but no rule uses them yet.

**Fix direction.** Exact duplicate (identical `sha256`) → immediate flag; near duplicate
(same `invoice_number` + vendor + amount inside the window) → review with the earlier
document linked. Both are two joins away in the SQLite schema.

---

## Regression tests for the fixed cases

Every case above that the layout extractor already handles has a test that fails if the
handling is removed:

| case | test |
| --- | --- |
| `Total Items` distractor | `tests/test_extract_rules.py::test_grand_total_wins_over_distractor_total_items` |
| invoice number summed into tax | `tests/test_extract_rules.py::test_invoice_number_never_sums_into_tax` |
| banner text as vendor name | `tests/test_extract_rules.py::test_layout_extractor_reads_all_core_fields` |
| arithmetic mismatch routing | `tests/test_pipeline.py::test_arithmetic_error_is_flagged_end_to_end` |
| partial totals block | `tests/test_validate.py::test_partial_totals_block_is_flagged` |
