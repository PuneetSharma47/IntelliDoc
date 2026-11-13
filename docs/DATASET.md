# The labelled corpus

## Why synthetic

Invoices are confidential. There is no public corpus of Indian GST invoices with
layout and field-level labels that could be shipped with this repository, and
redacting real documents reliably is a project of its own. So the evaluation corpus is
**generated**, deterministically, by `src/docintel/synth.py`:

```bash
python scripts/generate_dataset.py --n 300 --seed 20260101
```

Same seed → byte-identical PDFs and labels. That makes every number in
`docs/EVALUATION.md` reproducible and makes the suite usable as a regression test.

**This is a test harness, not an accuracy claim.** A model or rule set that scores
1.000 here has been shown to handle *these* layout variations; it has not been shown to
handle a real accounts-payable inbox.

## What is generated

300 documents, 4 layout families, 75 each:

| template | shape | notable properties |
| --- | --- | --- |
| `classic` | A4, vendor header left, invoice block right, item table, totals right-aligned | 2-page variant with terms on page 2 (15%) |
| `compact` | A4, dense: billed-to left, invoice number/date right, one-line items | smallest gutters, hardest column splits |
| `receipt` | 320×620 thermal-style, centred vendor, single column | no due date, `CASH MEMO` banner |
| `ledger` | A4, **held out**: seller/consignee wording, values printed left of their captions | unseen vocabulary (see below) |

Per-document variation:

* **Tax treatment**: CGST+SGST split (60%), IGST (25%), single GST row (15%);
  slabs 5 / 12 / 18 / 28%. Receipts use a single row.
* **Number formats**: `₹1,18,000.00`, `Rs. 1,18,000.00`, `INR 1,18,000.00`, bare
  `1,18,000.00`; Indian digit grouping throughout.
* **Date formats**: `05/03/2026`, `05-03-2026`, `05 Mar 2026`, `2026-03-05`; 30% of
  invoices have no due date.
* **Distractors** (40%): `Total Items: 3`, `PO Number: …`, `Freight: included`.
* **Tax identifiers**: GIStINs are built so that the mod-36 check digit is valid.
* **Arithmetic errors**: 13.3% of documents have `total ≠ subtotal + tax` by a random
  amount, which is what the validation stage is trained to catch.
* **Malformed dates**: 2% of invoices have a due date before the issue date.

## Splits

| split | documents | use |
| --- | --- | --- |
| train | 180 | fitting the linear tagger / LayoutLMv3 |
| dev | 60 | thresholds, calibrating scores, tuning |
| test | 60 | reported numbers only |

Each split contains all four templates in the same proportion. The `ledger` family is
additionally flagged `holdout: true` and reported as a separate slice, because it was
authored **after** the rule set was frozen and never used for tuning. That is what makes
the 1.000 → 0.838 gap in `docs/EVALUATION.md` meaningful.

## Weak supervision for the learned baselines

Token-level labels for training come from the label file, not from hand annotation:
`extract/learned.py::weak_labels()` matches each ground-truth value back to a token
(by parsed amount, parsed date, canonical identifier, or party-name span) and resolves
collisions by picking the match closest to the relevant label anchor.

Consequences:

* the tagger inherits the anchors' notion of *where* a value may live, which flatters it
  slightly on layouts similar to the training distribution;
* it cannot learn a field the anchors never locate, which is why `weak_labels` is
  explicitly documented as bootstrap-and-not-truth;
* for LayoutLMv3 the same labels are exported to BIO JSONL, so both learned paths are
  comparable to each other.

A production corpus would replace this with 2 annotators × 500 documents, an annotation
guideline and an inter-annotator agreement number. Doing that here would be dishonest —
the documents themselves would still be synthetic.

## Field schema

| field | type | notes |
| --- | --- | --- |
| `invoice_number` | string | compared after stripping separators and case |
| `issue_date`, `due_date` | date | compared as dates, not strings |
| `vendor_name`, `buyer_name` | string | exact match after normalisation, plus token-level F1 |
| `vendor_gstin` | string | 15 chars, checksum-valid |
| `currency` | ISO code | INR in the generated corpus |
| `subtotal`, `tax_amount`, `total_amount` | float | ±0.01 tolerance |
| `document_type` | enum | invoice / receipt / credit note |

Only fields present in the ground truth are scored. `tax_amount` for a CGST+SGST
document is the **sum of both components**, which is why the extractor has a
`sum_of_components` path.

## Limitations

1. Synthetic layouts are more regular than reality: no stamps, signatures, handwriting,
   skew, coffee stains, multi-column magazine layouts, or 40-page contracts.
2. The vocabulary is small — 8 vendors, 6 buyers, 8 products. A real corpus has
   unbounded strings, which is where character-level robustness matters.
3. All documents are digital PDFs with a text layer. Scan artefacts are simulated by
   character corruption (`degrade.py`), not by actual rasterisation + OCR.
4. No credit notes or negative invoices are generated, although the classifier supports
   the class.
5. Amounts are always INR and always positive.

## Privacy

No real company, person, address or taxpayer identifier is present. Vendor and buyer
names are invented, addresses are generic, and every GSTIN is generated: the check digit
is valid by construction, but the state code + PAN combination does not belong to a real
registered taxpayer.
