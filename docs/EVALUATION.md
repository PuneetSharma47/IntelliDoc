# Evaluation

All numbers below are produced by `python scripts/run_eval.py`. Every run writes its
full report and the settings fingerprint to `benchmarks/results/`, so the tables in
this document can be regenerated and diffed.

* Corpus: 300 synthetic documents, split 180 train / 60 dev / 60 test (deterministic,
  seed `20260101`). Test split = 45 documents from the three "seen" layout families +
  15 from the "ledger" holdout family.
* A field is scored only when the ground truth contains it: a document without a due
  date cannot produce a false negative for `due_date`.
* A wrong value costs one false positive **and** one false negative.
* `exact match` = TP / scored fields. `straight-through accuracy` = share of
  auto-approved documents that were completely correct.

## 1. Extractor comparison (test split)

| extractor | slice | n | micro F1 | macro F1 | doc EM | auto-approved | STP accuracy | review rate | arith P | arith R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rules_flat` | seen | 45 | 0.769 | 0.743 | 0.000 | 0.000 | – | 1.000 | 0.089 | 1.000 |
| `rules_layout` | seen | 45 | **1.000** | **1.000** | **1.000** | 0.911 | **1.000** | 0.089 | 1.000 | 1.000 |
| `learned` | seen | 45 | 0.942 | 0.932 | 0.378 | 0.000 | – | 1.000 | 0.800 | 1.000 |
| `rules_flat` | holdout | 15 | 0.716 | 0.630 | 0.000 | 0.000 | – | 1.000 | 0.000 | 0.000 |
| `rules_layout` | holdout | 15 | 0.838 | 0.709 | 0.000 | 0.000 | – | 1.000 | 0.000 | 0.000 |
| `learned` | holdout | 15 | 0.896 | 0.886 | 0.067 | 0.000 | – | 1.000 | 0.273 | 0.750 |

Read this table as three separate findings:

1. **Geometry is the difference between a demo and a tool.** Without it, the baseline
   reaches 0.769 micro F1 and never produces a fully correct document: it grabs the
   first number after the word "total" (`Total Items: 4`), takes the first date on the
   page, and returns the whole first line as the vendor name.
2. **Nothing is auto-approved when it is wrong.** The layout extractor auto-approves
   91% of seen documents at 100% accuracy, and auto-approves *zero* holdout documents —
   the review queue and the validation rules absorb the generalisation gap instead of
   leaking bad data downstream.
3. **The learned tagger loses to the rules but generalises slightly better.** 0.942 on
   seen layouts vs 0.896 on holdout is a much smaller gap than 1.000 → 0.838: a model
   that learns geometry transfers better than hand-written anchors, once it has enough
   data. It still auto-approves nothing, because its class probabilities are not
   calibrated against the routing thresholds (see §5).

## 2. Per-field results, `rules_layout`

**Seen layouts (test, n=45)** — every field at precision/recall 1.000, including
`tax_amount` reconstructed by summing CGST + SGST components in 60% of documents.

**Holdout layout (test, n=15)**

| field | precision | recall | F1 | scored |
| --- | --- | --- | --- | --- |
| `invoice_number` | 1.000 | 0.867 | 0.929 | 15 |
| `issue_date` | 1.000 | 1.000 | 1.000 | 15 |
| `due_date` | 0.000 | 0.000 | 0.000 | 9 |
| `vendor_name` | 0.867 | 0.867 | 0.867 | 15 |
| `vendor_gstin` | 1.000 | 1.000 | 1.000 | 15 |
| `buyer_name` | 1.000 | 1.000 | 1.000 | 15 |
| `currency` | 1.000 | 1.000 | 1.000 | 15 |
| `subtotal` | 0.000 | 0.000 | 0.000 | 15 |
| `tax_amount` | 1.000 | 1.000 | 1.000 | 15 |
| `total_amount` | 0.000 | 0.000 | 0.000 | 15 |
| `document_type` | 1.000 | 1.000 | 1.000 | 15 |

The three zeros are all vocabulary, not geometry: the holdout renders totals as
*"Value of Supply"*, *"Output Tax CGST"* and *"Net Receivable"*, and the due date as
*"Pay by"*. `Output Tax CGST` still matches the `CGST` anchor, which is why tax
survives. The obvious next iteration is synonym expansion — and it is deliberately not
done here, because the holdout exists to measure exactly this gap.

## 3. Robustness to OCR noise (test split, `rules_layout`)

Scanned documents are simulated by corrupting the text layer at a target character
error rate (`docintel/degrade.py`, seeded, with confusable-glyph substitutions such as
`0↔O`, `1↔l`, `5↔S`). This isolates "how fast does accuracy fall as text quality
falls" from "which OCR engine is installed".

| CER | micro F1 | doc EM | auto-approved | STP accuracy | subtotal EM | total EM |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.963 | 0.750 | 0.667 | 1.000 | 0.750 | 0.750 |
| 0.02 | 0.855 | 0.183 | 0.317 | 0.474 | 0.550 | 0.617 |
| 0.05 | 0.669 | 0.017 | 0.117 | 0.000 | 0.317 | 0.450 |
| 0.10 | 0.528 | 0.000 | 0.000 | – | 0.367 | 0.200 |

Observations:

* The pipeline degrades **gracefully but not sharply enough**: at 2% CER, 31.7% of
  documents are still auto-approved, and only 47.4% of those are fully correct. That is
  the single worst number in this report and the main argument for gating auto-approval
  on document-level OCR quality rather than only on field confidence (roadmap item 1).
* Amounts survive noise better than text fields: digits degrade into other digits, and
  the arithmetic check catches most corrupted totals even when extraction is wrong.
* `doc EM` at CER 0 is 0.750, not 1.000, because this ablation runs over the whole test
  split (including the 15 holdout documents, which no extractor handles).

## 4. Confidence threshold sweep (dev split, `rules_layout`)

What the review threshold actually buys. `coverage` is the share of all labelled fields
accepted without human review; `accepted precision` is the share of accepted fields
that were correct.

| threshold | field coverage | accepted fields | wrong accepted | accepted precision | STP accuracy |
| --- | --- | --- | --- | --- | --- |
| 0.50 | 0.920 | 583 | 2 | 0.997 | 0.967 |
| 0.70 | 0.845 | 536 | 1 | 0.998 | 0.983 |
| 0.80 | 0.789 | 500 | 1 | 0.998 | 0.983 |
| **0.85** | **0.661** | **419** | **0** | **1.000** | **1.000** |
| 0.90 | 0.539 | 342 | 0 | 1.000 | 1.000 |
| 0.95 | 0.128 | 81 | 0 | 1.000 | 1.000 |

0.85 is the shipped default: it is the lowest threshold at which no accepted field was
ever wrong on the dev split, and it still keeps 66% of fields out of the queue (419 of
634 labelled fields accepted automatically). Below 0.85 the
marginal coverage comes from low-confidence fields that are occasionally wrong — exactly
the trade a review team should not make silently.

## 5. The learned tagger does not auto-approve anything

`learned` produces 0 auto-approvals at 0.957 micro F1, because its score is
`0.55 + 0.42 × P(class)`, and a linear model over 536 features rarely exceeds
P=0.71 (score 0.85) on the winning token. Two honest readings:

* the score map was inherited from the rules so that one policy could apply to both, and
  it was not recalibrated for the model — a real defect, listed in the roadmap;
* it is the *correct* failure direction: an uncalibrated model should route everything
  to a human rather than assert.

Calibrating per-extractor score maps (isotonic regression on the dev split, with the
sweep in §4 as the acceptance test) is the fix.

## 6. Latency

| stage | mean | p95 |
| --- | --- | --- |
| ingest + layout + extract + validate + route (`rules_layout`) | 2.6 ms | 4.3 ms |
| same, `learned` | 5.6 ms | 8.8 ms |
| same, `rules_flat` | 0.9 ms | 1.2 ms |

Measured on the extraction stage only, single-threaded CPU, native text layer. Rendering
page images for the reviewer (≈40 ms/page at 120 dpi) is the dominant cost in the web
path and is deliberately done on demand.

## 7. Reproducing

```bash
python scripts/generate_dataset.py --n 300 --seed 20260101
python scripts/train_learned.py
python scripts/run_eval.py --split dev test --extractors rules_flat rules_layout learned
```

Outputs: `benchmarks/results/latest.json` (everything, including per-field metrics and
the exact settings object), `benchmarks/results/field_metrics.csv` (one row per
extractor × field), and one JSON per extractor/slice pair.
