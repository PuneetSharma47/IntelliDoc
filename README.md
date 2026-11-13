# IntelliDoc

[![ci](https://github.com/puneet047/IntelliDoc/actions/workflows/ci.yml/badge.svg)](https://github.com/puneet047/IntelliDoc/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Layout-aware extraction for invoices and receipts: **read the document, check that
the numbers add up, and ask a human only about the parts the system is unsure of.**

Most "document AI" demos stop at OCR → LLM → spreadsheet, which produces output you
cannot audit and cannot trust. This project is built the other way round: every value
carries its page and pixel box, arithmetic is validated against the document, field
confidence decides what is auto-approved, and everything a reviewer corrects is stored
so the next iteration can be measured against it.

```
invoice.pdf ──▶ ingest ──▶ classify ──▶ extract ──▶ score ──▶ validate ──▶ route
                (text layer                          (per-field  (arithmetic,  (auto-approve
                 or OCR)       invoice / receipt /    confidence  GSTIN, dates, / review /
                               credit note            + parts)    completeness) reject)
                                                          │
                                                          ├──▶ SQLite library + page-level search
                                                          └──▶ review console (web UI)
```

## What it does

* **Two extraction strategies plus a learned baseline.** A deliberately naive
  regex-on-text baseline (`rules_flat`), a geometry-aware label/value extractor
  (`rules_layout`), a trained linear token tagger over layout features (`learned`),
  and an adapter for a fine-tuned **LayoutLMv3** checkpoint (`layoutlmv3`). All four
  are scored by the same harness.
* **Field-level provenance.** Every value stores page number, bounding box, the rule
  that produced it, the raw text before normalisation, and a breakdown of how its
  confidence was computed.
* **Validation that a finance team would sign off on.** `subtotal + tax == total`
  (with paise tolerance), GST slab plausibility, due-date ordering, GSTIN format and
  mod-36 check digit, partial-totals detection, required-field checks.
* **Confidence-routed human review.** Per-field thresholds (money and identifiers
  need more certainty than a defaulted currency), ranked review tasks, and a
  straight-through-accuracy number that tells you whether the queue is worth the
  reviewer's time.
* **Searchable library with citations.** SQLite FTS5/BM25 over page text; every hit
  is a page, not a document.
* **Extraction history and corrections** kept separately from machine output, so
  corrections can become training data later.

## Results (synthetic labelled corpus, 300 documents, 60-document test split)

Two evaluation slices: **seen** = the three layout families the rules were written
against; **holdout** = a fourth layout family authored after the rules were frozen and
never used for tuning.

| extractor | slice | micro F1 | doc exact match | auto-approved | straight-through accuracy |
| --- | --- | --- | --- | --- | --- |
| `rules_flat` (no geometry) | seen | 0.769 | 0.000 | 0.0% | – |
| `rules_layout` | seen | **1.000** | **1.000** | 91.1% | **100%** |
| `rules_layout` | holdout | 0.838 | 0.000 | 0.0% | – |
| `learned` (linear tagger) | seen | 0.942 | 0.378 | 0.0% | – |
| `learned` (linear tagger) | holdout | 0.896 | 0.067 | 0.0% | – |

* Mean end-to-end latency **2.6 ms/document** on the test split (native text layer).
* The layout extractor never auto-approves a document it gets wrong: on the holdout
  layout every document is routed to review, and the review queue catches **79%** of
  the wrong fields.
* Robustness curve (simulated OCR character error rate, whole test split):
  micro F1 **0.963 → 0.855 → 0.669 → 0.528** at CER 0 / 2 / 5 / 10%.
* Operating point: at the field threshold of 0.85 the pipeline auto-accepts **66.1%**
  of labelled fields (419 of 634) at **100%** accepted-value precision, which is what
  makes a 91% straight-through rate defensible.

Full tables, per-field numbers, threshold sweeps and error analysis:
[`docs/EVALUATION.md`](docs/EVALUATION.md). Concrete failures with reproductions:
[`docs/FAILURE_CASES.md`](docs/FAILURE_CASES.md).

## Quickstart

```bash
git clone https://github.com/puneet047/IntelliDoc.git && cd IntelliDoc
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. build the labelled corpus (deterministic: same seed -> same documents)
python scripts/generate_dataset.py --n 300

# 2. extract one document
docintel ingest data/demo/demo-classic-00.pdf --review

# 3. train the linear tagger and evaluate every extractor
python scripts/train_learned.py
python scripts/run_eval.py --split dev test --extractors rules_flat rules_layout learned

# 4. review console at http://localhost:8000
docintel seed-demo
docintel serve --port 8000
```

`docintel ingest` prints one line per field with its value, confidence, page and the
rule that produced it:

```
demo-classic-00.pdf  [auto_approved]  invoice (0.99)  4.8 ms  via rules_layout
  buyer_name       Ashirwad Supermarket   conf 0.84   p0 bill_to_next_row
  invoice_number         INV-2025-770     conf 0.94   p0 label+same_chunk
  subtotal                 9,120.00       conf 0.91   p0 money_anchor
  tax_amount               1,641.60       conf 0.91   p0 sum_of_components
  total_amount            10,761.60       conf 0.95   p0 money_anchor
  ...
```

### Scanned documents (OCR)

Digital PDFs use their text layer (faster and lossless); pages with too little text
are routed to Tesseract automatically.

```bash
pip install -e ".[ocr]"      # plus the tesseract system binary
docintel ingest scan.pdf     # engine note explains which path was used
```

### LayoutLMv3

```bash
pip install -e ".[ml]"
python scripts/train_layoutlmv3.py --export-only   # BIO JSONL, no GPU needed
python scripts/train_layoutlmv3.py                  # fine-tune microsoft/layoutlmv3-base
python scripts/run_eval.py --extractors rules_layout learned layoutlmv3
```

If no checkpoint is present, the evaluation reports LayoutLMv3 as *not run* rather
than quietly substituting another extractor's numbers. See
[`docs/LAYOUTLM.md`](docs/LAYOUTLM.md).

## Repository layout

```
src/docintel/          (import package name: docintel, console script: docintel)
  ocr/          pluggable engines: pdf_text (PyMuPDF), tesseract
  ingest.py     file -> Document (pages, tokens, boxes, hashes)
  layout.py     line reconstruction, column splitting, token features
  normalize.py  money (incl. ₹1,18,000.00), dates, GSTIN check digit
  classify.py   invoice / receipt / credit-note cues
  extract/      rules_flat, rules_layout, learned, layoutlmv3, ensemble
  confidence.py per-field confidence with an auditable breakdown
  validate.py   arithmetic + cross-field rules
  review.py     thresholds, ranked human-review routing
  store.py      SQLite: documents, fields, issues, queue, corrections
  search.py     FTS5/BM25 page search with citations
  evaluate.py   field P/R/F1, EM, straight-through accuracy, ablation, sweeps
  render.py     page images with field highlights
api/            FastAPI app + review console (vanilla JS, no build step)
scripts/        dataset generation, evaluation, training
tests/          72 tests (pytest)
docs/           architecture, evaluation, failure cases, dataset, LayoutLMv3
benchmarks/     archived metrics from the last runs
```

## Design decisions worth defending

1. **Rules before models.** A 300-line geometry-aware extractor scores 1.000 micro F1
   on the layouts it was written for at 2.6 ms/document; the trained linear tagger
   reaches 0.942 and needs a training pipeline. Rules are also reviewable by the
   accounts team. The learned baselines exist to keep the rules honest, not to replace
   them by default.
2. **Confidence is decomposable.** `confidence_parts` records base score, consensus
   among candidates, OCR quality, geometry and validation penalties. A number you
   cannot explain cannot be tuned, and cannot be shown to an auditor.
3. **Never average disagreeing values.** Averaging two candidate totals invents a
   third value that appears nowhere in the document. The pipeline picks one and flags
   the disagreement.
4. **Per-field thresholds.** Money and identifiers need 0.85 confidence; a defaulted
   currency needs 0.55. A single global threshold drags every document into review
   over metadata nobody reads, and reviewers stop trusting the queue.
5. **The evaluation harness runs the deployed code path.** The API, CLI and metrics all
   call the same `Pipeline`. Numbers that come from a separate research script describe
   a system you did not ship.
6. **Holdout layout.** Rules tuned on a corpus score gloriously on that corpus. The
   fourth layout family was frozen before tuning and is reported separately, because
   the gap between 1.000 and 0.838 is the honest measure of generalisation.

## Limitations

* The labelled corpus is **synthetic**. It is reproducible and suitable for
  regression testing, not for claiming real-world accuracy. See `docs/DATASET.md`.
* Extraction targets a fixed 11-field schema. Line items are not extracted.
* Totals split across pages, right-to-left scripts and handwriting are out of scope.
* The `learned` tagger's probabilities are not calibrated for the routing policy, so it
  auto-approves nothing out of the box (documented in `docs/EVALUATION.md`).

## License and data

MIT © Puneet Sharma — see [`LICENSE`](LICENSE). The corpus is generated locally; no real invoices,
customer names or personal data are included. Company and GSTIN values are synthetic
(GSTINs are generator-valid but do not belong to real taxpayers).
