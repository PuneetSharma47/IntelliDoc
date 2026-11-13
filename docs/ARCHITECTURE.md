# Architecture

## Data flow

```
                    ┌────────────────────────────────────────────────────────┐
 file (pdf/png) ──▶ │ ingest.py   engine selection, page read, hashing        │
                    │             -> Document(pages[Page(tokens[Token])])     │
                    └───────────────┬────────────────────────────────────────┘
                                    ▼
                    ┌────────────────────────────────────────────────────────┐
                    │ layout.py   line reconstruction, column splitting,     │
                    │             token features (position, gaps, bands)      │
                    └───────────────┬────────────────────────────────────────┘
                                    ▼
                    ┌────────────────────────────────────────────────────────┐
                    │ classify.py invoice | receipt | credit note | unknown   │
                    └───────────────┬────────────────────────────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ extract/  each extractor returns [Candidate(field, value, score,  │
        │           provenance)]                                            │
        │   rules_flat    regexes over one text blob (baseline)             │
        │   rules_layout  label anchors + geometry (default)                │
        │   learned       linear tagger over layout features                │
        │   layoutlmv3    fine-tuned token classifier (optional)            │
        │   ensemble      merges candidates, rewards agreement              │
        └───────────────┬───────────────────────────────────────────────────┘
                        ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ confidence.py  winner + consensus + geometry + OCR -> FieldValue  │
        │                (carries confidence_parts and alternatives)         │
        └───────────────┬───────────────────────────────────────────────────┘
                        ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ validate.py  arithmetic, tax slab, dates, GSTIN, completeness      │
        │              -> [Issue], blocking issues penalise confidence       │
        └───────────────┬───────────────────────────────────────────────────┘
                        ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ review.py  per-field thresholds -> [ReviewTask], DocStatus         │
        └───────────────┬───────────────────────────────────────────────────┘
                        ▼
        ┌───────────────────────┐        ┌───────────────────────────────────┐
        │ store.py  SQLite      │───────▶│ search.py  FTS5/BM25 page search  │
        │ documents/fields/     │        └───────────────────────────────────┘
        │ issues/queue/         │        ┌───────────────────────────────────┐
        │ corrections/runs      │───────▶│ render.py  page PNG + highlights  │
        └───────────────────────┘        └───────────────────────────────────┘
```

`pipeline.Pipeline` is the only place that wires these together. The CLI, the FastAPI
app and `evaluate.py` all construct a `Pipeline` — there is no second code path.

## Why each stage is separate

| Stage | Runs on | Failure mode if merged into a neighbour |
| --- | --- | --- |
| ingest | raw file | OCR retries become entangled with parsing logic |
| layout | tokens | every extractor would re-implement line/column logic |
| extract | page geometry | rules and models could not be compared on equal terms |
| confidence | candidates | scores would be hard-coded per rule, impossible to audit |
| validate | typed fields | business rules would be sprinkled through parsing |
| review | fields + issues | the auto-approve policy would be untestable in isolation |

## Data model

`Token` is the atom: text, `BBox` in PDF points, page, optional `ocr_confidence`, and
`line_id` / `token_id` assigned by `layout.index_document`.

`FieldValue` is the only thing that leaves the pipeline:

```python
FieldValue(
  field=FieldName.TOTAL_AMOUNT,
  value=11800.0,
  confidence=0.95,
  provenance=Provenance(page=0, bbox=BBox(...), source="rules_layout",
                        raw_text="₹11,800.00", rule="label+next_chunk"),
  alternatives=[11900.0],
  confidence_parts={"base": 0.95, "consensus": 0.0, "ocr": 0.0,
                    "geometry": 0.0, "agreement": 0.0},
)
```

`ExtractionResult` bundles the document, typed fields, issues, review tasks and status.

## Database schema

| table | purpose |
| --- | --- |
| `documents` | one row per document: status, engine, extractor, latency, settings fingerprint |
| `pages` | page text + dimensions (used for rendering and search) |
| `fields` | one row per field: machine value **and** `corrected_value`, confidence, box, rule, per-signal breakdown |
| `issues` | validation findings with severity |
| `review_queue` | ranked tasks, resolved/approved with reviewer and timestamp |
| `corrections` | append-only audit trail of every human edit (`predicted` vs `corrected`) |
| `runs` | archived evaluation metrics with the config fingerprint that produced them |
| `page_fts` | FTS5 index over page text (BM25 ranked) |

Machine output is never overwritten by a correction, so `corrections` is directly
usable as a retraining set (`Store.export_corrections()`).

## Extension points

* **New field**: add to `FieldName` and `REQUIRED_INVOICE_FIELDS`/`MONEY_FIELDS` as
  appropriate, add anchors in `extract/rules.py`, add a `FIELD_RISK` weight and a
  per-field threshold in `config.py`. The evaluation picks it up from `SCORED_FIELDS`.
* **New OCR engine**: implement `read_pages(path) -> list[Page]` and register it in
  `ocr/registry.py`.
* **New validation rule**: write `check_*(fields, settings) -> list[Issue]` in
  `validate.py` and append it in `validate()`.
* **New extractor**: implement `extract(ctx) -> list[Candidate]`; the config key
  `extraction.extractor` and the CLI `--extractor` flag will accept it.

## Operational notes

* Everything is CPU-only by default; the pipeline has no network calls.
* Settings are fingerprinted (`Settings.fingerprint()`), and the fingerprint is stored
  with every document and evaluation run, so a metric change can always be traced to a
  config change.
* SQLite is deliberate: one file, no service to run, and `sqlite3 data/docintel.db`
  is a complete debugging tool. Swap `store.py` for Postgres when concurrency demands
  it; the interface is a handful of methods.
