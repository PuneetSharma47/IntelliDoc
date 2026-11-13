# LayoutLMv3 path

The repository ships everything needed to fine-tune and evaluate
[`microsoft/layoutlmv3-base`](https://huggingface.co/microsoft/layoutlmv3-base) for
token-level field tagging. The checkpoint itself is **not** committed (≈500 MB), so the
model is reported as *not run* in the evaluation until you train it. Nothing else in
the pipeline depends on it.

## Label scheme

Word-level BIO over 12 classes, shared with the linear tagger so the two are directly
comparable:

```
O
B_ID  I_ID              invoice number
B_ISSUE                 invoice date
B_DUE                   due date
B_VENDOR  I_VENDOR      seller name span
B_BUYER   I_BUYER       buyer name span
B_GSTIN                 seller GSTIN
B_SUBTOTAL              taxable value
B_TAX                   tax row (CGST/SGST/IGST component)
B_TOTAL                 grand total
```

Amounts are single tokens; party names are spans. Parameters that differ from a vanilla
token-classification setup:

* **Boxes** are normalised to the 0–1000 grid LayoutLMv3 was pretrained on
  (`layoutlm.normalize_boxes`), from PDF points.
* **Words, not subwords, are predicted.** Subword logits are merged by taking the first
  subword of each word (`word_ids()` + `seen` set), so the extractor always receives
  exactly one prediction per PDF word.
* **512-word windows with 128-word stride** (`chunk_document`), page by page. Multi-page
  documents are processed page-independently; cross-page fields are a known gap.
* **Tax components are summed at decode time**, mirroring the CGST/SGST handling in the
  rules extractor, so `tax_amount` means the same thing for every extractor.

## Training

```bash
pip install -e ".[ml]"

# 1. export BIO records (no GPU needed, no torch needed)
python scripts/train_layoutlmv3.py --export-only
#    -> models/train.jsonl, models/dev.jsonl, models/labels.json

# 2. fine-tune
python scripts/train_layoutlmv3.py \
    --epochs 18 --batch-size 2 --grad-accum 8 --lr 5e-5 --fp16
#    -> models/layoutlmv3-finetuned/  (config.json, weights, tokenizer)

# 3. evaluate on identical inputs as the other extractors
python scripts/run_eval.py --extractors rules_layout learned layoutlmv3
```

The script uses HuggingFace `Trainer` with `DataCollatorForTokenClassification`,
evaluates every epoch on the dev split and keeps the checkpoint with the lowest eval
loss. On a single 8 GB GPU the 180-document training split takes roughly 25 minutes at
those settings; on CPU, expect hours — export the JSONL and train on Colab or a rented
GPU instead.

## What to expect

The corpus is small (≈180 documents, ≈40k word-level labels for 12 classes, of which
`O` is >80%). Realistically:

* the model will beat the linear tagger on party names and on layouts it has not seen
  verbatim, because pretrained layout+text representations transfer;
* it will **not** automatically beat the rules on the three seen families, because those
  rules encode the same knowledge with zero training data and 1.8 ms latency;
* the interesting comparison is therefore `rules_layout` vs `layoutlmv3` **on the
  holdout slice**, which is exactly what `scripts/run_eval.py` reports separately.

If the checkpoint is missing when `--extractor layoutlmv3` is requested, the pipeline
logs `EXTRACTOR_FALLBACK` and uses `rules_layout`; the evaluation harness, by contrast,
reports the model as not run rather than substituting another extractor's numbers.

## Why not an LLM API call

Because the point of the project is auditability. A tokenised, box-aligned, locally
runnable model gives per-word confidence that can be routed, compared and regression
tested without sending a customer's invoice to a third party.
