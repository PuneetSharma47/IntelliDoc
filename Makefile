.PHONY: help install data train eval test serve seed lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## install the package with dev extras
	pip install -e ".[dev]"

data:  ## regenerate the labelled synthetic corpus (300 documents)
	python scripts/generate_dataset.py --n 300 --seed 20260101

train:  ## train the linear layout tagger
	python scripts/train_learned.py

eval:  ## evaluate all extractors on dev + test
	python scripts/run_eval.py --split dev test --extractors rules_flat rules_layout learned

test:  ## run the test suite
	python -m pytest -q

serve:  ## run the review console on :8000
	docintel seed-demo
	docintel serve --port 8000

seed:  ## index the bundled demo documents
	docintel seed-demo

lint:  ## lint with ruff
	ruff check src api scripts tests

clean:  ## remove generated artefacts
	rm -rf data/raw data/ground_truth data/uploads data/*.db .pytest_cache .ruff_cache build dist
