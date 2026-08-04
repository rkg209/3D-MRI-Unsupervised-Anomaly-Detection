.PHONY: help install slice sweep train eval report classical demo test lint check-data
.DEFAULT_GOAL := help

# Hydra overrides, e.g.: make eval HYDRA_OVERRIDES="+experiment=cluster model=unetr"
HYDRA_OVERRIDES ?=
PY := python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package + dev deps
	pip install -e ".[dev]"

check-data:  ## Verify data/ and checkpoints/ hold REAL files (not LFS stubs). Run this first.
	$(PY) scripts/check_data.py

slice:  ## Spec 000: one checkpoint, one volume -> Dice + figure
	$(PY) scripts/run_slice.py $(HYDRA_OVERRIDES)

# GPU-SPENDING. Manual invoke only. e.g. make sweep HYDRA_OVERRIDES="+experiment=cluster model=unetr"
sweep:  ## Spec 003: Dice-vs-threshold sweep on the VALIDATION split. MANUAL ONLY.
	$(PY) scripts/run_sweep.py $(HYDRA_OVERRIDES)

eval:  ## Full test-split evaluation from saved checkpoints (GPU preferred)
	$(PY) scripts/run_eval.py $(HYDRA_OVERRIDES)

classical:  ## Spec 006: classical-ML baseline (features -> gradient boosting -> CV)
	$(PY) scripts/run_classical.py $(HYDRA_OVERRIDES)

report:  ## Regenerate all tables/plots/README scorecard from saved artifacts. No GPU.
	$(PY) scripts/run_report.py $(HYDRA_OVERRIDES)

demo:  ## Spec 010: export the demo video from a saved ReconResult
	$(PY) scripts/run_demo.py $(HYDRA_OVERRIDES)

# GPU-SPENDING. Manual invoke only — the agent must never run this autonomously (see CLAUDE.md).
train:  ## Spec 009: synthetic-anomaly fine-tune. MANUAL ONLY.
	$(PY) scripts/run_train.py $(HYDRA_OVERRIDES)

test:  ## Run the test suite
	pytest --cov=src/mri_ad --cov-report=term-missing

lint:  ## Format + lint
	ruff format src tests scripts
	ruff check --fix src tests scripts
