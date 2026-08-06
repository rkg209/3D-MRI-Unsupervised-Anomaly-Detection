.PHONY: help install slice sweep recon train eval report report-check classical demo test lint check-data matrix slice-scores paradigm synth-table
.DEFAULT_GOAL := help

# Hydra overrides, e.g.: make eval HYDRA_OVERRIDES="+experiment=cluster model=unetr"
HYDRA_OVERRIDES ?=
PY := python

# make eval LEGACY_BUG_COMPAT=1 translates to a Hydra group selection at the shell boundary — no
# Python source reads this environment variable (Spec 004 acceptance test 6). Config stays the
# single source of truth: `make eval HYDRA_OVERRIDES="eval=legacy_compat"` is equivalent.
LEGACY_BUG_COMPAT ?= 0
EVAL_MODE := $(if $(filter 1 true yes,$(LEGACY_BUG_COMPAT)),eval=legacy_compat,)

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

# GPU-SPENDING. Manual invoke only. e.g. make recon HYDRA_OVERRIDES="+experiment=cluster model=unetr"
recon:  ## Spec 004: reconstruct the TEST split -> saved ReconResults. MANUAL ONLY.
	$(PY) scripts/run_recon.py $(HYDRA_OVERRIDES)

eval:  ## Spec 004: score saved test-split ReconResults. No model, no GPU. LEGACY_BUG_COMPAT=1 for bug-compat.
	$(PY) scripts/run_eval.py $(EVAL_MODE) $(HYDRA_OVERRIDES)

classical:  ## Spec 006: classical-ML baseline (features -> gradient boosting -> CV)
	$(PY) scripts/run_classical.py $(HYDRA_OVERRIDES)

report:  ## Regenerate all tables/plots/README scorecard from saved artifacts. No GPU.
	$(PY) scripts/run_report.py $(HYDRA_OVERRIDES)

report-check: report  ## CI guard: fails if `make report` would change the checked-in README.
	git diff --exit-code README.md

matrix:  ## Spec 005: render artifacts/tables/arch_loss_matrix.{csv,md,json}. No GPU.
	$(PY) scripts/run_arch_loss_matrix.py $(HYDRA_OVERRIDES)

slice-scores:  ## Spec 007: reduce one saved DL recon run to slice-level scores. No GPU (reads saved .pt).
	$(PY) scripts/run_slice_reduction.py $(HYDRA_OVERRIDES)

paradigm:  ## Spec 007: render the Classical vs UNETR vs Diffusion headline table. No GPU.
	$(PY) scripts/run_paradigm_comparison.py $(HYDRA_OVERRIDES)

demo:  ## Spec 010: export the demo video from a saved ReconResult
	$(PY) scripts/run_demo.py $(HYDRA_OVERRIDES)

# GPU-SPENDING. Manual invoke only — the agent must never run this autonomously (see CLAUDE.md).
train:  ## Spec 009: synthetic-anomaly fine-tune. MANUAL ONLY.
	$(PY) scripts/run_train.py $(HYDRA_OVERRIDES)

synth-table:  ## Spec 009: render the before/after synthetic-anomaly Dice/IoU table. No GPU.
	$(PY) scripts/run_synth_comparison.py $(HYDRA_OVERRIDES)

test:  ## Run the test suite
	pytest --cov=src/mri_ad --cov-report=term-missing

lint:  ## Format + lint
	ruff format src tests scripts
	ruff check --fix src tests scripts
