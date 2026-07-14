# CLAUDE.md — agent constitution

Unsupervised anomaly detection in 3D brain MRI: train reconstruction models on **healthy** brains
only, flag regions that reconstruct poorly as anomalous. Framed as a **comparative study**, never
a clinical tool.

Canonical reference: [`3d-mri-anomaly-detection-sdd-plan.md`](3d-mri-anomaly-detection-sdd-plan.md).
Requirements: [`planning/01-requirements.md`](planning/01-requirements.md).

---

## THE CENTRAL DOMAIN FACT

**Naive reconstruction models rebuild the tumor too well, so they fail to flag it.** Better
PSNR/SSIM made detection *worse* in the prior work. UNETR's patch-based features constrain this
best.

> **Optimize detection separability (Dice/IoU). NEVER optimize reconstruction fidelity.**
> Any change that improves PSNR/SSIM at the cost of Dice is a **regression**, not an improvement.

PSNR and SSIM exist in this codebase only to *explain why* high-fidelity models miss anomalies.
They are never a headline number and never a target.

---

## Locked decisions (constraints, not open questions)

| # | Decision |
|---|---|
| D1 | **Greenfield rebuild.** `legacy/` is *reference only* — port logic, not structure. |
| D2 | **Rigor & reproducibility first, then performance.** A clean, defensible study beats a fragile high score. |
| D3 | Synthetic-anomaly training is **in scope** (the one performance novelty). Multi-scale attention is a **stretch**. |
| D4 | **MRI only.** The old "CT" framing is dropped entirely. |
| D5 | Mobility-transfer is a **README paragraph only** — a conceptual analogy, not a tested result. |
| D6 | **Adopt MONAI** for 3D transforms, datasets, and backbones. Don't hand-roll. |
| D7 | **Spec-driven development**: `/specify` → `/plan` → `/tasks` → implement. Specs live in `specs/`. |
| D8 | Data & checkpoints are **user-supplied**. Run `make check-data` before any eval or training. Only Spec 009 needs fresh compute. |

**D8 caveat:** the seven `.pth` files under `legacy/` are **Git LFS pointer stubs — 133 bytes, no
weights.** Real weights come from the Google Drive folder in `README.md` and go in `checkpoints/`.
`make check-data` detects stubs. Never work around a checkpoint failure with `strict=False`: a
partially-loaded model produces plausible-looking garbage that scores cleanly.

---

## Non-negotiable rules

### 1. Never add a `Co-Authored-By:` trailer to a git commit
Not for any Claude model, not in any form. It breaks the user's GitHub push. This is enforced by a
`PreToolUse` hook that blocks such commits — but do not rely on the hook, just never write it.

### 2. Append to `progress_report.md` after every change
[`progress_report.md`](progress_report.md) is the project's narrative history: **what** was done,
**why**, and **how**, in sequence — including problems hit and how each was resolved. Append a new
entry after every meaningful change (a spec implemented, a bug fixed, a decision reversed). Never
rewrite history there; it is append-only. Follow the existing entry format.

### 3. No implementation code without an approved spec
The flow is `/specify` → `spec-reviewer` → `/plan` → `/tasks` → implement. Specs define **what**
and the acceptance tests; plans define **how**. If asked to build something with no spec, write
the spec first.

### 4. Never autonomously spend GPU or money
Training, sweeps, and any paid run are **manual-invoke only** (`/train`, `/run-experiment`). Never
launch one on your own initiative — propose it and let the user run it.

### 5. Never commit data or checkpoints
`data/`, `checkpoints/`, `*.nii*`, `*.npy`, `*.pt`, `*.pth` are git-ignored and must never be
committed — a medical-data licensing rule as much as repo hygiene. Never print patient
identifiers, file paths, or scan metadata to logs or artifacts.

### 6. Report honestly
If a result is worse, report it. A negative finding is a finding. Never fabricate a table cell,
never quietly drop an unmeasured condition, never present a number you cannot trace to an artifact.

---

## Repo map

```
CLAUDE.md            this file          progress_report.md   append-only project narrative
specs/               WHAT to build      configs/             Hydra YAML — no magic numbers in code
src/mri_ad/          the package        scripts/             one entry point per make target
  data/ models/ recon/ eval/            tests/               pytest
  classical/ synth/ viz/ utils/         artifacts/           generated tables, plots, runs
legacy/              prior repo — REFERENCE ONLY, never import
planning/            requirements & architecture   planning/future/  archived, DO NOT IMPLEMENT
data/ checkpoints/   git-ignored, user-supplied
```

## Commands

```bash
make check-data   # verify weights + data are real (not LFS stubs). RUN THIS FIRST.
make slice        # Spec 000: one checkpoint, one volume -> Dice + figure
make eval         # full test-split evaluation from saved checkpoints
make classical    # Spec 006: classical baseline
make report       # regenerate all tables/plots/README scorecard. No GPU.
make demo         # export the demo video
make test         # pytest + coverage
make lint         # ruff format + check
make train        # MANUAL ONLY — GPU spend. Never run this autonomously.

# Point at a config: make eval HYDRA_OVERRIDES="model=unet +experiment=cluster"
```

## Conventions

- **Config-driven.** Every path, hyperparameter, and loss lives in `configs/`. No magic numbers in code.
- **The shape invariant is `(1, 16, 128, 128)`** = `(C, D, H, W)`. A deviation at a module boundary
  is a bug, not a config option.
- **Determinism.** `seed_everything()` is called first in every entry point. Every run logs its git
  SHA, resolved config, and seed to `artifacts/runs/`.
- **One interface per layer.** Every model implements `AnomalyDetectionModel`. Thresholding happens
  only in `recon/`. Metrics are defined only in `eval/metrics.py`.
- Type hints and docstrings on all public functions. `ruff` runs automatically on save.

## Known traps (the prior work got these wrong — do not repeat them)

1. **Binarize the BraTS segmentation** (`seg > 0`; labels are `{0,1,2,4}`) and resize it
   **nearest-neighbour**. Trilinear-interpolating raw multi-class labels yields fractional values
   and an invalid Dice.
2. **Accumulate scores across all subjects and all depth-chunks.** The prior code re-initialized its
   score list inside the per-subject loop and scored only 1 of 8 chunks.
3. **Train and eval preprocessing must be the same function.** They were not.
4. **The UNETR weights do not load into `monai.networks.nets.UNETR`** — different output head, plus
   a final Sigmoid. Use the ported `UNETRReconstruction`. See `specs/002-model-registry.md`.
5. **Never tune a threshold on the test split.** Tune on validation.

## Pointers

- MRI/metric conventions → the `mri-domain` skill (auto-applies).
- MONAI house style → the `monai-patterns` skill.
- Before any run → the `repro-discipline` skill.
- Draft a spec → `/specify`, then the `spec-reviewer` agent. Research → `/lit` (`lit-scout`).
- Review PyTorch code → the `torch-reviewer` agent. Write results narrative → `results-analyst`.
