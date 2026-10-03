# Project results audit — verified state of `9_3D_MRI`

**Audit date:** 2026-09-03
**Audited commit:** `d40b83c` ("Implement Spec 013 (Phases 1-2)"), working tree dirty
**Auditor method:** repository inspection + live execution of every no-GPU path on this machine.
No number in this document is taken from a Markdown claim unless it was independently reproduced
from code, config, or a generated artifact. Where nothing was measured, this document says so.

---

## 0. Executive verdict

| Question | Answer |
|---|---|
| Is the software built? | **Yes.** 8,482 lines of `src/`, 7,206 lines of `tests/`, 14 entry-point scripts, 15 Makefile targets, full Hydra config tree. No `TODO`, no `FIXME`, no `NotImplementedError` anywhere in `src/` or `scripts/`. |
| Does the software work? | **On synthetic tensors, yes — verified this session.** 405 passed / 1 skipped, 94% line coverage on `src/mri_ad`. `ruff format --check` and `ruff check` both clean over 109 files. |
| Did the project achieve its objective? | **No.** The objective is a *comparative study producing measured results*. **Zero experimental results exist.** Every cell of every results table is `n/a`. |
| Completion status | **Functionally complete but entirely unvalidated — no experiment has ever been run.** |
| Resume-ready? | **The engineering is resume-ready. The results are not — because there are none.** Any claim about detection performance would today be fabrication. |

**The single most important fact in this audit:** across all 18 recorded runs in
`artifacts/runs/*/run_meta.json`, the maximum wall-clock duration is **0.975 seconds** and every
`metrics` block is either `{}` or `{"n_available": 0}`. No model has ever been trained. No model
has ever reconstructed a real volume. No Dice, IoU, ROC-AUC, PR-AUC, PSNR or SSIM value has ever
been produced by this codebase.

---

## 1. Original project objective

Reconstructed from `planning/01-requirements.md` §1 (Business Goals), `CLAUDE.md`, and
`3d-mri-anomaly-detection-sdd-plan.md` — cross-checked against what the code actually implements.

### Why the project was built

The author had prior work (`legacy/metric-uad.ipynb`) reporting **UNETR Dice 0.6255 / IoU 0.4551**
on unsupervised brain-MRI anomaly detection. That number was found to be produced by four
independent bugs. The project is a **greenfield, rigor-first rebuild** whose purpose is to replace
an untrustworthy headline number with a defensible one — and to turn a single-model notebook into a
three-paradigm comparative study.

### The problem being solved

Train reconstruction models on **healthy** brain MRI only (OpenBHB); at test time, feed a
tumor-bearing volume (BraTS20) and flag voxels that reconstruct poorly. The central domain fact —
stated in `CLAUDE.md` and encoded throughout the code — is that **naive reconstruction models
rebuild the tumor too well and therefore fail to flag it**, so higher PSNR/SSIM makes detection
*worse*. The study exists to measure that anti-correlation rather than assert it.

### Stated business goals (verbatim source: `planning/01-requirements.md:5-15`)

| ID | Goal |
|---|---|
| BG-1 | A rigorous, reproducible comparative study, defensible in a technical interview and suitable as a public portfolio artifact. |
| BG-2 | Close the classical-ML gap on the author's resume via a cross-validated gradient-boosting baseline and an honest DL-vs-classical comparison. |
| BG-3 | Demonstrate a **measurable performance improvement** over the prior work's best result (Dice ≈ 0.63) via synthetic-anomaly training, with a controlled before/after on an identical test split. |
| BG-4 | A clean, config-driven greenfield codebase reproducible by a third party from checkpoints + registered data alone. |
| BG-5 | A README paragraph on conceptual transfer to mobility/trajectory OOD detection (explicitly *not* a tested result). |

### Expected outcome / definition of success

Success as the repository itself defines it requires **all** of:

1. A populated architecture × loss matrix (`specs/005`) with real Dice/IoU/PSNR/SSIM per cell, and
   a measured Spearman rank correlation between PSNR and Dice.
2. A populated three-paradigm headline table (`specs/007`): Classical vs UNETR vs Diffusion.
3. A controlled before/after showing synthetic-anomaly fine-tuning changes Dice (BG-3).
4. A corrected baseline Dice, plus a demonstrated bug-compatibility reproduction of the prior
   0.6255 (FR-21) proving the old number was wrong.
5. Timing/cost evidence: inference < 5 s/volume (NFR-9), `make eval` < 30 min (NFR-10).
6. Engineering rigor: ≥80% coverage (NFR-17), zero ruff errors (NFR-18), full run provenance (NFR-2).

**Items 1–5 are entirely unmet. Item 6 is met and verified.**

### What would make this genuinely impressive to a top-tier reviewer

- A **measured, reported anti-correlation** between reconstruction fidelity and detection quality
  across a real model × loss grid — a mechanism finding, not a leaderboard entry.
- A **falsification of a previously published personal result** (0.6255) with the four causal bugs
  demonstrated in an executable bug-compat mode. This is rare and highly credible.
- A **three-paradigm comparison** including a classical baseline that might beat the deep models —
  reported honestly either way.
- **Reproducibility that actually holds**: git SHA + resolved config + seed per run, an auto-generated
  README scorecard that cannot drift from artifacts.

Note that **none of these require a state-of-the-art score.** The project's design is unusually
well-suited to being impressive *without* winning a benchmark — but it requires numbers to exist.

---

## 2. Completion status

### Verdict: **Functionally complete, entirely unvalidated.**

The harness is finished. The study is not started. Precisely: every component that can be built
without a GPU or real data is implemented and unit-tested; every component that produces a
*result* is blocked on the same two missing inputs (real `checkpoints/`, real `data/`), and those
have never been supplied on any machine this repo has run on.

### Evidence

`python scripts/check_data.py` (run this session):

```
MISSING  checkpoint  checkpoints/unetr_mse_ssim_aug.pth
MISSING  checkpoint  checkpoints/unet_mse.pth
MISSING  checkpoint  checkpoints/attunet_ssim_mse.pth
MISSING  data        data/openbhb/val_quasiraw
MISSING  data        data/brats/MICCAI_BraTS2020_TrainingData
5 problem(s). Eval and training are BLOCKED.
```

`artifacts/split_contract.json` — the deterministic split that every result must trace to — is
empty: `{"brats": {"test": [], "val": []}, "openbhb": {"train": [], "val": []}, "seed": 42}`.

### Component-by-component

| Spec | Component | Implemented | Unit-tested | Ever executed for real | Status |
|---|---|---|---|---|---|
| 000 | Vertical slice (`run_slice.py`) | Yes | Yes | No — fails `CheckpointError` | Blocked on checkpoint |
| 001 | Data layer (OpenBHB/BraTS, transforms, chunking, split contract) | Yes | Yes (`test_data.py`, 330 LOC) | No — no data | Blocked on data |
| 002 | Model registry (UNet, AttUNet, UNETR, diffusion, msa_unetr) | Yes | Yes | No real checkpoint ever loaded | Blocked on checkpoint |
| 003 | Recon engine + thresholding + sweep | Yes | Yes (`test_recon.py`) | No | Blocked |
| 004 | Eval harness + metrics + legacy bug-compat mode | Yes | Yes (`test_eval.py` 511, `test_eval_legacy.py` 241) | No | Blocked |
| 005 | Arch × loss matrix | Yes | Yes | Renderer runs; **0/10 cells evaluated** | Renders `n/a` only |
| 006 | Classical baseline (44 features/slice → XGBoost, StratifiedGroupKFold) | Yes | Yes (`test_classical.py` 609 LOC) | No — `ClassicalError: produced zero samples` | Blocked on data |
| 007 | Paradigm comparison | Yes | Yes (`test_paradigm.py` 472) | Renderer runs; **0/4 columns available** | Renders `n/a` only |
| 009 | Synthetic-anomaly (FPI) fine-tune harness | Yes | Yes (`test_synth.py` 340, `test_train.py` 368) | **Never trained** | Needs GPU run |
| 010 | Visualization + demo video export | Yes | Yes (`test_viz.py` 446) | No — needs a saved `ReconResult` | Blocked |
| 011 | Reporting + README scorecard | Yes | Yes (`test_report.py` 534) | Runs; **0/12 scorecard rows available** | Renders `n/a` only |
| 012 | Multi-scale attention UNETR (stretch) | Yes | Yes (`test_msa_unetr.py` 267) | Never trained (gate closed by design) | Out of scope for now |
| 013 | Diffusion / AnoDDPM | Phases 1–2 only (model + from-scratch training harness) | Yes (`test_diffusion.py` 242) | **Never trained.** Phases 3–4 (the run, then matrix/paradigm/report) pending | Needs GPU run |

There is **no Spec 008** — this is intentional and documented at `specs/README.md:50`, not a gap.

### Genuinely missing implementation (not merely unrun)

1. **Perceptual loss has no training code.** `configs/loss/perceptual.yaml` exists and warns "this
   loss has NEVER been run in this project"; `configs/matrix/arch_loss.yaml:19` records the cell's
   reason as `"no training code, no checkpoint"`. FR-23 requires perceptual loss as one of five
   loss arms. This is 1 of 10 matrix cells that cannot be filled even with GPU time.
2. **Spec 013 Phases 3–4** — the training run, `t_noise` sweep, recon, eval, and downstream
   report regeneration. Harness exists; the study step does not.

### Placeholders, mocks, fabricated numbers

**None found — and this is a genuine strength.** `grep -rniE 'TODO|FIXME|NotImplementedError'` over
`src/` and `scripts/` returns **0 matches**. Every "stub" string in the codebase refers to the
Git-LFS-pointer detection logic in `src/mri_ad/models/_checkpoint.py`, which *refuses* to load a
133-byte stub rather than silently degrading to `strict=False`. The renderers emit an explicit
`n/a (<reason>)` per cell with machine-readable `na_reason` fields rather than dropping rows or
inventing values (`src/mri_ad/eval/matrix.py:66`: *"Never blank, never a placeholder number."*).

**The repository has not lied about its own state anywhere I could find it.** The committed README
scorecard reads `0/12 rows available`. That is unusual discipline and is itself defensible material
in an interview.

### Can the intended workflow run end-to-end?

**Not on any machine to date.** The no-GPU half runs cleanly end-to-end and fails closed where it
should. The result-producing half has never had its inputs. Verified this session:

| Script | Behavior with no data | Correct? |
|---|---|---|
| `run_report.py` | prints `0/10 cells`, `0/12 rows`; leaves README unchanged | Yes |
| `run_arch_loss_matrix.py` | writes an all-`n/a` matrix | Yes |
| `run_paradigm_comparison.py` | writes an all-`n/a` table | Yes |
| `run_synth_comparison.py` | skips with a named reason | Yes |
| `run_classical.py` | `ClassicalError` (zero samples) | Yes — fails closed |
| `run_slice.py` | `CheckpointError` (named file) | Yes — fails closed |
| `run_slice_reduction.py` | `ArtifactError` (no results root) | Yes — fails closed |

---

## 3. Success metrics — definitions and targets

Targets below are cited to their source in this repository. Where no target was ever defined, the
row says so rather than manufacturing one.

### Tier 1 — headline scientific metrics

| # | Metric | What it measures | Why it matters here | Defined target | Impressive range | Target source |
|---|---|---|---|---|---|---|
| M1 | **Voxel-level Dice** (UNETR + MSE-SSIM, corrected pipeline) | Overlap of thresholded residual mask with binarized BraTS segmentation | The project's optimization target; the one number the whole harness exists to produce | **No absolute target defined.** The 0.6255 prior value is explicitly disowned as bug-produced | Unsupervised recon-based AD on BraTS commonly lands ~0.2–0.4 Dice; **no target range is defined in this repo, so none is asserted here** | `CLAUDE.md` ("optimize detection separability"); `planning/01-requirements.md` FR-18, NFR-5 |
| M2 | **Voxel-level IoU** | Same, stricter | Companion to M1 | None defined | Not defined in repo | FR-18 |
| M3 | **Spearman ρ(PSNR, Dice)** across matrix cells | Rank correlation between reconstruction fidelity and detection quality | **The central claim of the project.** A measured negative ρ *is* the finding | Direction: **negative**. No magnitude target | ρ ≤ −0.6 over ≥5 real cells would be a strong, quotable mechanism result | `CLAUDE.md` "THE CENTRAL DOMAIN FACT"; FR-25; `src/mri_ad/eval/matrix.py` `stats.spearman_psnr_dice` |
| M4 | **Synth-anomaly before/after ΔDice** (Spec 009) | Effect of FPI synthetic-anomaly fine-tuning on identical test split | BG-3, the project's stated performance novelty | "A **measurable** improvement" — direction only, **no magnitude specified** | A positive, controlled ΔDice on an identical split is publishable regardless of magnitude | `planning/01-requirements.md` BG-3, FR-37 |
| M5 | **Diffusion (AnoDDPM) Dice vs UNETR Dice** | Whether a generative prior escapes the reconstruct-the-tumor failure | `CLAUDE.md` D3 calls this the sharpest open question | **None — the outcome is deliberately open.** FR-33e mandates reporting a loss if it loses | Either direction is a result; "diffusion did not beat UNETR, here is the measured reason" is fully defensible | FR-33e, `CLAUDE.md` D3 |
| M6 | **Slice-level ROC-AUC / PR-AUC (classical baseline)** | Discriminative power of 44 engineered radiomic features + XGBoost, 5-fold StratifiedGroupKFold | BG-2; the honest control | None defined | ROC-AUC > 0.85 would be a strong classical row; **not a repo-defined target** | FR-28, BG-2 |
| M7 | **Legacy bug-compat reproduction** | Reproduce prior 0.6255 / 0.4551 *in bug-compat mode*, then show the corrected value differs | Turns "the old number was wrong" from assertion into demonstration — the most interview-durable result available | **Dice 0.6255 ± 0.005** (FR-21). ⚠️ `configs/eval/default.yaml:25` sets `tolerance: 0.02` — a real inconsistency with FR-21, flagged below | Hitting ±0.005 and then publishing a materially different corrected Dice is the single most credible outcome in this project | FR-21; `configs/eval/default.yaml:23-25` |

### Tier 2 — explanatory (never optimization targets)

| # | Metric | Role |
|---|---|---|
| M8 | **PSNR** | Explanatory only. NFR-6 and `CLAUDE.md` forbid using it as a target or headline. |
| M9 | **SSIM** | Same. Exists only to explain *why* high-fidelity models miss anomalies. |

### Tier 3 — systems / engineering metrics

| # | Metric | Defined target | Source |
|---|---|---|---|
| M10 | Inference latency per `(1,16,128,128)` volume | **< 5 s on target GPU**, measured and reported in the scorecard | NFR-9 |
| M11 | `make eval` wall-clock (full test split, no GPU) | **< 30 min** | NFR-10 |
| M12 | `make report` wall-clock | **< 2 min**, no GPU | NFR-11 |
| M13 | Determinism | Same config+seed+hardware reproduces within **±0.005** on Dice/IoU | NFR-1 |
| M14 | Line coverage on `src/mri_ad` | **≥ 80%** | NFR-17, T-9 |
| M15 | `ruff check` errors | **0** | NFR-18 |
| M16 | Run provenance | git SHA + resolved config + seed + timing persisted per run | NFR-2 |
| M17 | Demo video artifact | Exists, exported reproducibly | FR-40 |

---

## 4. Actual verified results

### Tier 1 and Tier 2 — every metric

| Metric | Verified result | Evidence |
|---|---|---|
| M1 Dice | **Not measured / no verified result available** | `artifacts/tables/arch_loss_matrix.md`: `n/a (not evaluated)`; `artifacts/metrics/` does not exist |
| M2 IoU | **Not measured** | same |
| M3 Spearman ρ | **Not measured** | `arch_loss_matrix.md`: `Spearman rho: n/a (fewer than 3 evaluated cells, n_pairs=0)` |
| M4 Synth ΔDice | **Not measured — model never trained** | `artifacts/tables/synth_before_after.md`: *"No aggregate metrics for 'unetr__mse_ssim' … Run `make eval` first."* |
| M5 Diffusion vs UNETR | **Not measured — model never trained** | matrix cell `diffusion__ddpm` → `na_reason: "untrained"`; `stats.diffusion_available: false` |
| M6 Classical ROC-AUC / PR-AUC | **Not measured — baseline never run** | `paradigm_comparison.md`: `n/a (not run)`; `run_classical.py` raises `ClassicalError` |
| M7 Legacy reproduction | **Not measured** | `README.md` scorecard: `Corrected Spec 004 baseline Dice: n/a (not yet evaluated)` |
| M8 PSNR | **Not measured** | all matrix cells `n/a` |
| M9 SSIM | **Not measured** | all matrix cells `n/a` |

**Aggregate: 0 of 10 matrix cells evaluated. 0 of 4 paradigm columns available. 0 of 12 README
scorecard rows available.** All three counts were reproduced by running the renderers this session.

The only numbers that appear anywhere in this repository's results surface are `0.6255` and
`0.4551` — and both are **prior-work values the repo explicitly labels untrustworthy**, stored in
`configs/eval/default.yaml` as *diff targets for a reproduction that has not been run*. They are not
results of this project and must never be presented as such.

### Tier 3 — systems metrics (several genuinely verified this session)

| Metric | Verified result | Target met | Evidence / conditions |
|---|---|---|---|
| M10 Inference latency | **Not measured** | — | README scorecard: `n/a (no recorded inference run… needs a cluster run)` |
| M11 `make eval` time | **Not measured** | — | never run with data |
| M12 `make report` time | **≈ 1 s** (0.975 s recorded) — but on an **empty** artifact tree, so not a valid test of the < 2 min target under load | Not meaningfully tested | `artifacts/runs/20260812T140109_206961Z/run_meta.json` `duration_seconds: 0.975` |
| M13 Determinism | **Not measured** | — | Requires two runs with real data; none exist |
| M14 Coverage | **94%** (3,560 statements, 229 missed) | ✅ **Exceeds 80%** | Run this session: `pytest <23 of 24 test files> --cov=src/mri_ad`. **Condition:** excludes `tests/test_models.py`, which is documented to spike swap to 27 GB+ on this 16 GB MacBook Air; includes `viz/`, which NFR-17 permits excluding — so the true NFR-17 figure is ≥94% |
| M15 ruff errors | **0**; `109 files already formatted`, `All checks passed!` | ✅ | Run this session: `ruff format --check` + `ruff check` over `src tests scripts` |
| M16 Provenance | **Working.** 18 runs each with git SHA, `git_dirty`, full resolved Hydra config (~350 keys), seed, hostname, start/end, and error class | ✅ | `artifacts/runs/*/run_meta.json`, inspected |
| M17 Demo video | **Not produced** | ❌ | `artifacts/figures/` contains only `paradigm_curves.png` (an empty-axes plot); no `artifacts/demo/` |

**Test suite, verified this session** — the strongest reproducible evidence the project currently has:

```
405 passed, 1 skipped, 2 warnings in 58.72s
TOTAL   3560 statements   229 missed   94%
```

Conditions: macOS 25.6.0, MacBook Air 16 GB, CPU only, `.venv` Python 3.11, `python -m pytest`
over 23 of 24 test files (all except `tests/test_models.py`). Reproducible via the batches in
`MANUAL_TESTING.md`. `tests/test_models.py` is **not currently runnable on this hardware** — 11 of
its cases are recorded as failing/aborted in `.pytest_cache/v/cache/lastfailed`, attributed in
`progress_report.md` §018 to memory exhaustion reproduced 3/3 times, not to a logic defect. That
attribution has **not been confirmed on GPU hardware** and remains an open item.

---

## 5. Expected vs actual

| Metric | Target | Impressive | Actual | Gap | Met | Evidence | Reason |
|---|---|---|---|---|---|---|---|
| M1 Dice | none defined | not defined in repo | **not measured** | total | ❌ | `arch_loss_matrix.md` | No checkpoints, no data, no GPU run |
| M2 IoU | none defined | — | **not measured** | total | ❌ | same | same |
| M3 Spearman ρ | negative | ρ ≤ −0.6 over ≥5 cells | **not measured** (`n_pairs=0`) | total | ❌ | `arch_loss_matrix.json` | Needs ≥3 evaluated cells; 0 exist |
| M4 Synth ΔDice | positive & measurable | any positive controlled Δ | **not measured** | total | ❌ | `synth_before_after.md` | Fine-tune never run (GPU-gated) |
| M5 Diffusion vs UNETR | open outcome, must be reported | either direction | **not measured** | total | ❌ | matrix `na_reason: untrained` | Spec 013 Phase 3 not run |
| M6 Classical AUC | none defined | ROC-AUC > 0.85 | **not measured** | total | ❌ | `paradigm_comparison.md` | `make classical` errors: no slices |
| M7 Legacy repro | 0.6255 ± 0.005 (FR-21) | hit tolerance + show corrected value differs | **not measured** | total | ❌ | README scorecard | Needs real UNETR checkpoint + BraTS |
| M8/M9 PSNR/SSIM | context only | n/a by design | **not measured** | total | ❌ | matrix | same |
| M10 latency | < 5 s/volume | < 1 s on A100 | **not measured** | total | ❌ | README scorecard | No GPU run |
| M11 eval time | < 30 min | — | **not measured** | total | ❌ | — | No GPU run |
| M12 report time | < 2 min | — | ~1 s on empty tree (not a real test) | untested under load | ⚠️ | `run_meta.json` | Nothing to render |
| M13 determinism | ±0.005 across runs | bit-identical | **not measured** | total | ❌ | — | Needs two real runs |
| M14 coverage | ≥ 80% | ≥ 90% | **94%** | +14 pts | ✅ | this session's pytest --cov | Test-first discipline held |
| M15 ruff | 0 errors | 0 | **0** | none | ✅ | this session | PostToolUse hook enforcement |
| M16 provenance | SHA+config+seed logged | + auto-generated README | **working, 18 runs** | none | ✅ | `artifacts/runs/` | Built in Spec 000 |
| M17 demo video | exists | — | **absent** | total | ❌ | `artifacts/` listing | Needs a saved `ReconResult` |

**Score: 3 of 17 metrics met. All three are engineering-hygiene metrics. Zero scientific metrics
have a value.**

---

## 6. Why the results are where they are

### Confirmed causes (directly evidenced)

1. **The experiment inputs have never existed on any machine this repo has run on.**
   `check_data.py` reports 5 missing items; `artifacts/split_contract.json` is empty. The three
   required checkpoints (`unetr_mse_ssim_aug.pth`, `unet_mse.pth`, `attunet_ssim_mse.pth`) live in
   a Google Drive folder linked from `README.md` and have never been downloaded here. OpenBHB and
   BraTS20 both require Kaggle registration and are absent. **This one cause accounts for every
   missing metric M1–M11 and M13, M17.**

2. **No GPU has ever been used.** `configs/train/ddpm_scratch.yaml` sets `allow_cpu: false`; the
   local machine is a 16 GB MacBook Air with no CUDA device. `GPU_SERVER_TASKS.md` documents the
   intended target (PARAM Rudra, IIT-B, account `24m1531`) and a complete SLURM workflow — but no
   job has been submitted. Every recorded run is sub-second and CPU-local.

3. **Deliberate policy prevented autonomous execution.** `CLAUDE.md` rule 4 makes training and
   sweeps manual-invoke-only, and D8 marks Specs 009 and 013 as `/train`-gated. The absence of
   results is therefore partly *by design* — the agent was correctly forbidden from spending GPU.
   The gap is that the manual invocation was never performed.

4. **Perceptual loss has no implementation.** Confirmed by absence in `src/mri_ad/train/objectives.py`
   (only `reconstruction_step` and `ddpm_step` exist) and by the explicit warning in
   `configs/loss/perceptual.yaml`. One of ten matrix cells is unfillable without new code.

5. **A requirements/config inconsistency exists on the FR-21 tolerance.** FR-21 specifies ±0.005;
   `configs/eval/default.yaml:25` sets `tolerance: 0.02` — 4× looser. Whichever is intended, the
   reproduction claim's strength depends on it, and today they disagree.

### Likely causes (evidence-supported, not confirmed)

6. **`tests/test_models.py` may hide real defects.** 11 cases are recorded as failing in the pytest
   cache, including `test_checkpoint_round_trip_both_layouts[unetr]` and
   `test_missing_checkpoint_raises[unetr]`. `progress_report.md` §018 attributes these to laptop
   memory exhaustion (reproduced 3/3 against a clean swap baseline), and every model *is* covered
   by a separate small-fixture file that passes. That attribution is plausible and evidenced — but
   **it has never been falsified by running the file on adequate hardware**, so a genuine
   UNETR checkpoint-loading defect cannot currently be ruled out. Given `CLAUDE.md` trap #4
   (UNETR weights do not load into stock MONAI `UNETR`), this is the specific risk worth retiring
   first on the cluster.

### Not causes (explicitly ruled out)

- **Not** insufficient engineering: 8.5k LOC src / 7.2k LOC tests, 94% coverage, zero lint errors.
- **Not** methodology error: the five known traps from the prior work are each addressed in code
  (binarized + nearest-neighbour segmentation resize, cross-subject *and* cross-chunk score
  accumulation, shared train/eval preprocessing, ported `UNETRReconstruction`, validation-only
  threshold tuning).
- **Not** dishonest reporting: no fabricated cell exists anywhere.
- **Not** a scale/load/concurrency problem — this project has no serving or throughput dimension;
  stress testing and concurrency benchmarks are not applicable metrics here.

### Untested

- Whether the real UNETR checkpoint loads under `strict=True`.
- Whether the diffusion model trains stably at full `(1,16,128,128)` resolution.
- Whether `make eval` meets NFR-10 at full split size.
- Whether the corrected pipeline reproduces 0.6255 in bug-compat mode.

---

## 7. Resume readiness

### Bottom line

**Do not put any performance number from this project on a resume today. There are none.** A single
interview question — *"what Dice did you get?"* — currently has no answer, and a fabricated one is
unrecoverable. The engineering, however, is already strong enough to defend.

### Safe to claim today (verified in this audit)

| Claim | Backing evidence |
|---|---|
| Built a config-driven, spec-driven 3D medical-imaging research codebase — 8.5k LOC source, 7.2k LOC tests, **405 tests, 94% line coverage**, zero lint errors | Reproduced this session |
| Designed a reproducibility harness that logs git SHA, fully-resolved config, seed and timing per run, and auto-generates the README scorecard from artifacts so published numbers cannot drift | `artifacts/runs/*/run_meta.json`, `src/mri_ad/eval/scorecard.py` |
| Identified **four specific defects** in a prior published result (score list re-initialized per subject; unbinarized `{0,1,2,4}` labels used in a Dice numerator; trilinear interpolation of categorical labels; only 1 of 8 depth chunks scored) and implemented an executable bug-compatibility mode to demonstrate them | `src/mri_ad/eval/legacy.py` (222 LOC), `README.md`, `tests/test_eval_legacy.py` |
| Implemented checkpoint loading that **refuses** Git-LFS pointer stubs and never falls back to `strict=False`, preventing partially-loaded models that score plausibly on garbage | `src/mri_ad/models/_checkpoint.py` |
| Implemented a classical radiomics baseline (44 features/slice: first-order, GLCM, gradient, hand-rolled GLRLM) with StratifiedGroupKFold to prevent subject leakage | `src/mri_ad/classical/` (5 modules), `tests/test_classical.py` |
| Implemented reporting that emits an attributed `n/a` per unmeasured cell rather than a placeholder number | `src/mri_ad/eval/matrix.py`, current tables |

Phrase these as **engineering and methodology**, never as results.

### Technically valid but not impressive on its own

- "94% test coverage" — good hygiene, not differentiating for an ML role by itself.
- "Implemented UNETR / AnoDDPM / FPI synthetic anomalies" — implementing a published architecture
  without running it reads as tutorial work to an experienced reviewer.

### Must NOT be claimed today

| Forbidden claim | Why |
|---|---|
| Any Dice / IoU / ROC-AUC / PR-AUC / PSNR / SSIM figure | **Zero measured values exist.** |
| "Improved Dice from 0.6255 to X" | Both halves are unmeasured, and 0.6255 is itself disowned as bug-produced. |
| "Showed synthetic-anomaly training improves detection" | The fine-tune has never been run. |
| "Compared three paradigms" (as a result) | The comparison table has 0 of 4 columns. *"Designed and implemented a three-paradigm comparison harness"* is true; *"compared"* is not. |
| "Demonstrated diffusion models resist reconstructing anomalies" | Never trained. |
| "Inference in under 5 s/volume" | Never timed. |
| Any mobility / trajectory claim | Explicitly a conceptual analogy; no such data was ever touched (NFR-22). |

### Evidence required before a stronger claim

1. Real checkpoints + real data on a GPU machine, `check_data.py` returning READY.
2. `make recon` + `make eval` for at least the UNETR headline cell → a real Dice with a split hash.
3. The classical baseline executed → real ROC-AUC/PR-AUC.
4. The legacy bug-compat run → the 0.6255 reproduction and its corrected counterpart.
5. At least three matrix cells → a real Spearman ρ, which is the project's actual thesis.
6. Ideally, the Spec 009 fine-tune → a real before/after ΔDice (BG-3).

Items 1–4 convert this from an unvalidated harness into a defensible study. Item 5 is what makes it
*interesting*. Item 6 is what makes it a *contribution*.

---

## 8. Reproducibility status

| Aspect | Status |
|---|---|
| Code reproducible from clean clone | **Yes** — `pip install -e ".[dev]"`, then the no-GPU paths run. Verified. |
| Tests reproducible | **Yes** — 405 pass on CPU, no data required, ≈59 s. Verified. `tests/test_models.py` excepted (hardware-limited on this machine). |
| Results reproducible | **N/A — there are no results to reproduce.** |
| Determinism verified | **No.** `seed_everything()` is called in every entry point and `deterministic: true` is set, but no two real runs have ever been compared (NFR-1 untested). |
| Provenance chain intact | **Yes**, and it is genuinely good — every future number will trace to a `run_meta.json` with SHA, config, and seed. |
| Data obtainable by a third party | **Yes** — OpenBHB and BraTS20 Kaggle links are in `README.md`; both require free registration. Checkpoints via the linked Drive folder. |
| Environment documented | **Yes** — `pyproject.toml` pins torch 2.2–3.0, MONAI 1.3–2.0; `GPU_SERVER_TASKS.md` documents the cluster setup end-to-end. |

Note: 12 of 18 commits' worth of `specs/`, `CLAUDE.md`, `progress_report.md` and `.claude/` changes
are **staged but uncommitted** in the working tree (`git status` shows them as added, and
`run_meta.json` records `git_dirty: true` for every run). Every provenance record so far therefore
points at a dirty tree — worth resolving before any real run, so the first real numbers trace to a
clean SHA.

---

## 9. Summary of what is safe versus what needs validation

**Safe to state as fact:** the harness is complete, tested at 94% coverage, lint-clean,
provenance-instrumented, and honest about being empty; four specific bugs in a prior result were
identified and made executable; a classical baseline, a synthetic-anomaly harness, and a diffusion
paradigm are all implemented and unit-tested.

**Requires validation before any statement:** every Dice, IoU, ROC-AUC, PR-AUC, PSNR, SSIM,
Spearman ρ, latency, and before/after delta in this project. All of them. Without exception.

The remaining work is documented in **`ACTION_PLAN_TO_RESULTS.md`**.
