# Action plan — from an unvalidated harness to defensible results

**Companion to:** [`PROJECT_RESULTS_AUDIT.md`](PROJECT_RESULTS_AUDIT.md) (2026-09-03)
**Premise:** the software is finished; **zero experimental results exist**. Every action below
exists to produce a number that does not yet exist, or to protect a number that is about to be
produced. Nothing here is refactoring, polish, or an optional feature.

**Reading this file cold:** each item is self-contained. Actions 1–5 are the critical path to
having *any* result. Actions 6–8 are what make the result interesting. Actions 9–10 are contingent.

> ## Corrections and the fast path — added 2026-09-30 (after reading the PARAM Rudra server log)
>
> The plan below is scientifically sound but **operationally wrong in five places**. Details and fixes:
> `GPU_SERVER_TASKS.md` §2 and `PARAM_Rudra_Quickstart_24m1531.md` §0–1.
>
> 1. **Command syntax.** Every `python scripts/run_*.py HYDRA_OVERRIDES="..."` below is invalid
>    (`HYDRA_OVERRIDES` is a Make variable). Use `make <target> HYDRA_OVERRIDES="..."` or
>    `python scripts/run_*.py +experiment=cluster model=...`.
> 2. **Paths — fixed in `cluster.yaml` 2026-09-30 (roots at `$SCR`); still export `MRI_AD_DATA`/`MRI_AD_CHECKPOINTS`.** Original problem: `configs/experiment/cluster.yaml` points at `/scratch/${USER}/...`; `/scratch/24m1531`
>    does not exist (real: `/scratch/IITB/ai-at-ieor/24m1531`), and `check_data.py` reads `MRI_AD_DATA` /
>    `MRI_AD_CHECKPOINTS`, not the config. Override `paths.*` on the command line or fix the config.
> 3. **No `git` on GPU nodes — fixed in `run_logger` 2026-09-30 (`.git/HEAD` fallback; `git_dirty: null` when unknown).** Original problem: `run_logger._git` returns `""`, so `run_meta.json` records an empty SHA and
>    `git_dirty: false`. Echo the SHA in the job script and/or add a `.git/HEAD` fallback *before*
>    publishing numbers (Action 3 onward).
> 4. **"Do it on the cluster" for downloads** means *login node* — GPU nodes are offline. `rsync`/`scp`
>    with `-p 4422` is untested; use `git pull` for code and VS Code drag/Download for files.
>    Env setup is uv-on-scratch, not `module load miniconda` + `conda create`; no lockfile, so `uv pip install`.
> 5. **Interactive GPU = `srun --pty`** (verified), not `salloc` + `ssh gpuNNN`. `make eval` needs no
>    GPU: run it (and `classical`) on the `small` CPU partition.
>
> **Parallelise instead of serialising.** The plan orders 5→6→7→8 to ration GPU time, but the cluster has
> 30 GPU nodes that started jobs in seconds on the account's prior project, and ~10 concurrent jobs are
> allowed. The critical path is the from-scratch diffusion run (Action 8), not the evaluations. Once
> Action 2's smoke test passes: submit Action 8 (after a ~20-step tiny sbatch) and Action 7 immediately —
> both need only data (+ the UNETR checkpoint for 007) and do *not* wait for Action 3's "before" eval —
> while Actions 3, 4, 5, 6 run concurrently as separate jobs (recon on `gpu`, eval/classical on `small`).
> Still manual-invoke by the user, per CLAUDE.md rule 4. Verify the ddpm harness can resume from a
> checkpoint before a 24 h walltime.
>
> **Pre-flight (Action 0):** commit + push the staged tree (server clones via git; user's call). The FR-21
> tolerance is **resolved, not a conflict**: Spec 004 explicitly replaces FR-21's ±0.005 with ±0.02 for the
> bug-compat reproduction (and explains why ±0.005 is meaningless), so `tolerance: 0.02` is correct.

**Standing constraints** (from `CLAUDE.md`, not negotiable):
- Never run `make train` / `run_train.py` autonomously — GPU spend is manual-invoke only.
- Never tune a threshold on the test split; validation only.
- Never use `strict=False` to work around a checkpoint failure.
- Never commit data, checkpoints, or `*.pth`.
- Append an entry to `progress_report.md` after every meaningful change.
- Report negative results as-is.

---

## 1. Acquire the real checkpoints and datasets, and get `check_data.py` to READY

**Action.** Download the three `.pth` checkpoints from the Google Drive folder linked in
`README.md` into `checkpoints/`, renamed to exactly `unetr_mse_ssim_aug.pth`, `unet_mse.pth`,
`attunet_ssim_mse.pth`. Download OpenBHB and BraTS20 from the Kaggle links in `README.md` into
`data/openbhb/val_quasiraw` and `data/brats/MICCAI_BraTS2020_TrainingData`. Do this **on the GPU
cluster**, not on the laptop (see Action 2 for transfer mechanics).

**Why.** This is the single blocking cause of every missing metric in the audit. `check_data.py`
reports 5 missing items and `artifacts/split_contract.json` is empty — nothing downstream can
produce a number until this is resolved. It gates literally every other action in this file.

**Current problem.** No checkpoint and no data has ever existed on any machine this repo has run
on. The seven `.pth` files under `legacy/` are 133-byte Git-LFS pointer stubs with no weights, and
`src/mri_ad/models/_checkpoint.py` correctly refuses to load them.

**Expected outcome.** `python scripts/check_data.py` prints no `MISSING` / `LFS STUB` lines and
reports 0 problems. This is not guaranteed — the Drive checkpoints have never been verified to load
under `strict=True` against `UNETRReconstruction` (that is Action 3).

**How to execute.** Follow `GPU_SERVER_TASKS.md` §4 verbatim. Kaggle CLI needs `~/.kaggle/kaggle.json`
uploaded first. For the Drive checkpoints, downloading in a browser on the Mac then `scp -P 4422`
to `/scratch/24m1531/mri_ad/checkpoints/` is more reliable than `gdown` on a headless login node.
Filenames must match `REQUIRED_CHECKPOINTS` in `scripts/check_data.py` and the `checkpoint:` field
in each `configs/model/*.yaml`.

**How to validate.** `python scripts/check_data.py` exits with "0 problem(s)". Spot-check that no
`.pth` is 133 bytes (`ls -l checkpoints/`).

**Metrics affected.** Unblocks M1–M11, M13, M17 — i.e. all of them.

**Resume impact.** None directly. Precondition for every claim.

**Dependencies.** Kaggle account (free registration) with both dataset licenses accepted; access to
the Drive folder.

---

## 2. Stand up the project on the GPU cluster and pass a real smoke test

**Action.** Set up the conda environment on PARAM Rudra, rsync the repo, `pip install -e ".[dev]"`,
then on an **interactive GPU allocation** run `pytest tests/test_models.py` (the one file this
laptop cannot run) followed by `python scripts/run_slice.py +experiment=cluster`.

**Why.** Two things must be established before spending real GPU hours: that the environment works,
and that `tests/test_models.py` fails on this laptop for memory reasons rather than because of a
genuine defect. The audit could not rule out the latter, and 11 of that file's cases are recorded
as failing — including `test_checkpoint_round_trip_both_layouts[unetr]` and
`test_missing_checkpoint_raises[unetr]`. `CLAUDE.md` trap #4 warns that UNETR weights do not load
into stock MONAI `UNETR`, so a UNETR checkpoint defect is a live, specific risk.

**Current problem.** 1 of 24 test files has never been executed anywhere. The laptop-memory
explanation in `progress_report.md` §018 is evidenced (3/3 reproductions against a clean swap
baseline) but never falsified on adequate hardware.

**Expected outcome.** `test_models.py` passes on the cluster, retiring the open question; and
`run_slice.py` produces the project's **first real Dice number** on one volume plus a saved figure.
If `test_models.py` fails on an A100, that is a real bug — stop and fix it before Action 3.

**How to execute.** `GPU_SERVER_TASKS.md` §2, §3, §5. Key points: `module load miniconda`,
`conda create --name mri_ad python=3.11`, torch must satisfy `>=2.2,<3.0` per `pyproject.toml`
(there is no `requirements.txt` — do not reuse the unrelated `PARAM_Rudra_Training_Workflow.md`,
which documents a different project). Allocate with
`salloc --nodes=1 --time=0:30:00 --gres=gpu:1 --partition=gpu`. Confirm CUDA with
`python -c "import torch; print(torch.cuda.is_available())"`.

**How to validate.** `pytest tests/test_models.py -q` reports all-passed with no OOM.
`run_slice.py` writes a `run_meta.json` under `artifacts/runs/` whose `metrics` block contains a
non-null Dice — the first non-empty metrics block in the project's history.

**Metrics affected.** M1 (first data point), and de-risks M1–M7.

**Resume impact.** Enables *"validated the full pipeline end-to-end on cluster GPU hardware"* and
converts the 405-test claim into a genuine 100%-of-suite claim.

**Dependencies.** Action 1.

---

## 3. Commit the deterministic split, then run the corrected UNETR headline evaluation

**Action.** Run `make recon` then `make eval` for the `unetr` + `mse_ssim` cell on the cluster,
producing `artifacts/metrics/unetr__mse_ssim/aggregate.json` and a populated
`artifacts/split_contract.json`. Time the run and record inference seconds/volume.

**Why.** This produces the project's headline number — the corrected Dice/IoU that the entire
rebuild exists to establish — and simultaneously satisfies NFR-9 (inference < 5 s/volume) and
NFR-10 (`make eval` < 30 min), neither of which has ever been measured. It also creates the split
contract that every subsequent comparison must hash against for the comparisons to be legitimate.

**Current problem.** M1, M2, M10, M11 are all "not measured". `split_contract.json` is empty, so
the Spec 007 acceptance test ("identical split, asserted by comparing hashes") is not even
demonstrable.

**Expected outcome.** A real, traceable Dice and IoU for UNETR+MSE-SSIM on the BraTS test split.
The magnitude is genuinely unknown and **may well be low** — unsupervised reconstruction-based
detection is a hard problem and the project's own thesis predicts that a faithful reconstructor
scores poorly. A low number is a valid result, not a failure.

**How to execute.**
```bash
python scripts/run_recon.py HYDRA_OVERRIDES="+experiment=cluster model=unetr"
python scripts/run_eval.py  HYDRA_OVERRIDES="+experiment=cluster model=unetr"
```
Threshold strategy comes from `configs/threshold/` (default: 95th-percentile). If tuning it, use
`make sweep` on the **validation** split only (`scripts/run_sweep.py`) and commit the chosen value
to config by hand — never select on test (`CLAUDE.md` trap #5). Wrap the eval in `/usr/bin/time -v`
or read `duration_seconds` from the emitted `run_meta.json` for NFR-9/NFR-10 evidence.

**How to validate.** `artifacts/metrics/unetr__mse_ssim/aggregate.json` exists with non-null
`dice`/`iou`. `artifacts/split_contract.json` lists real subject IDs and yields a stable hash. Then
`make report` on any machine turns the README scorecard rows from `n/a` to real values, and
`make report-check` passes. Confirm the eval accumulated scores across **all** subjects and **all 8**
depth chunks (prior-work traps #2) — the aggregate's sample count should reflect the full split.

**Metrics affected.** M1, M2, M8, M9, M10, M11, M17 (a saved `ReconResult` also unblocks the demo
video via `make demo`).

**Resume impact.** The first defensible performance claim in the project: *"corrected voxel-level
Dice of X on the BraTS20 test split, from a pipeline whose every published number regenerates from
logged artifacts."*

**Dependencies.** Actions 1, 2.

---

## 4. Run the legacy bug-compatibility reproduction and publish the delta

**Action.** Run `make eval LEGACY_BUG_COMPAT=1` (equivalently
`HYDRA_OVERRIDES="eval=legacy_compat"`) against the same UNETR checkpoint, and record whether it
reproduces the prior report's Dice 0.6255 / IoU 0.4551. Then publish the signed delta against the
corrected value from Action 3.

**Why.** This is, in my judgement, **the most interview-durable result this project can produce**,
and it does not depend on beating anyone. It converts *"the old number was wrong"* from an
assertion into an executable demonstration: run the pipeline with the four bugs re-enabled, get
0.6255; run it corrected, get something materially different. Very few portfolio projects
falsify their own prior published result.

**Current problem.** M7 is unmeasured. `src/mri_ad/eval/legacy.py` (222 LOC) and
`tests/test_eval_legacy.py` (241 LOC) exist and pass on synthetic tensors, but the reproduction has
never been attempted against the real checkpoint and real data — so the claim currently rests on
code review, not on a measurement.

**Expected outcome.** Bug-compat Dice lands within tolerance of 0.6255, and the corrected Dice
differs materially. Not guaranteed: if bug-compat mode does *not* hit 0.6255, that itself is
informative (it would mean a fifth, unidentified difference exists between this reimplementation
and the original notebook) and must be reported rather than tuned away.

**How to execute.**
```bash
python scripts/run_eval.py eval=legacy_compat HYDRA_OVERRIDES="+experiment=cluster model=unetr"
```
The bug-compat parameters live in `configs/eval/default.yaml` under `eval.legacy`
(`published_dice: 0.6255`, `published_iou: 0.4551`, `chunk_index: 4`, `threshold: 0.1`,
`subject_count: 100`, `crop_size: [160,130,170]`). **Resolve the tolerance inconsistency first:**
FR-21 in `planning/01-requirements.md:93` specifies ±0.005 but `configs/eval/default.yaml:25` sets
`tolerance: 0.02`. Pick one, make requirements and config agree, and note the decision in
`progress_report.md` — the strength of the reproduction claim depends on which bound was actually met.

**How to validate.** The eval reports the reproduction as within/outside tolerance, and the README
scorecard's `Corrected Spec 004 baseline Dice` and `Delta vs. prior published Dice` rows populate
with real values via `make report`.

**Metrics affected.** M7, and it contextualizes M1.

**Resume impact.** *"Reproduced a prior published Dice of 0.6255 in an executable
bug-compatibility mode, isolated four causal defects (per-subject score reinitialization,
unbinarized `{0,1,2,4}` labels in the Dice numerator, trilinear interpolation of categorical
labels, 1-of-8 depth-chunk coverage), and reported the corrected value with its signed delta."*
This is a strong, specific, unfakeable claim.

**Dependencies.** Action 3 (needs the corrected value to diff against).

---

## 5. Run the classical radiomics baseline

**Action.** Run `make classical` on the cluster to produce
`artifacts/classical/metrics/classical_metrics.json` with real slice-level ROC-AUC and PR-AUC from
5-fold StratifiedGroupKFold gradient boosting over the 44 engineered features.

**Why.** BG-2 exists specifically to close a classical-ML gap on the resume, and this baseline is
the honest control for the whole study — without it, "deep learning helps" is unfalsifiable. It is
also the **cheapest real result available**: feature extraction plus XGBoost, no GPU strictly
required, and it fills one of four paradigm columns on its own.

**Current problem.** M6 is unmeasured; `run_classical.py` currently raises
`ClassicalError: produced zero samples for the given pool` because there are no slices to extract
features from. The paradigm table shows `n/a (not run)`.

**Expected outcome.** A real ROC-AUC and PR-AUC with cross-validation fold statistics. There is a
genuine chance the classical baseline is **competitive with or better than** the reconstruction
models at slice-level detection — FR-31 explicitly requires reporting that if it happens.

**How to execute.**
```bash
python scripts/run_classical.py HYDRA_OVERRIDES="+experiment=cluster"
```
Config is `configs/classical/`. Two things to keep visible in the write-up: the operating
granularity is **slice-level, not voxel-level**, so its ROC-AUC is *not* comparable to voxel Dice
(FR-32 requires this caveat stated); and PyRadiomics was substituted with scikit-image plus a
hand-rolled GLRLM — every artifact already records `features.library: scikit-image` and
`pyradiomics_used: false`, and that substitution must stay visible. Verify the CV uses
`StratifiedGroupKFold` grouped by subject so no subject's slices straddle a fold boundary.

**How to validate.** `artifacts/classical/metrics/classical_metrics.json` exists with real AUC
values; the `classical` column in `artifacts/tables/paradigm_comparison.md` populates after
`make paradigm`; the split hash matches the contract from Action 3.

**Metrics affected.** M6, and one of four M5-table columns.

**Resume impact.** *"Cross-validated gradient-boosting baseline over 44 engineered radiomic
features (first-order, GLCM, gradient, GLRLM) with subject-grouped folds, achieving ROC-AUC X —
reported alongside the deep pipeline with the slice-vs-voxel granularity caveat stated explicitly."*
Directly satisfies BG-2.

**Dependencies.** Action 1. (Independent of Actions 2–4 — can run in parallel.)

---

## 6. Populate at least three more matrix cells and measure the Spearman ρ

**Action.** Run `make recon` + `make eval` for the remaining cells that have checkpoints —
`unet` + `mse` and `attention_unet` + `mse_ssim` — plus any additional UNETR loss variants for
which weights exist, then re-run `make matrix`.

**Why.** **This is the project's actual thesis.** `stats.spearman_psnr_dice` — the rank correlation
between reconstruction fidelity and detection quality — is the quantitative form of the central
claim, and it requires at least 3 evaluated cells to compute (currently `n_pairs = 0`). One
headline Dice is a data point; a measured anti-correlation across a model×loss grid is a *finding*.

**Current problem.** M3 is unmeasured. The matrix has 0 of 10 cells. FR-25 requires a written
analysis "demonstrating that higher reconstruction fidelity does not produce better anomaly
detection" — which cannot be written from zero cells.

**Expected outcome.** A negative Spearman ρ over ≥3–5 cells, with PSNR/SSIM and Dice side by side
making the trade-off visible. **The sign is not guaranteed.** If the anti-correlation does not
appear, or appears too weakly across too few cells to support, `CLAUDE.md` rule 6 requires
reporting that as the finding — and a well-measured null result is still defensible.

**How to execute.** Per cell:
```bash
python scripts/run_recon.py HYDRA_OVERRIDES="+experiment=cluster model=unet loss=mse"
python scripts/run_eval.py  HYDRA_OVERRIDES="+experiment=cluster model=unet loss=mse"
```
The declared cells are in `configs/matrix/arch_loss.yaml`. Note two cells that **cannot** be filled
without new work: `unetr__perceptual` has no training code at all (`configs/loss/perceptual.yaml`
warns it has never been run; `src/mri_ad/train/objectives.py` implements only `reconstruction_step`
and `ddpm_step`), and `msa_unetr__mse_ssim` is a deliberately gated stretch row. Leave both as
attributed `n/a` — do not invent a value, and do not build perceptual-loss training unless the grid
is otherwise too sparse for a ρ, in which case it is a scoped addition requiring a spec.
Then `python scripts/run_arch_loss_matrix.py` and `python scripts/run_report.py`.

**How to validate.** `artifacts/tables/arch_loss_matrix.json` reports `n_available >= 3` and a
non-null `spearman_psnr_dice` with `n_pairs > 0`. Regenerate the narrative with the
`results-analyst` agent against the real JSON.

**Metrics affected.** M1, M2, M3, M8, M9.

**Resume impact.** The strongest scientific claim available: *"measured a Spearman ρ of X between
reconstruction fidelity (PSNR) and detection quality (Dice) across N architecture×loss
configurations, quantifying why higher-fidelity reconstruction models fail to flag the anomaly they
reconstruct."* This is a mechanism result, not a leaderboard entry — and reviewers weight it higher.

**Dependencies.** Actions 1, 2, 3. Needs checkpoints for the additional cells to exist in the Drive
folder — verify before planning around them; cells without weights stay `n/a`.

---

## 7. Run the Spec 009 synthetic-anomaly fine-tune and the controlled before/after

**Action.** Submit the SLURM job for the FPI synthetic-anomaly fine-tune of UNETR, then evaluate
the resulting `unetr_synth` checkpoint on the **identical** test split and render the before/after
table with `make synth-table`.

**Why.** BG-3 — "demonstrate a measurable performance improvement via synthetic-anomaly training,
with a controlled before/after comparison on an identical test split" — is one of only two stated
performance novelties in the project. It is currently entirely unrealized: the model has never been
trained.

**Current problem.** M4 is unmeasured. `artifacts/tables/synth_before_after.md` reads *"No
aggregate metrics … Run `make eval` first."*

**Expected outcome.** A controlled ΔDice on an identical split. **Direction is not guaranteed** —
FPI fine-tuning may not help, and if ΔDice is negative or negligible that must be published as-is.
A rigorously controlled negative result here is still a legitimate finding and far better than no
comparison at all.

**How to execute.** `GPU_SERVER_TASKS.md` §7. Batch script with `--gres=gpu:1 --time=12:00:00`, then:
```bash
python scripts/run_train.py HYDRA_OVERRIDES="+experiment=cluster model=unetr_synth train=finetune"
```
Config: `configs/train/finetune.yaml` (`init_from: unetr`, lr 1e-5, `max_epochs: 20`,
`early_stopping_patience: 6`, selection on `val_dice` every 2 epochs over 8 val volumes). The FPI
generator has a **separability preflight** (`configs/synth/`: `abort_on_failure: true`,
`max_intensity_dice: 0.5`) that aborts if the synthetic anomalies are trivially separable by
intensity alone — if it fires, that is a real signal about the corruption parameters, not an
obstacle to bypass. Afterwards, `run_recon.py` + `run_eval.py` for `unetr_synth`, then
`python scripts/run_synth_comparison.py` (no GPU).

**How to validate.** `artifacts/tables/synth_before_after.{md,csv}` shows before and after Dice/IoU
with matching split hashes on both rows — the "identical test split" requirement of FR-37 is only
satisfied if those hashes match.

**Metrics affected.** M4, and one paradigm column.

**Resume impact.** *"Fine-tuned with foreign-patch-interpolation synthetic anomalies and measured a
controlled before/after ΔDice of X on an identical held-out split."* Note this is a *manual,
GPU-spending* run — do not launch it from an agent session.

**Dependencies.** Actions 1, 2, 3 (the "before" number must exist first).

---

## 8. Run Spec 013 Phase 3 — train the diffusion model and complete the three-paradigm comparison

**Action.** Train the AnoDDPM `DiffusionModelUNet` from scratch on healthy OpenBHB volumes, run
the validation-split `t_noise` sweep, commit the selected `t_noise` to config by hand, then recon +
eval + regenerate the paradigm table.

**Why.** `CLAUDE.md` D3 names the three-paradigm comparison (Classical vs UNETR vs Diffusion) as
**the headline comparison of the entire project**, and D3 calls the diffusion question the sharpest
form of the central domain fact: does a generative prior resist rebuilding the tumor, or fall to the
same failure? Without this, the "three-paradigm study" framing is not supportable.

**Current problem.** M5 is unmeasured; the `diffusion__ddpm` cell reads `untrained` and
`stats.diffusion_available: false`. Spec 013's own status line marks Phases 3–4 as pending.

**Expected outcome.** A real diffusion Dice, comparable to UNETR's on the same split. FR-33e is
explicit that if it does not improve on UNETR, that result is reported, not hidden. This is the
**most expensive and highest-risk** item here: from-scratch training over 100 epochs, and the
harness has never been exercised at full resolution.

**How to execute.** `GPU_SERVER_TASKS.md` §8. Budget generously (`--time=24:00:00` to start):
```bash
python scripts/run_train.py HYDRA_OVERRIDES="+experiment=cluster train=ddpm_scratch model=diffusion loss=ddpm"
python scripts/run_tnoise_sweep.py HYDRA_OVERRIDES="+experiment=cluster model=diffusion"
# copy the winning t_noise from tnoise_selection.json into configs/model/diffusion.yaml BY HAND
python scripts/run_recon.py HYDRA_OVERRIDES="+experiment=cluster model=diffusion"
python scripts/run_eval.py  HYDRA_OVERRIDES="+experiment=cluster model=diffusion"
```
Configs: `configs/train/ddpm_scratch.yaml` (from scratch, lr 1e-4, 100 epochs, `amp: false` to
preserve determinism per NFR-1, selection on `val_loss`) and `configs/model/diffusion.yaml`
(`t_noise: 250` starting value, DDIM sampler with 50 strided steps, sweep grid
`[100,150,200,250,300,400,500]`). The `t_noise` sweep is **validation-split only** and the winner is
committed by hand as a traceable change — never auto-written, never selected on test.

**How to validate.** `checkpoints/diffusion.pth` exists and loads under `strict=True`;
`artifacts/tables/paradigm_comparison.md` shows real values in all available columns with one split
hash across them; `artifacts/figures/paradigm_curves.png` shows actual overlaid ROC/PR curves rather
than empty axes.

**Metrics affected.** M5, M3 (adds a matrix row), and completes the paradigm table.

**Resume impact.** *"Compared three anomaly-detection paradigms — classical radiomics, UNETR
reconstruction, and an AnoDDPM diffusion prior — on an identical split, and reported which resists
reconstructing the anomaly."* Only claimable after this runs; before it, the honest phrasing is
"designed and implemented a three-paradigm comparison harness".

**Dependencies.** Actions 1, 2, 3. Do Actions 5–7 first — they are cheaper and each produces a
result independently, so if GPU time runs out you still have a study.

---

## 9. Regenerate every artifact and verify the README scorecard reflects only measured values

**Action.** Back on any machine, `rsync` `artifacts/` down from the cluster, then run
`make matrix`, `make paradigm`, `make synth-table`, `make report`, and `make demo`. Review
`git diff README.md` before committing.

**Why.** The auto-generated scorecard is the project's public face and its anti-drift guarantee —
NFR-2/FR-42 require every headline number to regenerate from artifacts without manual editing.
It is also the final honesty check: any cell that is still `n/a` must remain `n/a`.

**Current problem.** The README scorecard reads `0/12 rows available`. `make demo` has never
produced a video (FR-40, M17) because no saved `ReconResult` has ever existed.

**Expected outcome.** A README scorecard with real values in every row backed by a real run, and an
exported demo MP4 showing original / reconstruction / residual / anomaly mask / ground-truth overlay.

**How to execute.** `GPU_SERVER_TASKS.md` §9 for the rsync, §10 for regeneration. Then
`make report-check` (which runs `git diff --exit-code README.md`) must pass, proving the committed
README equals what the generator produces. `make demo` reads `configs/viz/` and writes
`artifacts/demo/demo.mp4`. **Before committing, confirm nothing under `data/`, `checkpoints/`, or
any `*.pth`/`*.nii*`/`*.npy`/`*.pt` entered the index**, and that no patient identifier or scan path
appears in any artifact or `run_meta.json` (NFR-12/NFR-13).

**How to validate.** `make report-check` exits 0. Every non-`n/a` scorecard row traces to a file
under `artifacts/`. `artifacts/demo/demo.mp4` plays.

**Metrics affected.** M12, M17; publishes M1–M7.

**Resume impact.** Enables the reproducibility claim in its strongest form, plus a demo video for
the non-technical reviewer persona (U-4/SH-3).

**Dependencies.** At least Action 3; ideally 3–8.

---

## 10. Verify determinism (NFR-1) by re-running one evaluation

**Action.** Re-run `make eval` for the UNETR headline cell a second time with identical config and
seed on the same hardware, and diff the resulting `aggregate.json` against the first.

**Why.** NFR-1 requires results reproducible within ±0.005 on Dice/IoU, and the whole
reproducibility story — the project's strongest differentiator — currently rests on
`seed_everything()` being *called* rather than on determinism being *observed*. This is the cheapest
possible verification of the claim the project leans on hardest.

**Current problem.** M13 unmeasured. No two real runs have ever existed to compare.

**Expected outcome.** Identical or within-tolerance Dice/IoU across runs. If they differ, a
nondeterministic op has slipped in and must be found before any number is published — worth
knowing *before* an interviewer asks rather than after.

**How to execute.** Re-run the exact Action 3 command, writing to a fresh run directory, and
compare the two `artifacts/metrics/unetr__mse_ssim/aggregate.json` snapshots. Note that
`configs/train/ddpm_scratch.yaml` already sets `amp: false` explicitly because mixed precision
breaks bitwise determinism — the same concern applies to any AMP use at eval time.

**How to validate.** |ΔDice| ≤ 0.005 and |ΔIoU| ≤ 0.005 between the two runs, with both
`run_meta.json` files recording the same seed and git SHA.

**Metrics affected.** M13; validates M1, M2.

**Resume impact.** Upgrades *"designed for reproducibility"* to *"verified reproducibility to within
±0.005 across independent runs"* — a claim most portfolio projects cannot make.

**Dependencies.** Action 3.

---

## Also worth doing (cheap, and each protects a result above)

- **Commit the staged working tree before any real run.** 12 commits' worth of `specs/`,
  `CLAUDE.md`, `progress_report.md`, `.claude/` and `planning/` changes are staged but uncommitted,
  and every `run_meta.json` to date records `git_dirty: true`. The first real results should trace
  to a clean SHA, or the provenance chain is weaker than it looks. (No `Co-Authored-By` trailer —
  `CLAUDE.md` rule 1, hook-enforced.)
- **Resolve the FR-21 tolerance conflict** (±0.005 in requirements vs `0.02` in
  `configs/eval/default.yaml:25`) before Action 4, so the reproduction claim has one unambiguous bound.

## What is deliberately NOT in this plan

Refactoring, documentation cleanup, new features, architecture changes, and stretch Spec 012 —
none of these change a metric. Also excluded: stress testing, concurrency and scalability
benchmarks, and production/serving evaluation. This is an offline research comparison with no
serving dimension; those metrics do not apply and manufacturing them would weaken the study, not
strengthen it.

## Minimum viable path if GPU time is scarce

Actions **1 → 2 → 3 → 4 → 5 → 9**. That yields: a corrected headline Dice/IoU, a demonstrated
falsification of the prior 0.6255 with four named causal bugs, a cross-validated classical
baseline, a populated README scorecard, and a demo video — with **no training run at all** (every
one of those steps uses existing checkpoints). Actions 6, 7, 8 add the scientific finding, the
performance novelty, and the third paradigm respectively, in that order of value-per-GPU-hour.
