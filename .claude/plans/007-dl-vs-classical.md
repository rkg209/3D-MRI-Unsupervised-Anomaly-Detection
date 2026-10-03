# Plan · Spec 007 — Paradigm comparison: Classical vs UNETR vs Diffusion

**Spec:** [`specs/007-dl-vs-classical.md`](../../specs/007-dl-vs-classical.md) (approved)
**Depends on:** 004 (eval harness), 005 (matrix conventions), 006 (classical baseline), 013 (diffusion — may be absent)
**Fresh compute:** none required for the headline table. The optional operating-point row needs a
val-split recon run (`make sweep`, GPU, manual-invoke only) — it renders `n/a` until then.

---

## Context

This is the project's headline table (D3): three paradigms, one test split. The spec exists because
the obvious version of this table is dishonest — a classical slice-level ROC-AUC printed beside a DL
voxel-level Dice invites a reader to declare a winner across two metrics that measure different
things at different granularities. The spec's fix is a **common footing**: reduce each DL model's
voxel anomaly mask to a slice-level score so all three columns share one genuinely comparable
number (slice ROC-AUC / PR-AUC), keep Dice/IoU for the thing only the DL models can do, and mark the
classical cells `n/a (no localization)` rather than leaving them blank.

Current repo state, verified:

- `classical/baseline.py:92` computes out-of-fold predicted probabilities and **discards them**
  after `binary_scores`. Only scalar AUC means/stds reach disk — so acceptance 3 (overlaid ROC/PR
  curves) is unsatisfiable from today's artifacts.
- No ROC/PR curve code exists anywhere outside `classical/metrics.py::binary_scores`.
- `per_volume.csv` carries no slice-level data, and the report-safe subgraph (test-enforced by
  `tests/test_eval_boundary.py`) forbids `torch`/`sklearn`, so the DL slice reduction **cannot**
  happen at report time. It has to happen where the `.pt` files are read.
- `scripts/run_recon.py` writes a manifest with `{model, loss, split, n_volumes}` — **no
  `split_hash`**. `run_classical.py` is the only thing in the repo that stamps
  `SplitContract.content_hash()`. Acceptance 1 ("asserted by comparing split hashes") therefore
  needs a hash on the DL side too.
- `make report` today renders exactly one cell (`cfg.model.name__cfg.loss.name`) and hard-fails on a
  missing artifact. It is not yet an orchestrator.

| Decision | Choice | Why |
|---|---|---|
| Classical AUC in the headline | Pooled out-of-fold, recomputed by one shared curve function | All three columns' AUCs then come from the *same* code path. 006's per-fold mean ± std is still printed alongside, and the small difference is stated, not hidden. |
| Curve implementation | Hand-rolled, pure-Python, in the report-safe subgraph | `sklearn` is forbidden in `eval/report.py`/`eval/matrix.py` and the report path. Precedent: `matrix.py::spearman` is hand-rolled for exactly this reason. A test pins it to sklearn to 1e-9. |
| DL slice score | Flagged-voxel **fraction** per slice, from `ReconResult.anomaly_mask` | The spec's own wording ("flagged-voxel count"). Fraction rather than count so slices of differing foreground area stay comparable. Declared in config, fixed before looking at any result. |
| Operating point row | Optional; `n/a (threshold not tuned)` until a val run exists | ROC/PR-AUC are threshold-free and need no GPU. Tuning the decision threshold needs val-split recon results — GPU, and rule 4 forbids spending it autonomously. |
| Missing diffusion column | `n/a (untrained)` read from config `na_reason` | The string already lives in `configs/matrix/arch_loss.yaml`. Mirror `matrix.py`'s policy exactly: absent artifact → declared `n/a`; **malformed** artifact → `ArtifactError`, never downgraded. |

The non-negotiable invariant this spec adds: **the three columns are scored on the identical sample
set, and that is asserted mechanically, not asserted in prose.** See "Same-sample assertion" below.

---

## Same-sample assertion — the core of acceptance 1

Split-hash equality is necessary but weak: it proves the same *subjects*, not the same *slices*.
Two facts make a stronger check available and cheap:

1. `classical/dataset.py::build_slice_dataset` calls `load_preprocessed_volume(cfg, subject_id)` and
   drops a slice when its foreground fraction `mean(image > foreground_eps) <= foreground_eps`.
2. `recon/engine.py::run_dataset_volume` reassembles chunks into `(1, D, 128, 128)` from the *same*
   preprocessed pipeline, plus chunking's zero-padded tail slices (155 → 160).

So if the DL reduction applies the **same** foreground-drop rule (same `foreground_eps`, read from
`cfg.classical.features.foreground_eps` — one source of truth) to `ReconResult.original`, the padded
tail slices fall out automatically (foreground 0) and the surviving `(subject_id, slice_index)` key
set must equal the classical one exactly.

The comparison therefore asserts, in order:

1. `classical_metrics.json:split.split_hash` == every DL `aggregate.json:split_hash`.
2. The `(volume_id, slice_index)` key set of each DL model's slice scores == the classical OOF key
   set, exactly. Any asymmetric key raises `ArtifactError` naming a few offending keys and the
   likely cause (`data.eval_subset` set on one side but not `classical.subject_limit` on the other).
3. The slice **labels** agree row-for-row between the two sides. Both derive from the same binarized
   seg, so a disagreement means a preprocessing drift bug — hard-fail.

Assertion 3 needs the label rule on the DL side. `eval/` must not import `classical/` (mirror of
acceptance 7's boundary, kept for symmetry), so `eval/slicelevel.py` restates the two-line rule and
`tests/test_slicelevel.py` asserts numeric agreement with `classical.dataset.slice_labels` on
synthetic volumes. Deliberate duplication, same reasoning as 006 R8.

---

## Files

### New

| Path | Purpose |
|---|---|
| `src/mri_ad/eval/curves.py` | Report-safe, pure-Python ROC/PR: curve points, ROC-AUC, average precision. No sklearn/scipy/numpy. |
| `src/mri_ad/eval/slicelevel.py` | The DL → slice-level reduction. Reads saved `ReconResult`s (torch), writes `slice_scores.csv`. Also the val-split threshold tuner. |
| `src/mri_ad/eval/paradigm.py` | Report-safe assembly: load the three columns, run the same-sample assertions, render `paradigm_comparison.{md,csv,json}`, plot the overlaid curves. |
| `scripts/run_slice_reduction.py` | `make slice-scores` — one DL model at a time; reduces a saved recon run to slice scores. Not report-safe (imports torch). |
| `scripts/run_paradigm_comparison.py` | `make paradigm` — report-safe; builds the table + figure from saved artifacts only. |
| `configs/paradigm/default.yaml` | Column declarations (`model`, `loss`, `role`, `na_reason`), score field, tuned threshold slot, paths. |
| `tests/test_curves.py` | ROC/PR correctness incl. ties, degenerate single-class input, agreement with sklearn. |
| `tests/test_slicelevel.py` | Reduction correctness, foreground drop, label agreement with `classical.dataset.slice_labels`, tuner refuses a non-val split. |
| `tests/test_paradigm.py` | Acceptance tests 1–6 (see mapping). |
| `artifacts/tables/paradigm_analysis.md` | The `results-analyst` narrative. Hand-written companion, precedent `arch_loss_analysis.md`. |

### Modified

- `src/mri_ad/classical/baseline.py` — `cross_validate` retains out-of-fold predictions. Add a
  frozen `OofPrediction(granularity, fold, volume_id, slice_index, y_true, y_score)` and a
  `@property oof_predictions -> list[OofPrediction]`, populated in the existing fold loop from
  `data.groups[test_idx]` / `data.slice_index[test_idx]` / `scores`. No change to CV mechanics, no
  change to any number 006 already publishes.
- `src/mri_ad/classical/report.py` — `write_predictions(rows) -> Path` →
  `artifacts/classical/metrics/oof_predictions.csv`. Carries a `granularity` column so the
  parametrized granularity test keeps its teeth. Fields:
  `granularity, fold, volume_id, slice_index, y_true, y_score`.
- `scripts/run_classical.py` — one extra `write_predictions` call. Nothing else.
- `scripts/run_recon.py` — add `"split_hash": contract.content_hash()` to the manifest dict. Two
  lines; the contract object is already in scope.
- `src/mri_ad/eval/loader.py` — no signature change; `read_manifest` already returns the raw dict.
- `scripts/run_eval.py` + `src/mri_ad/eval/report.py` — `write_aggregate` gains a required
  `split_hash: str` parameter, written as a top-level `split_hash` key in `aggregate.json`;
  `run_eval.py` reads it from the manifest and `_check_manifest` raises `ArtifactError` if absent
  ("re-run `make recon`"). Safe to make it required: `artifacts/metrics/` is empty on disk today.
- `scripts/run_report.py` — becomes a thin orchestrator: existing per-cell report, then the arch×loss
  matrix, then the paradigm comparison, each guarded so a missing input degrades to a printed
  skip-reason rather than a crash. This is what makes acceptance 6 true for `make report`.
- `Makefile` — add `slice-scores` and `paradigm` targets; add both to `.PHONY`; `slice-scores` is
  marked non-GPU (it reads saved `.pt`s, constructs no model).
- `configs/config.yaml` — add `paradigm: default` to `defaults`.
- `progress_report.md` — append the implementation entry (rule 2).

`configs/matrix/arch_loss.yaml` needs **no** change — 007 reads its own column list.

---

## Interfaces

### `eval/curves.py` — report-safe, no ML imports

```python
@dataclass(frozen=True)
class CurvePoint:
    threshold: float
    x: float  # FPR (ROC) or recall (PR)
    y: float  # TPR (ROC) or precision (PR)

@dataclass(frozen=True)
class BinaryCurves:
    roc: list[CurvePoint]
    pr: list[CurvePoint]
    roc_auc: float
    pr_auc: float
    n: int
    n_positive: int
    prevalence: float          # the PR-AUC's honest floor
    n_tied_score_groups: int   # ties are reported, never hidden

def binary_curves(y_true: Sequence[int], y_score: Sequence[float]) -> BinaryCurves
def roc_auc(y_true, y_score) -> float
def average_precision(y_true, y_score) -> float
```

Two implementation details are load-bearing, not stylistic. **Ties**: scores are sorted descending
and all rows sharing a score are consumed as one group before a curve point is emitted — otherwise
a threshold that cannot be realized produces an inflated AUC. Zero-flagged-voxel slices will form
one large tie group, so this is the common case here, not an edge case. **PR-AUC** is the step-wise
sum `Σ (R_i − R_{i−1}) · P_i`, matching `sklearn.average_precision_score`, *not* trapezoid — the
classical baseline's published numbers come from `average_precision_score`, and a trapezoid here
would silently disagree with 006. A single-class input raises `EvalError` rather than returning NaN
(the rule `classical/metrics.py` already follows).

### `eval/slicelevel.py` — the reduction (torch; **not** report-safe)

```python
SLICE_SCORE_FIELDS = ("volume_id", "slice_index", "flagged_voxels", "n_voxels",
                      "flagged_fraction", "foreground_fraction", "label")

@dataclass(frozen=True)
class SliceScore:
    volume_id: str; slice_index: int
    flagged_voxels: int; n_voxels: int; flagged_fraction: float
    foreground_fraction: float; label: int

def reduce_result(result: ReconResult, *, foreground_eps: float,
                  min_tumor_voxels: int = 1) -> list[SliceScore]
def reduce_results_dir(results_dir: Path, *, foreground_eps: float,
                       min_tumor_voxels: int = 1) -> list[SliceScore]
def write_slice_scores(rows: Sequence[SliceScore], path: Path) -> Path
def read_slice_scores(path: Path) -> list[dict[str, str]]        # report-safe reader, csv only

@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float; precision: float; recall: float; f1: float; split: str

def tune_slice_threshold(rows: Sequence[SliceScore], grid: Sequence[float], *,
                         split: str) -> list[ThresholdPoint]
```

`reduce_result` requires `result.ground_truth is not None` (an OpenBHB result has no label and must
not silently become an all-negative column) and drops a slice whose foreground fraction
`<= foreground_eps` — the identical rule `build_slice_dataset` applies, which is what makes the
key sets match and what removes chunking's zero-padded tail slices. `tune_slice_threshold` raises
`EvalError` for any `split != "val"`, mirroring `recon/sweep.py::run_threshold_sweep` — trap #5
enforced at the function boundary, not by convention.

`read_slice_scores` is a plain `csv.DictReader` and lives here only for locality; it imports no
torch at module scope, so `paradigm.py` may call it. The AST boundary test checks module-level
imports, so `slicelevel.py`'s torch import must stay module-level **and** `paradigm.py` must import
the reader lazily inside the function — or, cleaner and what this plan does: **the CSV reader moves
to `eval/paradigm.py`** and `slicelevel.py` keeps only the writer. One reader, unambiguously
report-safe.

### `eval/paradigm.py` — assembly and rendering (report-safe)

```python
NA_PREFIX = "n/a"                 # imported from eval.matrix — one spelling of "n/a"

@dataclass(frozen=True)
class ParadigmColumn:
    key: str                      # "classical" | "unetr__mse_ssim" | "diffusion__ddpm"
    paradigm: str                 # "Classical" | "UNETR" | "Diffusion (AnoDDPM)"
    localizes: bool
    split_hash: str | None
    roc_auc: float | None; pr_auc: float | None; prevalence: float | None
    dice: float | None; iou: float | None
    operating_point: ThresholdPoint | None
    curves: BinaryCurves | None
    na_reason: str | None
    n_slices: int | None

@dataclass(frozen=True)
class ParadigmStats:
    n_columns: int; n_available: int
    split_hash: str
    n_slices: int; n_positive: int; prevalence: float
    best_roc_auc_key: str; best_dice_key: str | None
    diffusion_available: bool

def load_classical_column(cfg, metrics_dir: Path) -> ParadigmColumn
def load_dl_column(cfg, column_cfg, slice_scores_root: Path,
                   metrics_root: Path) -> ParadigmColumn
def assert_same_samples(columns: Sequence[ParadigmColumn],
                        keysets: Mapping[str, frozenset]) -> None
def compute_stats(columns) -> ParadigmStats
def render_markdown(columns, stats, path: Path) -> Path
def render_csv(columns, path: Path) -> Path
def write_stats(columns, stats, path: Path) -> Path
def plot_curves(columns, path: Path) -> Path       # overlaid ROC + PR, one figure, two axes
```

`load_dl_column` pulls Dice/IoU from `artifacts/metrics/<cell_id>/aggregate.json` (`headline` block)
and slice scores from `artifacts/metrics/<cell_id>/slice_scores.csv`. Missing **either** file with a
declared `na_reason` → an `n/a` column; a malformed file → `ArtifactError`. A column with slice
scores but no `aggregate.json` (or the reverse) is malformed, not `n/a` — a half-present model is a
pipeline bug, and letting it render half a row is how a table cell stops being traceable.

`plot_curves` follows the house plotting pattern verbatim: `matplotlib.use("Agg")` inside the
function, `mkdir(parents=True, exist_ok=True)`, `fig.savefig`, `plt.close(fig)`, return `Path`. One
figure, two side-by-side axes (ROC left, PR right), same axes limits across models, the PR panel
carrying a dashed prevalence baseline — a PR curve without its prevalence floor is unreadable.

### Table shape

```
| Metric                        | Classical | UNETR   | Diffusion (AnoDDPM) |
|-------------------------------|-----------|---------|---------------------|
| Slice-level ROC-AUC           | 0.xxx     | 0.xxx   | n/a (untrained)     |
| Slice-level PR-AUC            | 0.xxx     | 0.xxx   | n/a (untrained)     |
| Slice-level F1 @ tuned thr.   | 0.xxx     | n/a (threshold not tuned) | n/a (untrained) |
| Voxel-level Dice              | n/a (no localization) | 0.xxx | n/a (untrained) |
| Voxel-level IoU               | n/a (no localization) | 0.xxx | n/a (untrained) |
```

Below the table, always emitted, never optional: the split hash, `n_slices` / `n_positive` /
prevalence (the PR-AUC floor), the granularity caveat verbatim from
`classical/metrics.py::GRANULARITY_NOTE`, and a line stating that the classical column's pooled-OOF
ROC-AUC differs slightly from 006's per-fold mean ± std, with both numbers printed.

---

## `configs/paradigm/default.yaml` (shape)

```yaml
# Spec 007 — the headline three-paradigm comparison.
# Columns are DATA, not code: adding Spec 013's diffusion row is a config edit (013 acceptance 2).
tables_dir: ${paths.artifact_root}/tables
figures_dir: ${paths.artifact_root}/figures
basename: paradigm_comparison

classical_metrics_dir: ${paths.artifact_root}/classical/metrics
metrics_root: ${eval.metrics_dir}

# The DL slice score. Declared here, BEFORE any result is looked at, so the choice of score can
# never be a post-hoc selection. Changing it is a config edit with a progress_report.md note.
score_field: flagged_fraction

# The slice-level decision threshold. MUST come from a VALIDATION sweep
# (`make slice-scores` on a val recon run -> tune -> a human edits this line). Never from test.
# Left null, the operating-point row renders "n/a (threshold not tuned)". AUC rows are unaffected.
slice_threshold: null
threshold_grid: [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

columns:
  - key: classical
    paradigm: Classical
    localizes: false
    na_reason: "not run"
  - {key: unetr__mse_ssim, model: unetr, loss: mse_ssim, paradigm: UNETR,
     localizes: true, na_reason: "not evaluated"}
  - {key: diffusion__ddpm, model: diffusion, loss: ddpm, paradigm: "Diffusion (AnoDDPM)",
     localizes: true, na_reason: "untrained"}
```

---

## Artifacts

```
artifacts/classical/metrics/oof_predictions.csv        # granularity,fold,volume_id,slice_index,y_true,y_score
artifacts/metrics/<cell_id>/slice_scores.csv           # SLICE_SCORE_FIELDS, one row per surviving slice
artifacts/metrics/<cell_id>/aggregate.json             # + new top-level "split_hash"
artifacts/tables/paradigm_comparison.md                # the headline table
artifacts/tables/paradigm_comparison.csv               # same cells, machine-readable
artifacts/tables/paradigm_comparison.json              # columns + stats + provenance
artifacts/tables/paradigm_analysis.md                  # results-analyst narrative (hand-written)
artifacts/figures/paradigm_curves.png                  # overlaid ROC + PR, one figure
artifacts/results/slice_threshold_sweep_<model>.csv    # VAL-split tuning, optional
```

`paradigm_comparison.json` top-level keys: `spec` (`"007"`), `granularity_note`, `split_hash`,
`sample{n_slices,n_positive,prevalence}`, `columns[]` (each with
`key,paradigm,localizes,roc_auc,pr_auc,dice,iou,operating_point,na_reason,source_paths`),
`stats{n_columns,n_available,best_roc_auc_key,best_dice_key,diffusion_available}`,
`provenance{run_id,eval_run_ids,recon_run_ids,classical_run_id}`. Every published number traces to
`source_paths` — that is what makes rule 6 checkable rather than aspirational.

---

## Tests → acceptance mapping

Synthetic data only, `tmp_path` for every artifact, fixtures written through the *real* writers so
a schema change breaks the test. No `.pt` file larger than `(1, 8, 16, 16)` — per the standing rule,
nothing in this suite runs a model or touches a real volume.

| Acc. | Tests |
|---|---|
| 1 | `test_mismatched_split_hashes_are_rejected` · `test_dl_and_classical_slice_keysets_must_match_exactly` · `test_subset_on_one_side_only_is_caught_and_named` · `test_slice_labels_agree_with_classical_slice_labels` |
| 2 | `test_table_has_auc_for_every_available_column` · `test_classical_dice_cell_reads_no_localization` · `test_absent_diffusion_column_reads_na_untrained` · `test_malformed_aggregate_raises_rather_than_rendering_na` · `test_half_present_column_is_malformed_not_na` |
| 3 | `test_plot_curves_writes_one_figure_with_all_available_models` · `test_pr_panel_carries_the_prevalence_baseline` |
| 4 | `test_markdown_contains_the_granularity_note_verbatim` · `test_markdown_states_pooled_oof_differs_from_per_fold_mean` |
| 5 | `test_stats_name_the_best_column_even_when_it_is_classical` · `test_diffusion_losing_to_unetr_is_rendered_not_dropped` |
| 6 | `test_paradigm_table_regenerates_byte_identically_from_saved_artifacts` · `test_report_skips_missing_inputs_without_crashing` |
| curves | `test_roc_auc_matches_sklearn_on_random_scores` · `test_average_precision_matches_sklearn_including_ties` · `test_all_zero_scores_give_auc_half_not_one` · `test_single_class_input_raises` |
| boundary | `test_paradigm_module_has_no_ml_imports` (extends the existing `tests/test_eval_boundary.py` allowlist) |
| reduction | `test_padded_tail_slices_are_dropped_by_the_foreground_rule` · `test_tuner_refuses_a_test_split` · `test_result_without_ground_truth_raises` |

`test_roc_auc_matches_sklearn_*` imports sklearn — that is fine in a *test*; the boundary test only
constrains `src/` and the report scripts.

Note: `tests/test_classical.py::test_every_generated_artifact_records_granularity` is parametrized
over artifact kinds — `oof_predictions.csv` must be added to that parametrize list, or the new
artifact silently escapes the check.

---

## Ordered steps

1. `eval/curves.py` + `tests/test_curves.py`. Riskiest novel code (tie handling, sklearn parity)
   goes first and is provable in isolation.
2. `classical/baseline.py` — retain OOF predictions. Existing 006 tests must stay green unchanged.
3. `classical/report.py::write_predictions` + `run_classical.py` wiring + granularity parametrize
   entry.
4. `scripts/run_recon.py` — `split_hash` into the manifest.
5. `eval/report.py::write_aggregate` + `run_eval.py` — `split_hash` into `aggregate.json`; update
   the existing `_aggregate` test fixture builders.
6. `eval/slicelevel.py` reduction + `tests/test_slicelevel.py` (foreground drop, label agreement,
   no-ground-truth refusal).
7. `eval/slicelevel.py` val tuner + its split guard test.
8. `scripts/run_slice_reduction.py` + `Makefile` target.
9. `configs/paradigm/default.yaml` + `configs/config.yaml` defaults entry; verify composition.
10. `eval/paradigm.py` loaders and the same-sample assertions + acceptance-1 tests.
11. `eval/paradigm.py` renderers (`md`/`csv`/`json`) + acceptance-2/4/5 tests.
12. `eval/paradigm.py::plot_curves` + acceptance-3 tests.
13. `scripts/run_paradigm_comparison.py` + `Makefile` target + boundary test extension.
14. `scripts/run_report.py` orchestration + acceptance-6 tests.
15. `results-analyst` narrative → `artifacts/tables/paradigm_analysis.md`.
16. Append to `progress_report.md`.

---

## Verification

- `make lint` — clean.
- `make test` — full suite; new tests are synthetic-only and CPU-only, so they run on the laptop.
- Config composition: `python -c "import hydra..."` compose check that `paradigm=default` resolves
  and `${eval.metrics_dir}` interpolates.
- Determinism: run `make paradigm` twice into a scratch `paths.artifact_root` and diff the three
  table files byte-for-byte (acceptance 6 is exactly this property).

**Proposed, not run by the agent** (needs real data/checkpoints on the cluster, and step 2 spends
GPU):

```bash
make check-data
make classical                                             # -> oof_predictions.csv
make recon HYDRA_OVERRIDES="+experiment=cluster model=unetr loss=mse_ssim"   # GPU, MANUAL
make eval  HYDRA_OVERRIDES="model=unetr loss=mse_ssim +eval.results_run_id=<run_id>"
make slice-scores HYDRA_OVERRIDES="model=unetr loss=mse_ssim +eval.results_run_id=<run_id>"
make paradigm
```

Post-hoc checks on the artifacts, in this order:
`paradigm_comparison.json:split_hash` equals `classical_metrics.json:split.split_hash`; `n_slices`
equals the classical `counts.n_samples`; the classical pooled ROC-AUC sits within a few thousandths
of 006's `headline.roc_auc_mean` (a large gap means the OOF join is wrong, not that the metric is
interesting); the diffusion column reads `n/a (untrained)`; every `n/a` cell carries a reason.

---

## Risks

**R1 · The table becomes the misleading thing the spec was written to prevent.** The failure mode is
a reader scanning one row and declaring a winner. Mitigation is structural, not editorial: the
`n/a` cells carry *reasons* (`no localization`, not blank), the granularity note is emitted verbatim
from `classical/metrics.py::GRANULARITY_NOTE` (so it cannot drift from 006's wording), and
acceptance-4 tests assert its presence in the rendered markdown. A test, not a good intention.

**R2 · Sample-set drift between the two sides.** `data.eval_subset` and `classical.subject_limit`
are independent knobs; `+experiment=laptop` sets both, but to *different* values (20 vs 10). Two
columns computed over different subject pools would look completely normal and be worthless. The
key-set equality assertion catches it and the error message names the two config keys explicitly.
This is why acceptance 1 is checked at slice granularity, not just by split hash.

**R3 · Ties in the slice score.** With percentile thresholding, slices with zero flagged voxels form
one large tie group. Naive ROC code that emits a point per row inflates AUC. `curves.py` consumes a
whole score group before emitting a point, `n_tied_score_groups` is reported in the JSON, and
`test_all_zero_scores_give_auc_half_not_one` pins the degenerate case.

**R4 · PR-AUC read without its floor.** A PR-AUC of 0.35 is excellent at 5% prevalence and terrible
at 40%. Prevalence is emitted next to every PR-AUC in the table, in the JSON, and as a dashed
baseline on the PR panel. 006 already established this discipline (`baselines.pr_auc_prevalence_mean`);
007 does not get to drop it.

**R5 · Test-split leakage through the reduction threshold.** The one new tunable in this spec is the
slice decision threshold, and the tempting shortcut is to pick it on the test scores that are
already in memory. Three defences: `tune_slice_threshold` raises on any `split != "val"`; the config
slot is `null` by default and a human writes the value with justification in `progress_report.md`
(the `run_sweep.py` precedent); and the AUC rows — the headline numbers — are threshold-free, so
there is no incentive to touch it at all.

**R6 · Two classical AUC numbers in one repo.** 006 publishes a per-fold mean ± std; 007 publishes a
pooled-OOF value from `curves.py`. They will differ in the third decimal. Left unexplained this
reads as an inconsistency or, worse, as a number chosen for being higher. Both are printed, in that
order, with the one-line reason. `test_markdown_states_pooled_oof_differs_from_per_fold_mean` keeps
the sentence present.

**R7 · The report-safe subgraph.** `tests/test_eval_boundary.py` AST-walks for
`{torch, monai, nibabel, torchmetrics, sklearn, scipy, skimage}`. `paradigm.py`, `curves.py`, and
`run_paradigm_comparison.py` must stay clean — which is precisely why the reduction lives in
`slicelevel.py` behind a CSV, and why the CSV **reader** lives in `paradigm.py` rather than next to
the writer. Add the two new report-safe modules to the boundary test's checked set in the same
commit that creates them, not later.

**R8 · Acceptance 5 is the one that gets quietly skipped.** "Report where each approach fails" is
prose, and prose is easy to soften when the classical baseline beats UNETR on slice ROC-AUC — a
plausible outcome, and one this project's central domain fact makes *likely*. `ParadigmStats`
computes `best_roc_auc_key` mechanically and the renderer states it whatever it is;
`test_stats_name_the_best_column_even_when_it_is_classical` fixes a synthetic case where classical
wins and asserts the table says so. Rule 6 with a test behind it.

**R9 · Scope creep into `scripts/run_report.py`.** Spec 011 owns full reporting (README scorecard,
all tables). 007 needs `make report` to regenerate *its* numbers (acceptance 6), which means adding
orchestration to a script that will be rewritten later. Keep the addition to a guarded call per
sub-report with a printed skip-reason — no discovery logic, no scorecard, no README edits. *That is
011's job and this spec must not pre-empt it.*
