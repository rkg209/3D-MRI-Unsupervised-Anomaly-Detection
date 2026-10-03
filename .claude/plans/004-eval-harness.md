# Plan · Spec 004 — Evaluation harness

**Spec:** [`specs/004-eval-harness.md`](../../specs/004-eval-harness.md) (status: approved) · **Depends on:** 003 (done) · **Fresh compute:** no (normal mode); legacy-compat mode needs checkpoints + BraTS on the cluster.

---

## Context

Every claim this project makes rests on Dice and IoU. In the prior work they were computed with four
independent bugs (`legacy/metric-uad.ipynb`), each sufficient on its own to invalidate the published
UNETR headline of Dice 0.6255 / IoU 0.4551. Spec 003 landed the reconstruction engine and persists
`ReconResult` files; nothing yet scores them.

This spec builds the canonical metric implementation, scores the full test split **from saved
`ReconResult` files with zero `forward()` calls**, and publishes an honest corrected baseline
alongside an explicit bug-compatibility mode that reproduces the old number. The corrected Dice may
well be lower than 0.6255 — per D2 and rule 6, that is reported as-is.

Current state: `src/mri_ad/eval/` contains only `metrics.py`, in which `MetricsComputer.dice` is
implemented but `iou`, `psnr`, `ssim` all `raise NotImplementedError("Spec 004")`. The `Makefile`
declares `eval` and `report` targets pointing at `scripts/run_eval.py` and `scripts/run_report.py`,
neither of which exists.

### Decisions taken before planning

- **D-A** — A new `scripts/run_recon.py` + `make recon` produces the test-split `ReconResult`s
  (GPU, manual-invoke only, reusing the existing `ReconstructionEngine`). `run_eval.py` never
  constructs a model in normal mode. This is a small scope addition beyond the spec text, required
  because `run_sweep.py` only ever saves **val**-split results; it must be called out in
  `progress_report.md`.
- **D-B** — `AggregateEvaluator.evaluate_split(results_dir, labels_dir=None)` defaults to the
  ground truth already embedded in each `ReconResult` (binarized + nearest-resized by the spec-001
  transform pipeline). `labels_dir` stays in the signature as an optional re-derivation override.
  A result with `ground_truth=None` raises `EvalError`.

---

## Governing invariants

| Invariant | How it is enforced |
|---|---|
| NFR-5: metrics defined **only** in `eval/metrics.py` | legacy no-eps/non-binarizing Dice+IoU live there too, in a separate `LegacyMetricsComputer` class |
| Binarization happens **only** in `recon/` | legacy path calls `AbsoluteThreshold(0.1)`; never an inline `residual > 0.1`. `tests/test_model_boundary.py::test_no_module_outside_recon_binarizes_a_residual` greps `src/mri_ad/eval/**` for `\b\w*residual\w*\s*[<>]=?\s*[\d.]` and will fail otherwise |
| Normal `make eval` never builds or calls a model | `mri_ad.models` / `mri_ad.recon.engine` imported **function-locally inside the legacy branch only**; asserted by AST test + patched `nn.Module.__call__` |
| `make report` has no ML imports | "report-safe subgraph" rule below + subprocess `sys.modules` test |
| No magic numbers in code | `configs/eval/{default,legacy_compat}.yaml` |

**Report-safe subgraph** (document in `eval/report.py`'s docstring): `mri_ad.exceptions`,
`mri_ad.utils.run_logger`, `mri_ad.eval.__init__`, `mri_ad.eval.report` must never transitively
import `torch`, `monai`, `nibabel`, `torchmetrics`, `sklearn`, `scipy`, `skimage`.
`src/mri_ad/eval/__init__.py` is docstring-only today — **keep it that way**, no re-exports, or
importing `mri_ad.eval` drags torch in via `metrics.py`. Likewise `mri_ad/utils/__init__.py` imports
torch, so `run_report.py` must use `from mri_ad.utils.run_logger import RunLogger`.

---

## Files

### Create

| File | Contents |
|---|---|
| `src/mri_ad/eval/loader.py` | results-directory discovery + streaming load |
| `src/mri_ad/eval/evaluator.py` | `VolumeEvaluator`, `AggregateEvaluator` |
| `src/mri_ad/eval/legacy.py` | quarantined bug-compat pipeline — the only eval module that touches a model |
| `src/mri_ad/eval/report.py` | `ReportGenerator` — ML-free |
| `scripts/run_recon.py` | D-A: test-split `ReconResult`s. GPU, manual only |
| `scripts/run_eval.py` | `make eval` |
| `scripts/run_report.py` | `make report` |
| `configs/eval/default.yaml`, `configs/eval/legacy_compat.yaml` | mode + paths + legacy knobs |
| `tests/test_eval.py` | acceptance 1, 2, 3, 5, 6, 7 |
| `tests/test_eval_boundary.py` | acceptance 4 and 8 (import-graph / subprocess) |

### Modify

| File | Change |
|---|---|
| `src/mri_ad/eval/metrics.py` | implement `iou`/`psnr`/`ssim`; add `aggregate()`; add `LegacyMetricsComputer` |
| `configs/config.yaml` | add `- eval: default` to `defaults` (before `_self_`) |
| `Makefile` | add `recon` target; translate `LEGACY_BUG_COMPAT` into a Hydra group selection |
| `progress_report.md` | append entry (CLAUDE.md rule 2) |

---

## Interfaces

### `eval/metrics.py`

```python
class MetricsComputer:
    @staticmethod
    def dice(pred: Tensor, gt: Tensor, eps: float = EPS) -> float: ...   # already implemented
    @staticmethod
    def iou(pred: Tensor, gt: Tensor, eps: float = EPS) -> float: ...
    @staticmethod
    def psnr(recon: Tensor, orig: Tensor) -> float: ...
    @staticmethod
    def ssim(recon: Tensor, orig: Tensor) -> float: ...

def aggregate(volumes: Sequence[VolumeMetrics]) -> AggregateMetrics: ...
```

- `iou` mirrors the existing `dice` exactly: **no internal binarization** (callers pass `seg > 0`),
  `union = pred.sum() + gt.sum() - inter`, `union == 0 -> 1.0`, else `inter / (union + eps)`.
- `psnr`: whole-volume `mse = ((recon - orig) ** 2).mean()`, clamped at `EPS`, `10*log10(1/mse)`.
  Data range fixed at 1.0 (images are min-max normalized in `data/transforms.py::_minmax`).
  The clamp caps PSNR at ~60 dB rather than returning `inf`, which would poison `fmean`/`pstdev`
  and emit invalid JSON. Reconstructions are **not** clamped to `[0,1]` — UNet/AttUNet are
  unbounded and clamping would silently flatter them. Document the cap in the docstring.
- `ssim`: `@lru_cache(maxsize=1)` factory with a **function-local** `from monai.metrics import
  SSIMMetric`, `SSIMMetric(spatial_dims=3, data_range=1.0)`. Given the `(1,D,128,128)` contract,
  call with `recon.unsqueeze(0)` -> `(1,1,D,H,W)`. Every spatial dim must be `>= win_size` (11);
  raise `EvalError` naming the observed shape if not, rather than letting MONAI throw an opaque
  conv error. Unit tests must therefore use `D >= 11`.
- `aggregate` uses `statistics.fmean` / `statistics.pstdev`, matching `recon/sweep.py`.

**Legacy metrics live here too** — NFR-5 says *defined only in `metrics.py`*, not *only one class*:

```python
class LegacyMetricsComputer:
    """Bug-compatible reproductions of legacy/metric-uad.ipynb. NEVER publish these numbers."""
    @staticmethod
    def dice_bugcompat(pred: Tensor, gt_raw: Tensor) -> float: ...
    @staticmethod
    def iou_bugcompat(pred: Tensor, gt_raw: Tensor) -> float: ...
```

`dice_bugcompat` binarizes only `pred` (`> 0.5`, a no-op on an already-binary mask, kept to mirror
the source), uses `gt_raw` raw/fractional, and has **no eps** in the denominator (guard a zero
denominator to `nan`, matching numpy's warn-and-nan). `iou_bugcompat` binarizes neither input and
adds `1e-6`. Name locals `pred`/`mask`, never `residual`, so the boundary-test regex cannot fire.

### `eval/loader.py`

```python
def resolve_results_dir(root: Path, run_id: str | None = None) -> Path: ...
def result_paths(results_dir: Path) -> list[Path]: ...
def iter_results(results_dir: Path) -> Iterator[ReconResult]: ...
```

`resolve_results_dir` returns `root/<run_id>`, or the most-recently-modified subdirectory holding
at least one `*.pt`; `ArtifactError` if the root is absent or no candidate qualifies.
`iter_results` **streams** — reuse the existing `recon.io.load_result`; 250 test volumes × 4 tensors
of `160×128×128` float32 is ~15 GB if eagerly materialized.

### `eval/evaluator.py`

```python
class VolumeEvaluator:
    def __init__(self, metrics: type[MetricsComputer] = MetricsComputer) -> None: ...
    def evaluate(self, result: ReconResult, label: Tensor | None = None) -> VolumeMetrics: ...

class AggregateEvaluator:
    def __init__(self, volume_evaluator: VolumeEvaluator | None = None) -> None: ...
    def evaluate_volumes(self, results_dir: Path, labels_dir: Path | None = None) -> list[VolumeMetrics]: ...
    def evaluate_split(self, results_dir: Path, labels_dir: Path | None = None) -> AggregateMetrics: ...
```

- `evaluate`: `label=None` -> `result.ground_truth`; `EvalError` if both are `None` (D-B).
  Asserts shape equality with `result.anomaly_mask`. Does **not** binarize (that already happened in
  the transform pipeline) but **does** assert `unique(label) ⊆ {0., 1.}` and raises `EvalError`
  naming trap #1 otherwise — this is acceptance test 1's runtime binarization guard.
- `evaluate_split` = `aggregate(self.evaluate_volumes(...))`, preserving the spec's literal return
  type while `run_eval.py` gets per-volume rows from `evaluate_volumes`.
- **Acceptance 3** — first statement of both methods, torch-free:
  ```python
  if not isinstance(results_dir, str | Path):
      raise TypeError(
          f"evaluate_split expects a results directory Path, got {type(results_dir).__name__}. "
          "The evaluation harness scores saved ReconResults and never re-runs forward()."
      )
  ```

### `eval/legacy.py` (quarantined)

```python
LEGACY_PUBLISHED_DICE: float = 0.6255
LEGACY_PUBLISHED_IOU: float = 0.4551

@dataclass(frozen=True)
class LegacyCompatConfig:
    subject_count: int = 10
    chunk_index: int = 4
    chunk_depth: int = 16
    threshold: float = 0.1
    crop_size: tuple[int, int, int] = (160, 130, 170)
    side: int = 128
    sort_subjects: bool = True   # see R2

@dataclass(frozen=True)
class LegacyCompatResult:
    dice: float
    iou: float
    n_subjects_visited: int
    n_scores_accumulated: int    # == 1: the re-initialized-list bug, made visible
    last_volume_id: str
    published_dice: float = LEGACY_PUBLISHED_DICE
    published_iou: float = LEGACY_PUBLISHED_IOU

def legacy_preprocess(t2: np.ndarray, seg: np.ndarray, cfg: LegacyCompatConfig
                     ) -> tuple[Tensor, Tensor]: ...   # pure, model-free, unit-testable

class LegacyCompatEvaluator:
    def __init__(self, model: AnomalyDetectionModel, config: LegacyCompatConfig,
                 device: torch.device) -> None: ...
    def evaluate_subject(self, t2: np.ndarray, seg: np.ndarray) -> tuple[float, float]: ...
    def evaluate_directory(self, brats_dir: Path) -> LegacyCompatResult: ...
```

`legacy_preprocess` reproduces, in order — each line carrying a `# BUG-COMPAT #n:` comment, with the
module docstring mapping 1–10 to the spec's "Notes / deviations":

1. min-max the **raw** T2 before transpose, no eps (bug 10 — current pipeline normalizes last);
2. `np.transpose(v, (2,0,1))` -> `(155, 240, 240)`;
3. H/W-only centre crop **computed from `crop_size`, never hardcoded**: `start_h = (240-130)//2 = 55`,
   `start_w = (240-170)//2 = 35` -> `[:, 55:185, 35:205]`. `start_d` is computed and deliberately
   **unused**, so depth stays at 155 (bug 5);
4. `F.interpolate(size=(D,128,128), mode="trilinear", align_corners=False)` applied to the **seg as
   well**, raw `{0,1,2,4}` labels never binarized -> fractional "labels" (bugs 2 + 3);
5. `range(8)` × 16 slices -> first 128 of 155 depth; take index `chunk_index` **only** (bug 4).

`evaluate_subject`: `out = model(chunk)`; re-min-max `out` with no eps (bug 6);
`diff = (chunk - out).abs()`; `mask = AbsoluteThreshold(self.config.threshold)(diff)` (bug 7, routed
through `recon/` so the boundary test passes); then `LegacyMetricsComputer.dice_bugcompat` /
`iou_bugcompat`.

`evaluate_directory`: iterate subjects, break at `subject_count`, and **re-initialize the score lists
inside the loop** (bug 1) — so the returned mean is the last subject only and
`n_scores_accumulated == 1`.

`legacy.py` deliberately does **not** reuse `data/transforms.py`; reusing the correct pipeline would
defeat the point. The duplication is intentional and confined to this one module.

### `eval/report.py` (no ML imports)

```python
CONTEXT_ONLY_NOTE = ("PSNR/SSIM are explanatory context only (NFR-6/NFR-22) — never an "
                     "optimization target and never a headline number.")
PER_VOLUME_FIELDS = ("volume_id", "dice", "iou", "psnr_db_context_only", "ssim_context_only")

class ReportGenerator:
    def __init__(self, metrics_dir: Path, figures_dir: Path) -> None: ...
    def write_per_volume(self, rows: Sequence[Mapping[str, object]]) -> Path: ...   # per_volume.csv
    def write_aggregate(self, aggregate: Mapping[str, object], *, mode: str, model: str,
                        split: str, n_volumes: int, run_id: str | None = None) -> Path: ...
    def read_per_volume(self) -> list[dict[str, str]]: ...
    def read_aggregate(self) -> dict: ...
    def plot_dice_distribution(self, rows: Sequence[Mapping[str, object]]) -> Path: ...
    def write_summary_markdown(self) -> Path: ...
```

`ReportGenerator` accepts **plain mappings, not `VolumeMetrics`** — that is exactly what keeps
`report.py` free of `metrics.py` (which imports torch). `run_eval.py` does the `dataclasses.asdict`
plus the key rename into `psnr_db_context_only` / `ssim_context_only`. `matplotlib` is imported
**lazily inside `plot_dice_distribution`** (`matplotlib.use("Agg")` before `pyplot`, `# noqa: E402`).

`artifacts/metrics/aggregate.json` — context-only labelling is structural, not cosmetic:

```json
{"mode": "normal", "model": "unetr", "split": "test", "n_volumes": 250, "run_id": "...",
 "headline":     {"dice_mean": 0.0, "dice_std": 0.0, "iou_mean": 0.0, "iou_std": 0.0},
 "context_only": {"psnr_db_mean": 0.0, "psnr_db_std": 0.0, "ssim_mean": 0.0, "ssim_std": 0.0,
                  "note": "PSNR/SSIM are explanatory context only ..."},
 "baseline_delta": {"published_dice": 0.6255, "corrected_dice": 0.0, "delta": 0.0}}
```

`summary.md` renders two tables, the second titled `### Context only — never a target`, with the note
beneath. Kept deliberately thin — spec 011 owns full reporting; the durable part here is the
read/write API.

---

## Configs

`configs/eval/default.yaml`:

```yaml
legacy_compat: false
results_run_id: null          # null -> newest run dir under recon.results_dir
metrics_dir: ${paths.artifact_root}/metrics
figures_dir: ${paths.artifact_root}/figures
labels_dir: null              # D-B override; null -> GT embedded in each ReconResult
legacy:
  subject_count: 10
  chunk_index: 4
  threshold: 0.1
  crop_size: [160, 130, 170]
  sort_subjects: true
  published_dice: 0.6255
  published_iou: 0.4551
  tolerance: 0.02
```

`configs/eval/legacy_compat.yaml`:

```yaml
# @package _global_
# Reproduces legacy/metric-uad.ipynb's bugs ON PURPOSE (spec 004 acceptance test 6).
# NEVER publish a number produced by this config.
eval:
  legacy_compat: true
threshold: absolute   # value 0.1 — the prior work's hardcoded cutoff
```

`configs/config.yaml`: add `- eval: default` to the `defaults` list before `_self_`.

---

## Entry points

**`scripts/run_recon.py`** (D-A, `make recon`, GPU, manual only) — structurally a clone of
`run_sweep.py` but over `SplitContract.load(...).brats["test"]`: `seed_everything` as the literal
first line, `with RunLogger(cfg) as run:`, `build_default_registry()` -> model -> `load_checkpoint`
-> `.eval()`, `ReconstructionEngine`, loop `engine.run_dataset_volume`,
`save_result(result, Path(cfg.recon.results_dir), run.run_id)`, `run.record(...)`, print the run_id
(the user passes it back as `+eval.results_run_id=...`). This is the only spec-004 script that
touches BraTS or weights.

**`scripts/run_eval.py`** (`make eval`):

```python
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    with RunLogger(cfg) as run:
        if bool(cfg.eval.legacy_compat):
            summary = _evaluate_legacy(cfg, run_id=run.run_id)   # imports models INSIDE
        else:
            summary = _evaluate_normal(cfg, run_id=run.run_id)   # never touches models
        run.record(**summary)
```

Module-level imports are strictly hydra/omegaconf, `mri_ad.eval.{loader,evaluator,report,metrics}`,
`mri_ad.exceptions`, `mri_ad.utils`. `_evaluate_normal(cfg, run_id) -> dict` (also the function the
tests call directly, bypassing hydra) resolves the results dir, runs `evaluate_volumes`, aggregates,
and writes `per_volume.csv` + `aggregate.json` + `dice_distribution.png`.
`_evaluate_legacy` does its `mri_ad.models` / device imports **locally**, and writes
`artifacts/metrics/legacy_compat.json` — a **separate file**, so a bug-compat number can never
overwrite the honest baseline in `aggregate.json`.

**`scripts/run_report.py`** (`make report`) — no `seed_everything`: nothing here is stochastic and
importing `mri_ad.utils.seed` would drag torch in, breaking acceptance test 8. Document that
trade-off in a module comment. Reads the two artifacts via `ReportGenerator.read_*`, regenerates the
plot and `summary.md`, prints the headline row, raises `ArtifactError` with a "run `make eval` first"
message if `aggregate.json` is absent.

**`Makefile`**:

```make
LEGACY_BUG_COMPAT ?= 0
EVAL_MODE := $(if $(filter 1 true yes,$(LEGACY_BUG_COMPAT)),eval=legacy_compat,)

recon:  ## Spec 004: reconstruct the TEST split -> saved ReconResults (GPU). MANUAL ONLY.
	$(PY) scripts/run_recon.py $(HYDRA_OVERRIDES)

eval:  ## Score saved test-split ReconResults. No model, no GPU. LEGACY_BUG_COMPAT=1 for bug-compat.
	$(PY) scripts/run_eval.py $(EVAL_MODE) $(HYDRA_OVERRIDES)
```

This satisfies the spec's literal `make eval LEGACY_BUG_COMPAT=1` while **no Python file ever reads
`os.environ`** — the make variable is translated into a Hydra group selection at the shell boundary,
so the config stays the single source of truth and `make eval HYDRA_OVERRIDES="eval=legacy_compat"`
is an equivalent invocation.

---

## Sequencing

1. `metrics.py`: `iou`, `psnr`, `ssim`, `aggregate`, `LegacyMetricsComputer` + unit tests. Unblocks everything, zero new deps.
2. `loader.py` + `evaluator.py` + acceptance 1/2/3 tests.
3. `report.py` + `configs/eval/*` + `config.yaml` defaults + acceptance 5/7 tests.
4. `scripts/run_eval.py` (normal branch) + acceptance 4 tests.
5. `scripts/run_report.py` + acceptance 8 tests.
6. `scripts/run_recon.py` + `make recon`.
7. `eval/legacy.py` + `_evaluate_legacy` + `LEGACY_BUG_COMPAT` wiring + acceptance 6 tests.
8. `torch-reviewer` on the diff, then append the `progress_report.md` entry.

The actual `make recon` / `make eval` / `make eval LEGACY_BUG_COMPAT=1` runs are **user-invoked** —
GPU and real data (CLAUDE.md rule 4).

---

## Verification

Everything below runs on the laptop with **no real data, no checkpoints, and no real 3D forward
passes** — synthetic seeded tensors and a `_StubModel` (copy the one in `tests/test_recon.py`).
Follow house conventions: no `conftest.py`, `REPO = Path(__file__).resolve().parent.parent`,
`pytest.importorskip("monai")` with `# noqa: E402`, `# ── Acceptance #N: … ───` separators,
long sentence-style test names.

```bash
make lint
make test                      # or: pytest tests/test_eval.py tests/test_eval_boundary.py -q
```

**Acceptance 1** — four properties, one test each:
`test_ground_truth_is_binarized_before_dice_and_raw_labels_are_rejected` (plant `{0,1,2,4}` in
`ReconResult.ground_truth` -> `EvalError`); `test_labels_are_resized_nearest_neighbour_and_stay_binary`
(run `build_transforms(cfg, dataset="brats")` on a synthetic `240×240×32` seg, assert uniques
`⊆ {0.,1.}` — trilinear would yield fractions);
`test_every_depth_chunk_is_scored_not_just_chunk_four` (plant a tumour only in chunk 7 of 10, assert
Dice > 0); `test_all_subjects_are_accumulated_not_just_the_last` (three results scoring ≈1, 0, 1 ->
`dice_mean ≈ 2/3`, `n_volumes == 3`).

**Acceptance 2** — `test_dice_and_iou_match_closed_form_values_on_hand_built_tensors` (parametrized,
hand-computed `2i/(p+g)` and `i/(p+g-i)`); `test_dice_and_iou_empty_mask_conventions` (both empty ->
1.0, one empty -> 0.0); `test_iou_and_dice_are_monotonically_consistent` (`iou <= dice`).

**Acceptance 3** — `test_evaluate_split_raises_type_error_when_handed_a_model_instead_of_a_path`,
passing both a `_StubModel` and a bare `object()`, `match="never re-runs forward"`.

**Acceptance 4** — three independent layers in `tests/test_eval_boundary.py`:
1. Runtime: `nn.Module` subclasses override `forward`, so patching `Module.forward` is useless —
   patch the dispatcher instead (`monkeypatch.setattr(torch.nn.Module, "__call__", _boom)` and
   `"_call_impl"`), then call `run_eval._evaluate_normal(cfg, run_id="test")` and require it to
   complete. Verified safe: `monai.metrics.SSIMMetric` is a plain `Metric`, not an `nn.Module`, and
   returns an identical value under exactly this patch, so the SSIM leg still runs.
2. Construction: patch `mri_ad.models.build_default_registry` to raise — proves not even a model
   *object* is built.
3. Static: AST walk of `scripts/run_eval.py`'s module-level `Import`/`ImportFrom` nodes (none may
   resolve under `mri_ad.models` or `mri_ad.recon.engine`), and of `src/mri_ad/eval/*.py` with
   `legacy.py` as the single allowlisted exception. Same technique as `tests/test_model_boundary.py`.

**Acceptance 5** — `test_per_volume_csv_and_aggregate_json_are_written_with_mean_and_std` (headers
exactly `PER_VOLUME_FIELDS`, one row per volume, `dice_std` matches `statistics.pstdev`);
`test_psnr_and_ssim_are_labelled_context_only_everywhere_they_appear` (CSV headers carry
`context_only`, JSON nests them under `context_only` and **never** under `headline`, `summary.md`
contains `CONTEXT_ONLY_NOTE`); `test_aggregate_json_round_trips_through_the_report_reader`.

**Acceptance 6** — the deterministic mechanics need no model:
`test_legacy_preprocess_reproduces_the_prior_crop_resize_and_chunk_selection` (synthetic
`(240,240,155)` T2 + `{0,1,2,4}` seg -> assert crop bounds `55:185, 35:205`, depth **untouched at
155**, chunk shape `(1,1,16,128,128)` covering global depth `64:80`, and that the seg chunk contains
**fractional** values); `test_legacy_dice_has_no_eps_and_does_not_binarize_the_ground_truth` (assert
it *differs* from `MetricsComputer.dice` on the same inputs);
`test_legacy_iou_binarizes_neither_input_and_adds_1e-6`;
`test_legacy_evaluator_reports_only_the_last_subject` (three stub subjects ->
`n_scores_accumulated == 1`, `dice ==` the third subject's score);
`test_legacy_compat_config_group_flips_the_mode_and_pins_the_absolute_threshold` (hydra `compose`
with `eval=legacy_compat`);
`test_makefile_translates_legacy_bug_compat_into_the_hydra_override`;
`test_no_python_source_reads_the_legacy_bug_compat_environment_variable` (grep `src/` + `scripts/`).
The ±0.02-of-0.6255 assertion itself needs real weights + real BraTS ->
`test_legacy_compat_lands_within_tolerance_of_the_published_dice`, `@pytest.mark.skipif`-gated,
reading `artifacts/metrics/legacy_compat.json` when present. **This is the one acceptance criterion
not verifiable on a laptop** — it is confirmed on the cluster.

**Acceptance 7** — `test_aggregate_json_records_the_delta_from_the_published_number`
(`baseline_delta` block present, rendered in `summary.md`). The narrative attribution of the delta
to specific bug fixes is human-written in `progress_report.md`, per rule 2.

**Acceptance 8** — subprocess tests asserting that importing `mri_ad.eval.report`, and separately
exec'ing `scripts/run_report.py` as a module (`main()` not invoked since `__name__ != "__main__"`),
leave **none** of `torch, monai, nibabel, torchmetrics, sklearn, scipy, skimage` in `sys.modules`.
Plus `test_report_generation_finishes_well_under_the_two_minute_budget`: synthesize a 250-row
`per_volume.csv` + `aggregate.json` in `tmp_path`, time `ReportGenerator` end-to-end, assert
`< 120 s` (actual ≈ 0.3 s).

**Also**: loader units (`resolve_results_dir` newest-run selection, `ArtifactError` on empty/missing
root, `EvalError` when a result has no ground truth, `labels_dir` override precedence) and metric
units (PSNR finite and capped at ~60 dB for identical volumes, PSNR monotonically decreasing with
noise, SSIM 1.0 for identical volumes, `EvalError` when a spatial dim is below `win_size`).

### End-to-end (user-invoked, cluster)

```bash
make check-data
make recon HYDRA_OVERRIDES="+experiment=cluster model=unetr"     # GPU. prints <run_id>
make eval  HYDRA_OVERRIDES="+experiment=cluster model=unetr +eval.results_run_id=<run_id>"
make eval  LEGACY_BUG_COMPAT=1 HYDRA_OVERRIDES="+experiment=cluster model=unetr"
make report
```

Expected artifacts: `artifacts/results/recon/<run_id>/*.pt`, `artifacts/metrics/per_volume.csv`,
`artifacts/metrics/aggregate.json`, `artifacts/metrics/legacy_compat.json`,
`artifacts/metrics/summary.md`, `artifacts/figures/dice_distribution.png`,
`artifacts/runs/<stamp>/run_meta.json`.

---

## Risks & open questions

| # | Risk | Handling |
|---|---|---|
| R1 | Legacy `os.listdir` ordering (bug 9) decides *which* subject is "last", so the ±0.02 match is filesystem-dependent. | `sort_subjects` defaults to `true` for determinism; `LegacyCompatResult` reports `last_volume_id` and `n_subjects_visited`. If the sorted run misses ±0.02 on the cluster, flip `sort_subjects=false` and record both numbers in `progress_report.md`. Genuinely open until it runs. |
| R2 | The ±0.02 assertion can't run on this laptop. | Skip-marked test reading `legacy_compat.json`; every deterministic mechanic (preprocessing, metric formulas, single-subject accumulation) is unit-tested model-free. |
| R3 | Test-split `ReconResult`s are ~63 MB each; 250 volumes ≈ 15 GB. | `iter_results` streams one at a time; `data.eval_subset` caps local runs. **Not** storing float16 residuals — it would perturb Dice in the 4th decimal. |
| R4 | The 60 dB PSNR cap is a design choice not in the spec. | Documented in the docstring and unit-tested; the alternative (`inf`) breaks JSON and aggregation. |
| R5 | `legacy.py` duplicates preprocessing rather than reusing `data/transforms.py`. | Intentional — reusing the correct pipeline defeats bug-compat. Confined to one quarantined module with per-line `# BUG-COMPAT #n` comments. |
| R6 | `run_recon.py` + `make recon` is scope beyond the spec text. | Approved as D-A; disclose explicitly in the `progress_report.md` entry so the addition is traceable. |
| R7 | Spec 011 owns full reporting; 004's `summary.md` may be discarded. | Kept deliberately thin; `ReportGenerator`'s read/write API is the durable surface. |
