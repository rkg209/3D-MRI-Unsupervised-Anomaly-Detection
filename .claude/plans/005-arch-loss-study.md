# Plan · Spec 005 — Fidelity-vs-detection study (architecture × loss + diffusion)

**Spec:** [`specs/005-arch-loss-study.md`](../../specs/005-arch-loss-study.md) (status: approved) · **Depends on:** 004 (done), 002 (diffusion row), 013 (to fill its cell) · **Fresh compute:** no

---

## Context

The project's central claim is that **higher reconstruction fidelity does not buy better anomaly
detection — it hurts**. That claim is only credible across a matrix, not from one data point. Spec
005 is the *mechanism* study: it orchestrates reconstruction models × losses into one generated
table plus one written analysis, with PSNR/SSIM shown *alongside* Dice/IoU precisely so the
anti-correlation is visible and quantifiable. (Spec 007 owns the headline three-paradigm table and
ROC/PR curves; 005 must not produce those.)

**The gap this plan closes.** Spec 004's harness gives us canonical metrics and a durable
`aggregate.json`, but its entire artifact layout is keyed by **model alone** —
`artifacts/metrics/<model>/`, `manifest.json = {model, split, n_volumes}`. There is no loss
dimension anywhere in the artifact tree, so `unetr` trained with MSE and `unetr` trained with
MSE+SSIM would overwrite each other's metrics. A (model × loss) matrix cannot be built on that.
This spec introduces the **cell** as a first-class artifact identity and the first
`artifacts/tables/` writer in the repo.

**The honest starting state.** `artifacts/` is empty; `checkpoints/` does not exist locally. Only
three checkpoints are known to exist at all (`unetr_mse_ssim_aug.pth`, `unet_mse.pth`,
`attunet_ssim_mse.pth`). So on first run **every cell renders `n/a`**, and even after a full
cluster pass most of the UNETR loss sweep stays `n/a` — there are no weights for MSE-only,
SSIM-only, multi-scale, or perceptual UNETR. That is the expected, correct output. An incomplete
matrix honest about being incomplete beats a complete one with a fabricated cell.

### Decisions taken with the user

- **D-A — Cell identity is `cell_id = f"{model}__{loss}"`.** Spec 004's `run_eval.py` and
  `report.py` are extended to namespace by `cell_id` and to record `model`, `loss`, and `cell_id`
  in `aggregate.json`. `artifacts/` is empty, so there is no migration cost. *This is a
  modification to already-shipped Spec 004 code and must be called out in `progress_report.md`.*
- **D-B — The narrative is a separate file.** The generator writes
  `arch_loss_matrix.{csv,md,json}`; the `results-analyst` agent writes
  `artifacts/tables/arch_loss_analysis.md`, **quoting** the stats the generator computed rather
  than computing its own. This keeps regeneration non-destructive of prose.
- **D-C — New `make matrix` entry point** (`scripts/run_arch_loss_matrix.py`), self-contained.
  Spec 011 may later fold it into `make report`.
- **D-D — `n/a` reasons are distinct and are data, not prose.** Each unavailable cell carries a
  specific reason string sourced from `configs/matrix/arch_loss.yaml`, never invented in code.

---

## Governing invariants

| Invariant | How it is enforced |
|---|---|
| No number in the table is hardcoded | Acceptance test 2: AST walk over the generator for numeric literals outside a small allowlist; plus a test that mutating an `aggregate.json` changes the rendered cell |
| Every number traces to an artifact | Each cell row carries `cell_id`, `metrics_path`, `eval_run_id`, `recon_run_id` — read from `aggregate.json`, emitted into `arch_loss_matrix.json` |
| PSNR/SSIM are context, never performance | Column headers read `PSNR (dB) — context only` / `SSIM — context only`; `CONTEXT_ONLY_NOTE` from `eval/report.py` is reused verbatim as the table footer (NFR-6/NFR-22) |
| No cell is blank or fabricated | A cell is either a float or `n/a (<reason>)` with a non-empty reason; a test asserts this over every cell of a synthetic full matrix |
| The generator is report-safe | `scripts/run_arch_loss_matrix.py` and `mri_ad/eval/matrix.py` are added to `tests/test_eval_boundary.py`'s `FORBIDDEN_ML_MODULES` guard — no torch/monai/scipy/sklearn, even transitively. **Spearman ρ is therefore hand-rolled in pure Python.** |
| Metrics are defined once | The generator computes no metrics. It only reads means already written by `MetricsComputer` via `ReportGenerator.write_aggregate`. |
| 005 does not encroach on 007 | No ROC/PR curves, no slice-level AUC, no classical column, no paradigm-winner claim |

**Rule:** the generator never touches `checkpoints/`, `data/`, or a `.pt` ReconResult. It reads
`artifacts/metrics/<cell_id>/aggregate.json` and nothing heavier.

---

## Files

### Create

| File | Contents |
|---|---|
| `configs/matrix/arch_loss.yaml` | The declared matrix: rows, columns, per-cell `na_reason` defaults, `context_only_columns`, output paths |
| `src/mri_ad/eval/matrix.py` | `MatrixCell`, `ArchLossMatrix`, `spearman`, `load_cells`, `render_csv`, `render_markdown`, `write_stats`. **No ML imports.** |
| `scripts/run_arch_loss_matrix.py` | `make matrix` entry point — Hydra, `seed_everything`, `RunLogger`, no GPU |
| `tests/test_matrix.py` | Acceptance tests 1–5 against synthetic `aggregate.json` fixtures |
| `artifacts/tables/arch_loss_analysis.md` | Authored by the `results-analyst` agent (last step, after the table exists) |

### Modify

| File | Change |
|---|---|
| `src/mri_ad/eval/report.py` | `write_aggregate` gains a required keyword `loss: str` and emits `loss` + `cell_id` in the payload; `write_summary_markdown` shows the loss in its title |
| `scripts/run_eval.py` | Namespace `metrics_dir`/`figures_dir` by `f"{cfg.model.name}__{cfg.loss.name}"`; pass `loss=` through to `write_aggregate` |
| `scripts/run_recon.py` | Write `loss` into `manifest.json` (`{model, loss, split, n_volumes}`) so a results dir is traceable to a cell |
| `scripts/run_report.py` | Read from the `cell_id` directory instead of `<model>` (mechanical follow-on of D-A) |
| `tests/test_eval.py`, `tests/test_eval_boundary.py` | Update the namespacing assertions; extend the boundary guard to the two new modules |
| `Makefile` | Add the `matrix` target |
| `configs/config.yaml` | Add `matrix: arch_loss` to the `defaults` list |
| `progress_report.md` | Append entry `## 009` |

---

## Interfaces

### `src/mri_ad/eval/matrix.py` (no ML imports)

```python
NA_PREFIX = "n/a"

@dataclass(frozen=True)
class MatrixCell:
    model: str            # "unetr"
    loss: str             # "mse_ssim"
    cell_id: str          # "unetr__mse_ssim"
    role: str             # "headline" | "reference" | "paradigm"
    dice: float | None
    iou: float | None
    psnr_db: float | None
    ssim: float | None
    na_reason: str | None       # non-None iff dice is None
    metrics_path: str | None    # relative path to the aggregate.json this row came from
    eval_run_id: str | None
    recon_run_id: str | None
    n_volumes: int | None

    @property
    def available(self) -> bool: ...
    def render(self, field: str, fmt: str) -> str:
        """Formatted number, or ``n/a (<reason>)``. Never blank, never a placeholder number."""

@dataclass(frozen=True)
class MatrixStats:
    n_cells: int
    n_available: int
    spearman_psnr_dice: float | None   # None if fewer than 3 available cells
    n_pairs: int
    best_cell_id: str | None           # highest Dice among available cells
    best_dice: float | None
    diffusion_available: bool

def load_cells(matrix_cfg: Mapping, metrics_root: Path) -> list[MatrixCell]:
    """Resolve every declared cell against ``metrics_root/<cell_id>/aggregate.json``.

    Present and readable -> a numeric cell. Absent -> an ``n/a`` cell carrying the declared
    ``na_reason``, defaulting to ``"not evaluated"`` when the config gives none. A malformed or
    unreadable aggregate.json raises ArtifactError — it is never silently downgraded to ``n/a``,
    because that would hide a broken pipeline behind an honest-looking absence.
    """

def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation, pure Python (no scipy — report-safe subgraph).

    Average ranks for ties, then Pearson on the ranks. Returns None for n < 3 or zero variance.
    """

def compute_stats(cells: Sequence[MatrixCell]) -> MatrixStats: ...
def render_csv(cells, path: Path) -> Path: ...
def render_markdown(cells, stats: MatrixStats, path: Path) -> Path: ...
def write_stats(cells, stats: MatrixStats, path: Path) -> Path:   # arch_loss_matrix.json
```

- `render_markdown` emits **one row per cell** (long form: `model | loss | role | Dice | IoU |
  PSNR (dB) — context only | SSIM — context only | source`), not a wide grid. Long form survives
  `n/a` cells and ragged coverage far better, and keeps the provenance column readable.
- The `source` column is the `cell_id` (a directory under `artifacts/metrics/`), so a reader can
  walk from any printed number to the JSON that produced it.
- Footer of the `.md`: the `CONTEXT_ONLY_NOTE` imported verbatim from `mri_ad.eval.report`, plus
  the Spearman line, plus a one-line count of `n/a` cells.
- `spearman` is computed over **available cells only**, pairing each cell's `psnr_db` and `dice`.
  If it cannot be computed the footer says `Spearman rho: n/a (fewer than 3 evaluated cells)` —
  which is the honest state today.

### `configs/matrix/arch_loss.yaml`

```yaml
# Spec 005 matrix declaration. Cells are (model x loss); a cell is real only if
# artifacts/metrics/<model>__<loss>/aggregate.json exists. na_reason explains WHY a cell is
# missing and is data, not prose invented in code.
name: arch_loss
metrics_root: ${paths.artifact_root}/metrics
tables_dir: ${paths.artifact_root}/tables
basename: arch_loss_matrix

cells:
  # --- Headline: UNETR swept across losses (the core fidelity-vs-detection evidence) ---
  - {model: unetr, loss: mse,            role: headline,  na_reason: "no checkpoint"}
  - {model: unetr, loss: ssim,           role: headline,  na_reason: "no checkpoint"}
  - {model: unetr, loss: mse_ssim,       role: headline,  na_reason: "not evaluated"}
  - {model: unetr, loss: multiscale_mse, role: headline,  na_reason: "no checkpoint"}
  - {model: unetr, loss: perceptual,     role: headline,  na_reason: "no training code, no checkpoint"}

  # --- Prior-work reference rows: the anti-correlation is not a UNETR artefact ---
  - {model: unet,           loss: mse,      role: reference, na_reason: "not evaluated"}
  - {model: attention_unet, loss: mse_ssim, role: reference, na_reason: "not evaluated"}

  # --- Paradigm row: does a generative prior escape the anti-correlation? ---
  - {model: diffusion, loss: ddpm, role: paradigm, na_reason: "untrained"}
```

Notes on the declaration:
- `unetr__mse_ssim` defaults to `"not evaluated"` because the checkpoint `unetr_mse_ssim_aug.pth`
  is expected to exist — it just needs `make recon` + `make eval`. The three cells with no known
  weights say `"no checkpoint"`. Perceptual carries the spec's exact wording (spec Notes,
  option 1). Diffusion says `"untrained"` per spec §Contract.
- `attention_unet` is declared under `mse_ssim` because `configs/model/attention_unet.yaml`'s
  checkpoint is `attunet_ssim_mse.pth` and its model card names the loss "SSIM + MSE".
- `diffusion`'s `loss: ddpm` is a label only — it is *not* in the loss sweep (spec §Contract).
  There is no `configs/loss/ddpm.yaml` and none is added; the generator never instantiates a loss.

### `scripts/run_arch_loss_matrix.py` (`make matrix`)

```python
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Render the Spec 005 architecture x loss matrix from saved aggregate.json files. No GPU."""
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    with RunLogger(cfg) as run:
        cells = load_cells(cfg.matrix, Path(str(cfg.matrix.metrics_root)))
        stats = compute_stats(cells)
        tables_dir = Path(str(cfg.matrix.tables_dir))
        render_csv(cells, tables_dir / f"{cfg.matrix.basename}.csv")
        render_markdown(cells, stats, tables_dir / f"{cfg.matrix.basename}.md")
        write_stats(cells, stats, tables_dir / f"{cfg.matrix.basename}.json")
        run.record(n_cells=stats.n_cells, n_available=stats.n_available,
                   spearman_psnr_dice=stats.spearman_psnr_dice)
    print(f"{stats.n_available}/{stats.n_cells} cells evaluated -> {tables_dir}")
```

Module-level imports restricted to hydra/omegaconf, `mri_ad.eval.matrix`, `mri_ad.exceptions`,
`mri_ad.utils` — the same restriction `run_eval.py` already lives under.

```make
matrix: ## Spec 005: render artifacts/tables/arch_loss_matrix.{csv,md,json}. No GPU.
	$(PY) scripts/run_arch_loss_matrix.py $(HYDRA_OVERRIDES)
```

### The `cell_id` change to Spec 004 code

`scripts/run_eval.py`, replacing the model-only namespacing at lines 87–91:

```python
cell_id = f"{cfg.model.name}__{cfg.loss.name}"
metrics_dir = Path(str(cfg.eval.metrics_dir)) / cell_id
figures_dir = Path(str(cfg.eval.figures_dir)) / cell_id
...
generator.write_aggregate(..., model=str(cfg.model.name), loss=str(cfg.loss.name), ...)
```

`report.py::write_aggregate` gains `loss: str` as a **required keyword** (not defaulted — a
defaulted loss would let a cell be written with the wrong label and never be noticed) and adds
`"loss"` and `"cell_id"` to the payload. The existing `_check_manifest` guard in `run_eval.py`
gains a matching loss check, so a results dir produced under one loss cannot be scored as another.

*Deviation from Spec 004's shipped interface:* this changes the on-disk metrics path and the
`write_aggregate` signature. Justified by D-A, costless today (`artifacts/` is empty), and
recorded in `progress_report.md` entry 009.

---

## Sequencing

1. `mri_ad/eval/matrix.py` — dataclasses, `spearman`, `load_cells`, renderers. Pure functions
   first; this is the whole substance of the spec.
2. `configs/matrix/arch_loss.yaml` + the `matrix: arch_loss` default in `configs/config.yaml`.
3. The `cell_id` migration: `report.py`, `run_eval.py`, `run_recon.py`, `run_report.py`, and the
   affected assertions in `tests/test_eval.py`.
4. `scripts/run_arch_loss_matrix.py` + the `Makefile` target.
5. `tests/test_matrix.py` (acceptance 1–5) and the boundary-guard extension in
   `tests/test_eval_boundary.py`.
6. `make lint`; run the laptop-safe test subset (below).
7. Run `make matrix` locally against the empty `artifacts/` — the correct output is an all-`n/a`
   table with `Spearman rho: n/a (fewer than 3 evaluated cells)`. Commit that table; it is a true
   artifact of the current state.
8. Invoke the `results-analyst` agent to write `artifacts/tables/arch_loss_analysis.md` from
   `arch_loss_matrix.json`. Today it will honestly report that nothing is measured yet; re-invoke
   after any cluster pass fills cells.
9. `torch-reviewer` on the diff (it touches `run_eval.py`/`run_recon.py`), then append
   `progress_report.md` entry `## 009`.

---

## Verification

Laptop-safe, no data, no checkpoints, no real forward pass:

```bash
make lint
pytest tests/test_matrix.py tests/test_eval.py tests/test_eval_boundary.py -q
make matrix          # renders an all-n/a table against the empty artifacts/
```

Per the standing constraint (memory: no heavy local runs), do **not** run the full `make test` —
`tests/test_slice.py` and `tests/test_models.py` hang on this machine. Use
`pytest tests/ --ignore=tests/test_slice.py --ignore=tests/test_models.py -k "not unetr"`.

Mapping acceptance criteria to tests in `tests/test_matrix.py` (house style: no `conftest.py`,
module-local `_aggregate(...)` fixture writing synthetic `aggregate.json` files under `tmp_path`,
long sentence-style test names):

- **Acceptance 1** — `test_every_declared_cell_renders_a_number_or_an_explicit_na_reason`,
  `test_no_cell_is_ever_blank_or_dropped_from_the_table`.
- **Acceptance 2** — `test_table_values_change_when_the_source_aggregate_json_changes` (the real
  anti-hardcoding proof), `test_matrix_generator_contains_no_hardcoded_metric_literals` (AST walk
  reusing `tests/test_eval_boundary.py`'s `_imported_modules` technique),
  `test_every_available_cell_names_its_metrics_path_and_run_ids`.
- **Acceptance 3** — `test_psnr_and_ssim_columns_are_labelled_context_only_in_csv_and_markdown`.
- **Acceptance 4** — `test_spearman_matches_a_known_hand_computed_value`,
  `test_spearman_handles_ties_with_average_ranks`,
  `test_spearman_is_na_when_fewer_than_three_cells_are_available`,
  `test_stats_json_records_whether_the_diffusion_row_is_available`.
- **Acceptance 5** — `test_best_cell_is_selected_only_from_available_cells`,
  `test_best_cell_is_none_when_no_cell_is_available`.
- **Boundary** — extend `tests/test_eval_boundary.py::FORBIDDEN_ML_MODULES` coverage to
  `mri_ad.eval.matrix` and `scripts/run_arch_loss_matrix.py`.

**End-to-end (user-invoked, cluster — GPU, manual only, never autonomous):** for each cell with a
real checkpoint,

```bash
make recon HYDRA_OVERRIDES="model=unetr loss=mse_ssim"
make eval  HYDRA_OVERRIDES="model=unetr loss=mse_ssim +eval.results_run_id=<run_id>"
make matrix
```

Expected artifacts: `artifacts/metrics/unetr__mse_ssim/{per_volume.csv,aggregate.json,summary.md}`,
`artifacts/figures/unetr__mse_ssim/dice_distribution.png`, and
`artifacts/tables/arch_loss_matrix.{csv,md,json}` with that cell populated and the rest `n/a`.

**Done when:** `make matrix` renders all eight declared cells with no blanks, every available cell
carries a traceable `metrics_path` + run ids, PSNR/SSIM are labelled context-only in both output
formats, `arch_loss_matrix.json` carries the Spearman ρ (or an explicit `null` with the reason),
and `arch_loss_analysis.md` states the relationship quantitatively without claiming an unmeasured
cell.

---

## Risks & open questions

| # | Risk | Handling |
|---|---|---|
| R1 | `mri_ad.losses.SSIMMSELoss` / `MultiScaleMSELoss` **do not exist** — `losses.py` is a one-line stub, so `configs/loss/{mse_ssim,multiscale_mse}.yaml` would crash anything that instantiates them | Spec 005 never instantiates a loss — it reads `cfg.loss.name` as a **string label** only. Flag the stub in `progress_report.md` as an open item for the training spec; do not fix it here (out of scope, no spec). |
| R2 | Changing the metrics namespace breaks Spec 004's shipped paths | Costless today: `artifacts/` is empty and nothing has been run. Update `tests/test_eval.py` in the same commit and record the deviation in the progress report. |
| R3 | Spearman ρ over 1–3 available cells is meaningless but will look authoritative | Return `None` for n < 3; render `n/a (fewer than 3 evaluated cells)` and put `n_pairs` in the JSON so the analyst must state the sample size. |
| R4 | The analyst infers a trend from cells that are mostly `n/a` | Acceptance 5 test enforces best-cell selection from available cells only; the analyst prompt hands it `arch_loss_matrix.json` and forbids computing its own numbers (agent rule 1). |
| R5 | The `n/a` reason in config drifts from reality (e.g. a checkpoint arrives but the cell still says "no checkpoint") | `load_cells` prefers the real `aggregate.json` whenever one exists — the reason is only consulted on absence. `make check-data` remains the source of truth for what weights exist. |
| R6 | A malformed `aggregate.json` silently degrades a cell to `n/a`, hiding a broken pipeline | `load_cells` raises `ArtifactError` on a present-but-unreadable file; only *absence* yields `n/a`. |
| R7 | The zero-padded-depth scoring issue (progress report 008, Problems hit #5) still biases every Dice in the matrix | Out of scope here, but the matrix `.md` footer names it as a known caveat affecting all rows equally — so cross-cell *comparison*, which is what this spec claims, remains valid. |
| R8 | 005 drifts into 007's territory | No ROC/PR, no slice-level AUC, no classical column, no paradigm-winner claim. The diffusion row states only whether it sits on or off the PSNR↔Dice trend. |
