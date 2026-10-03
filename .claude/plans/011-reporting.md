# Plan · Spec 011 — Reporting & public artifact

**Spec:** [`specs/011-reporting.md`](../../specs/011-reporting.md) (approved) · **Depends on:** 005, 007, 009, 010 · **Fresh compute:** no

---

## Context

The README is the deliverable most readers will ever see, and today it is the *pre-rebuild*
notebook README: title `IE643_TensorTitan`, the dropped "MRI/CT" framing (violates D4), a
`epoch: Default 100` hyperparameter block that contradicts the notebooks (50), and a 30-line GUI
setup section pointing at a `GUI/` directory and `tk_app.py` that **do not exist in this repo**.
Only the Spec 006 classical-baseline section at the bottom is rebuild-era content.

Meanwhile `Makefile:41` and `CLAUDE.md:100` both already advertise that `make report` "regenerates
the README scorecard" — and `scripts/run_report.py` never touches the README. There are no
`SCORECARD_START`/`SCORECARD_END` markers anywhere in the repo. That gap is this spec.

Outcome: a rewritten README whose every headline number is spliced in by one no-GPU command from
files under `artifacts/`, so it can never drift from what was actually measured; plus the framing
the project is contractually required to carry — comparative study not clinical tool (NFR-21), the
fidelity↔detection anti-correlation as the central finding (NFR-23), and the mobility-transfer
paragraph as a labelled conceptual analogy (D5, FR-44, NFR-22).

### Decisions taken with the user

1. **Timing:** instrument `RunLogger` + `run_recon.py` to emit per-volume inference seconds and GPU
   hours into `run_meta.json`. Code only — no GPU spend now; the scorecard renders
   `n/a (not measured)` until a real cluster run lands, then populates itself.
2. **Diagram:** a ```mermaid fence in the README. Plain text, diffs cleanly, renders natively on
   GitHub, no build step, no committed binary. (Nothing existing can serve — the only images in the
   repo are legacy notebook screenshots tied to the discredited 0.6255 numbers.)
3. **Baseline number:** `artifacts/metrics/` does not exist on this machine, so the corrected Spec
   004 Dice renders `n/a (not yet evaluated)` with its attributed reason. Tests assert the
   *machinery*, never a specific value. The delta-from-0.6255 explanation ships as static prose now.

---

## Files

### New

| File | Responsibility |
|---|---|
| `src/mri_ad/eval/scorecard.py` | Report-safe. Loads the declared source artifacts, joins them into `ScorecardEntry` rows, renders the markdown block. Holds `CENTRAL_FINDING_SENTENCE`. |
| `src/mri_ad/eval/readme_block.py` | Pure marker splice: find `START`/`END` in a README string, replace the block, leave every other byte untouched. |
| `configs/report/default.yaml` | Marker strings, README path, declared source artifact paths, row declarations, print precision. |
| `tests/test_report.py` | One test per acceptance criterion + splice/joining unit tests. |

### Modified

| File | Change |
|---|---|
| `README.md` | **Rewritten wholesale.** New structure below. |
| `configs/config.yaml` | Add `- report: default` to the `defaults:` list (same one-group pattern 007 used for `paradigm`, 010 for `viz`). |
| `scripts/run_report.py` | Add `_report_scorecard(cfg)` and `_report_synth(cfg)` as 4th/5th guarded sub-reports in the existing `for name, report_fn` loop. |
| `src/mri_ad/utils/run_logger.py` | Add `duration_seconds` to `run_meta.json`; add a `timer(name)` context manager accumulating into `metrics["timings"]`. Stays torch-free. |
| `scripts/run_recon.py` | Wrap the forward pass in `run.timer("inference")`; `run.record(device=..., n_volumes=..., seconds_per_volume=..., gpu_hours=...)`. |
| `Makefile` | Add `report-check` (exits nonzero if the README would change — CI guard). |
| `progress_report.md` | Append the Spec 011 entry (CLAUDE.md rule 2). |

**Report-safety is a hard constraint.** `scripts/run_report.py`'s docstring and
`tests/test_eval_boundary.py:27` (`FORBIDDEN_ML_MODULES`) forbid `torch`/`monai`/`nibabel`/
`sklearn`/`scipy`/`skimage` anywhere in the `make report` import graph. Both new `src/` modules are
stdlib-only (`json`, `csv`, `pathlib`, `dataclasses`, `re`). Import `RunLogger` from
`mri_ad.utils.run_logger`, never from `mri_ad.utils`.

---

## Interfaces

### `src/mri_ad/eval/scorecard.py`

```python
CENTRAL_FINDING_SENTENCE: str   # asserted verbatim in README by test AT-4

@dataclass(frozen=True)
class ScorecardEntry:
    key: str; label: str; value: float | None; fmt: str; unit: str
    source_path: str | None; run_id: str | None; na_reason: str | None
    @property
    def available(self) -> bool: ...
    def render(self) -> str:  # value formatted at `fmt`, or f"n/a ({na_reason})"

@dataclass(frozen=True)
class Scorecard:
    entries: tuple[ScorecardEntry, ...]
    split_hash: str | None
    sources: tuple[str, ...]        # every artifact path actually read
    n_available: int

def load_scorecard(report_cfg: Mapping[str, Any], repo_root: Path) -> Scorecard
def load_timing(runs_root: Path, *, model: str, loss: str) -> dict[str, Any]
def render_markdown(card: Scorecard) -> str
```

`load_scorecard` **never raises on a missing input** — a missing file becomes a per-entry
`na_reason`, matching the house style in `eval/matrix.py` and `eval/paradigm.py`. It raises
`ArtifactError` only for a malformed *config* (undeclared row key, unknown source).

Source join (all paths declared in YAML, none hardcoded):

| Scorecard row | Source artifact | Key path |
|---|---|---|
| Dice, IoU (DL) | `artifacts/tables/arch_loss_matrix.json` | `cells[cell_id == headline_cell].{dice,iou}` |
| ROC-AUC, PR-AUC (DL + classical) | `artifacts/tables/paradigm_comparison.json` | `columns[].{roc_auc,pr_auc}` per `key` |
| Classical headline | `artifacts/classical/metrics/classical_metrics.json` | `headline.{roc_auc_mean,pr_auc_mean}` |
| Synth before/after Dice | `artifacts/tables/synth_before_after.csv` | `dice_mean` per `cell` |
| Inference s/volume, GPU hours | newest matching `artifacts/runs/<stamp>/run_meta.json` | `metrics.{seconds_per_volume,gpu_hours,device}` |
| Corrected 004 baseline + delta | `artifacts/metrics/<cell>/aggregate.json` | `headline.dice_mean`, `baseline_delta.*` |

### `src/mri_ad/eval/readme_block.py`

```python
def splice(text: str, block: str, *, start: str, end: str) -> str
    # raises ArtifactError: marker absent, marker appearing more than once, end before start
def write_block(readme_path: Path, block: str, *, start: str, end: str) -> bool
    # returns True if the file changed; writing twice is byte-identical (idempotent)
def would_change(readme_path: Path, block: str, *, start: str, end: str) -> bool  # `report-check`
```

### `configs/report/default.yaml` (sketch — no magic numbers in `.py`)

```yaml
readme_path: ./README.md
marker_start: "<!-- SCORECARD_START -->"
marker_end: "<!-- SCORECARD_END -->"
headline_cell: { model: unetr, loss: mse_ssim }
sources:
  matrix_json:    ${matrix.tables_dir}/${matrix.basename}.json
  paradigm_json:  ${paradigm.tables_dir}/${paradigm.basename}.json
  classical_json: ${classical.report.metrics_dir}/classical_metrics.json
  synth_csv:      ${paths.artifact_root}/tables/synth_before_after.csv
  aggregate_dir:  ${eval.metrics_dir}
  runs_root:      ${paths.artifact_root}/runs
rows: [ ... ]           # declared list of {key, label, source, field, fmt, unit}
precision: { dice: ".4f", auc: ".4f", seconds: ".1f", gpu_hours: ".2f" }
```

### Timing instrumentation

`RunLogger.__exit__` gains `"duration_seconds": (end - start).total_seconds()`.
`RunLogger.timer(name)` is a `@contextmanager` writing `self.metrics["timings"][name]`.
`run_recon.py` records `device` (the resolved string from the existing device helper, not the
literal `auto`), `n_volumes`, `seconds_per_volume`, and `gpu_hours` — which is **`None` unless the
device starts with `cuda`**, so a laptop CPU run can never publish a fake GPU-hours figure.

`load_timing` selects the newest run dir (lexicographic max of the UTC stamps) whose
`config.model.name`/`config.loss.name` match the headline cell **and** whose `metrics` contain
`seconds_per_volume`, and returns its `run_id` for the provenance footer.

---

## README structure (rewritten wholesale)

1. **Title + framing banner** — comparative study of unsupervised anomaly detection in 3D brain
   MRI. Explicit "research comparison; **not** a clinical or diagnostic tool" disclaimer.
2. **The central finding** (NFR-23), above the fold: reconstruction fidelity and detection quality
   are anti-correlated; detection separability (Dice/IoU) is the target, PSNR/SSIM never are.
3. **Architecture** — mermaid fence: OpenBHB healthy → shared preprocess to `(1,16,128,128)` →
   train recon → BraTS test → residual → threshold (`recon/`) → metrics (`eval/`), with the
   Classical (006) and Diffusion (013) branches drawn alongside as the three compared paradigms.
4. **Scorecard** — `<!-- SCORECARD_START -->` … `<!-- SCORECARD_END -->`, generated.
5. **Reproducibility** — dataset registration links (OpenBHB, BraTS20), the Google Drive
   checkpoint folder → `checkpoints/`, the **LFS-stub caveat** (the seven `legacy/*.pth` are 133-byte
   pointers), `make check-data` first, then the exact command sequence, and the determinism
   contract (seed, git SHA, resolved config in `artifacts/runs/`).
6. **The corrected baseline & the 0.6255 delta** — the four bugs enumerated from
   `src/mri_ad/eval/legacy.py`'s docstring, with the corrected cell referenced from the generated
   block. Framed as evidence the audit was real (spec's note on AT-7).
7. **Classical-ML baseline** — preserve current README L63–77 verbatim in spirit.
8. **Mobility transfer** — one paragraph between `<!-- MOBILITY_START/END -->` markers (so the test
   can scope precisely), labelled **a conceptual analogy, not a tested result**. No number, no
   metric word, no driving-data code.
9. **Repo map + commands + data-use note.**

Deleted: the GUI section, the hyperparameters block, and every "CT" mention (D4).

---

## Risks — what would produce a *confidently wrong* number

| # | Failure | Guard | Test |
|---|---|---|---|
| R1 | Scorecard silently pairs Dice from one split with AUC from another. | `load_scorecard` collects `split_hash` from every contributing artifact; disagreement renders `n/a (split mismatch)` on the affected rows rather than averaging. | Build two tmp artifacts with different hashes → assert mismatch surfaced, not blended. |
| R2 | A stale number survives because splice silently no-op'd. | `splice` raises on absent/duplicate/inverted markers; `write_block` is the only writer; `report-check` fails CI when the README would change. | Missing/duplicate/inverted marker each → `ArtifactError`. |
| R3 | Someone hand-types a number into the block. | AT-2 test greps live README literals against artifact values. | `test_every_number_in_the_readme_scorecard_traces_to_an_artifact`. |
| R4 | CPU laptop timing published as GPU hours. | `gpu_hours` is `None` unless device starts with `cuda`; the device string is rendered next to the timing row. | Assert `gpu_hours` n/a and device shown for a `cpu` run_meta. |
| R5 | Rounding invents precision. | All formats come from `report.precision` YAML. | Formatted string round-trips to the source value at that precision. |
| R6 | A number appears with no traceable source. | `ScorecardEntry` with a `value` and `source_path is None` is rejected at construction. | Unit test asserts the raise. |

---

## Test plan — one per acceptance criterion

| Spec AT | Test in `tests/test_report.py` |
|---|---|
| 1 · rewrites between markers, zero manual editing | `test_scorecard_block_is_rewritten_between_markers` (stale junk inside the block is replaced; every byte outside is identical) + `test_running_report_twice_is_byte_identical` |
| 2 · every number traces to an artifact | `test_every_number_in_the_readme_scorecard_traces_to_an_artifact` — regex all numeric literals in the live block, assert each appears among values recursively extracted from the declared sources, formatted at the configured precision. Passes trivially in the all-`n/a` state (zero literals), which is the only state reproducible here. |
| 3 · no GPU, no ML importable, < 2 min | Extend `tests/test_eval_boundary.py`'s `FORBIDDEN_ML_MODULES` subprocess test to the new modules; the existing `test_report_generation_finishes_well_under_the_two_minute_budget` now covers the scorecard path. |
| 4 · states the central finding | `test_readme_states_the_central_finding` — asserts `CENTRAL_FINDING_SENTENCE` appears verbatim, so prose and constant cannot drift apart. |
| 5 · mobility paragraph is claim-free | `test_mobility_paragraph_has_no_performance_claim` — scoped to the mobility markers; asserts the disclaimer string present and no metric word (`Dice`, `AUC`, `accuracy`, `outperform`, `%`), no code fence. |
| 6 · comparative study, never clinical | `test_readme_uses_no_clinical_claim_words` — case-insensitive `{clinical, diagnostic, diagnose, FDA, patient-ready}`, permitted **only** inside the designated disclaimer lines. |
| 7 · corrected baseline + 0.6255 note | `test_readme_carries_the_corrected_baseline_and_explains_the_delta` — `0.6255` (resolved from `configs/eval/default.yaml`, not hardcoded in the test) present, delta-explanation heading present, corrected cell is either a value matching `aggregate.json` or the configured n/a text. |

Plus unit tests: splice error cases; `load_timing` newest-match selection and recorded `run_id`;
`load_scorecard` degrades to `na_reason` on every missing source; `RunLogger.timer` accumulation.

House conventions to follow (`tests/` has **no** `conftest.py` and one fixture repo-wide):
module-level `REPO = Path(__file__).resolve().parent.parent`, private `_builder()` helpers instead
of fixtures, `tmp_path`/`monkeypatch`/`capsys`, and `OmegaConf.create({...})` +
`run_report_main.__wrapped__(cfg)` to bypass Hydra — the pattern at
`tests/test_eval_boundary.py:360`.

---

## Task order

1. `configs/report/default.yaml` + wire into `configs/config.yaml` defaults.
2. `eval/readme_block.py` + its unit tests (pure, fastest to get right).
3. `RunLogger.timer` / `duration_seconds` + `run_recon.py` recording + tests.
4. `eval/scorecard.py` loaders and renderer + tests against synthetic `tmp_path` artifacts.
5. Wire `_report_scorecard` and `_report_synth` into `scripts/run_report.py`; add `report-check`.
6. Rewrite `README.md` (markers, mermaid diagram, all prose sections).
7. Acceptance tests AT-1…AT-7; extend `tests/test_eval_boundary.py`.
8. `make lint && make test`; append the `progress_report.md` entry.

**No GPU is spent at any step.** Step 3 only adds instrumentation that a future cluster run fills in.

## Verification

```bash
make lint
make test                      # includes the 7 acceptance tests + boundary tests
make report                    # < 2 min, no GPU; rewrites the README block in place
git diff README.md             # the ONLY change is between the scorecard markers
make report && git diff --exit-code README.md   # idempotent: second run changes nothing
make report-check              # exits 0 when the README is in sync
python -c "import sys; sys.argv=['x']; import scripts.run_report; \
  assert not {'torch','monai','sklearn'} & set(sys.modules)"   # report-safe subgraph
```

On the cluster, after a real `make recon && make eval`, re-run `make report` and confirm the n/a
cells populate with values matching `artifacts/metrics/unetr__mse_ssim/aggregate.json` — with no
edit to the README by hand.
