# Plan · Spec 010 — 3D visualization & demo video

**Spec:** [`specs/010-visualization.md`](../../specs/010-visualization.md) (status: approved) · **Depends on:** 003 (done), 004 (done) · **Fresh compute:** no — reads saved `ReconResult` files, zero `forward()` calls, no GPU.

---

## Context

The public artifact of this project is a **video**, not a hosted service (D5, C-8). A recruiter or
interviewer must be able to *see* the method work — original → reconstruction → residual → mask →
ground truth — without cloning the repo, installing MONAI, or obtaining BraTS.

Spec 003 persists `ReconResult` files and Spec 004 scores them; nothing renders them. The only
existing visual is `scripts/run_slice.py:47-70` — a single-slice, 5-panel PNG bolted into the Spec
000 vertical slice. `src/mri_ad/viz/` is an empty package (docstring-only `__init__.py`).
`Makefile:60-61` already declares `demo: $(PY) scripts/run_demo.py`, but **`scripts/run_demo.py`
does not exist** — `make demo` is a broken target today.

This spec builds the viewer and the headless export path, and nothing else: no web app, no
real-time inference, no model imports anywhere in the demo call graph.

Environment already verified: `matplotlib 3.10.1`, `imageio 2.37.3`, and `imageio_ffmpeg` with a
bundled ffmpeg binary are present in the venv; `ipywidgets` is not.

### Decisions taken before planning (user-confirmed)

- **D-A** — Frames carry **no volume_id by default**. Panel titles + `depth k / D` + a static
  caption only; `viz.annotate_volume_id: false` can turn the ID on. Strictest reading of NFR-13
  (acceptance 5); the volume_id still lives in `artifacts/runs/<stamp>/run_meta.json`.
- **D-B** — `scripts/run_slice.py` is **not** refactored to use the new module. Spec 000 stays
  frozen as the vertical slice; ~20 lines of duplication is the accepted cost of not widening a
  Spec-000-owned file (its `inferno` residual map stays as-is).
- **D-C** — The video covers the **auto-trimmed** depth range (`policy: nonzero`): slices where
  `original` has signal. BraTS volumes are 155 real slices zero-padded to 160
  (`recon/types.py:14`), so a literal full-range video would open and close on black. `all`,
  `gt_window`, and `explicit` policies are also implemented and selectable.
- **D-D** — Residual intensity is scaled **per volume** by default, not per slice. The legacy
  per-slice `(s - min)/(max - min)` renormalizes every slice to full scale, so an anomaly-free slice
  looks identical to a lesion slice — the demo would show a brain on fire in every frame and mislead
  the viewer about what the method actually detected. The `cm.hot` **colorization** is ported
  faithfully as the spec requires; only the scaling *window* differs, and `per_slice` stays
  available in config. Disclose in `progress_report.md` (rule 6).

---

## Governing invariants

| Invariant | How it is enforced |
|---|---|
| `make demo` performs no forward pass and needs no GPU (acceptance 2) | `mri_ad.models` / `mri_ad.recon.engine` / `monai` never imported by `viz/*` or `scripts/run_demo.py`; asserted by AST walk, by a subprocess `sys.modules` check, and by running a full export with `torch.cuda.is_available` patched to raise — the three-layer technique of `tests/test_eval_boundary.py:41-49` |
| Headless export never imports `ipywidgets` (C-8) | `ipywidgets` imported **function-locally inside `show_interactive()` only**; declared as an optional extra, deliberately absent from `dev` so CI exercises the fallback |
| No patient identifier or path in any frame (NFR-13, acceptance 5) | D-A; asserted **behaviourally** — two results identical except for `volume_id` must render byte-identical frames |
| Library code never sets a global matplotlib backend | `Figure` + `FigureCanvasAgg` used directly (no `pyplot`): Agg rendering without hijacking a notebook's backend. AST test asserts `matplotlib.use` appears nowhere under `src/mri_ad/viz/` |
| Thresholding happens only in `recon/` | `viz/` reads `result.anomaly_mask` as given and never compares a residual to a number; `tests/test_model_boundary.py`'s regex covers `src/**` |
| Metrics are defined only in `eval/metrics.py` | `viz/` computes no Dice — the demo renders, it does not score |
| No magic numbers in code | `configs/viz/default.yaml` |

**All colour math happens in numpy, not in matplotlib.** Every panel is converted to `(H, W, 3)`
uint8 *before* `imshow`, which is then called with no `cmap` and no `norm`. This makes the panel
pixels a pure numpy function of the input tensor — the property that makes acceptance 3 assertable —
and leaves only axis chrome and text sensitive to a matplotlib version bump.

---

## Files

### Create

| File | Contents |
|---|---|
| `src/mri_ad/viz/colormaps.py` | guarded normalization + the ported `cm.hot` recipe + overlay compositing |
| `src/mri_ad/viz/depth.py` | depth-range policies for the zero-padded 160-deep volume |
| `src/mri_ad/viz/panels.py` | `PanelSpec`, `RenderSpec`, `build_panels` — the 5-panel contract |
| `src/mri_ad/viz/frames.py` | `FrameRenderer`, `iter_frames`, `frames_digest` |
| `src/mri_ad/viz/video.py` | `write_video` (mp4 via ffmpeg, gif/webp via pillow) |
| `src/mri_ad/viz/viewer.py` | `VolumeViewer` — the spec's contract object |
| `src/mri_ad/viz/exporter.py` | `DemoExporter`, `DemoExport` |
| `scripts/run_demo.py` | `make demo` |
| `configs/viz/default.yaml` | fps, dpi, panels, depth policy, video encoding, output path |
| `tests/test_viz.py` | acceptance 1, 3, 4, 5 |
| `tests/test_viz_boundary.py` | acceptance 2 (AST + subprocess + runtime) |

### Modify

| File | Change |
|---|---|
| `src/mri_ad/viz/__init__.py` | re-export `VolumeViewer`, `DemoExporter`, `DemoExport`, `RenderSpec`, `DepthSpec`, `VideoSpec`, `normalize_slice`, `hot_rgb` — nothing ipywidgets-flavoured |
| `src/mri_ad/exceptions.py` | add a `# ── viz/ ──` section with `VizError`, matching the neighbouring classes' base and docstring style |
| `src/mri_ad/eval/loader.py` | add `load_by_volume_id(results_dir, volume_id)` — the gap found during exploration; better than duplicating the glob in `viz/` |
| `configs/config.yaml` | add `- viz: default` to `defaults`, between `- synth: fpi` and `- train: finetune` |
| `pyproject.toml` | add optional extra `viz = ["ipywidgets>=8.1"]` (precedent: `radiomics`, line 32); **remove** `omit = ["src/mri_ad/viz/*"]` at line 73 — viz is now tested |
| `tests/test_scaffold.py` | add `configs/viz/default.yaml` to the config-parse parametrization (~line 59-79) |
| `progress_report.md` | append entry (rule 2), disclosing D-B and D-D |

`Makefile` needs **no change** — `demo` already exists and is correctly not marked MANUAL ONLY.

---

## Interfaces

### `viz/colormaps.py` — the ported recipe, guarded

```python
EPS: Final = 1e-8

def normalize_slice(a: np.ndarray, *, vmin: float | None = None,
                    vmax: float | None = None, eps: float = EPS) -> np.ndarray: ...
def hot_rgb(a: np.ndarray, **kw) -> np.ndarray:        # (H,W) -> (H,W,3) uint8
def gray_rgb(a: np.ndarray, **kw) -> np.ndarray: ...
def overlay_rgb(base: np.ndarray, mask: np.ndarray,
                color: tuple[int, int, int], alpha: float) -> np.ndarray: ...
```

- **`normalize_slice` is the acceptance-4 guard.** `lo = vmin or a.min()`, `hi = vmax or a.max()`;
  **if `hi - lo <= eps` return `zeros_like(a)`** — no division is ever executed, so this is a guard,
  not an epsilon fudge (an epsilon still returns garbage for a constant-`c` slice and `nan` for an
  all-`nan` one). Output clipped to `[0, 1]`. The legacy code divides by a bare `np.max`
  (`legacy/GUI/tk_app.py:31-34`) and by an unguarded `max - min` (`:88-102`); BraTS's 5 zero-padded
  slices make that crash **guaranteed**, not hypothetical.
- **`hot_rgb` is `tk_app.py:88-99` minus the defects** listed in the spec's "Notes / deviations":
  `matplotlib.cm.hot(normalize_slice(a))[..., :3] * 255 → uint8`. Min-subtraction retained, division
  guarded, no PIL, no LANCZOS resize, no Tk.

### `viz/depth.py` — depth policy for the padded volume

```python
@dataclass(frozen=True)
class DepthSpec:
    policy: str = "nonzero"        # all | nonzero | gt_window | explicit
    window: int = 40
    min_intensity: float = 0.0
    explicit: tuple[int, int] | None = None
    max_frames: int = 160

def select_depths(original: Tensor, label: Tensor | None, spec: DepthSpec) -> list[int]: ...
```

`nonzero` (default, D-C) drops every `d` where `original[0, d].max() <= min_intensity` — removing
both the 5 pad slices and the empty superior/inferior anatomy. `gt_window` centres `window` slices
on `label.sum(dim=(-1,-2)).argmax()`, clipped and intersected with `nonzero`. Fallback chain
`gt_window → nonzero → all` when a policy yields `< 2` depths (keeps 16-deep synthetic test tensors
working). Always returns sorted unique ints, uniform-stride-subsampled to `max_frames` — so the
frame count is deterministic and bounded.

### `viz/panels.py` — the 5-panel contract

```python
@dataclass(frozen=True)
class PanelSpec:
    key: str; title: str; cmap: str; overlay: str | None = None

@dataclass(frozen=True)
class RenderSpec:
    panels: tuple[PanelSpec, ...]; figsize: tuple[float, float]; dpi: int
    caption: str; annotate_volume_id: bool; residual_scale: str
    overlay_color: tuple[int, int, int]; overlay_alpha: float; eps: float
    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> RenderSpec: ...   # validates EVEN pixel dims — see R2

def build_panels(result: ReconResult, label: Tensor | None, depth: int,
                 spec: RenderSpec, *, residual_vmax: float | None = None
                 ) -> list[tuple[str, np.ndarray]]: ...
```

`build_panels` returns exactly five `(title, (H,W,3) uint8)` pairs, **all sliced at the same
`depth`** — this is where "synchronized on one depth slider" is enforced, and it is unit-testable
with no rendering at all. A `_as_dhw` helper squeezes `(1, D, H, W) → (D, H, W)` and raises
`VizError` naming the observed shapes on a `result`/`label` depth mismatch.

| # | Panel | Rendering |
|---|---|---|
| 1 | original | `gray_rgb(x, vmin=0, vmax=1)` — fixed window, so brightness does not pump between frames |
| 2 | reconstruction | same fixed window, so the two are visually comparable |
| 3 | residual | `hot_rgb(x, vmin=0, vmax=residual_vmax)`; `residual_vmax` is the volume-wide max under `per_volume` (D-D), `None` under `per_slice` |
| 4 | anomaly mask | `gray_rgb(binary, vmin=0, vmax=1)` |
| 5 | ground truth | `overlay_rgb(original_gray, gt, overlay_color, overlay_alpha)`. If `ground_truth is None` (OpenBHB), a black panel titled `"ground truth (unavailable)"` — never a blank axis with no explanation |

### `viz/frames.py` — one depth → one RGB frame

```python
class FrameRenderer:                       # context manager
    def __init__(self, spec: RenderSpec) -> None: ...     # ONE Figure + FigureCanvasAgg, reused
    def render(self, result, label, depth: int) -> np.ndarray: ...   # (H, W, 3) uint8
    def close(self) -> None: ...

def iter_frames(result, label, depths: Sequence[int], spec: RenderSpec) -> Iterator[np.ndarray]: ...
def frames_digest(frames: Iterable[np.ndarray]) -> str: ...          # streaming sha256
```

Axes are created **once** via `fig.subplots(1, 5)` with explicit `fig.subplots_adjust(...)`
constants — **never `tight_layout` or `bbox_inches="tight"`**, whose output size varies with title
length and would break encoding mid-video. Per frame: `build_panels` → `im.set_data(rgb)` on the
five persistent artists → `canvas.draw()` → `np.asarray(canvas.buffer_rgba())[..., :3].copy()`.
160 frames therefore cost one figure, not 160.

Caption is `fig.suptitle(f"depth {d:03d} / {n}  ·  {spec.caption}")`, prefixed with the volume_id
only when `annotate_volume_id` (D-A). **No `Path` can reach this module** — `render` takes arrays,
ints, and a `RenderSpec`; that is asserted by signature introspection.

### `viz/video.py` — frames → file

```python
@dataclass(frozen=True)
class VideoSpec:
    codec: str = "libx264"; quality: int = 8
    pixelformat: str = "yuv420p"; macro_block_size: int = 1

def write_video(frames: Iterable[np.ndarray], path: Path, fps: int, video: VideoSpec) -> tuple[int, str]: ...
```

Dispatches on suffix: `.mp4` → `imageio.get_writer(...)` (ffmpeg); `.gif` / `.webp` → the pillow
plugin, **which needs no ffmpeg at all** — this is the escape hatch that lets the whole test suite
exercise the real export path on any machine. Frames are streamed with `append_data` and the sha256
is updated in the same pass, so nothing accumulates (160 × 1500×320×3 ≈ 230 MB if materialized).
Returns `(n_frames, digest)`. `imageio` is imported **function-locally** so importing `mri_ad.viz`
does not pull in ffmpeg. A missing or unusable ffmpeg is caught and re-raised as `VizError` naming
`pip install "imageio[ffmpeg]"` — never a bare imageio `RuntimeError`.

### `viz/viewer.py` — the spec's contract object

```python
class VolumeViewer:
    def __init__(self, result: ReconResult, label: Tensor | None = None,
                 spec: RenderSpec | None = None, depth: DepthSpec | None = None) -> None: ...
    @property
    def depths(self) -> list[int]: ...
    def frame(self, depth: int) -> np.ndarray: ...        # shared by BOTH paths
    def show_interactive(self) -> Any: ...                # ipywidgets, lazy import
    def export_video(self, path: Path, fps: int = 4, video: VideoSpec | None = None) -> Path: ...
```

`label` defaults to `result.ground_truth` when `None` (mirrors `VolumeEvaluator.evaluate`'s `label`
parameter, `eval/evaluator.py:29`). `export_video` touches only `frames.py` + `video.py` and writes
a sidecar `<output>.frames.json` — `{"n_frames", "fps", "frame_shape", "sha256"}`.

`show_interactive` imports `ipywidgets` **inside the function body**, converting `ImportError` into
`VizError("The interactive viewer needs the optional viz extra: pip install -e '.[viz]'")`. It wires
one `IntSlider(min=depths[0], max=depths[-1])` to an `Image` widget fed by `self.frame(d)` encoded
to PNG bytes. One slider, one frame, five panels — synchronized *by construction*, because the frame
**is** the composite, rather than by five callbacks that can drift.

### `viz/exporter.py` — the `make demo` object

```python
@dataclass(frozen=True)
class DemoExport:
    path: Path; volume_id: str; n_frames: int; depths: list[int]; frames_digest: str; fps: int

class DemoExporter:
    def __init__(self, render: RenderSpec, depth: DepthSpec, video: VideoSpec, fps: int = 4) -> None: ...
    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> DemoExporter: ...
    def pick_volume(self, results_dir: Path) -> str: ...
    def export(self, results_dir: Path, volume_id: str | None, output_path: Path) -> DemoExport: ...
```

`pick_volume` defaults to the volume with the largest ground-truth area (ties → lexicographic min,
for determinism) — a demo should show the method working on a visible lesion. It streams via
`iter_results` rather than loading all volumes. `export` loads through the new
`eval.loader.load_by_volume_id`, raising `ArtifactError` on a miss that reports the **count** of
available volumes, never a listing of their IDs — the same NFR-13 discipline as the frames. A
`ground_truth is None` result raises `ArtifactError`: the demo is a BraTS artifact and shipping a
silently black GT panel would misrepresent the result. (The interactive `VolumeViewer` still
degrades gracefully with a labelled panel.)

---

## Determinism (acceptance 3)

**The reproducible unit is the frame array, not the MP4 file.** H.264 bytes are not stable across
ffmpeg builds, versions, or thread counts, so asserting on `.mp4` bytes yields a test that passes
here and fails on the cluster for reasons unrelated to the code.

1. Frames are a pure function of `(tensors, RenderSpec, DepthSpec)` — no RNG, no `pyplot` global
   state, no wall-clock, no path in any label, fixed `figsize`/`dpi`, colour math in numpy.
2. `frames_digest` = streaming sha256 over each `frame.tobytes()` in order.
3. `export_video` writes the digest + frame count + shape to `<output>.frames.json`; `run_demo.py`
   records the same in `run_meta.json`, so any published video is traceable (rule 6).
4. Tests assert two independent exporters produce `np.array_equal` frames and equal digests, and
   that two **fresh subprocesses** print the same digest (catching dependence on global backend or
   RNG state). **No golden digest or golden PNG is committed** — a matplotlib/FreeType bump would
   make that a flaky test rather than a real regression.
5. The MP4 is asserted only to exist, be non-empty, and read back with the expected frame count.

Document in `viewer.py`'s docstring that reproducibility is scoped to a fixed environment.

---

## Config

`configs/viz/default.yaml`:

```yaml
# Spec 010: demo video + interactive viewer. `make demo`.
# Reads SAVED ReconResults only — no model, no GPU (acceptance test 2).
results_run_id: null      # null -> newest run dir under recon.results_dir (eval/loader.py)
volume_id: null           # null -> largest ground-truth area; ties -> lexicographic min
output_path: ${paths.artifact_root}/demo/demo.mp4
frames_dir: null          # optional PNG dump for debugging; null -> skip

fps: 4                    # slow enough to watch a lesion appear and vanish
dpi: 100
figsize: [15.0, 3.2]      # 1500x320 px — BOTH dims even, required by libx264/yuv420p (R2)
# Static caption. Never a volume_id, never a filesystem path (NFR-13 / acceptance 5).
caption: "saved reconstruction · no live inference"
annotate_volume_id: false # D-A

panels:
  - { key: original,       title: "original",       cmap: gray, overlay: null }
  - { key: reconstruction, title: "reconstruction", cmap: gray, overlay: null }
  - { key: residual,       title: "residual",       cmap: hot,  overlay: null }   # ported cm.hot
  - { key: anomaly_mask,   title: "anomaly mask",   cmap: gray, overlay: null }
  - { key: ground_truth,   title: "ground truth",   cmap: gray, overlay: original }

normalize:
  eps: 1.0e-8             # range < eps -> all-zero panel; guards the zero-pad slices (test 4)
  residual_scale: per_volume  # per_volume | per_slice. See D-D — per_slice is the legacy
                              # behaviour and makes an anomaly-free slice look like a lesion slice.
overlay:
  color: [255, 0, 0]
  alpha: 0.45

# The BraTS artifact is 160 deep (155 real + 5 zero-pad). Never render the pad.
depth:
  policy: nonzero         # all | nonzero | gt_window | explicit   (D-C)
  window: 40              # gt_window only: frames centred on the max-GT-area slice
  min_intensity: 0.0      # a slice is "empty" if original.max() <= this
  explicit: null          # [start, stop] inclusive when policy=explicit
  max_frames: 160         # hard cap; uniform-stride subsample beyond it

video:
  codec: libx264
  quality: 8
  pixelformat: yuv420p
  macro_block_size: 1     # frames are already even-sized; forbid imageio's silent resize (R2)
```

`configs/config.yaml`: insert `  - viz: default` into `defaults` between `- synth: fpi` and
`- train: finetune`.

---

## Entry point

`scripts/run_demo.py` (`make demo` — no GPU, no model):

```python
#!/usr/bin/env python
"""Spec 010: export the demo video from a saved ReconResult. ``make demo``.

No model is constructed, no checkpoint is read, no GPU is touched (acceptance test 2) — this
script reads persisted ``ReconResult`` .pt files exactly like ``make eval`` does. ipywidgets is
never imported: the interactive path lives behind ``VolumeViewer.show_interactive`` and the
optional ``[viz]`` extra.
"""
```

Body: `seed_everything(cfg.seed, deterministic=cfg.deterministic)` as the first statement (house
spine; unlike `run_report.py` there is no torch-free constraint to trade against, since
`recon.io.load_result` imports torch anyway) → `with RunLogger(cfg) as run:` →
`resolve_results_dir(Path(str(cfg.recon.results_dir)), run_id=cfg.viz.results_run_id)` →
`read_manifest` (**advisory**: a `None` manifest prints a warning rather than crashing — a demo
publishes no metric, so the split-leakage check that `run_eval.py` enforces does not apply, but the
resolved run id and split are printed and recorded, see R9) → `DemoExporter.from_cfg(cfg.viz)` →
`export(...)` → `run.record(volume_id=..., output=..., n_frames=..., fps=..., frames_digest=...)` →
`print(export.path)`.

Module-level imports: `hydra`, `omegaconf`, `mri_ad.viz`, `mri_ad.eval.loader`, `mri_ad.exceptions`,
`mri_ad.utils.{seed,run_logger}` — nothing else, and never `mri_ad.models` or `mri_ad.recon.engine`.
Failures are `ArtifactError` (no run dir / unknown volume / GT missing) or `VizError` (no ffmpeg),
each with a remediation sentence.

---

## Sequencing

1. `exceptions.VizError` → `viz/colormaps.py` + acceptance-4 tests. Zero new deps, unblocks all.
2. `viz/depth.py` + `viz/panels.py` + acceptance-1 tests (no rendering needed).
3. `viz/frames.py` + acceptance-3 tests.
4. `viz/video.py` (gif path first — no ffmpeg needed).
5. `eval/loader.load_by_volume_id` + `viz/exporter.py`.
6. `configs/viz/default.yaml` + `configs/config.yaml` + the `test_scaffold.py` row.
7. `scripts/run_demo.py` + `tests/test_viz_boundary.py` (acceptance 2).
8. `viz/viewer.py` `show_interactive` + the `[viz]` extra + coverage-omit removal.
9. `torch-reviewer` on the diff, then append the `progress_report.md` entry (disclosing D-B, D-D).

---

## Verification

Everything runs on this laptop with **no real data, no checkpoints, and no 3D forward passes** —
synthetic seeded tensors at `(1, 16, 32, 32)` via a shared `_result(volume_id, depth=16, hw=32,
pad=3)` builder modelled on `tests/test_eval.py:36-60`, with the **last `pad` depth slices set
exactly to zero** so every test exercises the padded-volume case. House conventions: no
`conftest.py`, `REPO = Path(__file__).resolve().parent.parent`, `pytest.importorskip("monai")` with
`# noqa: E402`, `# ── Acceptance #N: … ───` separators, long sentence-style test names. Target
runtime for the whole file: < 10 s.

```bash
make lint
pytest tests/test_viz.py tests/test_viz_boundary.py -q
make test
```

| Acceptance | Tests |
|---|---|
| **1** five panels, one slider | `test_build_panels_returns_five_titled_rgb_panels` (len 5, titles match config order, each `(H,W,3)` uint8) · `test_all_panels_are_driven_by_one_depth_index` — give each slice a signature (`original[0,d] = d/D`), render `d` and `d±1`, assert every one of the five horizontal fifths changes (proves *synchronization*, not merely presence) · `test_ground_truth_panel_degrades_when_ground_truth_is_none` · `test_frame_shape_equals_figsize_times_dpi` |
| **2** no forward, no GPU | `test_viz_modules_import_no_model_or_notebook_dependencies` (AST over `src/mri_ad/viz/*` + `scripts/run_demo.py`; forbidden `{mri_ad.models, mri_ad.recon.engine, monai, ipywidgets, IPython}`) · `test_no_viz_module_sets_a_global_matplotlib_backend` (AST for `matplotlib.use`) · `test_run_demo_exports_from_a_saved_result_in_a_subprocess` — `python scripts/run_demo.py` over a `tmp_path` results dir with `viz.output_path=<tmp>/demo.gif`, assert rc 0 and file non-empty · `test_importing_mri_ad_viz_leaves_models_out_of_sys_modules` (subprocess prints `sorted(sys.modules)`) · `test_export_succeeds_when_cuda_is_unavailable` (patch `torch.cuda.is_available` to raise) |
| **3** reproducible | `test_frames_are_bitwise_reproducible_across_two_exporters` (`np.array_equal` + equal digest) · `test_frames_digest_is_stable_across_processes` (two subprocess digest prints) · `test_export_sidecar_digest_matches_recomputed_frames` · `test_exported_video_reads_back_with_the_expected_frame_count` |
| **4** all-zero slice | `test_normalize_all_zero_slice_returns_zeros_without_dividing_by_zero` — under `warnings.simplefilter("error")` **and** `np.errstate(divide="raise", invalid="raise")`, so a `RuntimeWarning` fails the test; parametrized over all-zero, constant-nonzero, and single-nonzero-pixel · `test_hot_rgb_matches_the_legacy_recipe_on_a_non_degenerate_slice` (inline `cm.hot` reference — pins the port against a colormap swap) · `test_every_depth_of_a_padded_volume_renders_finite_uint8` (policy `all`) · `test_nonzero_policy_drops_the_zero_padded_slices` · `test_a_volume_of_only_empty_slices_still_exports_a_non_empty_video` (the D-C fallback chain) |
| **5** no identifiers | `test_frames_are_identical_for_two_volumes_differing_only_in_volume_id` — same tensors, ids `"BraTS20_Training_001"` vs `"/data/private/patient_x"`, `annotate_volume_id=False` → `np.array_equal`, so nothing identifying was rasterized into *any* pixel · `test_caption_contains_no_path_separator` · `test_render_signatures_accept_no_path_argument` (`inspect.signature` on `build_panels` / `FrameRenderer.render`) · `test_missing_volume_error_reports_a_count_not_a_listing_of_ids` |
| extras split | `test_show_interactive_raises_vizerror_when_ipywidgets_is_absent` (`monkeypatch.setitem(sys.modules, "ipywidgets", None)`; message contains `.[viz]`) |
| ffmpeg-gated | `test_mp4_export_when_ffmpeg_is_available` — module-level `_FFMPEG = _have_ffmpeg()` wrapping `imageio_ffmpeg.get_ffmpeg_exe()` in try/except, `@pytest.mark.skipif`. **Every other video test writes `.gif`** via the pillow plugin, so the suite has no ffmpeg dependency. |

Plus: `test_viz_config_group_parses_and_round_trips_into_the_specs` (Hydra `compose` with
`viz=default` → `RenderSpec.from_cfg` / `DepthSpec` / `VideoSpec`) ·
`test_render_spec_rejects_odd_pixel_dimensions` (R2) ·
`test_explicit_depth_range_overrides_the_auto_trim` ·
`test_iter_frames_is_a_generator_and_does_not_materialize_the_volume` (`inspect.isgenerator`).

### End-to-end (user-invoked, wherever saved `ReconResult`s exist)

```bash
make demo HYDRA_OVERRIDES="+viz.results_run_id=<run_id>"
make demo HYDRA_OVERRIDES="+viz.volume_id=<id> viz.fps=6 viz.depth.policy=gt_window"
```

Expected artifacts: `artifacts/demo/demo.mp4`, `artifacts/demo/demo.mp4.frames.json`,
`artifacts/runs/<stamp>/run_meta.json` carrying `frames_digest`.

Note `make demo` currently has **nothing to read** — `artifacts/results/recon/` does not exist until
someone runs `make recon` (GPU, manual). Until then the end-to-end path is exercised only against a
synthetic `tmp_path` results directory, and `run_demo.py` must fail with a readable `ArtifactError`
("run `make recon` first"), never an `IndexError`.

---

## Risks & open questions

| # | Risk | Handling |
|---|---|---|
| R1 | matplotlib/FreeType drift changes frame pixels; a golden-hash test would be flaky | Digests are compared *within* a run/environment; nothing golden is committed. Colour math lives in numpy, so only axis chrome and text are version-sensitive. |
| R2 | `imageio` silently resizes frames whose dims aren't a multiple of 16 (`macro_block_size` default), so the encoded video no longer matches the hashed frames | `macro_block_size: 1` in config **and** `RenderSpec.from_cfg` validates that `figsize × dpi` is an even integer pair, raising `VizError` otherwise. Directly unit-tested. |
| R3 | ffmpeg absent on a reviewer's machine → `make demo` dies with an opaque imageio error | `write_video` re-raises as `VizError` with `pip install "imageio[ffmpeg]"`; the `.gif` path works with no ffmpeg at all, and the test suite uses it. |
| R4 | 160 frames × 1500×320×3 ≈ 230 MB if materialized | Frames stream generator → writer, digest updated incrementally; asserted by `inspect.isgenerator`. |
| R5 | `ground_truth is None` (OpenBHB results) reaches the demo and ships a silently black GT panel | `DemoExporter.export` raises `ArtifactError`; only the interactive viewer degrades, and then with a labelled "unavailable" panel. |
| R6 | D-D deviates from the literal legacy per-slice normalization | Colorization ported faithfully; only the scaling window changes, `per_slice` stays in config, deviation disclosed in `progress_report.md`. |
| R7 | An identifier leaks into a frame later (someone adds `volume_id` to the caption) | Acceptance-5's pixel-equality assertion fails immediately — a *behavioural* guard, not a code-review convention. |
| R8 | `ipywidgets` is a new optional dep and the interactive path can't be tested headlessly | Optional extra `[viz]`, deliberately excluded from `dev` so CI proves the fallback. `show_interactive` is thin (one slider over the already-tested `frame()`); its only tested behaviour is the `VizError`. |
| R9 | `resolve_results_dir`'s newest-mtime fallback could pick a **val**-split sweep run for the demo | Harmless (no metric is published from a demo), but the resolved run id and manifest split are printed and recorded in `run_meta.json`; a missing manifest warns rather than crashes. |
