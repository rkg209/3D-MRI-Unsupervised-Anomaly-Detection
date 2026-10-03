# Implementation Plan — Spec 002 · Model registry & checkpoint loading

## Context

Four models across three paradigms — **UNet** and **Attention-UNet** (prior-work reference rows),
**UNETR** (the deterministic-reconstruction anchor), and **AnoDDPM** (diffusion, Spec 013) — must
sit behind one interface so that `recon/`, `eval/`, and `classical/` never branch on model type.
That uniform interface is what makes the diffusion model a drop-in with zero downstream changes.

Today only two of the four pieces exist:
- `src/mri_ad/models/base.py` — the `AnomalyDetectionModel` ABC + `ModelCard` (done).
- `src/mri_ad/models/unetr.py` — a fully-ported `UNETRReconstruction` (done, from Spec 000).

Missing and net-new for this spec: `UNetModel`, `AttentionUNetModel`, `DiffusionADModel`, a
`ModelRegistry`, a shared checkpoint-loading helper, `configs/model/diffusion.yaml`, and the
import-graph boundary test. The vertical slice (`scripts/run_slice.py`) currently instantiates
models with a provisional `_instantiate` helper explicitly flagged as "Spec 002's registry replaces
this" — so wiring the slice onto the registry is part of the done state.

**Central hazard this spec removes:** the trained weights fail to load cleanly into the wrong class,
and a `strict=False` fallback would load partially and silently produce plausible-looking garbage
that still scores. Every loader path here refuses stubs, refuses partial loads, and names the file.

## Design decisions (resolved)

1. **Shared checkpoint helper, one copy of the LFS logic.** Extract the dual-layout + LFS-stub +
   `strict=True` logic (currently inline in `models/unetr.py:184-209`, and duplicated in
   `scripts/check_data.py:53`) into `src/mri_ad/models/_checkpoint.py`:
   `load_checked_state_dict(module: nn.Module, path: Path) -> None`. It takes the **target module to
   load into** — this is what lets wrapper models load bare (unprefixed) legacy keys (see #2).

2. **Wrapper models load into the inner net, not `self`.** The legacy UNet weights were saved from a
   bare `monai.networks.nets.UNet` (keys like `model.0.conv...`, no wrapper prefix). If `UNetModel`
   holds `self.net = UNet(...)`, then `self.load_state_dict` would expect a `net.` prefix and fail
   `strict=True`. So wrapper models delegate: `load_checked_state_dict(self.net, path)` — keys match
   the bare checkpoint. `UNETRReconstruction` is *not* a wrapper (it subclasses the interface
   directly and its keys already match), so it loads into `self`.

3. **The registry is config-driven and dynamic — adding a model never edits `registry.py`.**
   Acceptance test 8 requires that adding a model touches only its module + a YAML + a checkpoint
   path. So `ModelRegistry` resolves `configs/model/*.yaml` by the existing `target` (dotted class
   path) + `params` schema (the same mechanism as `run_slice.py:_instantiate`), rather than a
   hardcoded `name -> class` table. A per-model `import` line in the registry would itself violate
   test 8.

4. **Interface validation via ABC introspection.** `register()` accepts a class and raises
   `ModelRegistrationError` unless it is a subclass of `AnomalyDetectionModel` **and** has no
   remaining `__abstractmethods__` (a subclass that omits `forward`/`load_checkpoint`/`model_card`
   keeps a non-empty `__abstractmethods__`). This satisfies test 6 without hand-listing method names.

5. **`UNETRReconstruction` port is already correct** — it never forwards `pos_embed`/`proj_type` to
   `ViT` (the legacy class didn't either), so the modern-MONAI rename is a non-issue. Only change to
   `unetr.py`: route its `load_checkpoint` through the shared helper (behavior identical).

6. **Diffusion is shape/interface-only in this spec.** Spec 013 owns the real schedule, `t_noise`,
   sampler, and training. Here `DiffusionADModel` must instantiate, run partial-noise→denoise on a
   **random-init** net and return `(B,1,16,128,128)`, and raise `CheckpointError` on its
   absent/stub checkpoint. `configs/model/diffusion.yaml` gets minimal, clearly-provisional defaults
   with a comment pointing to Spec 013's `/plan` for final values.

## Files to create / modify

### New — model modules (`src/mri_ad/models/`)
- **`_checkpoint.py`** — `LFS_MAGIC` constant + `load_checked_state_dict(module, path)`. The single
  source of truth for stub detection, dual-layout unwrap, `strict=True`, and `CheckpointError`
  messages ("not found", "Git-LFS pointer stub", "state-dict mismatch"), all naming `path.name`.
- **`unet.py`** — `UNetModel(AnomalyDetectionModel)`, wraps
  `monai.networks.nets.UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(16,32,64,128,256),
  strides=(2,2,2,2), num_res_units=2, norm=Norm.BATCH, dropout=0.2)`. No output activation.
  `forward -> self.net(x)`; `load_checkpoint -> load_checked_state_dict(self.net, path)`;
  `model_card` (training_loss="MSE", output_activation=None).
- **`attention_unet.py`** — `AttentionUNetModel(AnomalyDetectionModel)`, wraps
  `monai.networks.nets.AttentionUnet(spatial_dims=3, in_channels=1, out_channels=1,
  channels=(16,32,64,128,256), strides=(2,2,2,2), kernel_size=3)`. Same wrapper pattern.
  `model_card` (training_loss="SSIM + MSE", output_activation=None).
- **`diffusion.py`** — `DiffusionADModel(AnomalyDetectionModel)`, wraps
  `monai.networks.nets.DiffusionModelUNet(spatial_dims=3, ...)` + `DDPMScheduler`. `forward` adds
  noise to `t_noise` via the scheduler, runs the reverse loop to a healthy estimate, returns same
  shape. `load_checkpoint -> load_checked_state_dict(self.net, path)` (always raises until Spec 013
  trains weights). `model_card` documents the output range (not assumed `[0,1]`); training_loss=
  "DDPM denoising (untrained — Spec 013)".
- **`registry.py`** — `ModelRegistry` with:
  - `register(cls: type) -> None` — interface validation (decision #4) → stores keyed by resolved
    name; raises `ModelRegistrationError` on a non-conforming class.
  - `get(name: str) -> AnomalyDetectionModel` — instantiate from the YAML's `target` + `params`.
  - `checkpoint_path(name) -> Path` — resolve the YAML's `checkpoint` field.
  - `build_default_registry(config_dir="configs/model") -> ModelRegistry` — scan `*.yaml`, import
    each `target`, `register()` it. Dynamic import keeps the module free of per-model imports.

### Modify
- **`src/mri_ad/models/unetr.py`** — replace the inline body of `load_checkpoint` with a call to
  `load_checked_state_dict(self, path)`; drop the now-duplicated `_LFS_MAGIC`.
- **`src/mri_ad/models/__init__.py`** — export `ModelRegistry`, `build_default_registry`, and the
  four model classes.
- **`scripts/run_slice.py`** — replace `_instantiate` + manual `load_checkpoint` (lines ~90-92) with
  `registry = build_default_registry(); model = registry.get(cfg.model.name);
  model.load_checkpoint(registry.checkpoint_path(cfg.model.name))`. Keep behavior identical; do not
  break existing `tests/test_slice.py`.

### New — config
- **`configs/model/diffusion.yaml`** — `name: diffusion`,
  `target: mri_ad.models.diffusion.DiffusionADModel`, a `params` block (channels, `num_res_blocks`,
  `attention_levels`, `num_train_timesteps`, `t_noise`, scheduler), and
  `checkpoint: ${paths.checkpoint_root}/diffusion.pth`. Header comment: values provisional, finalized
  in Spec 013 `/plan`. (`unet.yaml`, `attention_unet.yaml`, `unetr.yaml` already exist and match.)

### New — tests (`tests/`)
- **`tests/test_models.py`** — parametrized across the four models (small/downsized archs for speed,
  mirroring the `small_unetr` fixture at `test_slice.py:51-66`; no real data/checkpoints):
  - forward on `torch.rand(2,1,16,128,128)` returns that shape (diffusion included, random init) —
    test 2.
  - `UNETRReconstruction` output in `[0,1]` (`out.min()>=0`, `out.max()<=1`); UNet/AttUNet **not**
    asserted bounded — test 3.
  - checkpoint round-trip: save each layout (bare `state_dict` and `{"model_state_dict": ...}`) with
    `torch.save`, load back with `strict=True`, assert no missing/unexpected keys — tests 1 & 5.
  - missing path **and** a written LFS-stub file each raise `CheckpointError`, message names the file
    and says "pointer stub" for the stub case — test 4.
  - `model_card` has a correct `param_count` (== `sum(p.numel())`) and the expected `training_loss` —
    test 7.
  - `ModelRegistry.register` on a class missing an interface method raises `ModelRegistrationError`;
    `get(name)` returns an `AnomalyDetectionModel` — test 6.
- **`tests/test_model_boundary.py`** — AST-parse every `src/mri_ad/models/*.py` and assert none
  import from `mri_ad.recon`, `mri_ad.eval`, `mri_ad.classical`, `mri_ad.synth`, or `mri_ad.viz`.
  This is the import-graph test for acceptance test 8 (and reused by 013.2 / 012.1). Use `ast.parse`
  + walk `Import`/`ImportFrom` nodes. No such test exists today — net-new.

## Risks / watch-items

- **MONAI ships `DiffusionModelUNet` + `DDPMScheduler`?** They moved from MONAI-Generative into core
  at a specific version. Verify the pinned `monai` in `pyproject.toml` exposes them; if not, bump the
  pin (flag to user before changing deps). If unavailable, gate `diffusion.py` imports so the other
  three models and the registry still import/test cleanly.
- **Wrapper key-prefix trap (decision #2)** — the single most likely correctness bug. The
  checkpoint round-trip test must save from a bare `monai` net and load through the wrapper to prove
  keys line up. Verify with `torch-reviewer` before merge.
- **Registry vs. test 8** — if `registry.py` ends up importing a concrete model class by name, test 8
  is undermined even if it passes (the test only scans `models/*.py`). Keep resolution dynamic.
- **Diffusion forward speed in tests** — a full reverse loop is slow; use a tiny `num_train_timesteps`
  / small `t_noise` and a downsized net in the fixture so the shape test stays fast.
- Do **not** touch `recon/`, `eval/`, `classical/` — if any change there seems required, the
  abstraction has failed and that is the finding (per `new-model` skill and spec 002.8).

## Verification

1. `make test` (or `pytest tests/test_models.py tests/test_model_boundary.py -q`) — all new tests
   green; existing `test_slice.py` / `test_scaffold.py` still pass after the `run_slice.py` rewire.
2. `make lint` — `ruff format` + `ruff check` clean.
3. Boundary check by hand: `git diff --stat` after implementation touches only `models/`,
   `configs/model/diffusion.yaml`, `scripts/run_slice.py`, and `tests/` — **zero** lines in
   `recon/`, `eval/`, `classical/`.
4. Smoke the registry from a REPL/script: `build_default_registry().get("unetr")` returns an
   `AnomalyDetectionModel` and a `(2,1,16,128,128)` forward returns that shape. (Real-checkpoint
   loading stays blocked until `make check-data` passes — checkpoints are user-supplied stubs today,
   so full `load_checkpoint` on real weights is verified when weights land, not in CI.)
5. Run `torch-reviewer` on the diff (shape/device/strict-load correctness) before opening the PR.
6. Append a `progress_report.md` entry (what/why/how, incl. the wrapper-prefix decision) after
   implementation — required by CLAUDE.md rule 2.

## Out of scope (deferred, per spec)

Training (009 UNETR/FPI, 013 diffusion weights), thresholding (003), metrics (004). This spec only
*registers* `DiffusionADModel` and proves its shape/interface contract.
