# Plan · Spec 013 — Diffusion-based anomaly detection (AnoDDPM)

**Spec:** [`specs/013-diffusion-anomaly.md`](../../specs/013-diffusion-anomaly.md)
**Depends on:** 002 (model registry), 003 (recon engine), 004 (eval harness), 009 (train harness)
**Fresh compute:** **YES.** This is the second and most GPU-hungry fresh-compute spec (with 009).
Everything below except Phase 4 is local, CPU-only, synthetic-tensor work. The training run and the
`t_noise` sweep are hand-offs the user opens manually via `/train`; the agent never launches them.

**Status:** drafted, not implemented.

---

## Context

The project's headline claim (D3) is a **three-paradigm** comparison: Classical (006) vs UNETR vs
**Diffusion**. Two of the three exist. The diffusion row is currently declared-but-empty — it
appears in `configs/matrix/arch_loss.yaml` and `configs/paradigm/default.yaml` as
`diffusion__ddpm` with `na_reason: "untrained"`, and renders as `n/a`. Until it is trained, the
headline table is a two-paradigm table wearing a three-paradigm label.

The question this spec answers is the sharpest form of THE CENTRAL DOMAIN FACT. UNet, Attention-UNet
and UNETR all learn a direct healthy→healthy map and all rebuild the tumor too faithfully to flag
it. A DDPM learns a *generative prior* over healthy anatomy instead. Partially noising a tumor-bearing
volume and denoising it back should, in principle, regenerate healthy tissue where the tumor was —
because the prior has never seen a tumor. **Does it? Or does a generative prior fall to the same
failure?** Either answer is the deliverable (acceptance 8); a diffusion row that loses to UNETR on
Dice gets reported as a finding, not buried.

**Intended outcome:** `diffusion__ddpm` becomes a real, honestly-scored row in the Spec 005 matrix
and the Spec 007 paradigm table — same test split, same threshold strategy, same everything as the
other rows — plus a trained checkpoint and a validation-tuned `t_noise`.

---

## Verified repo state

Checked first-hand; these facts shape every decision below.

| Fact | Evidence |
|---|---|
| **MONAI 1.6.0 is installed** and ships `DiffusionModelUNet`, `DDPMScheduler`, `DDIMScheduler`, `DiffusionInferer`. **No `pyproject.toml` bump is needed** (the spec's open question is closed) | `python -c "import monai; from monai.networks.nets import DiffusionModelUNet"` → OK |
| `src/mri_ad/models/diffusion.py` **already exists** (104 lines, Spec 002 shape-only stub): `DiffusionADModel` with `DiffusionModelUNet` + `DDPMScheduler`, a naive full reverse loop, and a `model_card` marked "untrained — Spec 013" | `src/mri_ad/models/diffusion.py` |
| `configs/model/diffusion.yaml` exists and is explicitly flagged **PROVISIONAL — "Spec 013's /plan finalizes them"** | `configs/model/diffusion.yaml` |
| The matrix and paradigm rows **already exist as data** — no config edit needed to make the row appear, only artifacts to fill it | `configs/matrix/arch_loss.yaml:26`, `configs/paradigm/default.yaml:27` |
| `cell_id = f"{model}__{loss}"`; matrix needs `<metrics_root>/diffusion__ddpm/aggregate.json`, paradigm needs that **plus** `slice_scores.csv` (half-present raises `ArtifactError`) | `scripts/run_eval.py:117`, `eval/matrix.py`, `eval/paradigm.py` |
| **There is no `configs/loss/ddpm.yaml`** — `configs/loss/` holds only mse, ssim, mse_ssim, multiscale_mse, perceptual. Without it `cfg.loss.name` can never be `ddpm`, so the cell id can never be produced | `ls configs/loss` |
| Preprocessing normalizes every volume to **`[0, 1]`** min-max | `data/transforms.py::_minmax` |
| `DDPMScheduler` defaults to `clip_sample=True` clipping to **`[-1, 1]`** — the standard DDPM data range | `inspect.signature(DDPMScheduler.__init__)` |
| MONAI schedulers are `nn.Module`s but their `state_dict()` is **empty** (alphas/betas are plain attributes, not buffers) — so wrapping them costs nothing in the checkpoint | `DDPMScheduler(...).state_dict()` → `{}` |
| `add_noise` moves `alphas_cumprod` onto the sample's device itself; no manual device juggling needed | `inspect.getsource(DDPMScheduler.add_noise)` |
| `DDIMScheduler.step(..., eta=0.0)` is **deterministic** and `set_timesteps(S)` gives S strided timesteps | `inspect.signature(DDIMScheduler.step)`, `set_timesteps` source |
| `CheckpointWriter` saves `model.state_dict()` of the **whole wrapper** (keys prefixed `net.`). `unetr.py`/`msa_unetr.py` load into `self`; `diffusion.py` currently loads into `self.net` — **a guaranteed `CheckpointError` on the first real checkpoint** | `train/checkpointing.py:31`, `models/diffusion.py:87` vs `models/unetr.py:187` |
| `Trainer.selection_key` is hard-restricted to `{"val_loss", "val_dice"}` (`ConfigError` otherwise) | `train/loop.py` |
| `_build_and_warm_start_model` requires **exactly one** of `init_from` / `warm_start_from`. There is no from-scratch branch | `scripts/run_train.py:153-174` |
| `run_train.py` unconditionally builds `AnomalyInformedDataset` (FPI) and runs `_preflight_separability_check` — both are Spec 009 concepts irrelevant to a from-scratch DDPM | `scripts/run_train.py:186-205` |
| `resolve_training_ids` already raises `SplitContractViolationError` on any BraTS-test intersection — **acceptance 7 is free if reused** | `train/splits.py` |
| `RunLogger` dumps the **fully resolved config** into `run_meta.json` — **acceptance 5 is free** provided no diffusion hyperparameter is hardcoded | `utils/run_logger.py` |
| `models/*.py` may not import `mri_ad.{recon,eval,classical,synth,viz}` — AST-enforced | `tests/test_model_boundary.py::test_models_never_import_downstream_layers` |
| `recon/sweep.py::run_threshold_sweep` **refuses any split but `"val"`** at the function boundary | `recon/sweep.py:62` |
| `run_sweep.py` writes `artifacts/results/threshold_sweep_{model}.csv` — a single fixed filename per model, so a hydra multirun over `t_noise` would silently overwrite itself 4 times | `scripts/run_sweep.py:95` |
| `tests/test_models.py::test_diffusion_has_no_checkpoint_yet` asserts `load_checkpoint` raises; `diffusion` is absent from `test_checkpoint_round_trip_both_layouts`'s parametrize list and from `tests/test_scaffold.py`'s config list | `tests/test_models.py:184`, `tests/test_scaffold.py:59-73` |

---

## Decisions

| # | Decision | Rationale |
|---|---|---|
| **D-1** | **Gaussian noise only.** No simplex noise. | AnoDDPM's published contribution is simplex (low-frequency) noise, but MONAI ships none and hand-rolling it fights D6 and doubles GPU spend. The paradigm comparison is the deliverable, not a noise ablation. Named as an explicit documented deviation in the model card, the README and `progress_report.md`, and listed as the natural future extension. |
| **D-2** | **Train with DDPM, sample with DDIM (`eta=0`, strided).** | ~50 strided steps instead of 250 sequential ones ≈ **5× faster** eval — and eval is 8 depth-chunks × every test volume × every sweep point, so this is the difference between a feasible and an infeasible sweep. `eta=0` makes the reverse process **fully deterministic**, so only the initial noise draw needs seeding (NFR-1). |
| **D-3** | **`t_noise` is tuned on the BraTS *val* split**, then frozen into `configs/model/diffusion.yaml` before the single test run. | `t_noise` is the detection knob — too low and the tumor survives the round trip (the central failure mode, back again); too high and the whole brain is resampled and the residual is noise everywhere. Every other model's threshold is val-tuned; not tuning `t_noise` would handicap the diffusion row and make a negative result uninterpretable. Trap #5 is enforced by reusing `run_threshold_sweep`'s `split != "val"` guard. |
| **D-4** | **Rescale `[0,1] → [-1,1]` on entry to `forward`, and `[-1,1] → [0,1]` on exit.** Confined inside `DiffusionADModel`. | The data is `[0,1]` but the DDPM noise schedule and `clip_sample` assume `[-1,1]`. Skipping this silently mis-scales the signal-to-noise ratio at every timestep and makes `clip_sample` a no-op. `forward` returning `[0,1]` keeps the residual directly comparable to every other model's — `recon/` stays untouched (acceptance 2). |
| **D-5** | **`load_checkpoint` loads into `self`, not `self.net`.** | Matches `CheckpointWriter`'s wrapper-level `state_dict()` and the `unetr`/`msa_unetr` precedent. The current `self.net` target would raise `CheckpointError` on the first real checkpoint. Scheduler `state_dict()` is empty, so `self` adds only the `net.` prefix. Still `strict=True`; **never** `strict=False`. |
| **D-6** | **Seed the initial noise draw from a `torch.Generator`** owned by the model, re-seeded per `forward` from a config `sample_seed`. | With DDIM `eta=0` this makes `forward` **bitwise reproducible** — the same volume scores the same Dice on every re-run, independent of chunk batching order. Without it, the one stochastic op in the pipeline would make every published diffusion number irreproducible. |
| **D-7** | **Training selection metric = `val_loss`** (DDPM denoising loss on held-out healthy OpenBHB). | `val_dice` would require a full DDIM sampling pass over BraTS-val volumes every N epochs — the most expensive validation in the project — and it would be measured at an *untuned* `t_noise`, so it would select on a knob setting we are about to change. Denoising loss on healthy data is the honest objective-aligned criterion; detection quality is then tuned separately in Phase 3. |
| **D-8** | **New from-scratch branch in `run_train.py`**, selected by `train.from_scratch: true`, which also **bypasses** the FPI dataset and the separability preflight. | `scripts/` is not `recon`/`eval`, so editing it does not violate acceptance 2. Reuses `Trainer`, `CheckpointWriter`, `resolve_training_ids`, `RunLogger`, the loaders and the optimizer/scheduler construction wholesale — a genuinely from-scratch path adds a branch, not a second training script. |
| **D-9** | **Add `configs/loss/ddpm.yaml`** (`name: ddpm`, `target: torch.nn.MSELoss`). | Mechanically required: it is what makes `cfg.loss.name == "ddpm"` and therefore `cell_id == "diffusion__ddpm"` at both train and eval time. It is also literally correct — the DDPM objective is MSE on the predicted noise. One config serves both roles. |
| **D-10** | **A dedicated `scripts/run_tnoise_sweep.py`** rather than a hydra multirun over `run_sweep.py`. | `run_sweep.py` writes one fixed filename per model, so a multirun would overwrite its own results three times and silently report only the last. The new script reuses `build_sweep_strategies` / `run_threshold_sweep` / `select_operating_point` unchanged and writes its own artifact. |

---

## Files

### New

| Path | Purpose |
|---|---|
| `configs/loss/ddpm.yaml` | D-9. `name: ddpm`, `target: torch.nn.MSELoss`, `params: {}`. |
| `configs/train/ddpm_scratch.yaml` | From-scratch training config: `from_scratch: true`, `save_as: diffusion`, `checkpoint_path`, `max_epochs`, Adam `lr: 1e-4` (from-scratch, unlike 009/012's `1e-5` fine-tune LR), scheduler, `grad_clip_norm`, `early_stopping_patience`, `amp: false`, `allow_cpu: false`, `selection: {metric: val_loss, mode: min}`. |
| `scripts/run_tnoise_sweep.py` | D-10. For each `t_noise` in `cfg.model.sweep.t_noise_values`: build the model at that `t_noise`, reconstruct the BraTS **val** volumes, run `run_threshold_sweep`, keep the best point. Writes `artifacts/results/tnoise_sweep_diffusion.csv` (one row per `(t_noise, threshold)`) and `tnoise_selection.json` (the winner). |
| `tests/test_diffusion.py` | Spec-013 deep tests, mirroring `tests/test_msa_unetr.py`'s structure. |

### Modified

| Path | Change |
|---|---|
| `src/mri_ad/models/diffusion.py` | The core of the spec. `__init__` gains `sampler` (`"ddim"`/`"ddpm"`), `num_inference_steps`, `sample_seed`. Add `[0,1]↔[-1,1]` rescaling (D-4). Rewrite `forward` to use strided DDIM timesteps filtered to `t ≤ t_noise` (D-2) with a seeded generator (D-6). Add a `training_step_output(x)`-style helper (or use `DiffusionInferer`) exposing noise prediction for the trainer **without importing `train/`**. Fix `load_checkpoint` → `self` (D-5). Update `model_card`: real training data/loss strings, and `known_characteristics` stating the Gaussian-not-simplex deviation. |
| `configs/model/diffusion.yaml` | Drop the PROVISIONAL banner. Finalize `channels`/`num_res_blocks`/`attention_levels`, add `sampler`, `num_inference_steps`, `sample_seed`, and a `sweep: {t_noise_values: [...]}` block. `t_noise` stays at its provisional value until Phase 3 writes the val-selected one back. |
| `src/mri_ad/train/objectives.py` | Add `ddpm_step(model, batch, device, *, criterion)` — sample `t ~ U[0, T)`, draw noise, `add_noise`, predict, `criterion(pred_noise, noise)`. Reads `batch["image"]` (`OpenBHBDataset`'s key), not `"corrupted"`/`"healthy"`. Export it. |
| `scripts/run_train.py` | D-8. Gate the synth generator + `AnomalyInformedDataset` + `_preflight_separability_check` behind `not from_scratch`; use bare `OpenBHBDataset` otherwise. Add the from-scratch branch to `_build_and_warm_start_model` (registry-build only, no checkpoint load). Select `ddpm_step` vs `reconstruction_step`. Guard `on_epoch_start` (no `set_epoch` without the synth dataset) and the `"synth"`/`"init_from"` checkpoint-meta fields. |
| `Makefile` | Add `tnoise-sweep:` (MANUAL ONLY, mirroring `sweep:`). |
| `tests/test_models.py` | Delete `test_diffusion_has_no_checkpoint_yet`; add `diffusion` to `test_checkpoint_round_trip_both_layouts`'s parametrize list. |
| `tests/test_scaffold.py` | Add `configs/model/diffusion.yaml`, `configs/loss/ddpm.yaml`, `configs/train/ddpm_scratch.yaml` to the must-parse list. |
| `README.md`, `progress_report.md` | Scorecard regenerated via `make report`; a `progress_report.md` entry per CLAUDE.md rule 2, including the D-1 simplex deviation. |

### Explicitly untouched (acceptance 2)

`src/mri_ad/recon/**`, `src/mri_ad/eval/**`, `src/mri_ad/classical/**`. If any of these needs an
edit, **that is the finding** — the Spec 002 abstraction failed, and it gets reported rather than
patched around.

---

## Phases

### Phase 1 — Model (local, CPU, synthetic tensors)

1. Rewrite `DiffusionADModel` per D-2/D-4/D-5/D-6; finalize `configs/model/diffusion.yaml`.
2. Add `configs/loss/ddpm.yaml`.
3. Write `tests/test_diffusion.py`; fix up `tests/test_models.py` and `tests/test_scaffold.py`.
4. `make lint && make test`.

**Exit:** the model registers from YAML alone, forward-passes at `(2,1,16,128,128)` on a downsized
net, returns `[0,1]`, is bitwise reproducible across two calls, and raises `CheckpointError` on both
a missing file and an LFS stub.

### Phase 2 — Training harness (local, CPU)

5. `ddpm_step` in `objectives.py`; `configs/train/ddpm_scratch.yaml`.
6. The `run_train.py` from-scratch branch.
7. Tests: `ddpm_step` returns a finite scalar with a live grad path; from-scratch config resolves and
   selects the right step fn; `train/` still imports nothing from `recon`/`eval`.
8. Dry-run on CPU with a downsized model and 2 synthetic volumes to prove the wiring, **not** to train.

**Exit:** `make train train=ddpm_scratch` is one command away from working, and the whole path is
proven on synthetic data at zero GPU cost.

### Phase 3 — GPU hand-off · **`/train`, user-invoked only** (CLAUDE.md rule 4)

9. `make check-data`.
10. **Train** (`make train train=ddpm_scratch model=diffusion loss=ddpm ...`) → `checkpoints/diffusion.pth`.
11. **`t_noise` sweep on val** (`make tnoise-sweep`) → best `(t_noise, threshold)` → write `t_noise`
    back into `configs/model/diffusion.yaml` and commit it as a tuned, traceable value.
12. **One** test-split run: `make recon model=diffusion loss=ddpm` → `make eval` → `make slice-scores`.

### Phase 4 — Report (local, no GPU)

13. `make matrix`, `make paradigm`, `make report`. The `n/a` cells become numbers automatically.
14. Write the narrative. **If diffusion loses to UNETR on Dice, that is the headline result**
    (acceptance 8) — `tests/test_paradigm.py::test_diffusion_losing_to_unetr_is_rendered_not_dropped`
    already guarantees the table renders it rather than dropping it. Append to `progress_report.md`.

---

## Risks

| Risk | Mitigation |
|---|---|
| **Sampling cost dominates.** 8 chunks/volume × N test volumes × S DDIM steps of a 3D UNet. | D-2 (DDIM ≈5×). Benchmark one volume's wall-clock **before** launching the sweep and report the extrapolation to the user; if it is still infeasible, cut `num_inference_steps` or the sweep grid — not the test-split size. |
| **`norm_num_groups=32` vs `channels`.** `DiffusionModelUNet` requires every `channels` entry to be a multiple of `norm_num_groups`. | Already documented in the stub's docstring; tests downsize both together (`channels=(4,8)`, `norm_num_groups=4`). Assert the constraint in `__init__` with a clear message. |
| **Memory.** Attention at `(16,128,128)` on the deepest level plus a 3D UNet can OOM. | `attention_levels=(False,False,True)` keeps attention at the coarsest level only. Batch size is already config-driven; tune it in the experiment config, never in code. |
| **The model reconstructs the tumor anyway** — the central failure mode surviving into the generative paradigm. | This is the *finding*, not a bug. Report it (rule 6, acceptance 8) with PSNR/SSIM as *explanatory context only*, never as a headline or target. |
| **`t_noise` too high resamples the whole brain**, inflating the residual everywhere and producing a meaningless Dice. | The val sweep's Dice objective naturally penalizes this. Include a residual-map figure at the selected `t_noise` in the report so the reader can see whether detection is localized or diffuse. |
| **Scope creep into simplex noise mid-implementation.** | D-1 is locked. Simplex is a documented deviation and a named future extension, out of scope for this spec. |

---

## Verification

**Local (every phase except 3), zero GPU:**

```bash
make lint
make test                       # incl. tests/test_diffusion.py, test_model_boundary.py
pytest tests/test_model_boundary.py -q   # acceptance 2: recon/eval/classical untouched
pytest tests/test_scaffold.py -q         # all three new configs parse
```

Per-acceptance mapping:

| Acceptance | Verified by |
|---|---|
| 1 — shape + finite, bounded output | `tests/test_diffusion.py::test_forward_preserves_shape`, `::test_output_is_finite_and_in_unit_range` |
| 2 — extension guarantee | `tests/test_model_boundary.py`, plus a grep test that `recon/`, `eval/`, `classical/` never name `diffusion` outside the pre-existing `diffusion_available` stat (`eval/matrix.py:245`, `eval/paradigm.py:397`) |
| 3 — appears in 005 and 007 on the identical split/threshold | `make matrix && make paradigm` after Phase 3; rows resolve from the existing config cells with no config edit |
| 4 — `CheckpointError`, never a silent partial load | `tests/test_models.py` parametrized missing/stub tests, now including `diffusion`; plus the round-trip test proving D-5's wrapper-level load actually works |
| 5 — all generation params in `run_meta.json` | `RunLogger` dumps the resolved config; a test asserts no diffusion hyperparameter is hardcoded (every one is read from `cfg`) |
| 6 — training only via `/train`/`make train` | No autonomous invocation; `make train` is the only entry point and `/train` carries `disable-model-invocation: true` |
| 7 — test split never seen in training | `resolve_training_ids` + `SplitContract`, already enforced; covered by the existing `train/splits.py` tests |
| 8 — a loss to UNETR is reported, not hidden | `tests/test_paradigm.py::test_diffusion_losing_to_unetr_is_rendered_not_dropped` (already present) |

**End-to-end (after the user's Phase 3 GPU run):** `checkpoints/diffusion.pth` exists and is not a
stub → `artifacts/metrics/diffusion__ddpm/{aggregate.json,slice_scores.csv}` exist with
`split: "test"` → `make matrix && make paradigm && make report` turn the `n/a` cells into numbers
traceable back to a `run_meta.json` under `artifacts/runs/`.
