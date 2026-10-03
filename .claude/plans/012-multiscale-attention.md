# Plan · Spec 012 — Multi-scale attention variant (STRETCH)

**Spec:** [`specs/012-multiscale-attention.md`](../../specs/012-multiscale-attention.md)
**Depends on:** 002 (model registry), 005 (arch×loss matrix), 009 (fine-tune harness)
**Fresh compute:** **YES, but gated.** Every step below is local, CPU-only, synthetic-tensor work.
The GPU run is a separate hand-off the user opens manually via `/train`, and only after the
evidence gate in *The evidence gate* section below is satisfied.

**Status:** approved, not implemented. No code has been written for Spec 012.

---

## Context

Spec 012 is the project's one remaining architectural stretch. The hypothesis: anomalies span
scales — a 5 mm lesion and a 4 cm mass need different receptive fields — so attention applied at
multiple scales *might* localize better than UNETR's uniform patch-based features.

Two facts from the current repo state shape this plan and cannot be wished away:

1. **The spec's own trigger condition is unresolvable today.** The spec says the one thing that
   makes this worth GPU hours is Spec 005 showing the fidelity/detection anti-correlation is
   *scale-dependent*. `artifacts/metrics/` **does not exist** — no eval has ever run, so every cell
   of `configs/matrix/arch_loss.yaml` is currently `n/a`. Spec 013 (diffusion), which owns the
   headline third-paradigm slot, is also unimplemented. Per constraint C-7 this spec must not block
   anything.
2. **The variant cannot warm-start naively.** Attention gates add new `state_dict` keys, so
   `unetr_mse_ssim_aug.pth` will not load under `strict=True` — and CLAUDE.md forbids `strict=False`
   outright ("a partially-loaded model produces plausible-looking garbage that scores cleanly").

So this plan takes a **gate-first posture**: build and fully unit-test the architecture now, so the
variant registers, forward-passes, and renders as an honest `n/a` row in the 005 matrix. That
discharges acceptance test 1 immediately and leaves acceptance 2 as a one-command hand-off if and
only if the evidence gate opens — with acceptance 3 (documented deferral) as the live, expected
fallback rather than a failure.

**Intended outcome:** a registered, tested, provably-identity-at-init multi-scale attention variant
of UNETR, costing zero GPU, with the compute decision left explicitly to the user and to evidence
that does not yet exist.

---

## Verified repo state

| Fact | Evidence |
|---|---|
| Registry is a runtime YAML scan; dropping `configs/model/<name>.yaml` registers a model with **zero code edits** elsewhere | `src/mri_ad/models/registry.py` (`add_from_yaml`, `build_default_registry`) |
| Config schema is `{name, target, params, checkpoint}` — plain dotted-path `importlib`, **not** Hydra `_target_` | `configs/model/unetr.yaml`; a nested `_target_` would be silently ignored |
| Matrix rows are declared data: one `cells:` line + an `aggregate.json` that may not exist yet | `configs/matrix/arch_loss.yaml`, `src/mri_ad/eval/matrix.py::load_cells` |
| `load_checked_state_dict` is the only sanctioned loader: dual layout, LFS-stub sniff, `strict=True`, `CheckpointError` on mismatch | `src/mri_ad/models/_checkpoint.py:22-50` |
| UNETR has **four** skip connections at four scales — `enc1` (`feature_size`), `enc2` (`×2`), `enc3` (`×4`), `enc4` (`×8`) — passed **raw** into `UnetrUpBlock`. There is no gate anywhere today | `src/mri_ad/models/unetr.py:165-177` |
| `AttentionUNetModel` is a thin MONAI wrapper with no gate code of its own; the reference gate is `monai/networks/nets/attentionunet.py::AttentionBlock` | `src/mri_ad/models/attention_unet.py` |
| `Trainer` is model-agnostic and imports nothing from `recon`/`eval`; a new model plugs in with no `train/` change | `src/mri_ad/train/loop.py` |
| `models/` may not import `recon|eval|classical|synth|viz` — AST-enforced | `tests/test_model_boundary.py` |
| Model tests use **downsized** architectures and synthetic tensors, parametrized over `MODEL_FACTORIES` | `tests/test_models.py` |
| `tests/test_scaffold.py` holds a hardcoded list of config paths that must parse — new YAMLs go there by house convention | `tests/test_scaffold.py:59-73` |

---

## Decisions

| # | Decision | Why | Rejected |
|---|---|---|---|
| D1 | **Gate-first posture.** Build the architecture; the GPU run is a separate, user-triggered hand-off behind an explicit evidence gate. | The spec's trigger (005 scale-dependence) is unresolvable — `artifacts/metrics/` does not exist. C-7 forbids blocking. | Train immediately (spends compute on an untested hypothesis); defer without code (delivers nothing, and the code costs no compute). |
| D2 | Vary **UNETR**, the best performer, via subclass `MultiScaleAttentionUNETR(UNETRReconstruction)`. | The spec says gate the best model. Subclassing reuses all block construction + `_proj_feat` verbatim, so the shared `state_dict` keys stay byte-identical and the warm-start shim is a pure key partition. | Building on `AttentionUnet` (a reference-row model, not the best); a standalone copy-pasted class (key drift risk). |
| D3 | **Gate output is `x * 2 * sigmoid(psi(...))`**, with `psi` conv **zero-initialized** → gate ≡ `1.0` exactly at init. | Makes the variant **bitwise identical** to plain UNETR at init. That is a testable claim, it makes the warm-start meaningful, and it means a failed run cannot be blamed on a broken initialization. | MONAI's `x * sigmoid(...)`: zero-init gives a constant `0.5` rescale of every skip, silently halving skip magnitude at step 0 — a confound indistinguishable from "attention didn't help". |
| D4 | Warm-start via a **separate, explicitly-named `load_from_unetr(path)`** that partitions keys and asserts `ckpt_keys == self_keys - gate_keys` exactly, then loads `strict=True`. | Satisfies "never `strict=False`" in substance, not just letter: no key is silently absent, the gate keys are named and accounted for, and any drift raises `CheckpointError`. This is the "compat shim + equivalence test" the `new-model` skill prescribes. | `strict=False` (forbidden); overwriting `load_checkpoint` semantics (its own checkpoint must still load strictly). |
| D5 | The **multi-scale** axis is two-part: (a) gates at all four skip scales; (b) each gate's gating signal also receives the deepest ViT feature `dec4`, trilinearly upsampled and 1×1-projected to that scale. Controlled by `multiscale_context: bool`. | (a) alone is just AttUNet-gating grafted onto UNETR. (b) is what makes it genuinely *multi-scale* attention: each scale attends conditioned on global context. The flag gives a **free ablation row** at no extra code. | Gates at one scale only; a full feature-pyramid rewrite (large, and the spec is a stretch). |
| D6 | **No paradigm column.** 012 appears only as a row in the 005 matrix. | The spec is explicit: 013 owns the third-paradigm slot; 012 "remains an optional architectural add-on, **not a comparison member**." | Adding it to `configs/paradigm/default.yaml` — would misrepresent the headline comparison. |
| D7 | Warm-start is wired through **one optional branch in `scripts/run_train.py`** (`train.warm_start_from`), not a change to `train/loop.py`. | `recon/`, `eval/`, `classical/`, `viz/`, and `train/` stay untouched — the spec's Contract. Entry-point scripts may cross layers (existing precedent). | A new `Trainer` parameter (pollutes a model-agnostic class). |
| D8 | Fine-tune on the **same FPI objective as 009** (`configs/train/msa_finetune.yaml`, `init_from` semantics reused), lr `1e-5`, 20 epochs, selection on **BraTS-val** Dice. | Keeps the only varying factor the architecture. Comparing a plain-reconstruction 012 against a synth-trained 009 would confound architecture with objective. | Plain-reconstruction fine-tune; a from-scratch run (many multiples of the cost for a stretch spec). |

---

## Files

### New

| Path | Responsibility |
|---|---|
| `src/mri_ad/models/attention_gate.py` | `MultiScaleAttentionGate` — `W_g`/`W_x`/`psi` 1×1 convs + optional context branch. `torch`/`monai` only; imports nothing from `mri_ad` outside `models/`. |
| `src/mri_ad/models/msa_unetr.py` | `MultiScaleAttentionUNETR(UNETRReconstruction)` — four gates, overridden `forward`, `model_card`, and `load_from_unetr`. |
| `configs/model/msa_unetr.yaml` | `params` = `unetr.yaml`'s block **plus** gate keys; `checkpoint: ${paths.checkpoint_root}/msa_unetr.pth`. |
| `configs/train/msa_finetune.yaml` | Clone of `configs/train/finetune.yaml` with `warm_start_from: unetr`, `save_as: msa_unetr`. |
| `tests/test_msa_unetr.py` | Equivalence-at-init, gate math, key-partition, ablation-flag, and boundary tests. |

### Modified

- `scripts/run_train.py` — optional `train.warm_start_from` branch (see §Interfaces).
- `configs/matrix/arch_loss.yaml` — **one** cell line under a new `# --- Spec 012 stretch ---` heading:
  `- {model: msa_unetr, loss: mse_ssim, role: stretch, na_reason: "stretch — not trained (Spec 012 gate closed)"}`
- `tests/test_models.py` — add `_small_msa_unetr` to `MODEL_FACTORIES`; it then inherits the
  parametrized shape/bounds/param-count tests for free.
- `tests/test_scaffold.py` — add the two new config paths to the parse list.
- `progress_report.md` — append an entry (see §Ordered steps, step 7).

**Untouched:** everything in `recon/`, `eval/`, `classical/`, `viz/`, `synth/`, `train/`,
`configs/paradigm/`, `run_eval.py`, `run_recon.py`, `run_report.py`. Per the spec's Contract, if
implementation forces a change in any of these, **stop and report it** — that finding outranks the
architecture.

---

## Interfaces

```python
# models/attention_gate.py ─────────────────────────────────────────────────
class MultiScaleAttentionGate(nn.Module):
    """Additive attention gate on one skip connection, optionally conditioned on global context.

    gate = 2 * sigmoid(psi(relu(W_g(g) + W_x(x) [+ W_c(up(context))])))
    out  = x * gate

    With `psi` zero-initialized, gate == 1.0 exactly, so out == x bitwise (D3).
    """
    def __init__(self, *, spatial_dims: int = 3, f_g: int, f_x: int, f_int: int,
                 f_context: int | None = None, norm_name: tuple | str = "instance") -> None: ...
        # psi.weight and psi.bias are nn.init.zeros_ — the identity-at-init guarantee.
        # norm_name follows UNETR's "instance", NOT MONAI AttentionBlock's hardcoded BatchNorm
        # (batch stats over a batch of 2 volumes are noise, and eval-mode running stats would
        # differ from train mode — a silent train/eval skew).

    def forward(self, g: Tensor, x: Tensor, context: Tensor | None = None) -> Tensor: ...
        # context is trilinearly upsampled to x.shape[2:] before W_c. Raises if f_context was
        # declared and context is None, or vice versa — never silently ignores an argument.


# models/msa_unetr.py ──────────────────────────────────────────────────────
class MultiScaleAttentionUNETR(UNETRReconstruction):
    def __init__(self, *,
                 gate_reduction: int = 2,        # f_int = f_x // gate_reduction
                 multiscale_context: bool = True,# D5(b); False == plain AttUNet-style gating
                 gated_scales: tuple[int, ...] = (1, 2, 3, 4),   # which skips are gated
                 **unetr_kwargs) -> None: ...
        # super().__init__(**unetr_kwargs) builds every shared submodule with IDENTICAL attribute
        # names, so the shared state_dict keys are byte-identical to UNETRReconstruction's.
        # Adds ONLY: self.gate1 .. self.gate4 (present per `gated_scales`).

    GATE_PREFIXES: ClassVar[tuple[str, ...]] = ("gate1.", "gate2.", "gate3.", "gate4.")

    def forward(self, x: Tensor) -> Tensor:
        hidden, hidden_states = self.vit(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self._proj_feat(hidden_states[3]))
        enc3 = self.encoder3(self._proj_feat(hidden_states[6]))
        enc4 = self.encoder4(self._proj_feat(hidden_states[9]))
        dec4 = self._proj_feat(hidden)
        ctx  = dec4 if self.multiscale_context else None

        dec3 = self.decoder5(dec4, self._gate(4, g=dec4, x=enc4, context=ctx))
        dec2 = self.decoder4(dec3, self._gate(3, g=dec3, x=enc3, context=ctx))
        dec1 = self.decoder3(dec2, self._gate(2, g=dec2, x=enc2, context=ctx))
        out  = self.decoder2(dec1, self._gate(1, g=dec1, x=enc1, context=ctx))
        return self.sigmoid(self.out(out))
        # _gate(i, ...) returns `x` unchanged when scale i is not in gated_scales.

    def load_checkpoint(self, path) -> None:
        load_checked_state_dict(self, path)      # UNCHANGED, strict=True, its own checkpoint

    def load_from_unetr(self, path: Path | str) -> None:
        """Warm-start from a plain UNETR checkpoint; gates keep their identity init (D4).

        Raises CheckpointError unless the checkpoint's key set equals exactly this model's key
        set minus the gate keys. Never strict=False: the gate keys are explicitly enumerated and
        carried over from the current (identity) init, and every other key must match.
        """

    @property
    def model_card(self) -> ModelCard: ...       # name="msa_unetr"; known_characteristics states
                                                 # identity-at-init and the stretch/gated status.
```

```python
# scripts/run_train.py — the ONLY modified source file, one branch
if cfg.train.get("warm_start_from"):
    model = registry.get(cfg.train.save_as)                 # the GATED model
    model.load_from_unetr(registry.checkpoint_path(cfg.train.warm_start_from))
else:
    model = registry.get(cfg.train.init_from)               # existing path, unchanged
    model.load_checkpoint(registry.checkpoint_path(cfg.train.init_from))
```

`warm_start_from` and `init_from` are mutually exclusive — setting both raises `ConfigError`.

### `configs/model/msa_unetr.yaml`

```yaml
# Spec 012 (STRETCH). Multi-scale attention gates on UNETR's four skip connections.
# Identity at initialization: psi convs are zero-init and the gate is 2*sigmoid, so the gate
# evaluates to exactly 1.0 and the forward pass is bitwise equal to plain UNETR (test-enforced).
name: msa_unetr
target: mri_ad.models.msa_unetr.MultiScaleAttentionUNETR
params:
  # --- byte-identical to configs/model/unetr.yaml (a test asserts this) ---
  in_channels: 1
  img_size: [16, 128, 128]
  feature_size: 32
  hidden_size: 768
  mlp_dim: 3072
  num_heads: 12
  num_layers: 12
  patch_size: [16, 16, 16]
  norm_name: instance
  res_block: true
  conv_block: false
  dropout_rate: 0.0
  # --- Spec 012 additions ---
  gate_reduction: 2
  multiscale_context: true      # false => plain AttUNet-style gating (the free ablation, D5)
  gated_scales: [1, 2, 3, 4]
checkpoint: ${paths.checkpoint_root}/msa_unetr.pth
```

---

## Tests → acceptance mapping

| Spec acceptance | Test | What it proves |
|---|---|---|
| **1** registers, loads, forwards through the same interface | `test_msa_unetr.py::test_registers_from_yaml_alone` (build a registry over a tmp config dir, `registry.get("msa_unetr")`) · `::test_forward_preserves_shape` and `::test_output_bounded_to_unit_interval` inherited via `MODEL_FACTORIES` in `test_models.py` · `tests/test_model_boundary.py` (unchanged, now covers the new module) | The variant is a first-class registry citizen with **no downstream special-casing** — the Contract. |
| **2** appears as a row in the 005 table on the identical split/threshold | `test_msa_unetr.py::test_matrix_declares_msa_cell` — asserts `configs/matrix/arch_loss.yaml` contains the `msa_unetr__mse_ssim` cell and that it renders as `n/a` with its `na_reason` while no `aggregate.json` exists | The row is declared and honest **before** any number exists. The real cell is produced by the gated cluster run, scored by the unmodified `run_eval.py`, so split and threshold are identical by construction. |
| **3** documented deferral is acceptable | `progress_report.md` entry written in step 7 records the gate, its current state (closed), and the exact condition that would open it | The expected outcome is recorded, not implied by silence. |
| **4** a non-improvement is reported | `role: stretch` in the matrix cell; the 005 renderer already prints every declared cell | No path exists to drop the row. Whatever it scores, it prints. |

### Additional regression tests (the ones that stop a confidently wrong number)

- `::test_identity_at_init_matches_plain_unetr` — build a downsized `UNETRReconstruction` and a
  downsized `MultiScaleAttentionUNETR` from the **same seed**, copy the shared `state_dict`, assert
  `torch.equal(msa(x), unetr(x))`. **This is the load-bearing test**: it proves D3, proves the key
  partition in D4 is complete, and proves the gates are wired to the right tensors.
- `::test_gate_is_exactly_one_at_init` — `gate(g, x) == x` bitwise for random `g`, `x`.
- `::test_gate_learns_away_from_identity` — perturb `psi` weights; assert the output changes. Guards
  against a gate that is wired but dead (zero-init that can never receive gradient).
- `::test_load_from_unetr_rejects_extra_or_missing_keys` — a checkpoint with a renamed key raises
  `CheckpointError`; a checkpoint that *includes* gate keys also raises (that means it is an
  `msa_unetr` checkpoint and belongs in `load_checkpoint`).
- `::test_load_from_unetr_leaves_gates_at_identity` — after warm-start, `msa(x) == unetr(x)` still.
- `::test_own_checkpoint_round_trips_strictly` — save → `load_checkpoint` → `strict=True` clean.
- `::test_ablation_flag_changes_parameter_count` — `multiscale_context: false` drops the `W_c`
  params, proving the flag is real and not decorative.
- `::test_params_match_unetr_yaml` — the shared `params` keys are byte-identical to `unetr.yaml`
  (mirrors the existing `unetr_synth.yaml` test), so the two rows differ only by the gates.
- `::test_no_downstream_special_casing` — grep `src/mri_ad/{recon,eval,classical,viz}` for
  `msa_unetr`; assert zero hits. Mechanizes the Contract.

All tests use **downsized** models (`feature_size=8, hidden_size=96, num_heads=4`) and synthetic
tensors — no real data, no checkpoints, no full-resolution forward passes (laptop constraint).

---

## Ordered steps

**Everything below is local and CPU-only. No GPU is spent in any of it.**

1. `src/mri_ad/models/attention_gate.py` + gate-level tests (identity-at-init, learns-away,
   context-argument validation). Pure tensor code, fastest feedback.
2. `src/mri_ad/models/msa_unetr.py` — subclass, four gates, `forward`, `model_card`.
3. `::test_identity_at_init_matches_plain_unetr`. **Do not proceed until this is green** — every
   later claim rests on it.
4. `load_from_unetr` + key-partition and round-trip tests.
5. Config surface: `configs/model/msa_unetr.yaml`, `configs/train/msa_finetune.yaml`; add both to
   `tests/test_scaffold.py`; add the one matrix cell; add `_small_msa_unetr` to `MODEL_FACTORIES`.
6. `scripts/run_train.py` warm-start branch + the mutual-exclusion `ConfigError`.
7. `progress_report.md` entry: What / Why / **Problems hit** (the `strict=True` key-set conflict and
   how D3+D4 resolved it) / Result (architecture built and registered, GPU deferred) / the explicit
   gate condition / **Next:**.
8. **Hand off. The agent runs nothing below this line.**

---

## The evidence gate (step 8 → the cluster)

The user opens this gate manually. It is **closed** as of this plan, and closing it permanently is
an acceptable, spec-sanctioned outcome (acceptance 3).

**Open the gate only if all three hold:**

1. Specs 000–011 are complete and defensible, and 013 has landed (C-7 — 012 must never block them).
2. `artifacts/metrics/` contains real cells, i.e. `make eval` has actually run.
3. The 005 matrix shows the fidelity/detection anti-correlation is **scale-dependent** — e.g.
   `multiscale_mse` separates differently from `mse`/`ssim`, or the gap between UNETR and the
   UNet/AttUNet reference rows tracks receptive field. If 005 shows the anti-correlation is
   uniform across scales, **there is no hypothesis to test and the gate stays shut** — record that
   in `progress_report.md` and stop. That is a finding, not a failure.

**If the gate opens**, the user runs, in order:

```bash
make check-data                                                        # must exit 0
make train HYDRA_OVERRIDES="train=msa_finetune +experiment=cluster"    # /train ONLY
make recon HYDRA_OVERRIDES="model=msa_unetr +experiment=cluster"
make eval  HYDRA_OVERRIDES="model=msa_unetr loss=mse_ssim +experiment=cluster"
make matrix                                                            # the n/a becomes a number
```

Optional ablation, if and only if the gated model beats the baseline — it isolates D5(b), the actual
multi-scale claim, from plain gating:

```bash
make train HYDRA_OVERRIDES="train=msa_finetune model.params.multiscale_context=false \
                            train.save_as=msa_unetr_nc +experiment=cluster"
```

Per CLAUDE.md rule 6 and acceptance 4: if `msa_unetr` does not beat `unetr_synth`, the row is
published with the losing number and the architecture is dropped. No architecture is kept for its
novelty.

---

## Verification

Local, no GPU, no data, no checkpoints:

```bash
make lint
pytest tests/test_msa_unetr.py -v            # the new acceptance + regression suite
pytest tests/test_models.py tests/test_model_boundary.py tests/test_scaffold.py
make test                                    # full suite; expect the existing green count + new
make matrix                                  # msa_unetr renders as n/a with its na_reason — present, not omitted
make report                                  # no regression in any existing table
```

The single most informative check is
`pytest tests/test_msa_unetr.py::test_identity_at_init_matches_plain_unetr`. If it passes, the
architecture is correctly wired, the warm-start is sound, and any future training result is
attributable to the gates rather than to a broken initialization.

---

## Risks

| # | Risk | Caught by |
|---|---|---|
| R1 | Gates are wired to the wrong tensor pair (`g`/`x` swapped, or a skip gated with its own decoder input) → a plausible model that is not the intended architecture. | `test_identity_at_init_matches_plain_unetr` catches shape/wiring errors; a deliberate `psi` perturbation test asserts each gate actually influences the output. |
| R2 | Zero-init `psi` is a dead branch that never receives gradient → "attention didn't help" is really "attention never trained". | `test_gate_learns_away_from_identity`; `TrainSummary.param_l2_delta` (already recorded by `Trainer`) must be non-zero **on the gate parameters specifically** — assert this in the training hand-off. |
| R3 | Warm-start silently drops a non-gate key (the `strict=False`-by-another-name failure). | `load_from_unetr` asserts exact set equality `ckpt == self - gates` and raises `CheckpointError` naming the offending keys. |
| R4 | An `msa_unetr` checkpoint gets loaded through `load_from_unetr`, leaving trained gates reset to identity → the model scores exactly like plain UNETR and the run looks like a null result. | `test_load_from_unetr_rejects_extra_or_missing_keys` — a checkpoint containing gate keys raises rather than silently discarding them. |
| R5 | `msa_unetr` is compared against a `unetr` baseline trained on a different objective → the delta measures the objective, not the architecture. | D8 pins the FPI objective; the honest comparator is `unetr_synth__mse_ssim`, stated in the progress-report entry and in the matrix `role`. |
| R6 | The stretch spec quietly consumes the compute budget that 013 needs. | The evidence gate (three conditions) is explicit, and the agent never launches training (CLAUDE.md rule 4). |
| R7 | BatchNorm inside the gate (MONAI's default) creates a train/eval skew at batch size 2. | D3/`norm_name: instance` follows UNETR's convention; a test asserts no `BatchNorm` module exists anywhere in the model. |
| R8 | The variant leaks into the headline paradigm table and misrepresents the three-paradigm comparison. | D6 — `configs/paradigm/default.yaml` is untouched; `test_no_downstream_special_casing` greps for stray references. |
