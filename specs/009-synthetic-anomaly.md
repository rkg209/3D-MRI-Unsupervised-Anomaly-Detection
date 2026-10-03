# Spec 009 · Synthetic-anomaly (anomaly-informed) training

**Status:** implemented
**Depends on:** 002, 003, 004
**Fresh compute required:** **YES — the only spec that needs it. `/train` is manual-invoke only.**

---

## Problem

This attacks the central failure mode head-on. A model trained only to reconstruct healthy brains
learns to reconstruct *anything* well, tumors included — so the residual at the tumor is small and
detection fails. If instead we corrupt healthy volumes with synthetic lesions and train the model
to **restore the healthy version**, it learns to *erase* anomalous-looking structure rather than
copy it. This is the project's one performance novelty and its before/after story.

## Contract

**Generation: Foreign Patch Interpolation (FPI).** Blend a patch from a donor healthy volume into
the target at a random location with random interpolation weight — spatially coherent but
intensity-inconsistent with surrounding tissue. Chosen over Poisson blending (needs boundary
computation, slower) and 3D CutPaste (sharp edges are trivially detectable, so the model learns the
edge rather than the anomaly).

```python
FPIAnomalyGenerator.generate(volume, donor) -> SynthResult(corrupted, healthy, synth_mask)
AnomalyInformedDataset  # wraps OpenBHB, applies FPI on the fly
```

Config `configs/synth/fpi.yaml`: `patch_size_range: [8,32]`, `n_patches_range: [1,4]`,
`alpha_range: [0.3,0.7]`, seed inherited.

**Objective:** input `corrupted`, target `healthy`. Fine-tune from the existing UNETR checkpoint.

## Acceptance tests

1. FPI produces `corrupted != healthy` exactly within `synth_mask` and **bit-identical outside it**
   — a test asserts the untouched region is unchanged.
2. Synthetic lesions are **not trivially separable**: a threshold on raw intensity alone must not
   recover `synth_mask` at high Dice. If it can, the task is too easy and the model will learn a
   shortcut rather than anatomy.
3. All generation parameters are logged in `run_meta.json` for every training run (FR-36).
4. **The before/after table.** Dice and IoU on the **identical** BraTS test split, before and after
   fine-tuning, same threshold strategy, same preprocessing. Anything else is not a controlled
   comparison.
5. The test split is **never** seen during fine-tuning — asserted against `SplitContract`.
6. Training is reachable **only** via `/train` or `make train`. The agent never launches it.
7. If Dice does **not** improve, that is reported as the result. A negative outcome is a finding.

## Out of scope

Any architecture change (that is 012). Training a model from scratch — we fine-tune. **The diffusion
model is a separate track (Spec 013): FPI/synthetic-anomaly here applies to UNETR only.** Fine-tuning
diffusion on FPI-corrupted data is a possible future extension, explicitly out of scope for both
specs.

## Notes / deviations

- Acceptance test 2 is an addition to `FR-33`/`FR-34`. Without it, FPI can degenerate into
  producing obvious artifacts that the model detects via a shortcut, giving a Dice improvement on
  *synthetic* data that does not transfer to real tumors. The whole point is transfer.
- **Budget guardrail (C-3).** This is the only fresh-compute spec. The agent must never launch
  training, a sweep, or any paid run autonomously. `/train` carries
  `disable-model-invocation: true`.
- The corrected Spec 004 baseline — not the prior report's 0.6255 — is the "before" number.
  Comparing against a number produced by different, buggy code would make the improvement fictional.
