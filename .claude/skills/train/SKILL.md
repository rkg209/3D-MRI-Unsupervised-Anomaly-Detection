---
name: train
description: Launch or fine-tune a model (e.g. the Spec 009 synthetic-anomaly fine-tune from the UNETR checkpoint, or the Spec 013 from-scratch diffusion model). MANUAL INVOKE ONLY — this spends GPU time and must never be triggered autonomously.
disable-model-invocation: true
---

# /train — MANUAL ONLY. This spends GPU time.

**Constraint C-3: the agent must never autonomously launch a training run, a sweep, or any run
that costs money.** This skill carries `disable-model-invocation: true`, so Claude cannot trigger
it — only the user can. If training seems warranted, **propose it and stop.** Do not run it.

Two specs need fresh compute (D8): **009** (synthetic-anomaly fine-tune, `make train`) and **013**
(diffusion/AnoDDPM, trained from scratch). Everything else in the project builds a harness around
already-saved outputs.

## Preflight — every box, before spending a single GPU-hour

- [ ] `make check-data` exits 0. Weights are real, not LFS stubs.
- [ ] The spec is approved and its plan is written.
- [ ] `seed_everything()` runs first in the entry point.
- [ ] Config is complete: no magic numbers — Spec 009's FPI params live in
      `configs/synth/fpi.yaml`, its schedule in `configs/train/finetune.yaml`.
- [ ] **The test split is excluded from training.** Verified against `SplitContract`
      (`resolve_training_ids` for Spec 009), not assumed.
- [ ] The "before" number is the **corrected Spec 004 baseline** — not the prior report's 0.6255,
      which came from buggy code. Comparing against it would make any improvement fictional.
- [ ] The git tree is clean, so the logged SHA describes the code that actually ran.

## Run

```bash
# Spec 009 — synthetic-anomaly fine-tune (needs configs/model/unetr.yaml's checkpoint first)
make train HYDRA_OVERRIDES="+experiment=cluster"
```

`run_train.py` refuses to start on a non-CUDA device unless `train.allow_cpu=true` is set
explicitly (a guard against an accidental laptop launch), and runs a real-data pre-flight
separability check (`configs/synth/fpi.yaml:check`) *before the first optimizer step* — it aborts
if FPI corruption turns out to be trivially separable by a global intensity threshold, so no
GPU-hours are spent training on a construction that was never going to produce a meaningful
before/after comparison.

Logs to `artifacts/runs/<timestamp>/run_meta.json`: git SHA, resolved config, seed, all FPI
generation parameters, wall-clock, and the resulting metrics. The checkpoint itself
(`checkpoints/unetr_synth.pth`) separately records the selected epoch, the selection metric/value,
and `param_l2_delta` (proof training actually moved the weights).

## After

1. `make eval` on the **identical** test split, **identical** threshold strategy, **identical**
   preprocessing. Change one thing at a time or you have measured nothing.
2. `make synth-table` to render the before/after Dice/IoU table
   (`eval/before_after.py` — hard-fails rather than rendering an uncontrolled comparison if the
   split hash, threshold, or preprocessing drifted between the two eval runs).
3. **Append to `progress_report.md`**: what was trained, why, the hyperparameters, the result —
   and the result even if it is worse. A negative finding is a finding (D2).
