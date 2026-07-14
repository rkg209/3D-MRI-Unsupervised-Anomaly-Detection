---
name: train
description: Launch or fine-tune a model (e.g. the Spec 009 synthetic-anomaly fine-tune from the UNETR checkpoint). MANUAL INVOKE ONLY — this spends GPU time and must never be triggered autonomously.
disable-model-invocation: true
---

# /train — MANUAL ONLY. This spends GPU time.

**Constraint C-3: the agent must never autonomously launch a training run, a sweep, or any run
that costs money.** This skill carries `disable-model-invocation: true`, so Claude cannot trigger
it — only the user can. If training seems warranted, **propose it and stop.** Do not run it.

Spec 009 (synthetic-anomaly fine-tune) is the **only** spec that needs fresh compute.

## Preflight — every box, before spending a single GPU-hour

- [ ] `make check-data` exits 0. Weights are real, not LFS stubs.
- [ ] The spec is approved and its plan is written.
- [ ] `seed_everything()` runs first in the entry point.
- [ ] Config is complete: no magic numbers, all synth params in `configs/synth/fpi.yaml`.
- [ ] **The test split is excluded from training.** Verified against `SplitContract`, not assumed.
- [ ] The "before" number is the **corrected Spec 004 baseline** — not the prior report's 0.6255,
      which came from buggy code. Comparing against it would make any improvement fictional.
- [ ] The git tree is clean, so the logged SHA describes the code that actually ran.

## Run

```bash
make train HYDRA_OVERRIDES="+experiment=cluster"
```

Logs to `artifacts/runs/<timestamp>/run_meta.json`: git SHA, resolved config, seed, all FPI
generation parameters, wall-clock, and the resulting metrics.

## After

1. `make eval` on the **identical** test split, **identical** threshold strategy, **identical**
   preprocessing. Change one thing at a time or you have measured nothing.
2. Produce the before/after Dice/IoU table.
3. **Append to `progress_report.md`**: what was trained, why, the hyperparameters, the result —
   and the result even if it is worse. A negative finding is a finding (D2).
