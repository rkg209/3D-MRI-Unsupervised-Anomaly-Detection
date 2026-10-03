---
name: run-experiment
description: Standardized experiment run — config in, metrics and artifacts out, fully logged. MANUAL INVOKE ONLY because it uses the GPU.
disable-model-invocation: true
---

# /run-experiment — MANUAL ONLY. This uses the GPU.

`disable-model-invocation: true` — Claude cannot trigger this. Only the user can (C-3).
If an experiment seems warranted, **propose it and stop.**

Use for anything that runs inference over the split (a full `make eval`, a threshold sweep, a new
architecture × loss cell). For fine-tuning, use `/train`.

## Contract

**Config in → metrics + artifacts out, fully logged.** Nothing about the run is implicit.

```bash
make eval HYDRA_OVERRIDES="model=unetr loss=mse_ssim +experiment=cluster"
```

## Preflight

- [ ] `make check-data` exits 0.
- [ ] `seed_everything()` runs first.
- [ ] Clean git tree — the logged SHA must describe the code that ran.
- [ ] Threshold operating point was chosen on **validation**. Tuning on test is leakage and is the
      easiest way to publish a fake result.

## Every run writes

`artifacts/runs/<timestamp>-<model>-<loss>/`
- `run_meta.json` — git SHA, resolved config, seed, hostname, timing
- `metrics/per_volume.csv`, `metrics/aggregate.json`
- `ReconResult` files under `artifacts/results/recon/<run_id>/` — **never a flat directory**, or
  two models' results collide on `volume_id` and silently overwrite each other.

## After

Report the metrics and the artifact paths. **Append to `progress_report.md`.** If the number moved,
say which change moved it — and if you cannot attribute it, say that too rather than guessing.
