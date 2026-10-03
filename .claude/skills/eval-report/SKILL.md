---
name: eval-report
description: Regenerate all metrics tables and plots from SAVED outputs — no training, no inference, no GPU. Cheap, deterministic, re-runnable. Use whenever results need refreshing after new artifacts land.
---

# /eval-report — regenerate results from saved artifacts

Reads `artifacts/` and regenerates every table, plot, and the README scorecard.
**No model is loaded. No `forward()` is called. No GPU. Under 2 minutes.**

```bash
make report
```

## The boundary this enforces

Inference and evaluation are separate stages. The reconstruction engine writes `ReconResult`
objects to `artifacts/results/recon/<run_id>/`; the evaluation harness reads **only those files**.
`AggregateEvaluator.evaluate_split()` takes a `Path` and raises `TypeError` if handed a model —
that guard exists to stop accidental re-inference during report generation.

So this command is always cheap and always safe to re-run.

## Outputs

- `artifacts/metrics/per_volume.csv`, `aggregate.json`
- `artifacts/tables/arch_loss_matrix.{csv,md}`, `dl_vs_classical.md`
- `artifacts/figures/` — ROC/PR curves, Dice distributions, the threshold sweep
- The README scorecard, rewritten between `<!-- SCORECARD_START -->` / `<!-- SCORECARD_END -->`

## Rules

- **Every number traces to an artifact.** Nothing is hardcoded in the table generator.
- **Unmeasured cells render `n/a (<reason>)`** — never blank, never estimated, never dropped.
  The perceptual-loss column has no training code and no checkpoint; it stays `n/a`.
- **PSNR/SSIM are labelled explanatory context**, never presented as performance. Higher fidelity
  means *worse* detection here — presenting PSNR as a win inverts the project's central finding.
- If an artifact is missing, fail loudly with the path. Never silently skip a row.
