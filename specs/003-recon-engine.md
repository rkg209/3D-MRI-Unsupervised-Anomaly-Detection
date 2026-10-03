# Spec 003 · Reconstruction & anomaly-map engine

**Status:** implemented
**Depends on:** 001, 002
**Fresh compute required:** no

---

## Problem

This is the heart of the method: model + volume → reconstruction → residual → binary anomaly
mask. The thresholding step is the single highest-leverage choice in the pipeline — it converts a
continuous residual into the mask that Dice is computed against, so it materially determines every
headline number. In the prior work it was a magic constant (`residual > 0.1`) buried in a notebook
cell. This spec centralizes it, makes it configurable, and requires it to be *justified by a sweep*
rather than asserted.

## Contract

```python
ReconstructionEngine(model: AnomalyDetectionModel, config: DictConfig)
    .run(volume: Tensor) -> ReconResult
```

`ReconResult` (scaffolded in `src/mri_ad/recon/types.py`): `original`, `reconstruction`,
`residual = abs(original - reconstruction)`, `anomaly_mask` (binary) — all `(1,16,128,128)`,
float32, CPU.

`ThresholdStrategy` implementations: `FixedPercentileThreshold`, `AbsoluteThreshold`,
`OtsuThreshold`. Injected via `configs/threshold/`.

**Thresholding happens here and nowhere else.** No other module may binarize a residual.

The engine persists `ReconResult` to `artifacts/results/recon/<run_id>/<volume_id>.pt`. The
`run_id` subdirectory is mandatory: a flat directory would let two models' results collide on
`volume_id` and silently overwrite each other.

## Acceptance tests

1. The engine emits all four tensors, correctly shaped, for any registered model.
2. `residual` is non-negative everywhere and `anomaly_mask` contains only `{0.0, 1.0}`.
3. `FixedPercentileThreshold(p)` flags exactly `(100-p)%` of voxels, ±1 voxel, by construction.
4. Swapping the threshold strategy requires **only** a config change — no code change.
5. **Threshold sweep:** the engine produces a Dice-vs-threshold curve over
   `configs/threshold/percentile.yaml:sweep` (percentiles 90→99 and absolute 0.05→0.2, which
   includes the prior work's 0.1). The chosen operating point is the sweep's argmax on the
   **validation** split, never the test split.
6. Two models evaluated in sequence do not overwrite each other's saved `ReconResult` files.
7. `ReconResult` round-trips through `torch.save`/`load` with bit-identical tensors.

## Out of scope

Metrics (004) — the engine produces the mask; it does not score it. Training (009).

## Notes / deviations

**Deviation from `planning/02-architecture.md:183`,** which sets the default to
`FixedPercentileThreshold` at the 95th percentile and claims this is "matching the prior work".
**It does not match the prior work.** The prior work used a hardcoded global *absolute* threshold,
`residual > 0.1`. These are different strategies with different behaviour:

- An **absolute** threshold lets the number of flagged voxels vary with how badly the model
  reconstructs a given volume — which is the actual anomaly signal.
- A **percentile** threshold *fixes* the flagged-voxel count by construction: at p95, exactly 5%
  of voxels are always flagged, no matter how healthy or how diseased the brain is.

That is a substantive change to what Dice measures. A percentile threshold may well be the better
choice — but it must be **selected by the sweep in acceptance test 5 and justified**, not adopted
because a planning doc mislabelled it as a reproduction.

Tuning the threshold on the test split would be leakage. Acceptance test 5 requires the operating
point be chosen on validation.
