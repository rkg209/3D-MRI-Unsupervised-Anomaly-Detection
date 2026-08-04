"""Threshold sweep: score every configured operating point on saved ``ReconResult``s (Spec 003).

Pure and model-free — it takes an iterable of :class:`~mri_ad.recon.types.ReconResult` (each
carrying ``residual`` + ``ground_truth``) and re-thresholds them. One expensive forward pass over
the validation split (done once, upstream, by the engine), then every sweep point is scored
offline. That is also what makes the whole sweep unit-testable on synthetic tensors.

Leakage guard: :func:`run_threshold_sweep` refuses any split but ``"val"`` — trap #5 (never tune
a threshold on the test split) enforced at the function boundary, not just by convention in the
calling script.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from omegaconf import DictConfig

from mri_ad.eval.metrics import MetricsComputer
from mri_ad.exceptions import ReconError
from mri_ad.recon.threshold import AbsoluteThreshold, FixedPercentileThreshold
from mri_ad.recon.types import ReconResult, ThresholdStrategy


@dataclass(frozen=True)
class SweepPoint:
    """One scored operating point: a threshold strategy + parameter, and its val-split Dice."""

    strategy: str  # "percentile" | "absolute"
    param: str  # "percentile" | "value"
    value: float | None
    dice_mean: float
    dice_std: float
    n_volumes: int
    split: str  # always "val"


def build_sweep_strategies(
    sweep_cfg: DictConfig,
) -> list[tuple[tuple[str, str, float], ThresholdStrategy]]:
    """Build every ``(strategy, param, value)`` key + its strategy instance from ``sweep_cfg``.

    ``sweep_cfg`` mirrors ``configs/threshold/percentile.yaml:sweep`` — ``percentiles: [...]``
    and ``absolute: [...]``.
    """
    strategies: list[tuple[tuple[str, str, float], ThresholdStrategy]] = []
    for p in sweep_cfg.get("percentiles", []):
        strategies.append((("percentile", "percentile", float(p)), FixedPercentileThreshold(p)))
    for v in sweep_cfg.get("absolute", []):
        strategies.append((("absolute", "value", float(v)), AbsoluteThreshold(v)))
    return strategies


def run_threshold_sweep(
    results: list[ReconResult],
    sweep_cfg: DictConfig,
    *,
    split: str,
) -> list[SweepPoint]:
    """Score every sweep point's Dice across ``results``. ``split`` must be ``"val"``."""
    if split != "val":
        raise ReconError(
            f"run_threshold_sweep refuses split={split!r}: the threshold must be tuned on "
            "'val' only, never 'test' (trap #5 — never tune a threshold on the test split)."
        )
    if not results:
        raise ReconError("run_threshold_sweep received an empty result list.")

    points: list[SweepPoint] = []
    for (strategy, param, value), threshold in build_sweep_strategies(sweep_cfg):
        dices = []
        for result in results:
            if result.ground_truth is None:
                raise ReconError(
                    f"ReconResult {result.volume_id!r} has no ground_truth; the sweep requires "
                    "BraTS results, not OpenBHB."
                )
            mask = threshold(result.residual)
            dices.append(MetricsComputer.dice(mask, result.ground_truth))
        points.append(
            SweepPoint(
                strategy=strategy,
                param=param,
                value=value,
                dice_mean=statistics.fmean(dices),
                dice_std=statistics.pstdev(dices) if len(dices) > 1 else 0.0,
                n_volumes=len(results),
                split=split,
            )
        )
    return points


def select_operating_point(points: list[SweepPoint]) -> SweepPoint:
    """Return the argmax-Dice point; ties broken by the lowest ``value``."""
    if not points:
        raise ReconError("select_operating_point received an empty point list.")
    return max(points, key=lambda p: (p.dice_mean, -(p.value if p.value is not None else 0.0)))


__all__ = ["SweepPoint", "build_sweep_strategies", "run_threshold_sweep", "select_operating_point"]
