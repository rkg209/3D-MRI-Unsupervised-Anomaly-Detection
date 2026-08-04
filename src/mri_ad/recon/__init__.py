"""Reconstruction and anomaly-map engine; the only place thresholding happens (Spec 003)."""

from __future__ import annotations

from mri_ad.recon.engine import ReconstructionEngine
from mri_ad.recon.io import load_result, result_path, save_result
from mri_ad.recon.sweep import SweepPoint, run_threshold_sweep, select_operating_point
from mri_ad.recon.threshold import AbsoluteThreshold, FixedPercentileThreshold, OtsuThreshold
from mri_ad.recon.types import ReconResult, ThresholdStrategy

__all__ = [
    "AbsoluteThreshold",
    "FixedPercentileThreshold",
    "OtsuThreshold",
    "ReconResult",
    "ReconstructionEngine",
    "SweepPoint",
    "ThresholdStrategy",
    "load_result",
    "result_path",
    "run_threshold_sweep",
    "save_result",
    "select_operating_point",
]
