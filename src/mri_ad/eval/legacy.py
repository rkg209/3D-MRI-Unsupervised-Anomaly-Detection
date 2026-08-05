"""Bug-compatible reproduction of ``legacy/metric-uad.ipynb`` — Spec 004 acceptance test 6.

**Quarantined on purpose.** This is the single module in ``eval/`` allowed to import
``mri_ad.models`` / touch a real forward pass (``tests/test_eval_boundary.py`` allowlists it by
name). Every other module in this package must stay model-free (acceptance test 4).

The point of this module is not to reproduce a good result — it is to reproduce a *known-bad*
one, on purpose, so we can point at exactly which four independent bugs produced the prior
work's published UNETR headline (Dice 0.6255, IoU 0.4551) and how each was fixed. See
``specs/004-eval-harness.md``'s "Notes / deviations" section for the full audit.

**Never publish a number produced by this module.** It deliberately does *not* reuse
``mri_ad.data.transforms`` — reusing the corrected preprocessing pipeline would defeat the
entire point of a bug-compatibility check. The duplication below is intentional and confined to
this one file.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as func
from torch import Tensor

from mri_ad.eval.metrics import LegacyMetricsComputer
from mri_ad.exceptions import DataError, EvalError
from mri_ad.models.base import AnomalyDetectionModel
from mri_ad.recon.threshold import AbsoluteThreshold

LEGACY_PUBLISHED_DICE: float = 0.6255
LEGACY_PUBLISHED_IOU: float = 0.4551


@dataclass(frozen=True)
class LegacyCompatConfig:
    """Parameters of the prior pipeline.

    Defaults match the real reproduction; tests may override ``crop_size``/``side``/
    ``chunk_depth`` to exercise the same bugs on smaller synthetic data.
    """

    subject_count: int = 100  # matches legacy/metric-uad.ipynb cell 14's `subject_count = 100`
    chunk_index: int = 4
    chunk_depth: int = 16
    threshold: float = 0.1
    crop_size: tuple[int, int, int] = (160, 130, 170)
    side: int = 128
    sort_subjects: bool = True  # see R1: os.listdir ordering is filesystem-dependent otherwise


@dataclass(frozen=True)
class LegacyCompatResult:
    """The bug-compat score, plus enough provenance to show *which* bug produced it."""

    dice: float
    iou: float
    n_subjects_visited: int
    n_scores_accumulated: int  # == 1: the re-initialized-list bug (#1), made visible
    last_volume_id: str
    published_dice: float = LEGACY_PUBLISHED_DICE
    published_iou: float = LEGACY_PUBLISHED_IOU


def _legacy_crop_starts(
    crop_size: tuple[int, int, int], height: int, width: int
) -> tuple[int, int]:
    """Centre-crop start indices for the H/W-only crop (depth is deliberately never cropped)."""
    _, crop_h, crop_w = crop_size
    return (height - crop_h) // 2, (width - crop_w) // 2


def legacy_preprocess(
    t2: np.ndarray, seg: np.ndarray, cfg: LegacyCompatConfig
) -> tuple[Tensor, Tensor]:
    """Reproduce the prior pipeline's crop/resize/chunk-selection, bug for bug.

    Returns ``(image_chunk, label_chunk)``, each ``(1, 1, chunk_depth, side, side)`` — the label
    chunk is **fractional** (never binarized), which is bug #2/#3 made visible downstream.
    """
    # BUG-COMPAT #10: min-max the RAW T2 before transpose, no eps (the corrected pipeline
    # normalizes LAST, after resize — see data/transforms.py::_minmax).
    t2 = t2.astype(np.float32)
    vmin, vmax = float(t2.min()), float(t2.max())
    denom = vmax - vmin
    image = (t2 - vmin) / denom if denom != 0 else np.zeros_like(t2)
    label = seg.astype(np.float32)

    # BUG-COMPAT: native BraTS (H, W, D) -> (D, H, W), same axis order as the corrected pipeline.
    image = np.transpose(image, (2, 0, 1))
    label = np.transpose(label, (2, 0, 1))

    depth, height, width = image.shape
    start_h, start_w = _legacy_crop_starts(cfg.crop_size, height, width)
    _, crop_h, crop_w = cfg.crop_size
    # BUG-COMPAT #5: start_d is computed and deliberately UNUSED — depth is never cropped.
    start_d = (depth - cfg.crop_size[0]) // 2
    del start_d
    image = image[:, start_h : start_h + crop_h, start_w : start_w + crop_w]
    label = label[:, start_h : start_h + crop_h, start_w : start_w + crop_w]

    # BUG-COMPAT #2/#3: trilinear-resize the SEG too — raw {0,1,2,4} labels never binarized,
    # producing fractional "labels".
    image_t = torch.from_numpy(np.ascontiguousarray(image)).float()[None, None]
    label_t = torch.from_numpy(np.ascontiguousarray(label)).float()[None, None]
    resized_depth = image_t.shape[2]
    image_t = func.interpolate(
        image_t, size=(resized_depth, cfg.side, cfg.side), mode="trilinear", align_corners=False
    )
    label_t = func.interpolate(
        label_t, size=(resized_depth, cfg.side, cfg.side), mode="trilinear", align_corners=False
    )

    # BUG-COMPAT #4: range(8) x chunk_depth slices -> only the first 8*chunk_depth of depth is
    # ever reachable, and only `chunk_index` is scored.
    start = cfg.chunk_index * cfg.chunk_depth
    end = start + cfg.chunk_depth
    return image_t[:, :, start:end], label_t[:, :, start:end]


def _find_modality(subject_dir: Path, volume_id: str, suffix: str) -> Path:
    for ext in (".nii.gz", ".nii"):
        candidate = subject_dir / f"{volume_id}_{suffix}{ext}"
        if candidate.is_file():
            return candidate
    raise DataError(f"Missing {suffix!r} volume for subject {volume_id}.")


class LegacyCompatEvaluator:
    """Runs the bug-compat pipeline end to end. The only ``eval/`` class that touches a model."""

    def __init__(
        self, model: AnomalyDetectionModel, config: LegacyCompatConfig, device: torch.device
    ) -> None:
        """Store the (already-loaded, already-``.eval()``'d) model, config, and device."""
        self.model = model
        self.config = config
        self.device = device

    def evaluate_subject(self, t2: np.ndarray, seg: np.ndarray) -> tuple[float, float]:
        """Preprocess, reconstruct, threshold, and score one subject. Returns ``(dice, iou)``."""
        if self.model.training:
            raise EvalError(
                "LegacyCompatEvaluator.evaluate_subject requires the model in eval mode "
                "(dropout/BN would make the bug-compat reproduction nondeterministic), mirroring "
                "ReconstructionEngine.run's guard."
            )
        image_chunk, label_chunk = legacy_preprocess(t2, seg, self.config)
        image_chunk = image_chunk.to(self.device)
        with torch.no_grad():
            out = self.model(image_chunk)
        out = out.detach().cpu()
        image_chunk = image_chunk.cpu()

        # BUG-COMPAT #6: re-min-max the model OUTPUT, no eps.
        vmin, vmax = float(out.min()), float(out.max())
        denom = vmax - vmin
        out_norm = (out - vmin) / denom if denom != 0 else torch.zeros_like(out)

        diff = (image_chunk - out_norm).abs()
        # BUG-COMPAT #7: the prior work's hardcoded absolute cutoff of 0.1 on the diff map,
        # routed through recon/ so thresholding still only ever happens in one place.
        mask = AbsoluteThreshold(self.config.threshold)(diff)

        dice = LegacyMetricsComputer.dice_bugcompat(mask, label_chunk)
        iou = LegacyMetricsComputer.iou_bugcompat(mask, label_chunk)
        return dice, iou

    def evaluate_directory(self, brats_dir: Path) -> LegacyCompatResult:
        """Score up to ``config.subject_count`` subjects.

        Reproduces bug #1: the score lists are re-initialized INSIDE the loop, so the returned
        mean is the **last subject only**.
        """
        subjects = [p.name for p in Path(brats_dir).iterdir() if p.is_dir()]
        if self.config.sort_subjects:
            subjects = sorted(subjects)
        subjects = subjects[: self.config.subject_count]
        if not subjects:
            raise DataError(f"No BraTS subjects found under {brats_dir}.")

        dice_scores: list[float] = []
        iou_scores: list[float] = []
        n_visited = 0
        last_volume_id = ""
        for volume_id in subjects:
            # BUG-COMPAT #1: re-initialized every iteration -> only the last subject survives.
            dice_scores = []
            iou_scores = []

            t2_path = _find_modality(Path(brats_dir) / volume_id, volume_id, "t2")
            seg_path = _find_modality(Path(brats_dir) / volume_id, volume_id, "seg")
            t2 = np.asarray(nib.load(str(t2_path)).get_fdata(), dtype=np.float32)
            seg = np.asarray(nib.load(str(seg_path)).get_fdata(), dtype=np.float32)

            dice, iou = self.evaluate_subject(t2, seg)
            dice_scores.append(dice)
            iou_scores.append(iou)
            n_visited += 1
            last_volume_id = volume_id

        return LegacyCompatResult(
            dice=statistics.fmean(dice_scores),
            iou=statistics.fmean(iou_scores),
            n_subjects_visited=n_visited,
            n_scores_accumulated=len(dice_scores),
            last_volume_id=last_volume_id,
        )


__all__ = [
    "LEGACY_PUBLISHED_DICE",
    "LEGACY_PUBLISHED_IOU",
    "LegacyCompatConfig",
    "LegacyCompatEvaluator",
    "LegacyCompatResult",
    "legacy_preprocess",
]
