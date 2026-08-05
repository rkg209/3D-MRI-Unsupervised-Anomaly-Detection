"""Gray-Level Run-Length Matrix (GLRLM) features — the one texture family skimage does not ship.

``skimage`` provides ``graycomatrix`` (GLCM) but no run-length equivalent, and PyRadiomics is the
fragile dependency Spec 006 permits substituting away from. This module is hand-rolled, isolated
from ``features.py`` on purpose, and every descriptor is cited so a reviewer can check the
arithmetic against the source formulas:

- Galloway, M. M. (1975). "Texture analysis using gray level run lengths." — SRE, LRE, GLN, RLN, RP.
- Chu, A., Sehgal, C. M., & Greenleaf, J. F. (1990). — LGRE, HGRE.
- Dasarathy, B. V., & Holder, E. B. (1991). — SRLGE, SRHGE, LRLGE, LRHGE.

Directions reduce to 1D lines: rows and columns of the quantized array, plus ``np.diagonal`` of
the array and of its ``fliplr`` for the two diagonal directions. Runs within a line come from
``np.flatnonzero(np.diff(line))`` — no Python loop over pixels, only over lines.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

GLRLM_FEATURE_NAMES: tuple[str, ...] = (
    "sre",
    "lre",
    "gln",
    "rln",
    "rp",
    "lgre",
    "hgre",
    "srlge",
    "srhge",
    "lrlge",
    "lrhge",
)


def _quantize(image: np.ndarray, *, levels: int) -> np.ndarray:
    """Min-max-normalize ``image`` then bin into ``levels`` integer gray levels ``[0, levels)``."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.int64)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax - vmin < 1e-12:  # constant slice -> a single gray level, never a division by zero
        return np.zeros(arr.shape, dtype=np.int64)
    scaled = (arr - vmin) / (vmax - vmin)
    quantized = np.round(scaled * (levels - 1)).astype(np.int64)
    return np.clip(quantized, 0, levels - 1)


def _lines_for_angle(quantized: np.ndarray, angle_deg: float) -> list[np.ndarray]:
    """Return the 1D lines ``quantized`` decomposes into for one of the four cardinal directions."""
    angle = int(round(angle_deg)) % 180
    if angle == 0:  # horizontal -> rows
        return [quantized[r, :] for r in range(quantized.shape[0])]
    if angle == 90:  # vertical -> columns
        return [quantized[:, c] for c in range(quantized.shape[1])]
    if angle == 45:  # main-diagonal direction
        n, m = quantized.shape
        return [quantized.diagonal(offset) for offset in range(-(n - 1), m)]
    if angle == 135:  # anti-diagonal direction: diagonals of the left-right flip
        flipped = np.fliplr(quantized)
        n, m = flipped.shape
        return [flipped.diagonal(offset) for offset in range(-(n - 1), m)]
    raise ValueError(f"Unsupported GLRLM angle {angle_deg} (expected one of 0/45/90/135).")


def _runs_in_line(line: np.ndarray) -> list[tuple[int, int]]:
    """Decompose one 1D line into ``(gray_level, run_length)`` pairs via a single diff pass."""
    if line.size == 0:
        return []
    change_idx = np.flatnonzero(np.diff(line)) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [line.size]))
    return [(int(line[s]), int(e - s)) for s, e in zip(starts, ends, strict=True)]


def run_length_matrix(quantized: np.ndarray, *, levels: int, angle_deg: float) -> np.ndarray:
    """Build the ``(levels, max_run_length)`` GLRLM ``P(i, j)`` for one direction.

    ``P[i, j]`` counts runs of gray level ``i`` and length ``j + 1`` (columns are 0-indexed run
    lengths, so a length-1 run lands in column 0). ``quantized`` must already be integer-binned
    into ``[0, levels)`` — see :func:`run_length_features` for the quantization step.
    """
    lines = _lines_for_angle(quantized, angle_deg)
    max_run = max((line.size for line in lines), default=0)
    max_run = max(max_run, 1)
    matrix = np.zeros((levels, max_run), dtype=np.float64)
    for line in lines:
        for value, length in _runs_in_line(line):
            matrix[value, length - 1] += 1.0
    return matrix


def glrlm_features(matrix: np.ndarray) -> dict[str, float]:
    """The 11 Galloway/Chu/Dasarathy-Holder descriptors of one run-length matrix.

    Every denominator (``Nr`` = total run count, ``Np`` = total pixel count implied by the
    matrix, and the 1-indexed gray-level axis so level 0 never divides by zero) is guarded: an
    empty matrix (``Nr == 0``, e.g. a zero-size slice) returns all-zero features, never NaN.
    """
    total_runs = float(matrix.sum())
    if total_runs == 0.0:
        return dict.fromkeys(GLRLM_FEATURE_NAMES, 0.0)

    levels, max_run = matrix.shape
    gray = np.arange(1, levels + 1, dtype=np.float64).reshape(-1, 1)  # 1-indexed, avoids i=0
    run_len = np.arange(1, max_run + 1, dtype=np.float64).reshape(1, -1)
    total_pixels = float((matrix * run_len).sum())

    sre = float((matrix / run_len**2).sum() / total_runs)
    lre = float((matrix * run_len**2).sum() / total_runs)
    gln = float(((matrix.sum(axis=1)) ** 2).sum() / total_runs)
    rln = float(((matrix.sum(axis=0)) ** 2).sum() / total_runs)
    rp = float(total_runs / total_pixels) if total_pixels > 0 else 0.0
    lgre = float((matrix / gray**2).sum() / total_runs)
    hgre = float((matrix * gray**2).sum() / total_runs)
    srlge = float((matrix / (gray**2 * run_len**2)).sum() / total_runs)
    srhge = float((matrix * gray**2 / run_len**2).sum() / total_runs)
    lrlge = float((matrix * run_len**2 / gray**2).sum() / total_runs)
    lrhge = float((matrix * gray**2 * run_len**2).sum() / total_runs)

    return {
        "sre": sre,
        "lre": lre,
        "gln": gln,
        "rln": rln,
        "rp": rp,
        "lgre": lgre,
        "hgre": hgre,
        "srlge": srlge,
        "srhge": srhge,
        "lrlge": lrlge,
        "lrhge": lrhge,
    }


def run_length_features(
    image: np.ndarray, *, levels: int, angles_deg: Sequence[float]
) -> dict[str, float]:
    """Quantize ``image`` to ``levels`` gray levels and average the 11 descriptors over angles."""
    quantized = _quantize(image, levels=levels)
    per_angle = [
        glrlm_features(run_length_matrix(quantized, levels=levels, angle_deg=angle))
        for angle in angles_deg
    ]
    return {
        name: float(np.mean([features[name] for features in per_angle]))
        for name in GLRLM_FEATURE_NAMES
    }


__all__ = ["GLRLM_FEATURE_NAMES", "glrlm_features", "run_length_features", "run_length_matrix"]
