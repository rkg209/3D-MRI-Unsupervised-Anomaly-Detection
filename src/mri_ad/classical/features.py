"""Slice-level radiomic/texture features (Spec 006) — 44 features/slice, scikit-image only.

**PyRadiomics substitution.** PyRadiomics is fragile to build on some platforms; Spec 006 permits
falling back to a ``scikit-image``-only feature set and documenting the substitution rather than
hiding it. That substitution is recorded here (``features.library == "scikit-image"``), in the
config, and stamped into every generated JSON artifact as ``pyradiomics_used: false``.

:func:`feature_names` is the single source of truth for column order: it is stamped into the
cache header, the metrics JSON, and the feature-importance CSV, so a mismatch anywhere is
detectable rather than a silent column-shuffle.

Five families, 44 features by default (``include_slice_index=False``):

- **First-order (15):** mean, std, min, max, range, skew, kurtosis, p05/25/50/75/95, IQR, energy,
  entropy.
- **GLCM (12):** 6 Haralick props x {mean, std} over 4 angles (skimage ``graycomatrix``).
- **Gradient (5):** Sobel magnitude -> mean, std, max, p90, energy.
- **GLRLM (11):** see :mod:`mri_ad.classical.runlength` (skimage ships no run-length equivalent).
- **Context (1):** ``fg_fraction``. ``include_slice_index`` adds a 45th feature (z-position) and
  defaults to ``False`` — it is an anatomical prior, not texture, and inflates AUC for free.

GLCM ``correlation`` is NaN on a constant slice, and skew/kurtosis are undefined there too, so
every value passes an explicit non-finite -> ``0.0`` sanitizer whose hit count is exposed via
:attr:`FeatureExtractor.sanitized_count` rather than swept away silently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel

from mri_ad.classical.runlength import GLRLM_FEATURE_NAMES, run_length_features

_FIRST_ORDER_NAMES: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
    "range",
    "skew",
    "kurtosis",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
    "iqr",
    "energy",
    "entropy",
)
_GRADIENT_NAMES: tuple[str, ...] = (
    "gradient_mean",
    "gradient_std",
    "gradient_max",
    "gradient_p90",
    "gradient_energy",
)
_ENTROPY_BINS = 32


@dataclass(frozen=True)
class FeatureConfig:
    """Every knob that changes a feature *value* (not dataset/labeling knobs — see ``dataset.py``).

    A change to any field must change :meth:`content_hash`, which keys the feature cache.
    """

    percentiles: tuple[float, ...] = (5, 25, 50, 75, 95)
    glcm_levels: int = 32
    glcm_distances: tuple[int, ...] = (1,)
    glcm_angles_deg: tuple[float, ...] = (0, 45, 90, 135)
    glcm_symmetric: bool = True
    glcm_normed: bool = True
    glcm_props: tuple[str, ...] = (
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    )
    runlength_levels: int = 16
    runlength_angles_deg: tuple[float, ...] = (0, 45, 90, 135)
    gradient_percentile: float = 90
    include_slice_index: bool = False
    foreground_eps: float = 1.0e-6
    library: str = field(default="scikit-image", compare=False)
    pyradiomics_used: bool = field(default=False, compare=False)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> FeatureConfig:
        """Build from ``cfg.classical.features`` (OmegaConf lists -> plain tuples)."""
        section = cfg.classical.features
        glcm = section.glcm
        runlength = section.runlength
        return cls(
            percentiles=tuple(OmegaConf.to_container(section.percentiles, resolve=True)),
            glcm_levels=int(glcm.levels),
            glcm_distances=tuple(OmegaConf.to_container(glcm.distances, resolve=True)),
            glcm_angles_deg=tuple(OmegaConf.to_container(glcm.angles_deg, resolve=True)),
            glcm_symmetric=bool(glcm.symmetric),
            glcm_normed=bool(glcm.normed),
            glcm_props=tuple(OmegaConf.to_container(glcm.props, resolve=True)),
            runlength_levels=int(runlength.levels),
            runlength_angles_deg=tuple(OmegaConf.to_container(runlength.angles_deg, resolve=True)),
            gradient_percentile=float(section.gradient.percentile),
            include_slice_index=bool(section.include_slice_index),
            foreground_eps=float(section.foreground_eps),
        )

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON of every feature-value-affecting field."""
        payload = {
            "percentiles": list(self.percentiles),
            "glcm_levels": self.glcm_levels,
            "glcm_distances": list(self.glcm_distances),
            "glcm_angles_deg": list(self.glcm_angles_deg),
            "glcm_symmetric": self.glcm_symmetric,
            "glcm_normed": self.glcm_normed,
            "glcm_props": list(self.glcm_props),
            "runlength_levels": self.runlength_levels,
            "runlength_angles_deg": list(self.runlength_angles_deg),
            "gradient_percentile": self.gradient_percentile,
            "include_slice_index": self.include_slice_index,
            "foreground_eps": self.foreground_eps,
        }
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def feature_names(config: FeatureConfig) -> list[str]:
    """The ordered column names :class:`FeatureExtractor` produces for ``config``."""
    names = list(_FIRST_ORDER_NAMES)
    names += [f"glcm_{prop}_{stat}" for prop in config.glcm_props for stat in ("mean", "std")]
    names += list(_GRADIENT_NAMES)
    names += [f"glrlm_{name}" for name in GLRLM_FEATURE_NAMES]
    names.append("fg_fraction")
    if config.include_slice_index:
        names.append("slice_index")
    return names


def _quantize_uint8(image: np.ndarray, *, levels: int) -> np.ndarray:
    """Min-max-normalize then bin ``image`` into ``[0, levels)`` as ``uint8`` for GLCM."""
    vmin, vmax = float(image.min()), float(image.max())
    if vmax - vmin < 1e-12:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image - vmin) / (vmax - vmin)
    quantized = np.round(scaled * (levels - 1)).astype(np.uint8)
    return np.clip(quantized, 0, levels - 1)


def _shannon_entropy(image: np.ndarray, *, bins: int = _ENTROPY_BINS) -> float:
    hist, _ = np.histogram(image, bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.0
    prob = hist[hist > 0] / total
    return float(-np.sum(prob * np.log2(prob)))


def _first_order(image: np.ndarray, config: FeatureConfig) -> dict[str, float]:
    flat = image.reshape(-1).astype(np.float64)
    std = float(flat.std())
    percentiles = np.percentile(flat, config.percentiles)
    p25, p75 = float(np.percentile(flat, 25)), float(np.percentile(flat, 75))
    return {
        "mean": float(flat.mean()),
        "std": std,
        "min": float(flat.min()),
        "max": float(flat.max()),
        "range": float(flat.max() - flat.min()),
        "skew": float(stats.skew(flat)) if std > 0 else 0.0,
        "kurtosis": float(stats.kurtosis(flat)) if std > 0 else 0.0,
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "iqr": p75 - p25,
        "energy": float(np.mean(flat**2)),
        "entropy": _shannon_entropy(image),
    }


def _glcm(image: np.ndarray, config: FeatureConfig) -> dict[str, float]:
    quantized = _quantize_uint8(image, levels=config.glcm_levels)
    angles_rad = [np.deg2rad(a) for a in config.glcm_angles_deg]
    matrix = graycomatrix(
        quantized,
        distances=list(config.glcm_distances),
        angles=angles_rad,
        levels=config.glcm_levels,
        symmetric=config.glcm_symmetric,
        normed=config.glcm_normed,
    )
    out: dict[str, float] = {}
    for prop in config.glcm_props:
        values = graycoprops(matrix, prop)[0]  # (n_distances, n_angles) -> distance 0
        out[f"glcm_{prop}_mean"] = float(np.mean(values))
        out[f"glcm_{prop}_std"] = float(np.std(values))
    return out


def _gradient(image: np.ndarray, config: FeatureConfig) -> dict[str, float]:
    grad = sobel(image)
    return {
        "gradient_mean": float(grad.mean()),
        "gradient_std": float(grad.std()),
        "gradient_max": float(grad.max()),
        "gradient_p90": float(np.percentile(grad, config.gradient_percentile)),
        "gradient_energy": float(np.mean(grad**2)),
    }


class FeatureExtractor:
    """Extracts the 44 (or 45) per-slice features defined by :func:`feature_names`."""

    def __init__(self, config: FeatureConfig) -> None:
        """Store ``config`` and reset the sanitizer counter."""
        self.config = config
        self._names = feature_names(config)
        self._sanitized_count = 0

    @property
    def sanitized_count(self) -> int:
        """Total non-finite values replaced by ``0.0`` across every call so far. Never hidden."""
        return self._sanitized_count

    def extract_slice(self, image: np.ndarray, *, slice_index: int = 0) -> np.ndarray:
        """One 2D slice in ``[0, 1]`` -> ``(F,)`` float32, ordered per :func:`feature_names`."""
        arr = np.asarray(image, dtype=np.float64)
        values: dict[str, float] = {}
        values.update(_first_order(arr, self.config))
        values.update(_glcm(arr, self.config))
        values.update(_gradient(arr, self.config))
        values.update(
            {
                f"glrlm_{k}": v
                for k, v in run_length_features(
                    arr,
                    levels=self.config.runlength_levels,
                    angles_deg=self.config.runlength_angles_deg,
                ).items()
            }
        )
        values["fg_fraction"] = float(np.mean(arr > self.config.foreground_eps))
        if self.config.include_slice_index:
            values["slice_index"] = float(slice_index)

        vector = np.array([values[name] for name in self._names], dtype=np.float64)
        non_finite = ~np.isfinite(vector)
        self._sanitized_count += int(non_finite.sum())
        vector[non_finite] = 0.0
        return vector.astype(np.float32)

    def extract(self, volume: np.ndarray) -> np.ndarray:
        """``(D,H,W)`` or ``(1,D,H,W)`` slice-stack, ``[0, 1]``-normalized -> ``(D, F)`` float32."""
        arr = np.asarray(volume)
        if arr.ndim == 4:
            arr = arr[0]
        depth = arr.shape[0]
        rows = [self.extract_slice(arr[d], slice_index=d) for d in range(depth)]
        return np.stack(rows, axis=0).astype(np.float32)


__all__ = ["FeatureConfig", "FeatureExtractor", "feature_names"]
