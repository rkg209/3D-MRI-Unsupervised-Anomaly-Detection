"""Foreign Patch Interpolation (Spec 009).

Corrupts a healthy volume by blending in one-to-several boxes cut from a *donor* volume:
``corrupted = (1 - alpha) * healthy + alpha * donor`` inside each box, bit-identical outside every
box. Chosen over Poisson blending (a boundary solve, slow) and 3D CutPaste (sharp edges are a
shortcut the model can learn instead of anatomy) — see ``configs/synth/fpi.yaml``.

Three properties are load-bearing and each has a dedicated test in ``tests/test_synth.py``:

1. **Foreground-only placement.** Volumes are min-max normalised with a large zero background;
   a patch landing entirely in background blends ``0`` with ``0`` — the mask is non-empty but
   nothing changed, silently training an identity map (R4).
2. **Same-coordinate donor alignment by default.** A donor patch pulled from an unrelated
   location is anatomically absurd and trivially separable by raw intensity (R5) — the FPI
   construction wants matching anatomy, mismatched texture/intensity.
3. **No global RNG.** Every draw goes through the caller-supplied ``torch.Generator`` so
   per-item, per-epoch determinism (Spec 009 D5) survives multi-worker dataloading.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from mri_ad.exceptions import ConfigError, DescriptiveValidationError

Range = tuple[int, int]
FloatRange = tuple[float, float]


@dataclass(frozen=True)
class PatchSpec:
    """One placed box: target coordinates, blend weight, and the donor's offset from it."""

    d0: int
    d1: int
    h0: int
    h1: int
    w0: int
    w1: int
    alpha: float
    donor_offset: tuple[int, int, int]


@dataclass(frozen=True)
class SynthResult:
    """The output of one :meth:`FPIAnomalyGenerator.generate` call."""

    corrupted: Tensor  # (1,D,H,W) f32 in [0,1] — convex combo of two [0,1] volumes
    healthy: Tensor  # (1,D,H,W) f32 — an OWNED CLONE of the input, never a view (R1)
    synth_mask: Tensor  # (1,D,H,W) f32 — alpha inside a placed box, 0.0 outside;
    # overlapping patches take the element-wise max(alpha), order-independent.
    patches: tuple[PatchSpec, ...]


def _randint(generator: torch.Generator, low: int, high: int) -> int:
    """Uniform integer in ``[low, high]`` inclusive, drawn from ``generator``."""
    if high <= low:
        return low
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def _rand_float(generator: torch.Generator, low: float, high: float) -> float:
    """Uniform float in ``[low, high]``, drawn from ``generator``."""
    return low + (high - low) * float(torch.rand((1,), generator=generator).item())


def _validate_volume(volume: Tensor, *, name: str) -> None:
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise DescriptiveValidationError(
            f"FPIAnomalyGenerator.generate: {name} must be (1, D, H, W), got {tuple(volume.shape)}."
        )
    if volume.dtype != torch.float32:
        raise DescriptiveValidationError(
            f"FPIAnomalyGenerator.generate: {name} must be float32, got {volume.dtype}."
        )
    vmin, vmax = float(volume.min()), float(volume.max())
    if vmin < -1e-4 or vmax > 1.0 + 1e-4:
        raise DescriptiveValidationError(
            f"FPIAnomalyGenerator.generate: {name} must be in [0, 1], observed "
            f"[{vmin}, {vmax}] (shape {tuple(volume.shape)})."
        )


class FPIAnomalyGenerator:
    """Places 1..N boxes of donor content into a healthy volume; see the module docstring."""

    def __init__(
        self,
        *,
        patch_size_range: Range,
        depth_size_range: Range,
        n_patches_range: Range,
        alpha_range: FloatRange,
        foreground_threshold: float,
        min_foreground_fraction: float,
        min_patch_contrast: float,
        max_placement_attempts: int,
        donor_alignment: str,
        corrupt_probability: float,
        seed: int,
    ) -> None:
        """Store parameters. Raises :class:`ConfigError` on an invalid ``donor_alignment``."""
        if donor_alignment not in ("same", "random"):
            raise ConfigError(
                f"donor_alignment must be 'same' or 'random', got {donor_alignment!r}."
            )
        self.patch_size_range = tuple(patch_size_range)
        self.depth_size_range = tuple(depth_size_range)
        self.n_patches_range = tuple(n_patches_range)
        self.alpha_range = tuple(alpha_range)
        self.foreground_threshold = float(foreground_threshold)
        self.min_foreground_fraction = float(min_foreground_fraction)
        self.min_patch_contrast = float(min_patch_contrast)
        self.max_placement_attempts = int(max_placement_attempts)
        self.donor_alignment = donor_alignment
        self.corrupt_probability = float(corrupt_probability)
        self.seed = int(seed)

    def _place_one(self, healthy: Tensor, donor: Tensor, generator: torch.Generator) -> PatchSpec:
        _, depth, height, width = healthy.shape
        ph = _randint(generator, self.patch_size_range[0], min(self.patch_size_range[1], height))
        pw = _randint(generator, self.patch_size_range[0], min(self.patch_size_range[1], width))
        pd = _randint(generator, self.depth_size_range[0], min(self.depth_size_range[1], depth))

        fallback: tuple[int, int, int, int, int, int] | None = None
        for _ in range(self.max_placement_attempts):
            d0 = _randint(generator, 0, depth - pd)
            h0 = _randint(generator, 0, height - ph)
            w0 = _randint(generator, 0, width - pw)
            target_patch = healthy[:, d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw]

            fg_fraction = (target_patch > self.foreground_threshold).float().mean().item()
            if fg_fraction < self.min_foreground_fraction:
                continue

            if self.donor_alignment == "same":
                do, ho, wo = 0, 0, 0
            else:
                dd0 = _randint(generator, 0, depth - pd)
                dh0 = _randint(generator, 0, height - ph)
                dw0 = _randint(generator, 0, width - pw)
                do, ho, wo = dd0 - d0, dh0 - h0, dw0 - w0

            donor_patch = donor[
                :, d0 + do : d0 + do + pd, h0 + ho : h0 + ho + ph, w0 + wo : w0 + wo + pw
            ]
            contrast = (target_patch.mean() - donor_patch.mean()).abs().item()
            if contrast < self.min_patch_contrast:
                fallback = (d0, h0, w0, do, ho, wo)
                continue

            alpha = _rand_float(generator, self.alpha_range[0], self.alpha_range[1])
            return PatchSpec(
                d0=d0,
                d1=d0 + pd,
                h0=h0,
                h1=h0 + ph,
                w0=w0,
                w1=w0 + pw,
                alpha=alpha,
                donor_offset=(do, ho, wo),
            )

        # Every attempt failed a soft constraint (foreground fraction or contrast floor).
        # Best-effort fall back to the last candidate that at least cleared placement, rather
        # than raising — a config/data mismatch here is caught by the acceptance-2 pre-flight
        # guard (configs/synth/fpi.yaml:check), not by an exception mid-epoch.
        if fallback is None:
            fallback = (0, 0, 0, 0, 0, 0)
        d0, h0, w0, do, ho, wo = fallback
        alpha = _rand_float(generator, self.alpha_range[0], self.alpha_range[1])
        return PatchSpec(
            d0=d0,
            d1=d0 + pd,
            h0=h0,
            h1=h0 + ph,
            w0=w0,
            w1=w0 + pw,
            alpha=alpha,
            donor_offset=(do, ho, wo),
        )

    def generate(
        self, volume: Tensor, donor: Tensor, *, generator: torch.Generator | None = None
    ) -> SynthResult:
        """Corrupt ``volume`` with content cut from ``donor``. All randomness via ``generator``."""
        _validate_volume(volume, name="volume")
        _validate_volume(donor, name="donor")
        if volume.shape != donor.shape:
            raise DescriptiveValidationError(
                f"FPIAnomalyGenerator.generate: volume {tuple(volume.shape)} and donor "
                f"{tuple(donor.shape)} must share a shape."
            )
        depth = volume.shape[1]
        if self.depth_size_range[1] > depth:
            raise ConfigError(
                f"depth_size_range upper bound {self.depth_size_range[1]} exceeds volume "
                f"depth {depth}."
            )

        gen = generator if generator is not None else torch.Generator().manual_seed(self.seed)

        healthy = volume.clone()
        corrupted = healthy.clone()
        mask = torch.zeros_like(healthy)

        if _rand_float(gen, 0.0, 1.0) > self.corrupt_probability:
            return SynthResult(corrupted=corrupted, healthy=healthy, synth_mask=mask, patches=())

        n_patches = _randint(gen, self.n_patches_range[0], self.n_patches_range[1])
        patches = tuple(self._place_one(healthy, donor, gen) for _ in range(n_patches))

        # Ascending-alpha order so the highest-alpha patch is written last and wins any overlap —
        # matching the mask's element-wise max(alpha), and making the result order-independent
        # with respect to `patches`' original (generation) order.
        for patch in sorted(patches, key=lambda p: p.alpha):
            target_box = (
                slice(patch.d0, patch.d1),
                slice(patch.h0, patch.h1),
                slice(patch.w0, patch.w1),
            )
            do, ho, wo = patch.donor_offset
            donor_box = (
                slice(patch.d0 + do, patch.d1 + do),
                slice(patch.h0 + ho, patch.h1 + ho),
                slice(patch.w0 + wo, patch.w1 + wo),
            )
            corrupted[(slice(None), *target_box)] = (1 - patch.alpha) * healthy[
                (slice(None), *target_box)
            ] + patch.alpha * donor[(slice(None), *donor_box)]
            mask[(slice(None), *target_box)] = torch.clamp(
                mask[(slice(None), *target_box)], min=patch.alpha
            )

        return SynthResult(corrupted=corrupted, healthy=healthy, synth_mask=mask, patches=patches)


__all__ = ["FPIAnomalyGenerator", "PatchSpec", "SynthResult"]
