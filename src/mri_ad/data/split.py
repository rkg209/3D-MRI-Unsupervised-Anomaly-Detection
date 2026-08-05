"""The deterministic split contract (Spec 001).

The prior work never recorded its train/test split. This module makes the split a first-class,
serialized artifact: given a seed and the sorted subject-id lists, the partition is fully
deterministic and reproducible; using a *different* split than the one on disk is a loud error,
not a silent divergence.

OpenBHB is partitioned into ``train``/``val`` (healthy, used for reconstruction training).
BraTS is partitioned into ``val``/``test`` (``val`` is the threshold-tuning holdout per trap #5 —
never tune a threshold on ``test``).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from mri_ad.exceptions import SplitContractViolationError

# Mirrors configs/data/default.yaml `split:` defaults. Real pipeline code should pass the
# resolved config fractions explicitly; these exist so `SplitContract.build` is usable standalone
# (e.g. in tests) without threading a full Hydra config through.
DEFAULT_OPENBHB_FRACTIONS: dict[str, float] = {"train": 0.85, "val": 0.15}
DEFAULT_BRATS_FRACTIONS: dict[str, float] = {"val": 0.2, "test": 0.8}


def _partition(ids: list[str], seed: int, fractions: dict[str, float]) -> dict[str, list[str]]:
    """Deterministically shuffle sorted ``ids`` by ``seed`` and split by cumulative fractions."""
    total = sum(fractions.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"Split fractions must sum to 1.0, got {total} ({fractions}).")

    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    keys = list(fractions)
    result: dict[str, list[str]] = {}
    start = 0
    for i, key in enumerate(keys):
        end = n if i == len(keys) - 1 else start + round(fractions[key] * n)
        result[key] = shuffled[start:end]
        start = end
    return result


@dataclass(frozen=True)
class SplitContract:
    """A serializable, deterministic partition of both datasets."""

    seed: int
    openbhb: dict[str, list[str]]
    brats: dict[str, list[str]]

    @classmethod
    def build(
        cls,
        seed: int,
        openbhb_ids: list[str],
        brats_ids: list[str],
        *,
        openbhb_fractions: dict[str, float] = DEFAULT_OPENBHB_FRACTIONS,
        brats_fractions: dict[str, float] = DEFAULT_BRATS_FRACTIONS,
    ) -> SplitContract:
        """Build a deterministic partition of both id lists from ``seed``."""
        return cls(
            seed=seed,
            openbhb=_partition(openbhb_ids, seed, openbhb_fractions),
            brats=_partition(brats_ids, seed, brats_fractions),
        )

    def save(self, path: Path) -> None:
        """Serialize to ``path`` (creating parent directories as needed)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"seed": self.seed, "openbhb": self.openbhb, "brats": self.brats}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> SplitContract:
        """Round-trip load from a file written by :meth:`save`."""
        data = json.loads(Path(path).read_text())
        return cls(seed=int(data["seed"]), openbhb=data["openbhb"], brats=data["brats"])

    def verify(self, other: SplitContract, *, override: bool = False) -> None:
        """Raise :class:`SplitContractViolationError` if ``other`` disagrees with ``self``.

        ``self`` is normally the contract recorded on disk; ``other`` is the split about to be
        used. ``override=True`` suppresses the check for an explicit, intentional re-split.
        """
        if override:
            return
        if self != other:
            raise SplitContractViolationError(
                f"Split contract mismatch: recorded seed={self.seed} disagrees with the "
                f"requested split (seed={other.seed}). Pass override=True to intentionally "
                "re-split — this will invalidate comparisons against prior runs."
            )

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON of this contract — keys Spec 006's feature cache."""
        payload = {"seed": self.seed, "openbhb": self.openbhb, "brats": self.brats}
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _list_openbhb_ids(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir() if p.is_file())


def _list_brats_ids(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir() if p.is_dir())


def resolve_contract(cfg: DictConfig, *, build_if_missing: bool) -> SplitContract:
    """Load the split contract recorded at ``cfg.data.split.contract_path``, or build it.

    Nothing in the repo writes ``split_contract.json`` on its own yet (``run_recon.py`` and
    ``run_sweep.py`` both dead-end on a missing file) — this is the one place that turns a data
    directory listing into the canonical, on-disk contract every downstream script trusts.

    A contract already on disk is **verified**, never silently replaced: the current directory
    listings are re-partitioned with the same seed and fractions, and any disagreement with the
    recorded contract (a subject gained or lost) raises
    :class:`~mri_ad.exceptions.SplitContractViolationError` rather than quietly re-splitting.
    """
    contract_path = Path(str(cfg.data.split.contract_path))
    openbhb_ids = _list_openbhb_ids(Path(str(cfg.data.openbhb.dir)))
    brats_ids = _list_brats_ids(Path(str(cfg.data.brats.dir)))
    openbhb_fractions = OmegaConf.to_container(cfg.data.split.openbhb, resolve=True)
    brats_fractions = OmegaConf.to_container(cfg.data.split.brats, resolve=True)

    fresh = SplitContract.build(
        int(cfg.seed),
        openbhb_ids,
        brats_ids,
        openbhb_fractions=openbhb_fractions,
        brats_fractions=brats_fractions,
    )

    if contract_path.is_file():
        recorded = SplitContract.load(contract_path)
        recorded.verify(fresh)
        return recorded

    if not build_if_missing:
        raise SplitContractViolationError(
            f"No split contract at {contract_path.name} and build_if_missing=False."
        )

    fresh.save(contract_path)
    return fresh


__all__ = [
    "DEFAULT_BRATS_FRACTIONS",
    "DEFAULT_OPENBHB_FRACTIONS",
    "SplitContract",
    "resolve_contract",
]
