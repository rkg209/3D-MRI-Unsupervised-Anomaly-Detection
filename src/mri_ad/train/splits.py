"""The single path that yields Spec 009 training ids (trap #5: never tune on BraTS test).

``resolve_training_ids`` is the only function in the codebase allowed to decide which ids feed
the fine-tune. It exists so that "which ids trained the model" is one auditable call, not
something re-derived ad hoc in ``scripts/run_train.py`` where a copy-paste mistake could quietly
leak the BraTS test split into checkpoint selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from mri_ad.data.split import SplitContract
from mri_ad.exceptions import SplitContractViolationError


@dataclass(frozen=True)
class TrainingIds:
    """The ids Spec 009's fine-tune is allowed to touch."""

    openbhb_train: tuple[str, ...]
    openbhb_val: tuple[str, ...]
    brats_select: tuple[str, ...]


def resolve_training_ids(cfg: DictConfig, contract: SplitContract) -> TrainingIds:
    """OpenBHB train/val for the fine-tune, BraTS **val** for checkpoint selection.

    Never BraTS **test** — :class:`~mri_ad.data.split.SplitContract` partitions ``openbhb`` and
    ``brats`` disjointly by construction, but this function re-asserts that guarantee explicitly
    (rather than trusting it silently) so a future refactor of ``SplitContract`` cannot
    reintroduce test-split leakage without this raising immediately.

    Raises:
        SplitContractViolationError: if any returned id intersects ``contract.brats["test"]``.
    """
    del cfg  # unused today; kept for a config-driven override extension point, precedent: recon/
    openbhb_train = tuple(contract.openbhb.get("train", []))
    openbhb_val = tuple(contract.openbhb.get("val", []))
    brats_select = tuple(contract.brats.get("val", []))

    test_ids = set(contract.brats.get("test", []))
    leaked = set(brats_select) & test_ids
    if leaked:
        raise SplitContractViolationError(
            f"resolve_training_ids: {len(leaked)} id(s) intersect the BraTS TEST split — the "
            f"test split must never be used for training or checkpoint selection: "
            f"{sorted(leaked)[:5]}."
        )

    return TrainingIds(
        openbhb_train=openbhb_train, openbhb_val=openbhb_val, brats_select=brats_select
    )


__all__ = ["TrainingIds", "resolve_training_ids"]
