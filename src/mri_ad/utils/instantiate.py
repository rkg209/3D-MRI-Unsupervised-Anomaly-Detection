"""Config-driven object construction (Spec 003), hoisted from ``scripts/run_slice.py``.

The ``{target, params}`` node shape is the one mechanism used to swap a threshold strategy (or
any other non-model object) by config alone — see ``configs/threshold/*.yaml``.
"""

from __future__ import annotations

import importlib
from typing import Any

from omegaconf import DictConfig, OmegaConf


def instantiate_from_config(node: DictConfig) -> Any:
    """Build an object from a ``{target, params}`` config node.

    ``target`` is a dotted class path (e.g. ``mri_ad.recon.threshold.FixedPercentileThreshold``);
    ``params`` is passed through as keyword arguments.
    """
    module_path, _, cls_name = str(node.target).rpartition(".")
    cls = getattr(importlib.import_module(module_path), cls_name)
    params = OmegaConf.to_container(node.get("params", {}), resolve=True) or {}
    return cls(**params)


__all__ = ["instantiate_from_config"]
