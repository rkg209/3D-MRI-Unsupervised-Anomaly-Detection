"""Config-driven model registry (Spec 002).

Resolves ``configs/model/*.yaml`` by their ``target`` (dotted class path) + ``params`` schema —
the same mechanism ``scripts/run_slice.py`` used provisionally before this spec. Adding a model
touches only its module + a YAML + a checkpoint path; this file never imports a concrete model
class by name (acceptance test 8 — see ``tests/test_model_boundary.py``).
"""

from __future__ import annotations

import importlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from mri_ad.exceptions import ModelRegistrationError
from mri_ad.models.base import AnomalyDetectionModel

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Mirrors configs/config.yaml's paths.checkpoint_root, so a standalone model YAML (loaded
# outside full Hydra composition) still resolves its ${paths.checkpoint_root} interpolation.
_BASE_PATHS_CFG = OmegaConf.create(
    {"paths": {"checkpoint_root": "${oc.env:MRI_AD_CHECKPOINTS,./checkpoints}"}}
)


class ModelRegistry:
    """Maps a model name to its validated class and resolved YAML config."""

    def __init__(self) -> None:
        """Start with no models registered."""
        self._classes: dict[str, type[AnomalyDetectionModel]] = {}
        self._configs: dict[str, DictConfig] = {}

    def register(self, name: str, cls: type) -> None:
        """Validate ``cls`` against :class:`AnomalyDetectionModel` and store it under ``name``.

        Raises :class:`ModelRegistrationError` if ``cls`` is not a conforming subclass — i.e. it
        either doesn't subclass the interface, or still has unimplemented abstract methods.
        """
        if not (isinstance(cls, type) and issubclass(cls, AnomalyDetectionModel)):
            raise ModelRegistrationError(f"{cls!r} does not subclass AnomalyDetectionModel.")
        if cls.__abstractmethods__:
            raise ModelRegistrationError(
                f"{cls.__name__} is missing required interface methods: "
                f"{sorted(cls.__abstractmethods__)}."
            )
        self._classes[name] = cls

    def add_from_yaml(self, yaml_path: Path) -> None:
        """Load one ``configs/model/*.yaml``, import its ``target``, and register it."""
        cfg = OmegaConf.merge(_BASE_PATHS_CFG, OmegaConf.load(yaml_path))
        name = str(cfg.name)
        module_path, _, cls_name = str(cfg.target).rpartition(".")
        cls = getattr(importlib.import_module(module_path), cls_name)
        self.register(name, cls)
        self._configs[name] = cfg

    def _config(self, name: str) -> DictConfig:
        if name not in self._configs:
            raise ModelRegistrationError(f"No config registered for model {name!r}.")
        return self._configs[name]

    def get(self, name: str) -> AnomalyDetectionModel:
        """Instantiate the model registered under ``name`` from its YAML's ``target``/``params``."""
        cls = self._classes.get(name)
        if cls is None:
            raise ModelRegistrationError(f"No model registered under name {name!r}.")
        params = OmegaConf.to_container(self._config(name).get("params", {}), resolve=True) or {}
        return cls(**params)

    def checkpoint_path(self, name: str) -> Path:
        """Resolve the checkpoint path from the YAML registered under ``name``."""
        return Path(str(self._config(name).checkpoint))


def build_default_registry(config_dir: str | Path = "configs/model") -> ModelRegistry:
    """Scan ``config_dir`` for ``*.yaml`` files and register each one's ``target`` class.

    ``config_dir`` is resolved relative to the current working directory first (matching how
    ``make`` targets run from the repo root), falling back to the repo root if that doesn't
    exist — so this also works when called from an arbitrary working directory (e.g. tests).
    """
    path = Path(config_dir)
    if not path.is_absolute():
        candidate = Path.cwd() / path
        if not candidate.is_dir():
            candidate = _REPO_ROOT / path
        path = candidate

    registry = ModelRegistry()
    for yaml_path in sorted(path.glob("*.yaml")):
        registry.add_from_yaml(yaml_path)
    return registry
