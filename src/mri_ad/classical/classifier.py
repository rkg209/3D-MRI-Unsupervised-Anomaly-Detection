"""The one file that imports ``xgboost`` (Spec 006, acceptance 7 / R6).

``xgboost`` fails to import on some machines this project runs on (missing ``libomp.dylib`` on
macOS without ``brew install libomp``) — every other ``classical/`` module must stay importable
regardless. ``baseline.py`` imports this module lazily, from inside
:meth:`~mri_ad.classical.baseline.ClassicalBaseline._build_estimator`, so merely doing
``import mri_ad.classical`` never touches ``xgboost``.
"""

from __future__ import annotations

from typing import Any


def build_xgboost(params: dict[str, Any], *, seed: int, scale_pos_weight: float) -> Any:
    """Build an ``XGBClassifier`` from config ``params`` plus the per-fold ``scale_pos_weight``.

    ``params`` must already pin ``tree_method`` and ``n_jobs`` (never ``-1``) — an unpinned
    ``n_jobs``/``tree_method`` makes tree construction machine-dependent (R1a).
    """
    import xgboost as xgb

    return xgb.XGBClassifier(
        **params,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        importance_type="gain",
    )


__all__ = ["build_xgboost"]
