"""Import-graph boundary test (Spec 002 acceptance test 8, reused by 013.2 / 012.1).

``src/mri_ad/models/*.py`` must never import from ``recon/``, ``eval/``, ``classical/``,
``synth/``, or ``viz/`` — that's what makes adding a model a change contained entirely to
``models/`` + a YAML + a checkpoint path.

Also guards Spec 003's thresholding rule: ``recon/`` is the **only** place a residual is
binarized. No other layer may reimplement it (e.g. ``residual > 0.1``-style comparisons).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "src" / "mri_ad" / "models"
FORBIDDEN = {"mri_ad.recon", "mri_ad.eval", "mri_ad.classical", "mri_ad.synth", "mri_ad.viz"}

# Directories that must never binarize a residual themselves — thresholding is recon/'s alone.
NO_THRESHOLD_DIRS = ["eval", "classical", "viz"]
# A float-residual comparison against a bare number, e.g. `residual > 0.1` or `x > 0.05`.
_THRESHOLD_PATTERN = re.compile(r"\b\w*residual\w*\s*[<>]=?\s*[\d.]")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_models_never_import_downstream_layers() -> None:
    violations: dict[str, set[str]] = {}
    for path in sorted(MODELS_DIR.glob("*.py")):
        modules = _imported_modules(path)
        hit = {m for m in modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)}
        if hit:
            violations[path.name] = hit
    assert not violations, f"models/ imports downstream layers: {violations}"


def test_no_module_outside_recon_binarizes_a_residual() -> None:
    violations: dict[str, list[str]] = {}
    for dirname in NO_THRESHOLD_DIRS:
        for path in sorted((REPO / "src" / "mri_ad" / dirname).rglob("*.py")):
            hits = _THRESHOLD_PATTERN.findall(path.read_text())
            if hits:
                violations[str(path.relative_to(REPO))] = hits
    assert not violations, f"residual thresholding found outside recon/: {violations}"
