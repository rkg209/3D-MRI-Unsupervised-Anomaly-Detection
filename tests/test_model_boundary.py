"""Import-graph boundary test (Spec 002 acceptance test 8, reused by 013.2 / 012.1).

``src/mri_ad/models/*.py`` must never import from ``recon/``, ``eval/``, ``classical/``,
``synth/``, or ``viz/`` — that's what makes adding a model a change contained entirely to
``models/`` + a YAML + a checkpoint path.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "src" / "mri_ad" / "models"
FORBIDDEN = {"mri_ad.recon", "mri_ad.eval", "mri_ad.classical", "mri_ad.synth", "mri_ad.viz"}


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
