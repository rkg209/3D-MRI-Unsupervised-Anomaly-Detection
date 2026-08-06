"""Results-directory discovery + streaming load (Spec 004).

The evaluation harness never re-runs ``forward()`` — it only ever reads persisted
:class:`~mri_ad.recon.types.ReconResult` files written by ``recon/io.py::save_result``. This
module resolves *which* ``<run_id>`` directory to read and streams its contents one volume at a
time (test-split results are ~63 MB each; 250 volumes eagerly materialized is ~15 GB).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from mri_ad.exceptions import ArtifactError
from mri_ad.recon.io import load_result
from mri_ad.recon.types import ReconResult

MANIFEST_FILENAME = "manifest.json"


def resolve_results_dir(root: Path, run_id: str | None = None) -> Path:
    """Return ``root/<run_id>``, or the most-recently-modified subdirectory holding a ``*.pt``.

    Raises :class:`ArtifactError` if ``root`` is absent, ``run_id`` doesn't exist under it, or
    no subdirectory of ``root`` contains any ``.pt`` file.
    """
    root = Path(root)
    if run_id is not None:
        candidate = root / run_id
        if not candidate.is_dir():
            raise ArtifactError(f"No results directory at {candidate} (run_id={run_id!r}).")
        return candidate

    if not root.is_dir():
        raise ArtifactError(f"No results root at {root}. Run `make recon` first.")

    candidates = [d for d in root.iterdir() if d.is_dir() and any(d.glob("*.pt"))]
    if not candidates:
        raise ArtifactError(
            f"No run directory under {root} contains any *.pt ReconResult. Run `make recon` first."
        )
    return max(candidates, key=lambda d: d.stat().st_mtime)


def result_paths(results_dir: Path) -> list[Path]:
    """Every ``*.pt`` ``ReconResult`` file under ``results_dir``, sorted for determinism."""
    return sorted(Path(results_dir).glob("*.pt"))


def iter_results(results_dir: Path) -> Iterator[ReconResult]:
    """Stream every :class:`ReconResult` under ``results_dir``, one at a time."""
    for path in result_paths(results_dir):
        yield load_result(path)


def load_by_volume_id(results_dir: Path, volume_id: str) -> ReconResult:
    """Load the single ``ReconResult`` for ``volume_id`` under ``results_dir``.

    Raises :class:`ArtifactError` on a miss, reporting the *count* of available volumes rather
    than listing their ids (NFR-13 discipline — the same rule the demo's frames follow).
    """
    path = Path(results_dir) / f"{volume_id}.pt"
    if not path.is_file():
        n = len(result_paths(results_dir))
        raise ArtifactError(
            f"No ReconResult for volume_id={volume_id!r} under {results_dir.name} "
            f"({n} volume(s) available)."
        )
    return load_result(path)


def read_manifest(results_dir: Path) -> dict | None:
    """Read ``<results_dir>/manifest.json`` if present, else ``None``.

    Written by ``run_recon.py``/``run_sweep.py`` at reconstruction time (``{"model", "split",
    "n_volumes", "run_id"}``) — both scripts persist ``ReconResult``s under the same
    ``recon.results_dir`` root with no other record of *which split* or *which model* produced a
    given run directory. Without this, ``resolve_results_dir``'s newest-mtime fallback can pick
    up a **val**-split sweep run and silently score it as if it were test (trap #5 wearing a new
    costume). Older/hand-built result directories without a manifest still work — the caller
    decides how strict to be about a missing one.
    """
    path = Path(results_dir) / MANIFEST_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


__all__ = [
    "MANIFEST_FILENAME",
    "iter_results",
    "load_by_volume_id",
    "read_manifest",
    "resolve_results_dir",
    "result_paths",
]
