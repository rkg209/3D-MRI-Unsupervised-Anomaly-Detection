"""Run provenance (Spec 000 spine, NFR-2).

Every entry point runs inside a :class:`RunLogger`. On exit it writes one
``artifacts/runs/<UTC-timestamp>/run_meta.json`` recording *exactly* what produced a number:
the git SHA (and whether the tree was dirty), the fully-resolved config, the seed, the host,
wall-clock start/end, and whatever metrics the run recorded.

**Privacy (NFR-13):** this file must never contain a patient identifier, an on-disk scan path,
or scan metadata. The de-identified BraTS subject key (a dataset folder name) is *not* a patient
identifier and may appear; a filesystem path to a ``.nii`` volume may not.
"""

from __future__ import annotations

import json
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _git(*args: str) -> str:
    """Run a git command from the repo, returning stripped stdout (``""`` on failure)."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
            check=False,
        )
        return out.stdout.strip()
    except OSError:
        return ""


class RunLogger:
    """Context manager that stamps a run directory with reproducible provenance.

    Usage::

        with RunLogger(cfg) as run:
            ...
            run.record(dice=0.42)
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Store the config; the run directory is created on ``__enter__``."""
        self.cfg = cfg
        self.metrics: dict[str, Any] = {}
        self.run_dir: Path | None = None
        self._start: datetime | None = None

    def __enter__(self) -> RunLogger:
        """Create ``artifacts/runs/<UTC-timestamp>/`` and start the wall clock."""
        self._start = datetime.now(UTC)
        stamp = self._start.strftime("%Y%m%dT%H%M%S_%fZ")
        artifact_root = Path(str(self.cfg.paths.artifact_root))
        self.run_dir = artifact_root / "runs" / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self

    def record(self, **metrics: Any) -> None:
        """Record scalar results (e.g. ``dice``) to be written into ``run_meta.json``."""
        self.metrics.update(metrics)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Write ``run_meta.json``. Runs even on failure, so a crash still leaves provenance."""
        if self.run_dir is None or self._start is None:
            return
        end = datetime.now(UTC)
        meta: dict[str, Any] = {
            "git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "seed": int(self.cfg.seed),
            "hostname": socket.gethostname(),
            "start_time": self._start.isoformat(),
            "end_time": end.isoformat(),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "metrics": self.metrics,
        }
        if exc_type is not None:
            meta["error"] = exc_type.__name__
        (self.run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
