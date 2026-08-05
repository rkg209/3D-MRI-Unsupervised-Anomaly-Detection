"""Feature-cache provenance (Spec 006). A header mismatch is a loud failure, never silent reuse.

Extraction is the expensive part of the classical baseline (~10-20 CPU-minutes). Its result is
cached to a ``.npz`` keyed by a header of everything that could invalidate it — granularity,
the split contract's content hash, the seed, the feature config's content hash, the exact
feature-name ordering, and a schema version. Loading a cache whose header disagrees with what is
requested raises :class:`~mri_ad.exceptions.FeatureCacheError` naming the offending field, rather
than silently returning stale features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mri_ad.classical.dataset import SliceDataset
from mri_ad.exceptions import FeatureCacheError

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FeatureCacheHeader:
    """Everything that must match for a cached feature matrix to be safely reused."""

    granularity: str
    split_hash: str
    seed: int
    feature_hash: str
    feature_names: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def stem(self) -> str:
        """The cache filename stem: ``<granularity>_<feathash12>_<splithash12>``."""
        return f"{self.granularity}_{self.feature_hash[:12]}_{self.split_hash[:12]}"

    def to_meta(self) -> dict:
        """Serializable provenance payload stored alongside the arrays."""
        return {
            "granularity": self.granularity,
            "split_hash": self.split_hash,
            "seed": self.seed,
            "feature_hash": self.feature_hash,
            "feature_names": list(self.feature_names),
            "schema_version": self.schema_version,
        }


class FeatureCache:
    """Reads/writes ``<dir>/<header.stem()>.npz``."""

    def __init__(self, directory: Path) -> None:
        """Store the cache directory; nothing is created until :meth:`save` runs."""
        self.directory = Path(directory)

    def _path(self, header: FeatureCacheHeader) -> Path:
        return self.directory / f"{header.stem()}.npz"

    def load(self, header: FeatureCacheHeader) -> tuple[SliceDataset, dict] | None:
        """Return ``(cached SliceDataset, counters)``, or ``None`` if no cache file exists.

        ``counters`` is the ``build_slice_dataset`` counter dict recorded at save time (``{}``
        for a cache file written before counters were tracked — never a hard failure over a
        purely informational field).

        Raises :class:`FeatureCacheError` naming the offending field if a cache file exists at
        this path but its stored provenance disagrees with ``header`` — this should be
        unreachable in practice (the path is itself keyed by the hashes) but guards against a
        hand-edited or corrupted cache file being trusted silently.
        """
        path = self._path(header)
        if not path.is_file():
            return None

        with np.load(path, allow_pickle=True) as payload:
            stored = json.loads(str(payload["meta"]))
            requested = header.to_meta()
            for field_name in (
                "granularity",
                "split_hash",
                "seed",
                "feature_hash",
                "schema_version",
            ):
                if stored[field_name] != requested[field_name]:
                    raise FeatureCacheError(
                        f"Feature cache {path.name} mismatch on {field_name!r}: "
                        f"stored={stored[field_name]!r} requested={requested[field_name]!r}."
                    )
            if stored["feature_names"] != requested["feature_names"]:
                raise FeatureCacheError(
                    f"Feature cache {path.name} mismatch on 'feature_names': "
                    f"stored has {len(stored['feature_names'])} columns, requested has "
                    f"{len(requested['feature_names'])}."
                )
            counters = json.loads(str(payload["counters"])) if "counters" in payload else {}
            data = SliceDataset(
                X=payload["X"],
                y=payload["y"],
                groups=payload["groups"],
                slice_index=payload["slice_index"],
                feature_names=list(header.feature_names),
            )
            return data, counters

    def save(
        self, header: FeatureCacheHeader, data: SliceDataset, *, counters: dict | None = None
    ) -> Path:
        """Write ``data`` (+ optional ``counters``) under ``header``'s provenance.

        Atomic: a crash mid-write leaves no ``.npz``.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(header)
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            with tmp_path.open("wb") as fh:
                np.savez(
                    fh,
                    X=data.X,
                    y=data.y,
                    groups=data.groups,
                    slice_index=data.slice_index,
                    meta=json.dumps(header.to_meta()),
                    counters=json.dumps(counters or {}),
                )
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return path


__all__ = ["SCHEMA_VERSION", "FeatureCache", "FeatureCacheHeader"]
