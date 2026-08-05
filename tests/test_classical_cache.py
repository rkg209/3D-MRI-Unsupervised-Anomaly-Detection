"""Spec 006 feature-cache provenance: round-trip, loud mismatch, absent cache, atomic save."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mri_ad.classical.cache import FeatureCache, FeatureCacheHeader
from mri_ad.classical.dataset import SliceDataset
from mri_ad.exceptions import FeatureCacheError


def _dataset(n: int = 6, f: int = 4) -> SliceDataset:
    rng = np.random.default_rng(0)
    return SliceDataset(
        X=rng.random((n, f)).astype(np.float32),
        y=rng.integers(0, 2, size=n).astype(np.uint8),
        groups=np.array([f"sub-{i % 2}" for i in range(n)], dtype=object),
        slice_index=np.arange(n, dtype=np.int64),
        feature_names=[f"feat_{i}" for i in range(f)],
    )


def _header(**overrides: object) -> FeatureCacheHeader:
    base = {
        "granularity": "slice-level",
        "split_hash": "a" * 64,
        "seed": 42,
        "feature_hash": "b" * 64,
        "feature_names": ("feat_0", "feat_1", "feat_2", "feat_3"),
    }
    base.update(overrides)
    return FeatureCacheHeader(**base)  # type: ignore[arg-type]


def test_round_trip(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    header = _header()
    data = _dataset()

    cache.save(header, data, counters={"n_samples": 6})
    result = cache.load(header)

    assert result is not None
    loaded, counters = result
    np.testing.assert_array_equal(loaded.X, data.X)
    np.testing.assert_array_equal(loaded.y, data.y)
    np.testing.assert_array_equal(loaded.groups, data.groups)
    np.testing.assert_array_equal(loaded.slice_index, data.slice_index)
    assert loaded.feature_names == data.feature_names
    assert counters == {"n_samples": 6}


def test_absent_cache_returns_none(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    assert cache.load(_header()) is None


@pytest.mark.parametrize(
    "field_name,override",
    [
        ("seed", {"seed": 7}),
        ("schema_version", {"schema_version": 999}),
        ("feature_names", {"feature_names": ("other_0", "other_1", "other_2", "other_3")}),
    ],
)
def test_same_path_mismatch_raises_naming_the_offending_field(
    tmp_path: Path, field_name: str, override: dict
) -> None:
    """Fields that do NOT change the cache path (unlike the hashes) must still be verified."""
    cache = FeatureCache(tmp_path)
    header = _header()
    cache.save(header, _dataset())

    mismatched = _header(**override)
    assert cache._path(mismatched) == cache._path(header)  # same stem, disagreeing provenance

    with pytest.raises(FeatureCacheError, match=field_name):
        cache.load(mismatched)


@pytest.mark.parametrize("field_name", ["split_hash", "feature_hash", "granularity"])
def test_hash_prefix_collision_still_raises_on_full_mismatch(
    tmp_path: Path, field_name: str
) -> None:
    """A truncated-stem collision must not let a differing full field through silently."""
    cache = FeatureCache(tmp_path)
    base_value = "a" * 12 + "1" * 52 if field_name != "granularity" else "slice-level"
    other_value = "a" * 12 + "2" * 52 if field_name != "granularity" else "slice-level-other"
    header = _header(**{field_name: base_value})
    cache.save(header, _dataset())

    colliding = _header(**{field_name: other_value})
    # Force a path collision even for `granularity` (part of the stem verbatim, not truncated)
    # by writing the saved file's bytes to the path `colliding` would read from.
    colliding_path = cache._path(colliding)
    if colliding_path != cache._path(header):
        colliding_path.write_bytes(cache._path(header).read_bytes())

    with pytest.raises(FeatureCacheError, match=field_name):
        cache.load(colliding)


def test_interrupted_save_leaves_no_partial_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FeatureCache(tmp_path)
    header = _header()

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(np, "savez", _boom)
    with pytest.raises(RuntimeError):
        cache.save(header, _dataset())

    assert not cache._path(header).exists()
    assert list(tmp_path.glob("*.tmp")) == []
