"""Spec 011: pure marker splice. Stdlib-only — report-safe subgraph.

``splice`` never guesses: an absent, duplicated, or inverted marker pair is a loud
:class:`ArtifactError`, not a silent no-op. That is what keeps a stale scorecard number from ever
surviving a broken README edit (plan risk R2).
"""

from __future__ import annotations

from pathlib import Path

from mri_ad.exceptions import ArtifactError


def splice(text: str, block: str, *, start: str, end: str) -> str:
    """Replace everything between the first ``start``/``end`` marker pair with ``block``.

    Every byte outside the markers (including the markers themselves) is left untouched. Raises
    :class:`ArtifactError` if either marker is absent, either appears more than once, or ``end``
    precedes ``start`` — each is a config/README-drift bug, never silently patched around.
    """
    start_count = text.count(start)
    end_count = text.count(end)
    if start_count == 0:
        raise ArtifactError(f"splice: start marker {start!r} not found in the target text.")
    if end_count == 0:
        raise ArtifactError(f"splice: end marker {end!r} not found in the target text.")
    if start_count > 1:
        raise ArtifactError(
            f"splice: start marker {start!r} appears {start_count} times — must appear exactly "
            "once so the splice target is unambiguous."
        )
    if end_count > 1:
        raise ArtifactError(
            f"splice: end marker {end!r} appears {end_count} times — must appear exactly once so "
            "the splice target is unambiguous."
        )

    start_idx = text.index(start)
    end_idx = text.index(end)
    if end_idx < start_idx:
        raise ArtifactError(
            f"splice: end marker {end!r} appears before start marker {start!r} — inverted markers."
        )

    head = text[: start_idx + len(start)]
    tail = text[end_idx:]
    return f"{head}\n{block}\n{tail}"


def write_block(readme_path: Path, block: str, *, start: str, end: str) -> bool:
    """Splice ``block`` into ``readme_path`` in place. Returns ``True`` iff the file changed.

    Writing the identical block twice is byte-identical (idempotent) — the second call always
    returns ``False``.
    """
    readme_path = Path(readme_path)
    original = readme_path.read_text()
    updated = splice(original, block, start=start, end=end)
    if updated == original:
        return False
    readme_path.write_text(updated)
    return True


def would_change(readme_path: Path, block: str, *, start: str, end: str) -> bool:
    """``True`` iff splicing ``block`` into ``readme_path`` would change its bytes. No write."""
    readme_path = Path(readme_path)
    original = readme_path.read_text()
    updated = splice(original, block, start=start, end=end)
    return updated != original


__all__ = ["splice", "would_change", "write_block"]
