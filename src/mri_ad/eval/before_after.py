"""Spec 009 acceptance 4: the synthetic-anomaly before/after table. Report-safe, no ML imports.

Like ``eval/report.py`` and ``eval/matrix.py``, this module (and everything it imports) must
never pull in ``torch``, ``monai``, or any other heavy dependency — only ``json``/``csv``/
``pathlib``. That keeps ``make report``/``make synth-table`` fast and GPU-free.

``aggregate.json`` records ``split_hash`` but not the threshold strategy or preprocessing config
acceptance 4 demands identical between the two runs. Rather than widen ``write_aggregate``, this
follows ``aggregate.json -> run_id -> artifacts/runs/<run_id>/run_meta.json``, which already
holds the fully-resolved config — zero harness change, strictly more coverage.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mri_ad.exceptions import ArtifactError

# The subtree paths (dotted, into run_meta.json's "config") that must match between the before
# and after runs for the comparison to be controlled (Spec 009 acceptance 4).
_CONTROLLED_SUBTREES: tuple[str, ...] = (
    "threshold",
    "data.preprocess",
    "data.eval_subset",
    "shape",
    "model.params",
)


@dataclass(frozen=True)
class CellRecord:
    """One before/after cell.

    The headline numbers plus everything needed to prove it's controlled.
    """

    model: str
    loss: str
    cell_id: str
    split: str
    n_volumes: int
    split_hash: str
    run_id: str | None
    results_run_id: str | None
    dice_mean: float
    dice_std: float
    iou_mean: float
    iou_std: float
    config: dict[str, Any]


def _dotted_get(mapping: dict[str, Any], dotted: str) -> Any:
    cur: Any = mapping
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_cell(metrics_root: Path, runs_root: Path, *, model: str, loss: str) -> CellRecord:
    """Load ``<metrics_root>/<model>__<loss>/aggregate.json`` + its run's ``run_meta.json``.

    Never reads ``baseline_delta.published_dice`` — only ``headline`` — so the legacy
    (buggy-code) 0.6255 number can never leak in as a "before" cell (spec's own note).

    Raises:
        ArtifactError: if the aggregate or its ``run_meta.json`` is missing/malformed.
    """
    cell_id = f"{model}__{loss}"
    aggregate_path = Path(metrics_root) / cell_id / "aggregate.json"
    if not aggregate_path.is_file():
        raise ArtifactError(
            f"No aggregate metrics for {cell_id!r} at {aggregate_path}. Run `make eval` first."
        )
    aggregate = json.loads(aggregate_path.read_text())

    run_id = aggregate.get("run_id")
    if not run_id:
        raise ArtifactError(f"{aggregate_path} has no run_id — cannot trace its config.")
    run_meta_path = Path(runs_root) / str(run_id) / "run_meta.json"
    if not run_meta_path.is_file():
        raise ArtifactError(f"No run_meta.json for run_id {run_id!r} at {run_meta_path}.")
    run_meta = json.loads(run_meta_path.read_text())

    try:
        headline = aggregate["headline"]
        return CellRecord(
            model=str(aggregate["model"]),
            loss=str(aggregate["loss"]),
            cell_id=str(aggregate["cell_id"]),
            split=str(aggregate["split"]),
            n_volumes=int(aggregate["n_volumes"]),
            split_hash=str(aggregate["split_hash"]),
            run_id=run_id,
            results_run_id=aggregate.get("results_run_id"),
            dice_mean=float(headline["dice_mean"]),
            dice_std=float(headline["dice_std"]),
            iou_mean=float(headline["iou_mean"]),
            iou_std=float(headline["iou_std"]),
            config=dict(run_meta.get("config", {})),
        )
    except KeyError as exc:
        raise ArtifactError(f"{aggregate_path} is missing required key {exc}.") from exc


def assert_controlled(before: CellRecord, after: CellRecord) -> None:
    """Raise :class:`ArtifactError` naming the first offending key if not controlled.

    Checks ``split_hash``, ``n_volumes``, ``split``, ``loss`` (both ``aggregate.json``), then
    every subtree in :data:`_CONTROLLED_SUBTREES` (both ``run_meta.json``'s resolved config).
    """
    for field in ("split_hash", "n_volumes", "split", "loss"):
        before_value, after_value = getattr(before, field), getattr(after, field)
        if before_value != after_value:
            raise ArtifactError(
                f"assert_controlled: {field!r} differs between before ({before_value!r}) and "
                f"after ({after_value!r}) — this is not a controlled comparison."
            )

    for subtree in _CONTROLLED_SUBTREES:
        before_value = _dotted_get(before.config, subtree)
        after_value = _dotted_get(after.config, subtree)
        if before_value != after_value:
            raise ArtifactError(
                f"assert_controlled: config.{subtree!r} differs between the before run "
                f"({before.run_id}) and after run ({after.run_id}) — this is not a controlled "
                "comparison."
            )


def render_markdown(before: CellRecord, after: CellRecord) -> str:
    """A markdown before/after table.

    A worse Dice renders a signed delta and ``REGRESSION`` — never a dropped row (Spec 009
    acceptance 7, CLAUDE.md rule 6: report honestly).
    """
    dice_delta = after.dice_mean - before.dice_mean
    iou_delta = after.iou_mean - before.iou_mean
    dice_marker = " **(REGRESSION)**" if dice_delta < 0 else ""
    iou_marker = " (regression)" if iou_delta < 0 else ""

    lines = [
        "# Spec 009 — synthetic-anomaly fine-tune: before/after",
        "",
        f"Before: `{before.cell_id}` (run_id={before.run_id}, "
        f"results_run_id={before.results_run_id})",
        f"After: `{after.cell_id}` (run_id={after.run_id}, results_run_id={after.results_run_id})",
        "",
        "| metric | before | after | delta |",
        "|---|---|---|---|",
        f"| Dice | {before.dice_mean:.4f} | {after.dice_mean:.4f} | "
        f"{dice_delta:+.4f}{dice_marker} |",
        f"| IoU | {before.iou_mean:.4f} | {after.iou_mean:.4f} | {iou_delta:+.4f}{iou_marker} |",
        "",
        f"`split_hash={before.split_hash}`, `n_volumes={before.n_volumes}`, "
        f"`split={before.split}`.",
        "",
    ]
    return "\n".join(lines)


def render_csv(before: CellRecord, after: CellRecord, path: Path) -> Path:
    """Write ``path``: one row per cell (``before``/``after``). Returns ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("cell", "model", "loss", "dice_mean", "dice_std", "iou_mean", "iou_std", "run_id")
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for label, record in (("before", before), ("after", after)):
            writer.writerow(
                [
                    label,
                    record.model,
                    record.loss,
                    record.dice_mean,
                    record.dice_std,
                    record.iou_mean,
                    record.iou_std,
                    record.run_id,
                ]
            )
    return path


__all__ = ["CellRecord", "assert_controlled", "load_cell", "render_csv", "render_markdown"]
