"""Spec 007: the three-paradigm comparison — assembly and rendering. Report-safe (no ML imports).

Reads only already-written artifacts (``classical_metrics.json`` + ``oof_predictions.csv`` for the
classical column; ``aggregate.json`` + ``slice_scores.csv`` for each DL column) and recomputes
every curve/AUC through the one shared, hand-rolled :mod:`mri_ad.eval.curves` — the classical
column's ``oof_predictions.csv`` recompute intentionally differs in the third decimal from 006's
own per-fold mean +/- std (pooled-OOF vs. per-fold aggregation); both numbers are reported, never
one silently substituted for the other.

**Boundary duplication, on purpose.** ``eval/`` must not import ``classical/`` (mirror of Spec
006 acceptance 7, kept for symmetry) — so :data:`GRANULARITY_NOTE` below is a verbatim copy of
``classical/metrics.py::GRANULARITY_NOTE``, not an import of it.
``tests/test_paradigm.py::test_granularity_note_matches_classical_verbatim`` pins the two equal so
they cannot silently drift apart.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mri_ad.eval.curves import BinaryCurves, binary_curves
from mri_ad.exceptions import ArtifactError

NA_PREFIX = "n/a"
_DEFAULT_NA_REASON = "not evaluated"

GRANULARITY_NOTE = (
    "Operating granularity is slice-level: each 2D axial slice is one sample. This does NOT "
    "localize an anomaly, so ROC-AUC here is NOT comparable to voxel-level Dice."
)


@dataclass(frozen=True)
class ThresholdPoint:
    """One scored slice-level decision threshold applied to a paradigm column."""

    threshold: float
    precision: float
    recall: float
    f1: float
    split: str


@dataclass(frozen=True)
class ParadigmColumn:
    """One (paradigm) column of the headline table — a real measurement or an explicit absence."""

    key: str
    paradigm: str
    localizes: bool
    split_hash: str | None
    roc_auc: float | None
    pr_auc: float | None
    prevalence: float | None
    dice: float | None
    iou: float | None
    operating_point: ThresholdPoint | None
    curves: BinaryCurves | None
    na_reason: str | None
    n_slices: int | None
    n_positive: int | None = None
    source_paths: tuple[str, ...] = field(default_factory=tuple)
    per_fold_roc_auc_mean: float | None = None  # classical only — 006's own published number

    @property
    def available(self) -> bool:
        """``True`` iff this column resolved against real artifacts (has a slice-level AUC)."""
        return self.roc_auc is not None

    def render(self, field_name: str, fmt: str) -> str:
        """Formatted number, or ``n/a (<reason>)``.

        Dice/IoU always render ``n/a (no localization)`` for a column that does not localize,
        regardless of ``na_reason`` — that caveat is a property of the paradigm, not of whether
        this particular run happened.
        """
        if field_name in ("dice", "iou") and not self.localizes:
            return f"{NA_PREFIX} (no localization)"
        value = getattr(self, field_name)
        if value is None:
            reason = self.na_reason or _DEFAULT_NA_REASON
            return f"{NA_PREFIX} ({reason})"
        return format(value, fmt)

    def render_f1(self) -> str:
        """The tuned-operating-point F1 cell, or ``n/a (threshold not tuned)``."""
        if self.operating_point is None:
            if not self.available:
                reason = self.na_reason or _DEFAULT_NA_REASON
                return f"{NA_PREFIX} ({reason})"
            return f"{NA_PREFIX} (threshold not tuned)"
        return format(self.operating_point.f1, ".4f")


@dataclass(frozen=True)
class ParadigmStats:
    """Cross-column statistics — computed only from columns that are actually available."""

    n_columns: int
    n_available: int
    split_hash: str | None
    n_slices: int | None
    n_positive: int | None
    prevalence: float | None
    best_roc_auc_key: str | None
    best_dice_key: str | None
    diffusion_available: bool


def _threshold_point(
    y_true: Sequence[int], y_score: Sequence[float], *, threshold: float, split: str
) -> ThresholdPoint:
    tp = fp = fn = 0
    for t, s in zip(y_true, y_score, strict=True):
        predicted = s >= threshold
        if predicted and t == 1:
            tp += 1
        elif predicted and t == 0:
            fp += 1
        elif not predicted and t == 1:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ThresholdPoint(
        threshold=float(threshold), precision=precision, recall=recall, f1=f1, split=split
    )


def _na_column(key: str, paradigm: str, *, localizes: bool, na_reason: str) -> ParadigmColumn:
    return ParadigmColumn(
        key=key,
        paradigm=paradigm,
        localizes=localizes,
        split_hash=None,
        roc_auc=None,
        pr_auc=None,
        prevalence=None,
        dice=None,
        iou=None,
        operating_point=None,
        curves=None,
        na_reason=na_reason,
        n_slices=None,
    )


def load_classical_column(
    cfg: Mapping, metrics_dir: Path
) -> tuple[ParadigmColumn, dict[tuple[str, int], int]]:
    """Load the classical column from ``classical_metrics.json`` + ``oof_predictions.csv``.

    Returns the column plus its ``(volume_id, slice_index) -> label`` sample map, used by
    :func:`assert_same_samples`. Missing both files -> an ``n/a`` column. Present/absent
    disagreement between the two -> :class:`ArtifactError` ("half-present" is a pipeline bug,
    never silently downgraded to an honest absence).
    """
    metrics_dir = Path(metrics_dir)
    metrics_path = metrics_dir / "classical_metrics.json"
    predictions_path = metrics_dir / "oof_predictions.csv"
    col_cfg = _column_cfg(cfg, "classical")
    na_reason = str(col_cfg.get("na_reason") or _DEFAULT_NA_REASON)

    metrics_present = metrics_path.is_file()
    predictions_present = predictions_path.is_file()
    if not metrics_present and not predictions_present:
        return _na_column("classical", "Classical", localizes=False, na_reason=na_reason), {}
    if metrics_present != predictions_present:
        raise ArtifactError(
            f"classical column: {metrics_path.name} present={metrics_present}, "
            f"{predictions_path.name} present={predictions_present} — a half-present column is "
            "a pipeline bug, not an honest absence."
        )

    try:
        payload = json.loads(metrics_path.read_text())
        split_hash = str(payload["split"]["split_hash"])
        per_fold_roc_auc_mean = float(payload["headline"]["roc_auc_mean"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"{metrics_path}: malformed classical_metrics.json ({exc}).") from exc

    with predictions_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ArtifactError(f"{predictions_path}: zero rows.")

    try:
        y_true = [int(row["y_true"]) for row in rows]
        y_score = [float(row["y_score"]) for row in rows]
        samples = {(row["volume_id"], int(row["slice_index"])): int(row["y_true"]) for row in rows}
    except KeyError as exc:
        raise ArtifactError(f"{predictions_path}: missing column {exc}.") from exc

    curves = binary_curves(y_true, y_score)
    operating_point = _threshold_point(y_true, y_score, threshold=0.5, split="test")

    column = ParadigmColumn(
        key="classical",
        paradigm="Classical",
        localizes=False,
        split_hash=split_hash,
        roc_auc=curves.roc_auc,
        pr_auc=curves.pr_auc,
        prevalence=curves.prevalence,
        dice=None,
        iou=None,
        operating_point=operating_point,
        curves=curves,
        na_reason=None,
        n_slices=curves.n,
        n_positive=curves.n_positive,
        source_paths=(str(metrics_path), str(predictions_path)),
        per_fold_roc_auc_mean=per_fold_roc_auc_mean,
    )
    return column, samples


def load_dl_column(
    cfg: Mapping, column_cfg: Mapping, metrics_root: Path
) -> tuple[ParadigmColumn, dict[tuple[str, int], int]]:
    """Load one DL column from ``<metrics_root>/<cell_id>/{aggregate.json,slice_scores.csv}``.

    Missing **either** file with a declared ``na_reason`` -> an ``n/a`` column. Missing **one**
    but not the other is malformed, not ``n/a`` — a half-present model is a pipeline bug, and
    letting it render half a row is how a table cell stops being traceable.
    """
    key = str(column_cfg["key"])
    model = str(column_cfg["model"])
    loss = str(column_cfg["loss"])
    paradigm = str(column_cfg["paradigm"])
    na_reason = str(column_cfg.get("na_reason") or _DEFAULT_NA_REASON)
    cell_id = f"{model}__{loss}"
    if cell_id != key:
        raise ArtifactError(
            f"paradigm column {key!r}: model__loss ({cell_id!r}) disagrees with its own "
            "declared key — configs/paradigm/default.yaml is internally inconsistent."
        )

    metrics_root = Path(metrics_root)
    agg_path = metrics_root / cell_id / "aggregate.json"
    slice_scores_path = metrics_root / cell_id / "slice_scores.csv"

    agg_present = agg_path.is_file()
    slices_present = slice_scores_path.is_file()
    if not agg_present and not slices_present:
        return _na_column(key, paradigm, localizes=True, na_reason=na_reason), {}
    if agg_present != slices_present:
        raise ArtifactError(
            f"{key} column: aggregate.json present={agg_present}, slice_scores.csv "
            f"present={slices_present} — a half-present column is a pipeline bug, not an "
            "honest absence."
        )

    try:
        agg = json.loads(agg_path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{agg_path}: not valid JSON ({exc}).") from exc
    if agg.get("cell_id") != cell_id:
        raise ArtifactError(
            f"{agg_path}: payload cell_id {agg.get('cell_id')!r} does not match {cell_id!r}."
        )
    if agg.get("split") != "test":
        raise ArtifactError(f"{agg_path}: split={agg.get('split')!r}, not test.")

    try:
        dice = float(agg["headline"]["dice_mean"])
        iou = float(agg["headline"]["iou_mean"])
        raw_split_hash = agg.get("split_hash")
        split_hash = str(raw_split_hash) if raw_split_hash is not None else None
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"{agg_path}: malformed aggregate.json ({exc}).") from exc

    with slice_scores_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ArtifactError(f"{slice_scores_path}: zero rows.")

    score_field = str(cfg["score_field"]) if isinstance(cfg, dict) else str(cfg.score_field)
    try:
        y_true = [int(row["label"]) for row in rows]
        y_score = [float(row[score_field]) for row in rows]
        samples = {(row["volume_id"], int(row["slice_index"])): int(row["label"]) for row in rows}
    except KeyError as exc:
        raise ArtifactError(f"{slice_scores_path}: missing column {exc}.") from exc

    curves = binary_curves(y_true, y_score)

    operating_point = None
    slice_threshold = (
        cfg.get("slice_threshold") if isinstance(cfg, dict) else cfg.get("slice_threshold")
    )
    if slice_threshold is not None:
        operating_point = _threshold_point(
            y_true, y_score, threshold=float(slice_threshold), split="test"
        )

    column = ParadigmColumn(
        key=key,
        paradigm=paradigm,
        localizes=True,
        split_hash=split_hash,
        roc_auc=curves.roc_auc,
        pr_auc=curves.pr_auc,
        prevalence=curves.prevalence,
        dice=dice,
        iou=iou,
        operating_point=operating_point,
        curves=curves,
        na_reason=None,
        n_slices=curves.n,
        n_positive=curves.n_positive,
        source_paths=(str(agg_path), str(slice_scores_path)),
    )
    return column, samples


def _column_cfg(cfg: Mapping, key: str) -> Mapping:
    columns = cfg["columns"] if isinstance(cfg, dict) else cfg.columns
    for col in columns:
        if (col["key"] if isinstance(col, dict) else col.key) == key:
            return col
    raise ArtifactError(f"configs/paradigm/default.yaml declares no column with key {key!r}.")


def assert_same_samples(
    samples: Mapping[str, Mapping[tuple[str, int], int]],
    split_hashes: Mapping[str, str | None],
) -> None:
    """The non-negotiable invariant: every available column shares the identical sample set.

    Checked in order — (1) split hashes agree; (2) ``(volume_id, slice_index)`` key sets agree
    **exactly**, naming a few offending keys and the likely config-drift cause
    (``data.eval_subset`` vs. ``classical.subject_limit``); (3) labels agree row-for-row for the
    shared keys — both sides derive from the same binarized segmentation, so a disagreement here
    is a preprocessing drift bug, not a config issue.
    """
    hashes = {h for h in split_hashes.values() if h is not None}
    if len(hashes) > 1:
        raise ArtifactError(
            f"Split-hash mismatch across paradigm columns: {dict(split_hashes)} — the columns "
            "were not scored on the identical test split."
        )

    names = [name for name, rows in samples.items() if rows]
    if len(names) < 2:
        return

    reference_name = names[0]
    reference = frozenset(samples[reference_name])
    for name in names[1:]:
        other = frozenset(samples[name])
        if other != reference:
            missing = sorted(reference - other)[:5]
            extra = sorted(other - reference)[:5]
            raise ArtifactError(
                f"Sample-set mismatch between {reference_name!r} and {name!r}: "
                f"missing_from_{name}={missing}, extra_in_{name}={extra}. Likely cause: "
                "`data.eval_subset` set on one side but not `classical.subject_limit` on the "
                "other (or vice versa) — the two subject-pool knobs are independent."
            )

    for key in reference:
        label_ref = samples[reference_name][key]
        for name in names[1:]:
            label_other = samples[name][key]
            if label_ref != label_other:
                raise ArtifactError(
                    f"Label disagreement at {key} between {reference_name!r} ({label_ref}) and "
                    f"{name!r} ({label_other}) — both derive from the same binarized "
                    "segmentation, so this is a preprocessing drift bug, not a config issue."
                )


def compute_stats(columns: Sequence[ParadigmColumn]) -> ParadigmStats:
    """Cross-column stats — best ROC-AUC / best Dice, computed only from available columns."""
    available = [c for c in columns if c.available]
    reference = available[0] if available else None

    best_roc_auc_key: str | None = None
    best_roc_auc: float | None = None
    for c in available:
        if c.roc_auc is not None and (best_roc_auc is None or c.roc_auc > best_roc_auc):
            best_roc_auc = c.roc_auc
            best_roc_auc_key = c.key

    best_dice_key: str | None = None
    best_dice: float | None = None
    for c in available:
        if c.dice is not None and (best_dice is None or c.dice > best_dice):
            best_dice = c.dice
            best_dice_key = c.key

    diffusion_available = any(
        c.available and c.paradigm.lower().startswith("diffusion") for c in columns
    )

    return ParadigmStats(
        n_columns=len(columns),
        n_available=len(available),
        split_hash=reference.split_hash if reference else None,
        n_slices=reference.n_slices if reference else None,
        n_positive=reference.n_positive if reference else None,
        prevalence=reference.prevalence if reference else None,
        best_roc_auc_key=best_roc_auc_key,
        best_dice_key=best_dice_key,
        diffusion_available=diffusion_available,
    )


def render_markdown(columns: Sequence[ParadigmColumn], stats: ParadigmStats, path: Path) -> Path:
    """The headline table + the caveats that keep it from being misread (R1/R6 in the plan)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = " | ".join(["Metric", *(c.paradigm for c in columns)])
    sep = " | ".join(["---"] * (len(columns) + 1))

    def _row(label: str, field_name: str, fmt: str) -> str:
        cells = [c.render(field_name, fmt) for c in columns]
        return f"| {label} | " + " | ".join(cells) + " |"

    def _f1_row() -> str:
        cells = [c.render_f1() for c in columns]
        return "| Slice-level F1 @ tuned thr. | " + " | ".join(cells) + " |"

    lines = [
        "# Paradigm comparison — Classical vs UNETR vs Diffusion (Spec 007)",
        "",
        f"**{GRANULARITY_NOTE}**",
        "",
        f"| {header} |",
        f"| {sep} |",
        _row("Slice-level ROC-AUC", "roc_auc", ".4f"),
        _row("Slice-level PR-AUC", "pr_auc", ".4f"),
        _f1_row(),
        _row("Voxel-level Dice", "dice", ".4f"),
        _row("Voxel-level IoU", "iou", ".4f"),
        "",
        f"Split hash: `{stats.split_hash or 'n/a'}`. n_slices={stats.n_slices}, "
        f"n_positive={stats.n_positive}, prevalence={stats.prevalence}.",
        "",
        f"Best slice-level ROC-AUC: **{stats.best_roc_auc_key or 'n/a'}**. "
        f"Best voxel-level Dice: **{stats.best_dice_key or 'n/a'}**.",
        "",
    ]

    classical = next((c for c in columns if c.key == "classical"), None)
    if classical is not None and classical.per_fold_roc_auc_mean is not None:
        lines += [
            "The classical column's pooled-OOF ROC-AUC "
            f"({classical.roc_auc:.4f}) differs slightly from Spec 006's own published "
            f"per-fold mean +/- std ROC-AUC ({classical.per_fold_roc_auc_mean:.4f}) — both are "
            "computed from the same predictions, aggregated two different ways (pooled vs. "
            "per-fold), and both are reported here rather than one silently standing in for "
            "the other.",
            "",
        ]

    lines += [GRANULARITY_NOTE, ""]
    path.write_text("\n".join(lines))
    return path


def render_csv(columns: Sequence[ParadigmColumn], path: Path) -> Path:
    """Long-form CSV: one row per column, machine-readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "key",
        "paradigm",
        "localizes",
        "roc_auc",
        "pr_auc",
        "dice",
        "iou",
        "f1_at_tuned_threshold",
        "na_reason",
        "split_hash",
        "n_slices",
    )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for c in columns:
            writer.writerow(
                {
                    "key": c.key,
                    "paradigm": c.paradigm,
                    "localizes": c.localizes,
                    "roc_auc": c.render("roc_auc", ".4f"),
                    "pr_auc": c.render("pr_auc", ".4f"),
                    "dice": c.render("dice", ".4f"),
                    "iou": c.render("iou", ".4f"),
                    "f1_at_tuned_threshold": c.render_f1(),
                    "na_reason": c.na_reason or "",
                    "split_hash": c.split_hash or "",
                    "n_slices": c.n_slices if c.n_slices is not None else "",
                }
            )
    return path


def write_stats(columns: Sequence[ParadigmColumn], stats: ParadigmStats, path: Path) -> Path:
    """Write ``paradigm_comparison.json`` — every column plus cross-column stats + provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec": "007",
        "granularity_note": GRANULARITY_NOTE,
        "split_hash": stats.split_hash,
        "sample": {
            "n_slices": stats.n_slices,
            "n_positive": stats.n_positive,
            "prevalence": stats.prevalence,
        },
        "columns": [
            {
                "key": c.key,
                "paradigm": c.paradigm,
                "localizes": c.localizes,
                "roc_auc": c.roc_auc,
                "pr_auc": c.pr_auc,
                "dice": c.dice,
                "iou": c.iou,
                "operating_point": (
                    {
                        "threshold": c.operating_point.threshold,
                        "precision": c.operating_point.precision,
                        "recall": c.operating_point.recall,
                        "f1": c.operating_point.f1,
                        "split": c.operating_point.split,
                    }
                    if c.operating_point is not None
                    else None
                ),
                "na_reason": c.na_reason,
                "source_paths": list(c.source_paths),
                "available": c.available,
            }
            for c in columns
        ],
        "stats": {
            "n_columns": stats.n_columns,
            "n_available": stats.n_available,
            "best_roc_auc_key": stats.best_roc_auc_key,
            "best_dice_key": stats.best_dice_key,
            "diffusion_available": stats.diffusion_available,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def plot_curves(columns: Sequence[ParadigmColumn], path: Path) -> Path:
    """Overlaid ROC + PR curves, one figure, two axes — every *available* column, same axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 5))
    for c in columns:
        if c.curves is None:
            continue
        roc_x = [p.x for p in c.curves.roc]
        roc_y = [p.y for p in c.curves.roc]
        ax_roc.plot(roc_x, roc_y, label=f"{c.paradigm} (AUC={c.roc_auc:.3f})")

        pr_x = [p.x for p in c.curves.pr]
        pr_y = [p.y for p in c.curves.pr]
        ax_pr.plot(pr_x, pr_y, label=f"{c.paradigm} (AP={c.pr_auc:.3f})")
        ax_pr.axhline(c.curves.prevalence, linestyle="--", alpha=0.4, color="gray")

    ax_roc.plot([0, 1], [0, 1], linestyle=":", color="black", alpha=0.5)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.set_title("ROC")
    ax_roc.legend(fontsize=8)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.set_title("PR (dashed = prevalence floor)")
    ax_pr.legend(fontsize=8)

    fig.suptitle(GRANULARITY_NOTE, wrap=True, fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


__all__ = [
    "GRANULARITY_NOTE",
    "NA_PREFIX",
    "ParadigmColumn",
    "ParadigmStats",
    "ThresholdPoint",
    "assert_same_samples",
    "compute_stats",
    "load_classical_column",
    "load_dl_column",
    "plot_curves",
    "render_csv",
    "render_markdown",
    "write_stats",
]
