"""Spec 005: the architecture x loss fidelity-vs-detection matrix. No ML imports.

Reads only ``artifacts/metrics/<cell_id>/aggregate.json`` files already written by Spec 004's
``ReportGenerator.write_aggregate`` — the generator here computes no metrics of its own, only
Spearman rank correlation over numbers already on disk (hand-rolled, no scipy, to keep this
module and ``scripts/run_arch_loss_matrix.py`` in the report-safe subgraph alongside
``mri_ad.eval.report``). PSNR/SSIM are rendered as explanatory context only (NFR-6/NFR-22),
never a performance column — see :data:`mri_ad.eval.report.CONTEXT_ONLY_NOTE`.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mri_ad.eval.report import CONTEXT_ONLY_NOTE
from mri_ad.exceptions import ArtifactError

NA_PREFIX = "n/a"
_DEFAULT_NA_REASON = "not evaluated"

_CSV_FIELDS = (
    "cell_id",
    "model",
    "loss",
    "role",
    "dice",
    "iou",
    "psnr_db",
    "ssim",
    "na_reason",
    "metrics_path",
    "eval_run_id",
    "recon_run_id",
    "n_volumes",
)


@dataclass(frozen=True)
class MatrixCell:
    """One (model, loss) row of the matrix — either a real measurement or an explicit absence."""

    model: str
    loss: str
    cell_id: str
    role: str
    dice: float | None
    iou: float | None
    psnr_db: float | None
    ssim: float | None
    na_reason: str | None
    metrics_path: str | None
    eval_run_id: str | None
    recon_run_id: str | None
    n_volumes: int | None

    @property
    def available(self) -> bool:
        """``True`` iff this cell was resolved against a real ``aggregate.json``."""
        return self.dice is not None

    def render(self, field: str, fmt: str) -> str:
        """Formatted number, or ``n/a (<reason>)``. Never blank, never a placeholder number."""
        value = getattr(self, field)
        if value is None:
            reason = self.na_reason or _DEFAULT_NA_REASON
            return f"{NA_PREFIX} ({reason})"
        return format(value, fmt)


@dataclass(frozen=True)
class MatrixStats:
    """Cross-cell statistics — computed only from cells that are actually available."""

    n_cells: int
    n_available: int
    spearman_psnr_dice: float | None
    n_pairs: int
    best_cell_id: str | None
    best_dice: float | None
    diffusion_available: bool


def _cell_from_aggregate(
    *, model: str, loss: str, cell_id: str, role: str, metrics_path: Path, agg: Mapping
) -> MatrixCell:
    if agg.get("cell_id") != cell_id:
        raise ArtifactError(
            f"{metrics_path}: payload cell_id {agg.get('cell_id')!r} does not match its "
            f"directory {cell_id!r} — a stale or misplaced aggregate.json, never silently "
            "attributed to the wrong cell."
        )
    if agg.get("split") != "test":
        raise ArtifactError(f"{metrics_path}: split={agg.get('split')!r}, not test.")
    if agg.get("mode") != "normal":
        raise ArtifactError(
            f"{metrics_path}: mode={agg.get('mode')!r} — legacy-compat numbers are never "
            "published into the matrix."
        )

    try:
        headline = agg["headline"]
        context_only = agg["context_only"]
        dice = float(headline["dice_mean"])
        iou = float(headline["iou_mean"])
        psnr_db = float(context_only["psnr_db_mean"])
        ssim = float(context_only["ssim_mean"])
        n_volumes = int(agg["n_volumes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"{metrics_path}: malformed aggregate.json ({exc}).") from exc

    return MatrixCell(
        model=model,
        loss=loss,
        cell_id=cell_id,
        role=role,
        dice=dice,
        iou=iou,
        psnr_db=psnr_db,
        ssim=ssim,
        na_reason=None,
        metrics_path=str(metrics_path),
        eval_run_id=agg.get("run_id"),
        recon_run_id=agg.get("results_run_id"),
        n_volumes=n_volumes,
    )


def load_cells(matrix_cfg: Mapping, metrics_root: Path) -> list[MatrixCell]:
    """Resolve every declared cell against ``metrics_root/<cell_id>/aggregate.json``.

    Present and readable -> a numeric cell. Absent -> an ``n/a`` cell carrying the declared
    ``na_reason``, defaulting to ``"not evaluated"`` when the config gives none. A malformed or
    unreadable aggregate.json raises :class:`ArtifactError` — it is never silently downgraded to
    ``n/a``, because that would hide a broken pipeline behind an honest-looking absence.
    """
    metrics_root = Path(metrics_root)
    cells: list[MatrixCell] = []
    seen_cell_ids: set[str] = set()
    for row in matrix_cfg["cells"]:
        model = str(row["model"])
        loss = str(row["loss"])
        role = str(row["role"])
        cell_id = f"{model}__{loss}"
        if cell_id in seen_cell_ids:
            raise ArtifactError(
                f"configs/matrix declares {cell_id!r} more than once — a duplicate row would "
                "double-count that cell in the table and in the Spearman correlation."
            )
        seen_cell_ids.add(cell_id)
        agg_path = metrics_root / cell_id / "aggregate.json"

        if not agg_path.is_file():
            cells.append(
                MatrixCell(
                    model=model,
                    loss=loss,
                    cell_id=cell_id,
                    role=role,
                    dice=None,
                    iou=None,
                    psnr_db=None,
                    ssim=None,
                    na_reason=str(row.get("na_reason") or _DEFAULT_NA_REASON),
                    metrics_path=None,
                    eval_run_id=None,
                    recon_run_id=None,
                    n_volumes=None,
                )
            )
            continue

        try:
            agg = json.loads(agg_path.read_text())
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"{agg_path}: not valid JSON ({exc}).") from exc

        cells.append(
            _cell_from_aggregate(
                model=model, loss=loss, cell_id=cell_id, role=role, metrics_path=agg_path, agg=agg
            )
        )
    return cells


def _rank(values: Sequence[float]) -> list[float]:
    """Average ranks (1-indexed), ties sharing the mean of their would-be ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation, pure Python (no scipy — report-safe subgraph).

    Average ranks for ties, then Pearson on the ranks. Returns ``None`` for ``n < 3`` or zero
    variance in either series.
    """
    n = len(xs)
    if n != len(ys) or n < 3:
        return None

    rx = _rank(list(xs))
    ry = _rank(list(ys))
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x**0.5 * var_y**0.5)


def compute_stats(cells: Sequence[MatrixCell]) -> MatrixStats:
    """Cross-cell stats: Spearman(PSNR, Dice), the best available Dice, diffusion availability."""
    available = [c for c in cells if c.available]
    # Paired by construction (not two independently filtered lists) so a cell with one metric
    # missing can never desynchronize psnr/dice at the same index.
    pairs = [(c.psnr_db, c.dice) for c in available if c.psnr_db is not None and c.dice is not None]
    psnr_vals = [p for p, _ in pairs]
    dice_vals = [d for _, d in pairs]
    rho = spearman(psnr_vals, dice_vals)

    best_cell_id: str | None = None
    best_dice: float | None = None
    for c in available:
        if c.dice is not None and (best_dice is None or c.dice > best_dice):
            best_dice = c.dice
            best_cell_id = c.cell_id

    diffusion_available = any(c.model == "diffusion" and c.available for c in cells)

    return MatrixStats(
        n_cells=len(cells),
        n_available=len(available),
        spearman_psnr_dice=rho,
        n_pairs=len(psnr_vals),
        best_cell_id=best_cell_id,
        best_dice=best_dice,
        diffusion_available=diffusion_available,
    )


def render_csv(cells: Sequence[MatrixCell], path: Path) -> Path:
    """Write one row per cell — long form, survives ``n/a`` cells and ragged coverage.

    Numeric columns are pre-formatted strings (``0.4231`` or ``n/a (...)`` in the same column,
    rounded for display) — a downstream ``float()`` fails loudly on the ``n/a`` rows, which is
    the correct failure. ``arch_loss_matrix.json`` (:func:`write_stats`) keeps full-precision
    ``float | None`` and is the machine-readable source of truth, not this CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_CSV_FIELDS))
        writer.writeheader()
        for c in cells:
            writer.writerow(
                {
                    "cell_id": c.cell_id,
                    "model": c.model,
                    "loss": c.loss,
                    "role": c.role,
                    "dice": c.render("dice", ".4f"),
                    "iou": c.render("iou", ".4f"),
                    "psnr_db": c.render("psnr_db", ".2f"),
                    "ssim": c.render("ssim", ".4f"),
                    "na_reason": c.na_reason or "",
                    "metrics_path": c.metrics_path or "",
                    "eval_run_id": c.eval_run_id or "",
                    "recon_run_id": c.recon_run_id or "",
                    "n_volumes": c.n_volumes if c.n_volumes is not None else "",
                }
            )
    return path


def render_markdown(cells: Sequence[MatrixCell], stats: MatrixStats, path: Path) -> Path:
    """One row per cell (long form). Footer: context-only note, Spearman line, n/a count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Architecture x loss matrix (Spec 005)",
        "",
        "| model | loss | role | Dice | IoU | PSNR (dB) — context only | "
        "SSIM — context only | source |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        lines.append(
            f"| {c.model} | {c.loss} | {c.role} | {c.render('dice', '.4f')} | "
            f"{c.render('iou', '.4f')} | {c.render('psnr_db', '.2f')} | "
            f"{c.render('ssim', '.4f')} | {c.cell_id} |"
        )

    n_na = stats.n_cells - stats.n_available
    if stats.spearman_psnr_dice is None:
        rho_line = f"Spearman rho: n/a (fewer than 3 evaluated cells, n_pairs={stats.n_pairs})"
    else:
        rho_line = f"Spearman rho (PSNR, Dice): {stats.spearman_psnr_dice:.4f} (n={stats.n_pairs})"

    lines += [
        "",
        CONTEXT_ONLY_NOTE,
        "",
        rho_line,
        "",
        f"{stats.n_available}/{stats.n_cells} cells evaluated; {n_na} cell(s) n/a.",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def write_stats(cells: Sequence[MatrixCell], stats: MatrixStats, path: Path) -> Path:
    """Write ``arch_loss_matrix.json`` — every cell plus the cross-cell stats, machine-readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cells": [
            {
                "cell_id": c.cell_id,
                "model": c.model,
                "loss": c.loss,
                "role": c.role,
                "dice": c.dice,
                "iou": c.iou,
                "psnr_db": c.psnr_db,
                "ssim": c.ssim,
                "na_reason": c.na_reason,
                "metrics_path": c.metrics_path,
                "eval_run_id": c.eval_run_id,
                "recon_run_id": c.recon_run_id,
                "n_volumes": c.n_volumes,
                "available": c.available,
            }
            for c in cells
        ],
        "stats": {
            "n_cells": stats.n_cells,
            "n_available": stats.n_available,
            "spearman_psnr_dice": stats.spearman_psnr_dice,
            "n_pairs": stats.n_pairs,
            "best_cell_id": stats.best_cell_id,
            "best_dice": stats.best_dice,
            "diffusion_available": stats.diffusion_available,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


__all__ = [
    "NA_PREFIX",
    "MatrixCell",
    "MatrixStats",
    "compute_stats",
    "load_cells",
    "render_csv",
    "render_markdown",
    "spearman",
    "write_stats",
]
