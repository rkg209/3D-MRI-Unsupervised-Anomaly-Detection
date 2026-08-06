"""Spec 011: the README scorecard — joins already-written artifacts, renders the generated block.

Report-safe (no ML imports): only ``json``/``csv``/``pathlib``/``dataclasses``, mirroring
``eval/matrix.py``/``eval/paradigm.py``. ``load_scorecard`` **never raises on a missing input** —
a missing artifact becomes a per-entry ``na_reason``. It raises :class:`ArtifactError` only for a
malformed artifact (bad JSON/CSV) or a malformed *config* (an undeclared source), because those
are pipeline bugs, not honest absences (house convention, see ``eval/matrix.py``'s docstring).

Six independent source joins, one per scorecard fact (plan's "Source join" table):

* Dice / IoU (DL headline)                -> ``arch_loss_matrix.json``
* ROC-AUC / PR-AUC (DL headline)          -> ``paradigm_comparison.json``
* ROC-AUC / PR-AUC (classical)            -> ``classical_metrics.json``
* Synthetic-anomaly Dice, before / after  -> ``synth_before_after.csv``
* Inference s/volume, GPU hours, device   -> newest matching run dir under ``artifacts/runs/``,
  matched on ``config.model.name``/``config.loss.name``
* Corrected Spec 004 baseline + delta     -> ``artifacts/metrics/<cell>/aggregate.json``
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mri_ad.exceptions import ArtifactError

NA_PREFIX = "n/a"

CENTRAL_FINDING_SENTENCE = (
    "Reconstruction fidelity and detection quality are anti-correlated: the model that rebuilds "
    "a held-out brain most faithfully (highest PSNR/SSIM) is the one that reconstructs the tumor "
    "too, and so is the worst at flagging it — detection separability (Dice/IoU) is the target, "
    "PSNR/SSIM are never optimized for and never a headline number."
)

MOBILITY_DISCLAIMER = "a conceptual analogy, not a tested result"


def _get(mapping: Any, key: str) -> Any:
    """Read ``key`` from a plain ``dict`` or an ``omegaconf`` mapping, whichever was passed."""
    return mapping[key] if isinstance(mapping, dict) else getattr(mapping, key)


def _resolve_path(raw: Any, repo_root: Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root / path


def _rel(path: Path, repo_root: Path) -> str:
    """Render a source path relative to the repo root — never an absolute local filesystem path."""
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path}: not valid JSON ({exc}).") from exc


@dataclass(frozen=True)
class ScorecardEntry:
    """One scorecard row — either a real measurement or an explicit, traceable absence."""

    key: str
    label: str
    value: float | None
    fmt: str
    unit: str
    source_path: str | None
    run_id: str | None
    na_reason: str | None

    def __post_init__(self) -> None:
        """R6: a value with no traceable source is a pipeline bug, never a silent number."""
        if self.value is not None and self.source_path is None:
            raise ArtifactError(
                f"ScorecardEntry {self.key!r}: has a numeric value but no source_path — every "
                "published number must trace to an artifact."
            )

    @property
    def available(self) -> bool:
        """``True`` iff this entry resolved against a real artifact value."""
        return self.value is not None

    def render(self) -> str:
        """Formatted ``value + unit``, or ``n/a (<reason>)``. Never a bare, untraceable number."""
        if self.value is None:
            reason = self.na_reason or "not available"
            return f"{NA_PREFIX} ({reason})"
        return f"{format(self.value, self.fmt)}{self.unit}"


@dataclass(frozen=True)
class Scorecard:
    """Every scorecard row plus the provenance needed to render and audit the block."""

    entries: tuple[ScorecardEntry, ...]
    split_hash: str | None
    sources: tuple[str, ...]
    n_available: int


def _entry(
    key: str,
    label: str,
    *,
    value: float | None,
    fmt: str,
    unit: str = "",
    source_path: str | None,
    run_id: str | None = None,
    na_reason: str | None,
) -> ScorecardEntry:
    return ScorecardEntry(
        key=key,
        label=label,
        value=value,
        fmt=fmt,
        unit=unit,
        source_path=source_path if value is not None else None,
        run_id=run_id,
        na_reason=na_reason if value is None else None,
    )


def load_timing(runs_root: Path, *, model: str, loss: str) -> dict[str, Any]:
    """Newest ``artifacts/runs/<stamp>/run_meta.json`` matching ``model``/``loss`` with timing.

    "Newest" is the lexicographic max of the UTC-stamped directory names (they sort
    chronologically by construction, ``RunLogger``'s ``%Y%m%dT%H%M%S_%fZ``). A run whose
    ``metrics`` lack ``seconds_per_volume`` is skipped — it never recorded inference timing (e.g.
    an eval-only or report-only run), not a candidate for this row.

    Returns ``{}`` if no matching run exists. Never raises on a missing/empty ``runs_root``.
    """
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return {}

    candidates = sorted((p for p in runs_root.iterdir() if p.is_dir()), reverse=True)
    for candidate in candidates:
        meta_path = candidate / "run_meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        config = meta.get("config", {})
        if _dotted(config, "model.name") != model or _dotted(config, "loss.name") != loss:
            continue
        metrics = meta.get("metrics", {})
        if metrics.get("seconds_per_volume") is None:
            continue
        return {**metrics, "run_id": candidate.name}
    return {}


def _dotted(mapping: Mapping, dotted: str) -> Any:
    cur: Any = mapping
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_scorecard(report_cfg: Any, repo_root: Path) -> Scorecard:
    """Join every declared source into the scorecard. Never raises on a missing artifact."""
    repo_root = Path(repo_root)
    sources_cfg = _get(report_cfg, "sources")
    headline_cfg = _get(report_cfg, "headline_cell")
    precision = _get(report_cfg, "precision")
    na_default = str(_get(report_cfg, "na_reason_default"))

    model = str(_get(headline_cfg, "model"))
    loss = str(_get(headline_cfg, "loss"))
    cell_id = f"{model}__{loss}"

    def _prec(name: str) -> str:
        return str(_get(precision, name))

    def _source(name: str) -> Path:
        return _resolve_path(_get(sources_cfg, name), repo_root)

    sources_read: list[str] = []
    split_hashes: dict[str, str | None] = {}

    # ── Dice / IoU (DL headline) <- arch_loss_matrix.json ───────────────────────────────────
    matrix_path = _source("matrix_json")
    dice_val = iou_val = None
    dice_na = iou_na = "arch x loss matrix not generated (`make matrix`)"
    matrix_source = None
    if matrix_path.is_file():
        payload = _read_json(matrix_path)
        sources_read.append(_rel(matrix_path, repo_root))
        matrix_source = _rel(matrix_path, repo_root)
        cell = next((c for c in payload.get("cells", []) if c.get("cell_id") == cell_id), None)
        if cell is None:
            dice_na = iou_na = f"{cell_id} not declared in arch_loss_matrix.json"
        elif cell.get("dice") is None:
            dice_na = iou_na = str(cell.get("na_reason") or na_default)
        else:
            dice_val = float(cell["dice"])
            iou_val = float(cell["iou"])

    # ── ROC-AUC / PR-AUC (DL headline + classical) <- paradigm_comparison.json ─────────────
    paradigm_path = _source("paradigm_json")
    dl_roc = dl_pr = classical_roc_p = classical_pr_p = None
    dl_auc_na = "paradigm comparison not generated (`make paradigm`)"
    paradigm_source = None
    if paradigm_path.is_file():
        payload = _read_json(paradigm_path)
        sources_read.append(_rel(paradigm_path, repo_root))
        paradigm_source = _rel(paradigm_path, repo_root)
        split_hashes["paradigm_json"] = payload.get("split_hash")
        columns = {c.get("key"): c for c in payload.get("columns", [])}
        dl_col = columns.get(cell_id)
        if dl_col is None:
            dl_auc_na = f"{cell_id} not declared in paradigm_comparison.json"
        elif dl_col.get("roc_auc") is None:
            dl_auc_na = str(dl_col.get("na_reason") or na_default)
        else:
            dl_roc = float(dl_col["roc_auc"])
            dl_pr = float(dl_col["pr_auc"])
        classical_col = columns.get("classical")
        if classical_col is not None and classical_col.get("roc_auc") is not None:
            classical_roc_p = float(classical_col["roc_auc"])
            classical_pr_p = float(classical_col["pr_auc"])

    # ── ROC-AUC / PR-AUC (classical) <- classical_metrics.json (headline, own source) ──────
    classical_path = _source("classical_json")
    classical_roc = classical_pr = None
    classical_na = "classical baseline not run (`make classical`)"
    classical_source = None
    if classical_path.is_file():
        payload = _read_json(classical_path)
        sources_read.append(_rel(classical_path, repo_root))
        classical_source = _rel(classical_path, repo_root)
        split_hashes["classical_json"] = _dotted(payload, "split.split_hash")
        try:
            headline = payload["headline"]
            classical_roc = float(headline["roc_auc_mean"])
            classical_pr = float(headline["pr_auc_mean"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(
                f"{classical_path}: malformed classical_metrics.json ({exc})."
            ) from exc
    elif classical_roc_p is not None:
        # Fall back to the paradigm comparison's own recompute if the classical report itself
        # is absent but the paradigm table already carries the classical column.
        classical_roc, classical_pr = classical_roc_p, classical_pr_p
        classical_na = None
        classical_source = paradigm_source

    # ── Synthetic-anomaly Dice, before / after <- synth_before_after.csv ───────────────────
    synth_path = _source("synth_csv")
    synth_before = synth_after = None
    synth_na = "synthetic-anomaly comparison not generated (`make synth-table`)"
    synth_source = None
    if synth_path.is_file():
        with synth_path.open(newline="") as fh:
            rows = {row["cell"]: row for row in csv.DictReader(fh)}
        sources_read.append(_rel(synth_path, repo_root))
        synth_source = _rel(synth_path, repo_root)
        if "before" in rows and "after" in rows:
            synth_before = float(rows["before"]["dice_mean"])
            synth_after = float(rows["after"]["dice_mean"])
        else:
            synth_na = "synth_before_after.csv missing a before/after row"

    # ── Inference s/volume, GPU hours, device <- newest matching run_meta.json ─────────────
    runs_root = _source("runs_root")
    timing = load_timing(runs_root, model=model, loss=loss)
    timing_na = "no recorded inference run for this cell (Spec 011 D1: needs a cluster run)"
    seconds_per_volume = timing.get("seconds_per_volume")
    gpu_hours = timing.get("gpu_hours")
    device = timing.get("device")
    timing_run_id = timing.get("run_id")
    timing_source = (
        _rel(runs_root / str(timing_run_id) / "run_meta.json", repo_root) if timing_run_id else None
    )
    if timing_source is not None:
        sources_read.append(timing_source)
    # R4: a real (non-cuda) run was found, so "n/a" here means "not GPU", never "not measured".
    gpu_hours_na = timing_na if timing_run_id is None else f"device={device} (not GPU)"

    # ── Corrected Spec 004 baseline + delta <- artifacts/metrics/<cell>/aggregate.json ─────
    aggregate_dir = _source("aggregate_dir")
    aggregate_path = aggregate_dir / cell_id / "aggregate.json"
    baseline_val = delta_val = None
    baseline_na = "not yet evaluated (`make recon && make eval`)"
    aggregate_source = None
    if aggregate_path.is_file():
        payload = _read_json(aggregate_path)
        sources_read.append(_rel(aggregate_path, repo_root))
        aggregate_source = _rel(aggregate_path, repo_root)
        split_hashes["aggregate_json"] = payload.get("split_hash")
        try:
            baseline_val = float(payload["headline"]["dice_mean"])
            delta_val = float(payload["baseline_delta"]["delta"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"{aggregate_path}: malformed aggregate.json ({exc}).") from exc

    # ── R1: refuse to blend AUC/baseline rows scored on different splits ───────────────────
    distinct_hashes = {h for h in split_hashes.values() if h is not None}
    split_hash = next(iter(distinct_hashes)) if len(distinct_hashes) == 1 else None
    if len(distinct_hashes) > 1:
        mismatch_reason = f"split mismatch across sources: {split_hashes!r}"
        dl_roc = dl_pr = classical_roc = classical_pr = baseline_val = delta_val = None
        dl_auc_na = classical_na = baseline_na = mismatch_reason

    entries = (
        _entry(
            "dice_dl",
            f"Dice ({cell_id}, DL headline)",
            value=dice_val,
            fmt=_prec("dice"),
            source_path=matrix_source,
            na_reason=dice_na,
        ),
        _entry(
            "iou_dl",
            f"IoU ({cell_id}, DL headline)",
            value=iou_val,
            fmt=_prec("iou"),
            source_path=matrix_source,
            na_reason=iou_na,
        ),
        _entry(
            "roc_auc_dl",
            f"ROC-AUC ({cell_id}, DL)",
            value=dl_roc,
            fmt=_prec("auc"),
            source_path=paradigm_source,
            na_reason=dl_auc_na,
        ),
        _entry(
            "pr_auc_dl",
            f"PR-AUC ({cell_id}, DL)",
            value=dl_pr,
            fmt=_prec("auc"),
            source_path=paradigm_source,
            na_reason=dl_auc_na,
        ),
        _entry(
            "roc_auc_classical",
            "ROC-AUC (classical baseline)",
            value=classical_roc,
            fmt=_prec("auc"),
            source_path=classical_source,
            na_reason=classical_na,
        ),
        _entry(
            "pr_auc_classical",
            "PR-AUC (classical baseline)",
            value=classical_pr,
            fmt=_prec("auc"),
            source_path=classical_source,
            na_reason=classical_na,
        ),
        _entry(
            "synth_dice_before",
            "Synth-anomaly fine-tune — Dice before",
            value=synth_before,
            fmt=_prec("dice"),
            source_path=synth_source,
            na_reason=synth_na,
        ),
        _entry(
            "synth_dice_after",
            "Synth-anomaly fine-tune — Dice after",
            value=synth_after,
            fmt=_prec("dice"),
            source_path=synth_source,
            na_reason=synth_na,
        ),
        _entry(
            "inference_seconds_per_volume",
            "Inference time (s/volume)",
            value=seconds_per_volume,
            fmt=_prec("seconds"),
            unit=" s",
            source_path=timing_source,
            run_id=timing_run_id,
            na_reason=timing_na,
        ),
        _entry(
            "gpu_hours",
            f"GPU hours (device={device or 'n/a'})",
            value=gpu_hours,
            fmt=_prec("gpu_hours"),
            unit=" h",
            source_path=timing_source,
            run_id=timing_run_id,
            na_reason=gpu_hours_na,
        ),
        _entry(
            "baseline_corrected_dice",
            "Corrected Spec 004 baseline Dice",
            value=baseline_val,
            fmt=_prec("dice"),
            source_path=aggregate_source,
            na_reason=baseline_na,
        ),
        _entry(
            "baseline_delta",
            "Delta vs. prior published Dice (see 'corrected baseline' section below)",
            value=delta_val,
            fmt="+.4f",
            source_path=aggregate_source,
            na_reason=baseline_na,
        ),
    )

    return Scorecard(
        entries=entries,
        split_hash=split_hash,
        sources=tuple(sources_read),
        n_available=sum(1 for e in entries if e.available),
    )


def render_markdown(card: Scorecard) -> str:
    """The generated scorecard block — spliced verbatim between the README markers."""
    lines = [
        "*Generated by `make report` from files under `artifacts/` — do not edit by hand.*",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for entry in card.entries:
        lines.append(f"| {entry.label} | {entry.render()} |")
    lines += [
        "",
        f"Split hash: `{card.split_hash or 'n/a'}`. "
        f"{card.n_available}/{len(card.entries)} rows available.",
        "",
        "Sources: " + (", ".join(f"`{s}`" for s in card.sources) if card.sources else "none yet."),
    ]
    return "\n".join(lines)


__all__ = [
    "CENTRAL_FINDING_SENTENCE",
    "MOBILITY_DISCLAIMER",
    "NA_PREFIX",
    "Scorecard",
    "ScorecardEntry",
    "load_scorecard",
    "load_timing",
    "render_markdown",
]
