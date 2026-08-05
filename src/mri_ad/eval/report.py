"""Tables and plots from saved metrics — the "report-safe subgraph" (Spec 004, `make report`).

**No ML imports.** This module (and its transitive imports — `mri_ad.exceptions`,
`mri_ad.utils.run_logger`, `mri_ad.eval.__init__`) must never pull in ``torch``, ``monai``,
``nibabel``, ``torchmetrics``, ``sklearn``, ``scipy``, or ``skimage``. That is what lets
``make report`` finish in under two minutes with no GPU (acceptance test 8). ``matplotlib`` is
the one plotting dependency, and even it is imported lazily inside :meth:`plot_dice_distribution`
so merely importing this module stays free of it too.

``ReportGenerator`` reads and writes **plain mappings**, never
:class:`~mri_ad.eval.metrics.VolumeMetrics`/``AggregateMetrics`` — importing ``metrics.py`` would
drag ``torch`` in. ``run_eval.py`` (which does construct those dataclasses) is responsible for
``dataclasses.asdict`` plus the key rename into ``psnr_db_context_only`` / ``ssim_context_only``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from mri_ad.exceptions import ArtifactError

CONTEXT_ONLY_NOTE = (
    "PSNR/SSIM are explanatory context only (NFR-6/NFR-22) — never an "
    "optimization target and never a headline number."
)
PER_VOLUME_FIELDS = ("volume_id", "dice", "iou", "psnr_db_context_only", "ssim_context_only")


class ReportGenerator:
    """Reads/writes ``artifacts/metrics/*`` + ``artifacts/figures/*``. No ML imports."""

    def __init__(self, metrics_dir: Path, figures_dir: Path) -> None:
        """Store the two output directories; nothing is created until a write method runs."""
        self.metrics_dir = Path(metrics_dir)
        self.figures_dir = Path(figures_dir)

    @property
    def _per_volume_path(self) -> Path:
        return self.metrics_dir / "per_volume.csv"

    @property
    def _aggregate_path(self) -> Path:
        return self.metrics_dir / "aggregate.json"

    def write_per_volume(self, rows: Sequence[Mapping[str, object]]) -> Path:
        """Write ``per_volume.csv`` with exactly the :data:`PER_VOLUME_FIELDS` header."""
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        with self._per_volume_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(PER_VOLUME_FIELDS))
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in PER_VOLUME_FIELDS})
        return self._per_volume_path

    def write_aggregate(
        self,
        aggregate: Mapping[str, object],
        *,
        mode: str,
        model: str,
        loss: str,
        split: str,
        n_volumes: int,
        split_hash: str,
        run_id: str | None = None,
        results_run_id: str | None = None,
    ) -> Path:
        """Write ``aggregate.json`` — headline (Dice/IoU) separate from context-only (PSNR/SSIM).

        ``aggregate`` must carry ``dice_mean``/``dice_std``/``iou_mean``/``iou_std``/
        ``psnr_mean``/``psnr_std``/``ssim_mean``/``ssim_std`` plus ``published_dice`` (the
        number to diff against — sourced from ``configs/eval/default.yaml``, never hardcoded
        here). Raises :class:`ArtifactError` if a required key is missing.

        ``loss`` is a **required keyword** (Spec 005 D-A) — a defaulted loss would let a cell be
        written with the wrong label and never be noticed. ``model``/``loss`` together form the
        ``cell_id`` (``f"{model}__{loss}"``) that namespaces this matrix cell in
        ``artifacts/metrics/`` and ``artifacts/figures/``.

        ``split_hash`` is a **required keyword** (Spec 007 acceptance 1) — the
        ``SplitContract.content_hash()`` stamped into the recon manifest this cell's
        ``ReconResult``s were produced from. It is what lets the paradigm comparison assert the
        classical and DL columns were scored on the identical test split, by comparison rather
        than by eye.

        ``run_id`` is the **eval** run that produced this file; ``results_run_id`` is the
        **recon** run whose persisted ``ReconResult``s were scored. They are different runs and
        both matter for tracing a published number back to the artifact that produced it.
        """
        try:
            headline = {
                "dice_mean": aggregate["dice_mean"],
                "dice_std": aggregate["dice_std"],
                "iou_mean": aggregate["iou_mean"],
                "iou_std": aggregate["iou_std"],
            }
            context_only = {
                "psnr_db_mean": aggregate["psnr_mean"],
                "psnr_db_std": aggregate["psnr_std"],
                "ssim_mean": aggregate["ssim_mean"],
                "ssim_std": aggregate["ssim_std"],
                "note": CONTEXT_ONLY_NOTE,
            }
            published_dice = float(aggregate["published_dice"])  # type: ignore[arg-type]
        except KeyError as exc:
            raise ArtifactError(f"write_aggregate: aggregate mapping is missing {exc}.") from exc

        corrected_dice = float(headline["dice_mean"])  # type: ignore[arg-type]
        payload = {
            "mode": mode,
            "model": model,
            "loss": loss,
            "cell_id": f"{model}__{loss}",
            "split": split,
            "n_volumes": n_volumes,
            "split_hash": split_hash,
            "run_id": run_id,
            "results_run_id": results_run_id,
            "headline": headline,
            "context_only": context_only,
            "baseline_delta": {
                "published_dice": published_dice,
                "corrected_dice": corrected_dice,
                "delta": corrected_dice - published_dice,
            },
        }
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._aggregate_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return self._aggregate_path

    def read_per_volume(self) -> list[dict[str, str]]:
        """Read ``per_volume.csv`` back as a list of string-valued dicts."""
        if not self._per_volume_path.is_file():
            raise ArtifactError(f"No per-volume metrics at {self._per_volume_path}.")
        with self._per_volume_path.open(newline="") as fh:
            return list(csv.DictReader(fh))

    def read_aggregate(self) -> dict:
        """Read ``aggregate.json`` back as a dict. Raises :class:`ArtifactError` if absent."""
        if not self._aggregate_path.is_file():
            raise ArtifactError(
                f"No aggregate metrics at {self._aggregate_path}. Run `make eval` first."
            )
        return json.loads(self._aggregate_path.read_text())

    def plot_dice_distribution(self, rows: Sequence[Mapping[str, object]]) -> Path:
        """Histogram of per-volume Dice. ``matplotlib`` imported lazily, here only."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        path = self.figures_dir / "dice_distribution.png"

        dice_values = [float(row["dice"]) for row in rows]
        fig, ax = plt.subplots()
        ax.hist(dice_values, bins=20, range=(0.0, 1.0))
        ax.set_xlabel("Dice")
        ax.set_ylabel("Volume count")
        ax.set_title("Per-volume Dice distribution")
        fig.savefig(path)
        plt.close(fig)
        return path

    def write_summary_markdown(self) -> Path:
        """Render ``summary.md`` from the already-written ``aggregate.json``.

        Kept deliberately thin — Spec 011 owns full reporting; this is the durable
        read/write surface that spec builds on.
        """
        agg = self.read_aggregate()
        headline = agg["headline"]
        context_only = agg["context_only"]
        delta = agg["baseline_delta"]

        lines = [
            f"# Evaluation summary — {agg['model']}__{agg['loss']} ({agg['mode']}, "
            f"split={agg['split']}, n={agg['n_volumes']})",
            "",
            "### Headline — Dice / IoU",
            "",
            "| metric | mean | std |",
            "|---|---|---|",
            f"| Dice | {headline['dice_mean']:.4f} | {headline['dice_std']:.4f} |",
            f"| IoU | {headline['iou_mean']:.4f} | {headline['iou_std']:.4f} |",
            "",
            f"Published baseline Dice: {delta['published_dice']:.4f}. Corrected: "
            f"{delta['corrected_dice']:.4f}. Delta: {delta['delta']:+.4f}.",
            "",
            "### Context only — never a target",
            "",
            "| metric | mean | std |",
            "|---|---|---|",
            f"| PSNR (dB) | {context_only['psnr_db_mean']:.2f} "
            f"| {context_only['psnr_db_std']:.2f} |",
            f"| SSIM | {context_only['ssim_mean']:.4f} | {context_only['ssim_std']:.4f} |",
            "",
            context_only["note"],
            "",
        ]

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        path = self.metrics_dir / "summary.md"
        path.write_text("\n".join(lines))
        return path


__all__ = ["CONTEXT_ONLY_NOTE", "PER_VOLUME_FIELDS", "ReportGenerator"]
