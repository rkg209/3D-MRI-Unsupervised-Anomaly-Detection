"""Tables and plots for the classical baseline (Spec 006, acceptance 6).

Deliberately re-implements part of ``eval/report.py``'s discipline (plain mappings in/out, lazy
``matplotlib``) rather than importing it — acceptance test 7 forbids ``classical/`` from
importing ``eval/``. The duplication is intentional; see R8 in the plan and
``eval/metrics.aggregate``'s ``statistics.pstdev`` convention, which :mod:`mri_ad.classical.metrics`
also reimplements rather than shares.

**Granularity is stamped everywhere**: a column on every CSV row, a top-level JSON key plus
``granularity_note``, the markdown H1 plus a bolded caveat, and every plot's ``suptitle`` — so a
reader can never mistake this slice-level ROC-AUC for something comparable to voxel-level Dice.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from mri_ad.classical.metrics import GRANULARITY_NOTE, ClassicalMetrics
from mri_ad.exceptions import ArtifactError

PER_FOLD_FIELDS = (
    "granularity",
    "fold",
    "n_train",
    "n_test",
    "n_train_subjects",
    "n_test_subjects",
    "n_test_positive",
    "positive_rate",
    "roc_auc",
    "pr_auc",
    "pr_auc_prevalence_baseline",
    "roc_auc_majority_baseline",
    "scale_pos_weight",
)
FEATURE_IMPORTANCE_FIELDS = ("granularity", "rank", "feature", "mean_gain")
OOF_PREDICTION_FIELDS = ("granularity", "fold", "volume_id", "slice_index", "y_true", "y_score")
BASELINES_NOTE = (
    "A majority-class classifier scores ROC-AUC 0.5 by construction; the honest PR-AUC floor is "
    "the positive rate — read PR-AUC against that floor, never against 0."
)


class ClassicalReportGenerator:
    """Reads/writes ``artifacts/classical/{metrics,figures}/*``. No ML imports, plotting is lazy."""

    def __init__(self, metrics_dir: Path, figures_dir: Path) -> None:
        """Store the two output directories; nothing is created until a write method runs."""
        self.metrics_dir = Path(metrics_dir)
        self.figures_dir = Path(figures_dir)

    @property
    def _per_fold_path(self) -> Path:
        return self.metrics_dir / "per_fold.csv"

    @property
    def _metrics_json_path(self) -> Path:
        return self.metrics_dir / "classical_metrics.json"

    @property
    def _feature_importance_path(self) -> Path:
        return self.metrics_dir / "feature_importance.csv"

    @property
    def _summary_path(self) -> Path:
        return self.metrics_dir / "summary.md"

    @property
    def _oof_predictions_path(self) -> Path:
        return self.metrics_dir / "oof_predictions.csv"

    def write_predictions(self, rows: Sequence) -> Path:
        """Write ``oof_predictions.csv`` — one row per held-out slice (Spec 007 acceptance 1/3).

        ``rows`` is a sequence of :class:`~mri_ad.classical.baseline.OofPrediction`. Retained so
        Spec 007's paradigm comparison can recompute the classical ROC/PR curve with the exact
        same curve function every other column uses, rather than trusting a hand-summarized mean.
        """
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        with self._oof_predictions_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(OOF_PREDICTION_FIELDS))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        return self._oof_predictions_path

    def write_per_fold(self, metrics: ClassicalMetrics) -> Path:
        """Write ``per_fold.csv`` — every row carries ``granularity`` (acceptance 6)."""
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        with self._per_fold_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(PER_FOLD_FIELDS))
            writer.writeheader()
            for fold in metrics.folds:
                writer.writerow({"granularity": metrics.granularity, **asdict(fold)})
        return self._per_fold_path

    def write_metrics_json(
        self,
        metrics: ClassicalMetrics,
        *,
        run_id: str,
        seed: int,
        complete: bool,
        subject_limit: int | None,
        split_info: Mapping[str, object],
        features_info: Mapping[str, object],
        counts_info: Mapping[str, object],
        classifier_info: Mapping[str, object],
    ) -> Path:
        """Write ``classical_metrics.json`` — the full provenance + headline + baselines record."""
        payload = {
            "spec": "006",
            "granularity": metrics.granularity,
            "granularity_note": GRANULARITY_NOTE,
            "run_id": run_id,
            "seed": seed,
            "complete": complete,
            "subject_limit": subject_limit,
            "split": dict(split_info),
            "features": dict(features_info),
            "counts": dict(counts_info),
            "class_balance": {
                "n_positive": metrics.n_positive,
                "n_negative": metrics.n_samples - metrics.n_positive,
                "positive_rate": metrics.positive_rate,
                "per_fold_positive_rate": [fold.positive_rate for fold in metrics.folds],
            },
            "headline": {
                "roc_auc_mean": metrics.roc_auc_mean,
                "roc_auc_std": metrics.roc_auc_std,
                "pr_auc_mean": metrics.pr_auc_mean,
                "pr_auc_std": metrics.pr_auc_std,
            },
            "baselines": {
                "roc_auc_majority": metrics.roc_auc_majority_baseline,
                "pr_auc_prevalence_mean": metrics.pr_auc_prevalence_baseline_mean,
                "note": BASELINES_NOTE,
            },
            "classifier": dict(classifier_info),
            "folds": [asdict(fold) for fold in metrics.folds],
        }
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return self._metrics_json_path

    def read_metrics_json(self) -> dict:
        """Read ``classical_metrics.json`` back. Raises :class:`ArtifactError` if absent."""
        if not self._metrics_json_path.is_file():
            raise ArtifactError(
                f"No classical metrics at {self._metrics_json_path}. Run `make classical` first."
            )
        return json.loads(self._metrics_json_path.read_text())

    def write_feature_importance(self, importance: Mapping[str, float]) -> Path:
        """Write ``feature_importance.csv``, ranked by mean gain descending."""
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        with self._feature_importance_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(FEATURE_IMPORTANCE_FIELDS))
            writer.writeheader()
            for rank, (feature, gain) in enumerate(ranked, start=1):
                writer.writerow(
                    {
                        "granularity": "slice-level",
                        "rank": rank,
                        "feature": feature,
                        "mean_gain": gain,
                    }
                )
        return self._feature_importance_path

    def write_summary_markdown(self) -> Path:
        """Render ``summary.md`` from the already-written ``classical_metrics.json``."""
        payload = self.read_metrics_json()
        headline = payload["headline"]
        baselines = payload["baselines"]
        class_balance = payload["class_balance"]

        lines = [
            f"# Classical baseline — {payload['granularity']}",
            "",
            f"**{payload['granularity_note']}**",
            "",
            f"n_samples={payload['counts'].get('n_samples', '?')}, "
            f"n_positive={class_balance['n_positive']}, "
            f"positive_rate={class_balance['positive_rate']:.4f}",
            "",
            "### Headline — ROC-AUC / PR-AUC (mean +/- std across folds)",
            "",
            "| metric | mean | std |",
            "|---|---|---|",
            f"| ROC-AUC | {headline['roc_auc_mean']:.4f} | {headline['roc_auc_std']:.4f} |",
            f"| PR-AUC | {headline['pr_auc_mean']:.4f} | {headline['pr_auc_std']:.4f} |",
            "",
            "### Baselines",
            "",
            f"Majority-class ROC-AUC: {baselines['roc_auc_majority']:.4f}. "
            f"Prevalence PR-AUC floor: {baselines['pr_auc_prevalence_mean']:.4f}.",
            "",
            baselines["note"],
            "",
        ]

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._summary_path.write_text("\n".join(lines))
        return self._summary_path

    def plot_class_balance(self, class_balance: Mapping[str, object]) -> Path:
        """Bar chart of positive vs. negative slice counts. ``matplotlib`` imported lazily."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        path = self.figures_dir / "class_balance.png"

        n_positive = int(class_balance["n_positive"])
        n_negative = int(class_balance["n_negative"])
        fig, ax = plt.subplots()
        ax.bar(["negative", "positive"], [n_negative, n_positive])
        ax.set_ylabel("Slice count")
        fig.suptitle(f"Class balance ({GRANULARITY_NOTE})", wrap=True, fontsize=8)
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_fold_auc(self, metrics: ClassicalMetrics) -> Path:
        """Per-fold ROC-AUC / PR-AUC bar chart. ``matplotlib`` imported lazily."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        path = self.figures_dir / "fold_auc.png"

        folds = [fold.fold for fold in metrics.folds]
        roc = [fold.roc_auc for fold in metrics.folds]
        pr = [fold.pr_auc for fold in metrics.folds]
        fig, ax = plt.subplots()
        width = 0.35
        ax.bar([f - width / 2 for f in folds], roc, width, label="ROC-AUC")
        ax.bar([f + width / 2 for f in folds], pr, width, label="PR-AUC")
        ax.set_xlabel("Fold")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        fig.suptitle(f"Per-fold AUC ({GRANULARITY_NOTE})", wrap=True, fontsize=8)
        fig.savefig(path)
        plt.close(fig)
        return path


__all__ = [
    "BASELINES_NOTE",
    "FEATURE_IMPORTANCE_FIELDS",
    "OOF_PREDICTION_FIELDS",
    "PER_FOLD_FIELDS",
    "ClassicalReportGenerator",
]
