"""Spec 011 acceptance tests + unit tests for the reporting/README-scorecard machinery.

Mirrors the house convention (``tests/test_matrix.py``, ``tests/test_paradigm.py``): synthetic
artifacts written under ``tmp_path`` via the real writers, never a real forward pass. No
``conftest.py`` — module-level ``REPO`` + private ``_builder()`` helpers instead of fixtures.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mri_ad.eval.readme_block import splice, would_change, write_block
from mri_ad.eval.scorecard import ScorecardEntry, load_scorecard, load_timing, render_markdown
from mri_ad.exceptions import ArtifactError

REPO = Path(__file__).resolve().parent.parent

START = "<!-- SCORECARD_START -->"
END = "<!-- SCORECARD_END -->"


# ── readme_block.splice ─────────────────────────────────────────────────────────────────────
def test_splice_replaces_only_the_content_between_the_markers() -> None:
    text = f"# Title\n\nintro\n\n{START}\nstale junk\n{END}\n\noutro\n"
    result = splice(text, "fresh block", start=START, end=END)
    assert result == f"# Title\n\nintro\n\n{START}\nfresh block\n{END}\n\noutro\n"


def test_splice_raises_on_missing_start_marker() -> None:
    with pytest.raises(ArtifactError, match="not found"):
        splice(f"no markers here\n{END}\n", "x", start=START, end=END)


def test_splice_raises_on_missing_end_marker() -> None:
    with pytest.raises(ArtifactError, match="not found"):
        splice(f"{START}\nno end\n", "x", start=START, end=END)


def test_splice_raises_on_duplicate_start_marker() -> None:
    text = f"{START}\na\n{START}\nb\n{END}\n"
    with pytest.raises(ArtifactError, match="appears 2 times"):
        splice(text, "x", start=START, end=END)


def test_splice_raises_on_duplicate_end_marker() -> None:
    text = f"{START}\na\n{END}\nb\n{END}\n"
    with pytest.raises(ArtifactError, match="appears 2 times"):
        splice(text, "x", start=START, end=END)


def test_splice_raises_on_inverted_markers() -> None:
    text = f"{END}\nbetween\n{START}\n"
    with pytest.raises(ArtifactError, match="inverted"):
        splice(text, "x", start=START, end=END)


# ── readme_block.write_block / would_change ─────────────────────────────────────────────────
def test_write_block_changes_the_file_and_reports_true(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{START}\nold\n{END}\nafter\n")
    changed = write_block(readme, "new", start=START, end=END)
    assert changed is True
    assert "new" in readme.read_text()
    assert "old" not in readme.read_text()


def test_write_block_is_idempotent_on_the_second_call(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{START}\nold\n{END}\nafter\n")
    write_block(readme, "new", start=START, end=END)
    second = write_block(readme, "new", start=START, end=END)
    assert second is False


def test_would_change_is_true_before_a_write_and_false_after(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{START}\nold\n{END}\nafter\n")
    assert would_change(readme, "new", start=START, end=END) is True
    write_block(readme, "new", start=START, end=END)
    assert would_change(readme, "new", start=START, end=END) is False


# ── RunLogger.timer ──────────────────────────────────────────────────────────────────────────
def _run_logger_cfg(tmp_path: Path):
    from omegaconf import OmegaConf

    return OmegaConf.create({"seed": 42, "paths": {"artifact_root": str(tmp_path)}})


def test_timer_accumulates_wall_clock_seconds_into_metrics_timings(tmp_path: Path) -> None:
    from mri_ad.utils.run_logger import RunLogger

    with RunLogger(_run_logger_cfg(tmp_path)) as run:
        with run.timer("inference"):
            pass
        with run.timer("inference"):
            pass

    assert "inference" in run.metrics["timings"]
    assert run.metrics["timings"]["inference"] >= 0.0


def test_timer_keeps_separate_names_independent(tmp_path: Path) -> None:
    from mri_ad.utils.run_logger import RunLogger

    with RunLogger(_run_logger_cfg(tmp_path)) as run:
        with run.timer("inference"):
            pass
        with run.timer("io"):
            pass

    assert set(run.metrics["timings"]) == {"inference", "io"}


def test_run_meta_json_records_duration_seconds(tmp_path: Path) -> None:

    from mri_ad.utils.run_logger import RunLogger

    with RunLogger(_run_logger_cfg(tmp_path)) as run:
        pass

    meta = json.loads((run.run_dir / "run_meta.json").read_text())
    assert "duration_seconds" in meta
    assert meta["duration_seconds"] >= 0.0


# ── eval/scorecard.py ────────────────────────────────────────────────────────────────────────
def _report_cfg(tmp_path: Path) -> dict:
    return {
        "sources": {
            "matrix_json": str(tmp_path / "tables" / "arch_loss_matrix.json"),
            "paradigm_json": str(tmp_path / "tables" / "paradigm_comparison.json"),
            "classical_json": str(tmp_path / "classical" / "metrics" / "classical_metrics.json"),
            "synth_csv": str(tmp_path / "tables" / "synth_before_after.csv"),
            "aggregate_dir": str(tmp_path / "metrics"),
            "runs_root": str(tmp_path / "runs"),
        },
        "headline_cell": {"model": "unetr", "loss": "mse_ssim"},
        "precision": {
            "dice": ".4f",
            "iou": ".4f",
            "auc": ".4f",
            "seconds": ".1f",
            "gpu_hours": ".2f",
        },
        "na_reason_default": "not yet evaluated",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _matrix_json(tmp_path: Path, *, dice: float, iou: float) -> None:
    _write_json(
        tmp_path / "tables" / "arch_loss_matrix.json",
        {"cells": [{"cell_id": "unetr__mse_ssim", "dice": dice, "iou": iou, "na_reason": None}]},
    )


def _paradigm_json(
    tmp_path: Path,
    *,
    dl_roc: float,
    dl_pr: float,
    classical_roc: float | None = None,
    classical_pr: float | None = None,
    split_hash: str = "hash-a",
) -> None:
    columns = [
        {"key": "unetr__mse_ssim", "roc_auc": dl_roc, "pr_auc": dl_pr, "na_reason": None},
    ]
    if classical_roc is not None:
        columns.append(
            {
                "key": "classical",
                "roc_auc": classical_roc,
                "pr_auc": classical_pr,
                "na_reason": None,
            }
        )
    _write_json(
        tmp_path / "tables" / "paradigm_comparison.json",
        {"split_hash": split_hash, "columns": columns},
    )


def _classical_json(tmp_path: Path, *, roc: float, pr: float, split_hash: str = "hash-a") -> None:
    _write_json(
        tmp_path / "classical" / "metrics" / "classical_metrics.json",
        {"split": {"split_hash": split_hash}, "headline": {"roc_auc_mean": roc, "pr_auc_mean": pr}},
    )


def _synth_csv(tmp_path: Path, *, before: float, after: float) -> None:
    path = tmp_path / "tables" / "synth_before_after.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell", "dice_mean"])
        writer.writerow(["before", before])
        writer.writerow(["after", after])


def _run_meta(
    tmp_path: Path,
    stamp: str,
    *,
    model: str = "unetr",
    loss: str = "mse_ssim",
    seconds_per_volume: float | None = 12.5,
    gpu_hours: float | None = None,
    device: str = "cpu",
) -> None:
    metrics: dict = {"device": device}
    if seconds_per_volume is not None:
        metrics["seconds_per_volume"] = seconds_per_volume
        metrics["gpu_hours"] = gpu_hours
    _write_json(
        tmp_path / "runs" / stamp / "run_meta.json",
        {"config": {"model": {"name": model}, "loss": {"name": loss}}, "metrics": metrics},
    )


def _aggregate_json(
    tmp_path: Path, *, dice: float, delta: float, split_hash: str = "hash-a"
) -> None:
    _write_json(
        tmp_path / "metrics" / "unetr__mse_ssim" / "aggregate.json",
        {
            "split_hash": split_hash,
            "headline": {"dice_mean": dice},
            "baseline_delta": {"delta": delta},
        },
    )


def test_load_scorecard_degrades_to_na_reason_on_every_missing_source(tmp_path: Path) -> None:
    card = load_scorecard(_report_cfg(tmp_path), tmp_path)
    assert card.n_available == 0
    for entry in card.entries:
        assert not entry.available
        assert entry.na_reason is not None
        assert "n/a (" in entry.render()


def test_load_scorecard_joins_every_declared_source(tmp_path: Path) -> None:
    _matrix_json(tmp_path, dice=0.55, iou=0.4)
    _paradigm_json(tmp_path, dl_roc=0.8, dl_pr=0.7)
    _classical_json(tmp_path, roc=0.6, pr=0.5)
    _synth_csv(tmp_path, before=0.5, after=0.6)
    _run_meta(tmp_path, "20260101T000000_000000Z", seconds_per_volume=12.5, device="cpu")
    _aggregate_json(tmp_path, dice=0.55, delta=0.55 - 0.6255)

    card = load_scorecard(_report_cfg(tmp_path), tmp_path)
    by_key = {e.key: e for e in card.entries}

    assert by_key["dice_dl"].value == pytest.approx(0.55)
    assert by_key["iou_dl"].value == pytest.approx(0.4)
    assert by_key["roc_auc_dl"].value == pytest.approx(0.8)
    assert by_key["pr_auc_dl"].value == pytest.approx(0.7)
    assert by_key["roc_auc_classical"].value == pytest.approx(0.6)
    assert by_key["pr_auc_classical"].value == pytest.approx(0.5)
    assert by_key["synth_dice_before"].value == pytest.approx(0.5)
    assert by_key["synth_dice_after"].value == pytest.approx(0.6)
    assert by_key["inference_seconds_per_volume"].value == pytest.approx(12.5)
    assert by_key["baseline_corrected_dice"].value == pytest.approx(0.55)
    assert card.split_hash == "hash-a"
    assert card.n_available == len(card.entries) - 1  # gpu_hours stays n/a on a cpu run


def test_load_scorecard_gpu_hours_na_on_cpu_run_shows_device(tmp_path: Path) -> None:
    _run_meta(tmp_path, "20260101T000000_000000Z", seconds_per_volume=12.5, device="cpu")
    card = load_scorecard(_report_cfg(tmp_path), tmp_path)
    gpu_entry = next(e for e in card.entries if e.key == "gpu_hours")
    assert not gpu_entry.available
    assert "cpu" in gpu_entry.render()
    seconds_entry = next(e for e in card.entries if e.key == "inference_seconds_per_volume")
    assert seconds_entry.available


def test_load_scorecard_never_publishes_gpu_hours_from_a_non_cuda_device(tmp_path: Path) -> None:
    _run_meta(
        tmp_path,
        "20260101T000000_000000Z",
        seconds_per_volume=9.0,
        gpu_hours=None,
        device="mps",
    )
    card = load_scorecard(_report_cfg(tmp_path), tmp_path)
    gpu_entry = next(e for e in card.entries if e.key == "gpu_hours")
    assert gpu_entry.value is None


def test_load_scorecard_surfaces_split_hash_mismatch_instead_of_blending(tmp_path: Path) -> None:
    _paradigm_json(tmp_path, dl_roc=0.8, dl_pr=0.7, split_hash="hash-a")
    _classical_json(tmp_path, roc=0.6, pr=0.5, split_hash="hash-b")
    _aggregate_json(tmp_path, dice=0.55, delta=0.0, split_hash="hash-a")

    card = load_scorecard(_report_cfg(tmp_path), tmp_path)
    by_key = {e.key: e for e in card.entries}

    assert card.split_hash is None
    assert by_key["roc_auc_dl"].value is None
    assert "split mismatch" in by_key["roc_auc_dl"].na_reason
    assert by_key["roc_auc_classical"].value is None
    assert by_key["baseline_corrected_dice"].value is None


def test_load_scorecard_rounds_at_configured_precision_only(tmp_path: Path) -> None:
    _matrix_json(tmp_path, dice=0.123456789, iou=0.4)
    card = load_scorecard(_report_cfg(tmp_path), tmp_path)
    dice_entry = next(e for e in card.entries if e.key == "dice_dl")
    assert dice_entry.render() == "0.1235"


def test_scorecard_entry_raises_when_a_value_has_no_source_path() -> None:
    with pytest.raises(ArtifactError, match="source_path"):
        ScorecardEntry(
            key="x",
            label="x",
            value=0.5,
            fmt=".4f",
            unit="",
            source_path=None,
            run_id=None,
            na_reason=None,
        )


def test_render_markdown_lists_every_entry_and_the_source_list(tmp_path: Path) -> None:
    _matrix_json(tmp_path, dice=0.55, iou=0.4)
    card = load_scorecard(_report_cfg(tmp_path), tmp_path)
    block = render_markdown(card)
    assert "0.5500" in block
    assert "tables/arch_loss_matrix.json" in block
    for entry in card.entries:
        assert entry.label in block


def test_load_scorecard_sources_are_relative_never_an_absolute_local_path(tmp_path: Path) -> None:
    _matrix_json(tmp_path, dice=0.55, iou=0.4)
    card = load_scorecard(_report_cfg(tmp_path), tmp_path)
    for source in card.sources:
        assert not Path(source).is_absolute()
        assert str(tmp_path) not in source


# ── load_timing ──────────────────────────────────────────────────────────────────────────────
def test_load_timing_picks_the_newest_matching_run(tmp_path: Path) -> None:
    _run_meta(tmp_path, "20260101T000000_000000Z", seconds_per_volume=10.0)
    _run_meta(tmp_path, "20260102T000000_000000Z", seconds_per_volume=20.0)
    timing = load_timing(tmp_path / "runs", model="unetr", loss="mse_ssim")
    assert timing["seconds_per_volume"] == pytest.approx(20.0)
    assert timing["run_id"] == "20260102T000000_000000Z"


def test_load_timing_skips_runs_missing_seconds_per_volume(tmp_path: Path) -> None:
    _run_meta(tmp_path, "20260102T000000_000000Z", seconds_per_volume=None)
    _run_meta(tmp_path, "20260101T000000_000000Z", seconds_per_volume=15.0)
    timing = load_timing(tmp_path / "runs", model="unetr", loss="mse_ssim")
    assert timing["run_id"] == "20260101T000000_000000Z"


def test_load_timing_skips_runs_for_a_different_model_or_loss(tmp_path: Path) -> None:
    _run_meta(tmp_path, "20260101T000000_000000Z", model="unet", seconds_per_volume=10.0)
    timing = load_timing(tmp_path / "runs", model="unetr", loss="mse_ssim")
    assert timing == {}


def test_load_timing_returns_empty_for_a_missing_runs_root(tmp_path: Path) -> None:
    timing = load_timing(tmp_path / "does-not-exist", model="unetr", loss="mse_ssim")
    assert timing == {}


# ══ Spec 011 acceptance tests — run against the real, committed README.md ═════════════════════
def _block(text: str, start: str, end: str) -> str:
    """The interior text strictly between the first ``start``/``end`` marker pair."""
    start_idx = text.index(start) + len(start)
    end_idx = text.index(end)
    return text[start_idx:end_idx]


def _readme_text() -> str:
    return (REPO / "README.md").read_text()


def _all_numeric_leaves(obj: object) -> set[str]:
    """Every ``int``/``float`` leaf in a nested JSON-like structure, stringified."""
    found: set[str] = set()
    if isinstance(obj, bool):
        return found
    if isinstance(obj, int | float):
        found.add(str(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= _all_numeric_leaves(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            found |= _all_numeric_leaves(v)
    return found


# ── AT-1: `make report` rewrites the scorecard between the markers, zero manual editing ────────
def test_scorecard_block_is_rewritten_between_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Title\n\nintro text\n\n{START}\nSTALE JUNK FROM A PRIOR RUN\n{END}\n\noutro text\n"
    )
    write_block(readme, "fresh generated content", start=START, end=END)
    result = readme.read_text()
    assert "STALE JUNK FROM A PRIOR RUN" not in result
    assert "fresh generated content" in result
    assert result.startswith("# Title\n\nintro text\n\n")
    assert result.endswith("\n\noutro text\n")


def test_running_report_twice_is_byte_identical(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{START}\nstale\n{END}\nafter\n")
    write_block(readme, "generated", start=START, end=END)
    first = readme.read_bytes()
    write_block(readme, "generated", start=START, end=END)
    second = readme.read_bytes()
    assert first == second


# ── AT-2: every headline number in the README scorecard traces to a source artifact ────────────
def test_every_number_in_the_readme_scorecard_traces_to_an_artifact() -> None:
    import re

    import yaml

    text = _readme_text()
    block = _block(text, START, END)
    literals = set(re.findall(r"-?\d+\.\d+", block))

    report_cfg = yaml.safe_load((REPO / "configs" / "report" / "default.yaml").read_text())
    source_values: set[str] = set()
    for rel_source in ("tables/arch_loss_matrix.json", "tables/paradigm_comparison.json"):
        path = REPO / "artifacts" / rel_source
        if path.is_file():
            source_values |= _all_numeric_leaves(json.loads(path.read_text()))
    del report_cfg  # only used to confirm the config exists; sources enumerated explicitly above

    for literal in literals:
        assert literal in source_values, f"{literal!r} in the live scorecard traces to no source"


def test_hand_typed_number_in_the_scorecard_block_is_detectable(tmp_path: Path) -> None:
    """A regression guard on the detector itself: a hand-typed number must fail the AT-2 check."""
    import re

    fake_readme = f"{START}\n| Dice | 0.9999 |\n{END}\n"
    block = _block(fake_readme, START, END)
    literals = set(re.findall(r"-?\d+\.\d+", block))
    source_values: set[str] = set()  # no artifacts on disk in this scenario
    offending = literals - source_values
    assert offending == {"0.9999"}


# ── AT-3: no GPU, no ML dependency importable — covered by test_eval_boundary.py's ─────────────
# FORBIDDEN_ML_MODULES subprocess checks (extended there for readme_block/scorecard) and the
# existing test_report_generation_finishes_well_under_the_two_minute_budget.


# ── AT-4: the README states the central finding, verbatim, in sync with the constant ───────────
def test_readme_states_the_central_finding() -> None:
    from mri_ad.eval.scorecard import CENTRAL_FINDING_SENTENCE

    normalized = " ".join(_readme_text().split())
    expected = " ".join(CENTRAL_FINDING_SENTENCE.split())
    assert expected in normalized


# ── AT-5: the mobility paragraph carries no performance claim ──────────────────────────────────
def test_mobility_paragraph_has_no_performance_claim() -> None:
    from mri_ad.eval.scorecard import MOBILITY_DISCLAIMER

    text = _readme_text()
    block = _block(text, "<!-- MOBILITY_START -->", "<!-- MOBILITY_END -->")
    assert MOBILITY_DISCLAIMER in block
    assert "```" not in block
    for banned in ("Dice", "IoU", "AUC", "accuracy", "outperform", "%"):
        assert banned not in block, f"mobility paragraph contains a performance claim: {banned!r}"


# ── AT-6: comparative study, never a clinical tool — forbidden words confined to the disclaimer ─
def test_readme_uses_no_clinical_claim_words_outside_the_disclaimer() -> None:
    forbidden = ("clinical", "diagnostic", "diagnose", "fda", "patient-ready")
    for line in _readme_text().splitlines():
        if line.strip().startswith(">"):
            continue  # the designated disclaimer blockquote is exempt
        lowered = line.lower()
        for word in forbidden:
            assert word not in lowered, (
                f"forbidden claim word {word!r} outside the disclaimer: {line!r}"
            )


def test_readme_disclaimer_actually_contains_the_forbidden_words() -> None:
    """A regression guard: the disclaimer exemption above must not be vacuously true."""
    blockquote_lines = [
        line for line in _readme_text().splitlines() if line.strip().startswith(">")
    ]
    lowered = " ".join(blockquote_lines).lower()
    assert "clinical" in lowered
    assert "diagnostic" in lowered


# ── AT-7: the README carries the corrected Spec 004 baseline and explains the 0.6255 delta ─────
def test_readme_carries_the_corrected_baseline_and_explains_the_delta() -> None:
    import yaml

    eval_cfg = yaml.safe_load((REPO / "configs" / "eval" / "default.yaml").read_text())
    published_dice = eval_cfg["legacy"]["published_dice"]

    text = _readme_text()
    assert f"{published_dice}" in text
    assert "corrected" in text.lower()
    assert "four independent bugs" in text.lower() or "four bugs" in text.lower()

    aggregate_path = REPO / "artifacts" / "metrics" / "unetr__mse_ssim" / "aggregate.json"
    if aggregate_path.is_file():
        agg = json.loads(aggregate_path.read_text())
        corrected = f"{agg['headline']['dice_mean']:.4f}"
        assert corrected in _block(text, START, END)
    else:
        assert "n/a" in _block(text, START, END)
