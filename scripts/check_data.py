#!/usr/bin/env python
"""Verify that data/ and checkpoints/ hold REAL files before any eval or training run.

This gate exists because the inherited repo's seven ``.pth`` files are **Git LFS pointer
stubs** — 133-byte text files, not weights. Loading one raises a confusing unpickling error,
and the failure mode people actually hit is worse: they reach for ``strict=False``, the model
loads with random weights, and it produces plausible-looking garbage reconstructions.

Run ``make check-data`` before ``make slice`` / ``make eval`` / ``make train``.
Exit code 0 = everything present and real. Exit code 1 = something is missing or is a stub.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("MRI_AD_DATA") or REPO / "data")
CKPT = Path(os.environ.get("MRI_AD_CHECKPOINTS") or REPO / "checkpoints")

# What the pipeline needs. Checkpoint filenames match configs/model/*.yaml.
REQUIRED_CHECKPOINTS = [
    "unetr_mse_ssim_aug.pth",  # the primary model — best prior result
    "unet_mse.pth",
    "attunet_ssim_mse.pth",
]
REQUIRED_DATA_DIRS = [
    ("openbhb/val_quasiraw", "*.npy", "healthy T1 volumes (training distribution)"),
    ("brats/MICCAI_BraTS2020_TrainingData", "*", "tumor volumes + segmentation maps (eval)"),
]

HELP = f"""
How to fix:

  1. Checkpoints -> {CKPT}
     Download from the Google Drive folder linked in README.md and place the .pth files there,
     renamed to match configs/model/*.yaml (see REQUIRED_CHECKPOINTS in this script).
     The .pth files under legacy/ are LFS stubs and are NOT usable.

  2. Data -> {DATA}
     OpenBHB: https://www.kaggle.com/datasets/rahulkumargop/openbhb
     BraTS20: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
     Both require registration. Neither may ever be committed to git.

  Or point elsewhere:  export MRI_AD_DATA=/path  MRI_AD_CHECKPOINTS=/path
"""


def is_lfs_stub(path: Path) -> bool:
    """True if `path` is a Git LFS pointer file rather than real content."""
    try:
        with path.open("rb") as fh:
            return fh.read(len(LFS_MAGIC)) == LFS_MAGIC
    except OSError:
        return False


def main() -> int:
    problems: list[str] = []
    ok: list[str] = []

    for name in REQUIRED_CHECKPOINTS:
        path = CKPT / name
        if not path.exists():
            problems.append(f"MISSING  checkpoint  {path}")
        elif is_lfs_stub(path):
            problems.append(
                f"LFS STUB checkpoint  {path}  ({path.stat().st_size} bytes, no weights)"
            )
        else:
            mb = path.stat().st_size / 1e6
            ok.append(f"ok       checkpoint  {name}  ({mb:.1f} MB)")

    for rel, pattern, desc in REQUIRED_DATA_DIRS:
        path = DATA / rel
        if not path.is_dir():
            problems.append(f"MISSING  data        {path}  — {desc}")
            continue
        n = sum(1 for _ in path.glob(pattern))
        if n == 0:
            problems.append(f"EMPTY    data        {path}  — {desc}")
        else:
            ok.append(f"ok       data        {rel}  ({n} entries)")

    for line in ok:
        print(line)
    for line in problems:
        print(line, file=sys.stderr)

    if problems:
        print(f"\n{len(problems)} problem(s). Eval and training are BLOCKED.", file=sys.stderr)
        print(HELP, file=sys.stderr)
        return 1

    print("\nAll checkpoints and datasets present. Eval and training may proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
