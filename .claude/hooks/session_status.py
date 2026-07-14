#!/usr/bin/env python3
"""SessionStart hook — orient the agent in one short block.

Prints spec status, whether real weights/data are present, and the current branch. Keeps
long-running work from drifting: the single most common way to waste a session here is to start
implementing against checkpoints that turn out to be LFS stubs.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def spec_status() -> tuple[int, int]:
    specs = sorted(p for p in (REPO / "specs").glob("[0-9]*.md"))
    implemented = 0
    for spec in specs:
        text = spec.read_text(errors="ignore")
        if re.search(r"^\*\*Status:\*\*.*implemented", text, re.MULTILINE | re.IGNORECASE):
            implemented += 1
    return implemented, len(specs)


def data_ready() -> tuple[bool, str]:
    ckpt_dir = REPO / "checkpoints"
    if not ckpt_dir.is_dir():
        return False, "checkpoints/ missing"
    real = [
        p
        for p in ckpt_dir.glob("*.pth")
        if p.read_bytes()[: len(LFS_MAGIC)] != LFS_MAGIC
    ]
    if not real:
        return False, "no real checkpoints (LFS stubs or empty)"
    if not (REPO / "data").is_dir():
        return False, f"{len(real)} checkpoint(s), but data/ missing"
    return True, f"{len(real)} checkpoint(s) + data/ present"


def main() -> None:
    done, total = spec_status()
    ready, detail = data_ready()

    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=REPO, capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "(detached)"
    except (subprocess.SubprocessError, OSError):
        branch = "(unknown)"

    print(f"[mri-ad] branch: {branch} | specs implemented: {done}/{total}")
    if ready:
        print(f"[mri-ad] data: READY — {detail}")
    else:
        print(f"[mri-ad] data: NOT READY — {detail}. Eval/training blocked; run `make check-data`.")
        print("[mri-ad] You can still build and unit-test against synthetic tensors.")
    print("[mri-ad] Reminders: no code without an approved spec · never autonomously train ·")
    print("[mri-ad]            append to progress_report.md · NO Co-Authored-By in commits.")


if __name__ == "__main__":
    main()
