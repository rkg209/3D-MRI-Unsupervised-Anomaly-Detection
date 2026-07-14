#!/usr/bin/env python3
"""PostToolUse hook — validate a written spec against the template schema.

Rejects incomplete specs early: a spec without falsifiable acceptance tests is the main way the
SDD loop degrades into "write some code and hope". Warns (exit 0 with stderr) rather than blocking,
since the file is already written by this point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = ["## Problem", "## Acceptance tests", "## Out of scope"]


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    path_str = payload.get("tool_input", {}).get("file_path", "")
    path = Path(path_str)

    # Only real spec files: specs/NNN-*.md. Skip _template.md and README.md.
    if "specs" not in path.parts or path.suffix != ".md":
        sys.exit(0)
    if path.name.startswith("_") or path.name == "README.md":
        sys.exit(0)
    if not path.name[:3].isdigit():
        sys.exit(0)
    if not path.exists():
        sys.exit(0)

    text = path.read_text(errors="ignore")
    missing = [s for s in REQUIRED if s not in text]

    if missing:
        print(
            f"[spec-validator] {path.name} is missing required section(s): "
            f"{', '.join(missing)}. See specs/_template.md.",
            file=sys.stderr,
        )
        sys.exit(0)

    # An acceptance-tests section with no numbered items is not an acceptance-tests section.
    body = text.split("## Acceptance tests", 1)[1].split("##", 1)[0]
    if not any(line.strip().startswith(("1.", "2.")) for line in body.splitlines()):
        print(
            f"[spec-validator] {path.name}: 'Acceptance tests' has no numbered, falsifiable "
            "criteria. Each must be something a test can assert or a command can demonstrate.",
            file=sys.stderr,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
