#!/usr/bin/env python3
"""PreToolUse guard on Bash. Blocks commands that violate the project's hard rules.

Reads the hook payload on stdin. Exit 0 = allow; exit 2 = block, with the reason on stderr
(which is fed back to Claude so it can correct course).

Enforces three rules from CLAUDE.md:
  1. No `Co-Authored-By:` trailer in any commit message — it breaks the user's GitHub push.
  2. No committing data/, checkpoints/, or imaging/weight binaries — licensing + repo hygiene.
  3. No `rm -rf` on a path that isn't clearly scratch.
"""

from __future__ import annotations

import json
import re
import sys

# 1. The commit-trailer rule. Matches any Co-Authored-By, any model, any spacing.
CO_AUTHORED = re.compile(r"co-authored-by\s*:", re.IGNORECASE)

# 2. Paths that must never enter a commit.
FORBIDDEN_PATHS = re.compile(
    r"(^|[\s/'\"])(data|checkpoints)/|\.(pt|pth|nii|nii\.gz|npy)(\s|$|['\"])",
    re.IGNORECASE,
)

# 3. Destructive removal.
RM_RF = re.compile(r"\brm\s+(-[a-zA-Z]*[rR][a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*[rR])\b")
SAFE_RM_TARGETS = re.compile(r"/(tmp|scratchpad|\.pytest_cache|__pycache__|\.ruff_cache)")


def block(msg: str) -> None:
    print(f"BLOCKED by .claude/hooks/guard_bash.py\n\n{msg}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    raw = sys.stdin.read()

    try:
        command = json.loads(raw).get("tool_input", {}).get("command", "")
    except (json.JSONDecodeError, ValueError):
        # Unparseable payload. Don't break the harness for the soft rules, but do NOT fail open
        # on the commit-trailer rule — scan the raw text instead. A malformed payload must not
        # become a way for a Co-Authored-By commit to slip through.
        if re.search(r"\bgit\s+commit\b", raw) and CO_AUTHORED.search(raw):
            block(
                "This commit message appears to contain a `Co-Authored-By:` trailer.\n\n"
                "CLAUDE.md rule 1: NEVER add a Co-Authored-By trailer to a git commit."
            )
        sys.exit(0)

    if not command:
        sys.exit(0)

    is_commit = re.search(r"\bgit\s+commit\b", command) is not None
    is_staging = re.search(r"\bgit\s+(add|commit)\b", command) is not None

    if is_commit and CO_AUTHORED.search(command):
        block(
            "This commit message contains a `Co-Authored-By:` trailer.\n\n"
            "CLAUDE.md rule 1: NEVER add a Co-Authored-By trailer to a git commit. It breaks the\n"
            "user's push to GitHub. Rewrite the commit message without it and try again."
        )

    if is_staging and FORBIDDEN_PATHS.search(command):
        block(
            "This command would stage or commit data, checkpoints, or an imaging/weight binary.\n\n"
            "CLAUDE.md rule 5: data/, checkpoints/, *.pt, *.pth, *.nii*, and *.npy must NEVER be\n"
            "committed — a medical-data licensing rule as well as repo hygiene. They are already\n"
            "in .gitignore; do not force-add them."
        )

    if RM_RF.search(command) and not SAFE_RM_TARGETS.search(command):
        block(
            "`rm -rf` on a non-scratch path.\n\n"
            "If this is genuinely needed, the user should run it themselves. Deleting data or\n"
            "checkpoints is unrecoverable — they are large, registration-gated downloads."
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
