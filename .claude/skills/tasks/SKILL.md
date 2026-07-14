---
name: tasks
description: Decompose an approved plan into an ordered, individually testable task list. Step 3 of the SDD loop, run after /plan and before implementation.
---

# /tasks — plan → ordered task list

Step 3 of the SDD loop. Turn the plan into tasks that can be implemented and verified **one at a
time**, in order.

## Each task must be

- **Independently testable** — it lands with a passing test, not "we'll test it at the end".
- **Small** — one file, one interface, one behaviour. If it needs three sentences to describe, split it.
- **Ordered** — dependencies first. Never a task that cannot be verified until a later one lands.
- **Traceable** — say which acceptance criterion from the spec it moves toward.

## Order

1. Types, dataclasses, exceptions (no logic — makes the shape of things reviewable).
2. The unit under test + its test, one at a time.
3. Wiring: config, entry point, Makefile target.
4. The acceptance tests from the spec, run end to end.

Use `TaskCreate`/`TaskUpdate` so progress is visible.

## After each task

Run `make test`. Then **append to `progress_report.md`** if the task represents a meaningful change
— what, why, how, and anything that went wrong. Do not batch this up at the end; the point of the
report is the sequence.
