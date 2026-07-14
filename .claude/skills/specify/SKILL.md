---
name: specify
description: Turn a feature description into a spec file in specs/ following specs/_template.md. Step 1 of the SDD loop. Use when starting any new piece of work, or when asked to build something that has no spec yet.
---

# /specify — draft a spec

Step 1 of `/specify` → `spec-reviewer` → `/plan` → `/tasks` → implement.

A spec defines **WHAT** and its **acceptance tests**. It does not define HOW — that is `/plan`.
Deferring implementation detail is deliberate: it keeps the spec reviewable and stops us
committing to a design before we've agreed on the goal.

## Steps

1. Read `specs/_template.md` and `specs/README.md` (build order, existing specs).
2. Pick the number. Follow the existing sequence. **There is no Spec 008 — do not fill the gap.**
3. Draft `specs/NNN-kebab-name.md` with every template section:
   **Problem · Contract · Acceptance tests · Out of scope · Notes/deviations.**
4. Hand it to the `spec-reviewer` agent. Do not skip this.
5. Get the user's approval before any code is written.

## What makes an acceptance test real

It must be **falsifiable** — something a test can assert or a command can demonstrate.

- ✗ "The data layer works correctly."
- ✓ "Both dataloaders yield `(B,1,16,128,128)` float32 batches with values in `[0,1]`, and
   `DataValidator` raises `DescriptiveValidationError` naming the observed shape on a malformed batch."

Prefer tests that catch the *specific* way this could go wrong. The prior work's bugs were all
things a targeted test would have caught: a Dice averaged over one subject, a segmentation left
un-binarized, seven-eighths of a volume never scored. Write the test that would have caught the
bug you are most afraid of.

## Always check the spec against reality

The planning docs contain claims that do not survive contact with the code (a checkpoint that
cannot load into the class the architecture doc names; a threshold described as "matching prior
work" that does not). If the spec you are drafting depends on such a claim, **verify it first** and
record any departure in **Notes / deviations**, with the reason. Silent deviation is how the wrong
thing gets built.
