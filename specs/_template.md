# Spec NNN · <Title>

**Status:** draft | in-review | approved | implemented
**Depends on:** <spec ids, or "none">
**Fresh compute required:** no | yes (must be gated behind `/train` — manual invoke only)

---

## Problem

What we are building and *why it must exist*. State the failure or gap it closes.
Two to five sentences. No implementation detail — that belongs in the `/plan` step.

## Contract

The interface this spec must satisfy: inputs, outputs, shapes, types, invariants, and the
public API other specs may depend on. Be exact about tensor shapes — the system-wide invariant
is `(1, 16, 128, 128)`.

## Acceptance tests

Numbered, **concrete and falsifiable**. Each must be something a test can assert or a command
can demonstrate. "Works correctly" is not an acceptance test; "`make slice` prints a Dice in
[0,1] and writes `artifacts/figures/slice.png`" is.

1. …
2. …

## Out of scope

What this spec explicitly does *not* do, so the boundary with neighbouring specs is unambiguous.

## Notes / deviations

Any place this spec knowingly departs from `3d-mri-anomaly-detection-sdd-plan.md` or
`planning/*.md`, **with the reason**. Silent deviation is how the wrong thing gets built.
