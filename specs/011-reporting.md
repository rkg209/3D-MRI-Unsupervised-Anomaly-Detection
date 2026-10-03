# Spec 011 · Reporting & public artifact

**Status:** implemented
**Depends on:** 005, 007, 009, 010
**Fresh compute required:** no

---

## Problem

The README *is* the deliverable for most readers. It must carry the diagram, an honest metric
scorecard, reproducibility instructions, and the mobility-transfer paragraph — and every number in
it must be regenerable from saved artifacts by one command, so it can never drift from the truth.

## Contract

`make report` (no GPU, no ML imports, < 2 min) regenerates the README scorecard **in place**
between markers:

```
<!-- SCORECARD_START -->  ... generated ...  <!-- SCORECARD_END -->
```

Scorecard: Dice, IoU (DL), ROC-AUC/PR-AUC (both), inference time per volume, GPU hours.

README also carries: an architecture diagram; reproducibility instructions (data registration
links, checkpoint source, exact commands); and the **mobility-transfer paragraph** — the conceptual
transfer of "reconstruct normal → flag deviations" to OOD detection in mobility/trajectory data,
explicitly labelled **a conceptual analogy, not a tested result** (D5, FR-44, NFR-22).

## Acceptance tests

1. `make report` rewrites the scorecard between the markers with zero manual editing.
2. **Every headline number traces to a file in `artifacts/`.** A test greps the README for numeric
   literals inside the scorecard block and asserts each appears in a source artifact — no number
   is typed by hand.
3. `make report` runs with no GPU and no ML dependency importable, in under 2 minutes.
4. The README states the **central finding** explicitly: reconstruction fidelity and detection
   quality are anti-correlated; detection separability is the target (NFR-23).
5. The mobility paragraph contains no performance claim and no driving-data code (NFR-22).
6. The README frames the project as a **comparative study**, never a clinical tool (NFR-21). A test
   greps for forbidden claim words ("clinical", "diagnostic", "FDA", "patient-ready").
7. The README carries the corrected Spec 004 baseline, and a short note on why it differs from the
   prior report's 0.6255.

## Out of scope

Hosting (C-8). Any new experiment.

## Notes / deviations

The current README has **no scorecard markers** and describes the old notebook project (it even
lists the wrong epoch count — 100; the notebooks trained 50). It is rewritten wholesale here.

Acceptance test 7 is an addition: shipping a README whose headline silently differs from the prior
report's published number, with no explanation, would look like an error rather than a correction.
The delta is a *feature* of this rebuild — it is the evidence that the audit was real.
