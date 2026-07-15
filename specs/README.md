# Specs — the source of truth for WHAT to build

No implementation code is written without an approved spec here. The loop is
`/specify` → `spec-reviewer` → `/plan` → `/tasks` → implement → acceptance tests pass.
Specs define **what** and the acceptance tests; plans define **how**.

## Build order

```
000 vertical slice ─► 001 data ─► 002 models ─► 003 recon engine ─► 004 eval harness
                                                                          │
   ┌──────────────┬──────────────┬──────────────┬──────────────┐─────────┘
   ▼              ▼              ▼              ▼              ▼
 005 fidelity   006 classical  009 synth-    013 diffusion   (004 feeds all)
 vs-detection   baseline       anomaly       AnoDDPM
 study          │              (TRAIN)       (TRAIN)
   │            │                              │
   └────────────┴──► 007 paradigm comparison ◄─┘
                     (Classical vs UNETR vs Diffusion)
                                      │
                                      ▼
                              010 visualization ─► 011 reporting/README
                                                          │
                                                          ▼
                                            012 multi-scale attention (stretch)
```

Critical path: **000 → 001 → 002 → 003 → 004**. Then 005 / 006 / 009 / 013 run in parallel
(009 and 013 are the two `/train`-gated specs). 007 is the headline three-paradigm table and
depends on 005, 006, and 013.

## Index

| Spec | Title | Fresh compute | Status |
|---|---|---|---|
| [000](000-vertical-slice.md) | Vertical slice & reproducibility spine | no | approved |
| [001](001-data-layer.md) | Data layer | no | approved |
| [002](002-model-registry.md) | Model registry & checkpoint loading | no | approved |
| [003](003-recon-engine.md) | Reconstruction & anomaly-map engine | no | approved |
| [004](004-eval-harness.md) | Evaluation harness | no | approved |
| [005](005-arch-loss-study.md) | Fidelity-vs-detection study (arch × loss + diffusion) | no | approved |
| [006](006-classical-baseline.md) | Classical-ML baseline | no | approved |
| [007](007-dl-vs-classical.md) | Paradigm comparison — Classical vs UNETR vs Diffusion | no | approved |
| [009](009-synthetic-anomaly.md) | Synthetic-anomaly training (UNETR/FPI) | **YES** | approved |
| [010](010-visualization.md) | 3D visualization & demo video | no | approved |
| [011](011-reporting.md) | Reporting & public artifact | no | approved |
| [012](012-multiscale-attention.md) | Multi-scale attention variant | **YES** | stretch |
| [013](013-diffusion-anomaly.md) | Diffusion-based anomaly detection (AnoDDPM) | **YES** | draft |

## There is no Spec 008

The numbering jumps 007 → 009 **deliberately**. This mirrors
`3d-mri-anomaly-detection-sdd-plan.md` §5, and renumbering would break every cross-reference in
the plan and the requirements doc. Do not "fix" the gap.

## Why some specs deviate from the planning docs

The pre-rebuild audit found that parts of `planning/` describe a system that cannot be built as
written. Where a spec knowingly departs, it says so in its **Notes / deviations** section. The
three that matter:

- **002** — `planning/02-architecture.md:139` says wrap `monai.networks.nets.UNETR`. The trained
  weights will not load into it. We port the custom `UNETR_Reconstruction` instead.
- **003** — `planning/02-architecture.md:183` claims a 95th-percentile threshold "matches the
  prior work". It does not; the prior work used a hardcoded `residual > 0.1`.
- **004** — `FR-21` demands reproducing Dice 0.6255 ± 0.005. That number came from buggy code and
  is unreachable once the bugs are fixed. Replaced with an honest re-baseline.
