---
name: plan
description: Turn an approved spec into a technical plan — file list, interfaces, model/loss choices, the experiment matrix, and risks. Step 2 of the SDD loop. Use after a spec is approved and before writing any implementation code.
---

# /plan — spec → technical plan

Step 2 of the SDD loop. **The spec must be approved first.** This is where the implementation
detail the spec deliberately deferred gets decided.

## Produce

1. **Files** — every file created or modified, with its responsibility in one line.
2. **Interfaces** — exact signatures, shapes, dtypes, and the exceptions raised.
3. **Decisions** — the choices the spec left open, each with a *reason* and the alternatives rejected.
   (E.g. Spec 006's slice-level vs supervoxel granularity; Spec 010's viewer technology.)
4. **Experiment matrix**, if any — the exact runs, and which are gated behind `/train`.
5. **Risks** — what could silently produce a wrong number, and how a test catches it.
6. **Test plan** — one test per acceptance criterion in the spec. Map them explicitly.

## Constraints the plan must respect

- The shape invariant `(1,16,128,128)`.
- Config-driven: no magic numbers. If the plan puts a number in a `.py`, it belongs in a YAML.
- Layering: `utils` ← `data`/`models`/`synth` ← `recon` ← `eval`/`classical` ← `viz`.
  `classical/` imports nothing from `models/`, `recon/`, or `eval/`.
- Thresholding lives only in `recon/`. Metrics live only in `eval/metrics.py`.
- Adding a model must touch only its module + a YAML + a checkpoint path (FR-14).
- Anything needing GPU or money is `/train` or `/run-experiment` — **manual invoke only**.

## Before finishing

Ask what would make this plan produce a **confidently wrong number**, and make sure a test in the
plan would catch it. That is the failure mode this project actually has — not crashes, but clean
metrics computed on garbage.
