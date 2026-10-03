---
name: repro-discipline
description: Reproducibility checklist to apply before and during any run (slice, eval, train, report) — seeds set, config resolved and logged, git SHA recorded, no nondeterministic ops, no test-set leakage, every published number traceable to an artifact. Use before running any experiment or publishing any metric.
---

# Reproducibility checklist

D2: **rigor first, then performance.** An unreproducible number is not a result.

## Before any run

- [ ] `seed_everything(cfg.seed, deterministic=True)` is the **first** thing the entry point does —
      before any model, dataloader, or transform is constructed.
- [ ] `make check-data` passes. Weights are real files, not Git LFS pointer stubs.
- [ ] No uncommitted changes to `src/` (the logged git SHA must describe the code that actually ran).
- [ ] Every parameter comes from `configs/`. If you are about to type a number into a `.py` file,
      it belongs in a YAML instead.

## Logged for every run — `artifacts/runs/<timestamp>/run_meta.json`

git SHA · fully-resolved config · seed · hostname · start/end time · resulting metrics.

A metric without a `run_meta.json` beside it cannot be published.

## Determinism

- `torch.backends.cudnn.deterministic = True`, `benchmark = False`.
- `shuffle=False` and a seeded `generator` on every eval dataloader.
- No random transform anywhere on the eval path.
- Same input + same seed + same hardware → identical output. If it does not, stop and find out why.

## Leakage — the things that silently fake a good result

- [ ] **Thresholds are tuned on validation, never on test.** Picking the threshold that maximizes
      test Dice is the single easiest way to publish a fake improvement.
- [ ] **The classical baseline's CV split is grouped by subject.** Adjacent slices of one brain are
      near-duplicates; an ungrouped split leaks and inflates AUC dramatically.
- [ ] **Spec 009 fine-tuning never sees the test split.** Check against `SplitContract`.
- [ ] Before/after comparisons use the **identical** split, threshold, and preprocessing. Changing
      two things at once means you have measured nothing.

## Publishing

- [ ] Every number in the README traces to a file in `artifacts/`, and `make report` regenerates it.
- [ ] Unmeasured conditions render as `n/a (<reason>)` — never blank, never estimated, never dropped.
- [ ] A worse result is reported as-is. A negative finding is a finding (D2).
