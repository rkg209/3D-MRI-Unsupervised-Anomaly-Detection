# Spec 000 · Vertical slice & reproducibility spine

**Status:** implemented
**Depends on:** none
**Fresh compute required:** no

---

## Problem

Nothing else may start until one thing works end to end. This spec cuts the thinnest possible
path through the entire stack — load the UNETR checkpoint, reconstruct **one** BraTS volume,
compute the residual, threshold it, score Dice against ground truth, save one figure — plus the
reproducibility spine everything hangs on. A vertical slice surfaces integration failures (shape
mismatches, checkpoint-format surprises, device placement) immediately, rather than after four
specs of work have been built on a wrong assumption.

## Contract

**Entry point:** `scripts/run_slice.py`, Hydra `@hydra.main`, invoked by `make slice`.

**The spine (deliverables other specs depend on):**

- `mri_ad.utils.seed.seed_everything(seed, deterministic=True)` — called first in *every* entry point.
- `mri_ad.utils.RunLogger` — writes `artifacts/runs/<timestamp>/run_meta.json` containing the
  git SHA, the fully-resolved config, the seed, hostname, and start/end time (NFR-2).
- `mri_ad.utils.DeviceManager.get_device()` — cuda → mps → cpu.
- `mri_ad.exceptions` — the hierarchy rooted at `MRIAnomalyDetectionError` (already scaffolded).
- `configs/` Hydra tree; `Makefile`; `pyproject.toml`.

**Flow:** checkpoint + one BraTS volume → `(1,16,128,128)` tensor → forward → residual →
threshold → Dice vs binarized ground truth → figure.

## Acceptance tests

1. `make check-data` exits 0 (real weights and data are present, not LFS stubs).
2. `make slice` completes with exit code 0 and prints a single Dice value in `[0, 1]`.
3. It writes `artifacts/figures/slice_<volume_id>.png` showing original / reconstruction /
   residual / mask / ground-truth.
4. It writes `artifacts/runs/<timestamp>/run_meta.json` containing a non-empty `git_sha`, the
   resolved config, and the seed.
5. **Determinism:** `make slice` run twice on the same hardware yields a Dice identical to
   within 1e-6, and two `run_meta.json` files with the same seed.
6. The ground truth is binarized (`seg > 0`) before scoring — asserted by a unit test, not by eye.
7. No patient identifier, file path, or scan metadata appears in stdout or in any artifact.

## Out of scope

Batch evaluation (004), the full transform pipeline (001), the model registry abstraction (002),
threshold *strategies* (003 — 000 may hardcode one and say so), any training.

## Notes / deviations

- This spec may reach past the abstractions that 001–003 will later introduce. That is the point
  of a vertical slice. It must be refactored onto them as they land, not left as a parallel path.
- The checkpoint loader here must already handle **both** on-disk formats (a dict with
  `model_state_dict`, and a bare `state_dict`) — both exist among the real weights.
