# Spec 006 · Classical-ML baseline

**Status:** implemented
**Depends on:** 001, 004
**Fresh compute required:** no (CPU only)

---

## Problem

The prior body of work is deep-learning-only. A classical baseline — engineered radiomic/texture
features into a gradient-boosting classifier — closes that gap and, more importantly, provides an
honest control: if hand-crafted features plus XGBoost match the reconstruction pipeline, that is
a finding worth reporting, not one worth hiding.

## Contract

**Operating granularity: slice-level.** Each 2D axial slice (128×128) is one sample; label = 1 if
the corresponding binarized segmentation slice contains any tumor voxel, else 0. This yields clean
ROC-AUC comparable to standard binary-classification benchmarks. It **does not localize**, which
creates a genuine metric-comparability gap against voxel-level Dice — that gap must be stated
plainly in Spec 007, not glossed.

- `FeatureExtractor.extract(volume) -> np.ndarray` of shape `(n_slices, n_features)`.
  Features: first-order intensity (mean, std, skew, kurtosis, percentiles 5/25/50/75/95);
  GLCM texture (contrast, dissimilarity, homogeneity, energy, correlation; 4 angles, averaged);
  gradient magnitude (Sobel); run-length features. ≈40 features per slice.
- `ClassicalBaseline` — XGBoost or LightGBM, `StratifiedKFold(n_splits=k, shuffle=True,
  random_state=seed)`. Reports ROC-AUC and PR-AUC, mean ± std across folds.
- Config: `configs/classical/`. No hyperparameter in code.

## Acceptance tests

1. Feature extraction yields a finite, non-NaN matrix of the documented shape.
2. Cross-validated ROC-AUC and PR-AUC are reported as mean ± std over k folds.
3. **The split is grouped by subject.** Slices from one subject never appear in both train and
   test of the same fold. (Adjacent slices of one brain are near-duplicates; an ungrouped split
   leaks and inflates AUC dramatically. This is the single easiest way to get a fake result here.)
4. **Class balance is reported.** Tumor-bearing slices are a minority; PR-AUC must be read against
   the positive rate, and a majority-class baseline AUC is reported alongside.
5. The test split is the **identical** `SplitContract` split the DL pipeline uses (enables 007).
6. Granularity ("slice-level") is recorded in `ClassicalMetrics` and in every generated artifact.
7. `classical/` imports nothing from `models/`, `recon/`, or `eval/` — asserted by an import test.

## Out of scope

The three-paradigm comparison narrative (007). Voxel-level classical localization — see below.

## Notes / deviations

- The build plan defers the granularity choice to `/plan`; `planning/02-architecture.md:226`
  already resolves it to **slice-level**. We keep that, and record the trade-off: clean AUC, but
  no localization, so it is **not** directly comparable to Dice. Spec 007 owns that caveat.
- PyRadiomics is fragile to build on some platforms. If it does not install cleanly, fall back to
  a `scikit-image` GLCM + first-order feature set and **document the substitution** — the feature
  family matters more than the library.
