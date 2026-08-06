# Paradigm comparison — Classical vs UNETR vs Diffusion (Spec 007)

**Operating granularity is slice-level: each 2D axial slice is one sample. This does NOT localize an anomaly, so ROC-AUC here is NOT comparable to voxel-level Dice.**

| Metric | Classical | UNETR | Diffusion (AnoDDPM) | UNETR (synth-anomaly) |
| --- | --- | --- | --- | --- |
| Slice-level ROC-AUC | n/a (not run) | n/a (not evaluated) | n/a (untrained) | n/a (not trained) |
| Slice-level PR-AUC | n/a (not run) | n/a (not evaluated) | n/a (untrained) | n/a (not trained) |
| Slice-level F1 @ tuned thr. | n/a (not run) | n/a (not evaluated) | n/a (untrained) | n/a (not trained) |
| Voxel-level Dice | n/a (no localization) | n/a (not evaluated) | n/a (untrained) | n/a (not trained) |
| Voxel-level IoU | n/a (no localization) | n/a (not evaluated) | n/a (untrained) | n/a (not trained) |

Split hash: `n/a`. n_slices=None, n_positive=None, prevalence=None.

Best slice-level ROC-AUC: **n/a**. Best voxel-level Dice: **n/a**.

Operating granularity is slice-level: each 2D axial slice is one sample. This does NOT localize an anomaly, so ROC-AUC here is NOT comparable to voxel-level Dice.
