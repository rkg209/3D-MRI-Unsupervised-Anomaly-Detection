"""Unsupervised anomaly detection in 3D brain MRI.

Train reconstruction models on healthy brains only; flag regions the model reconstructs
poorly as anomalous. The optimization target is **detection separability**, never
reconstruction fidelity — see CLAUDE.md and the `mri-domain` skill.

The system-wide shape invariant is ``(1, 16, 128, 128)`` = ``(C, D, H, W)``.
"""

__version__ = "0.1.0"

# The shape invariant. Any tensor deviating from this at a module boundary is a bug,
# not a configuration option.
VOLUME_SHAPE: tuple[int, int, int, int] = (1, 16, 128, 128)
