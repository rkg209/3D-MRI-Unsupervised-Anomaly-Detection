"""The exception hierarchy for the whole package (a Spec 000 deliverable).

Every failure raised by ``mri_ad`` derives from :class:`MRIAnomalyDetectionError`, so a caller
can catch the package's failures without catching unrelated ones. Errors are *loud and
descriptive* by policy: a data-contract violation must say what shape/range it actually saw.
"""


class MRIAnomalyDetectionError(Exception):
    """Base class for every error raised by ``mri_ad``."""


# ── data/ ─────────────────────────────────────────────────────────────────────
class DataError(MRIAnomalyDetectionError):
    """Base class for data-layer failures."""


class DescriptiveValidationError(DataError):
    """A batch violated the shape/dtype/intensity contract.

    The message must include the *observed* shape and value range, not just the expected one.
    """


class SplitContractViolationError(DataError):
    """An attempt was made to use a split other than the recorded contract without an override."""


# ── models/ ───────────────────────────────────────────────────────────────────
class ModelError(MRIAnomalyDetectionError):
    """Base class for model-registry failures."""


class CheckpointError(ModelError):
    """A checkpoint failed to load.

    Raised on a missing file, an LFS pointer stub (the inherited ``.pth`` files are stubs),
    or a ``state_dict`` key mismatch. Never fall back to ``strict=False`` to make this go away:
    a partially-loaded model produces plausible-looking garbage reconstructions.
    """


class ModelRegistrationError(ModelError):
    """A class that does not satisfy the ``AnomalyDetectionModel`` interface was registered."""


# ── recon/ ────────────────────────────────────────────────────────────────────
class ReconError(MRIAnomalyDetectionError):
    """Base class for reconstruction-engine failures."""


# ── eval/ ─────────────────────────────────────────────────────────────────────
class EvalError(MRIAnomalyDetectionError):
    """Base class for evaluation-harness failures."""


# ── classical/ ────────────────────────────────────────────────────────────────
class ClassicalError(MRIAnomalyDetectionError):
    """Base class for classical-baseline (radiomic features + gradient boosting) failures."""


class FeatureCacheError(ClassicalError):
    """A cached feature matrix's provenance disagreed with what was requested.

    Raised on a split-hash, feature-hash, granularity, feature-name, or schema-version
    mismatch — never silently re-used, per Spec 006's "loud failure" rule for the cache.
    """


# ── config / artifacts ────────────────────────────────────────────────────────
class ConfigError(MRIAnomalyDetectionError):
    """A required config key was missing or invalid."""


class ArtifactError(MRIAnomalyDetectionError):
    """A required artifact was absent or malformed."""
