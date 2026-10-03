---
name: lit
description: Research methods and library APIs (3D synthetic-lesion techniques, PyRadiomics features, MONAI APIs) via the lit-scout sub-agent, keeping reference-reading noise out of the main context. Use when a method or API needs looking up rather than recalling.
---

# /lit — method research, isolated

Delegates to the `lit-scout` sub-agent so that reading docs and papers does not flood the main
context. You get back a summary and citations, not the raw material.

## Use it for

- Synthetic-anomaly generation methods (FPI, Poisson blending, CutPaste-style 3D corruption) — Spec 009.
- Radiomic/texture feature families for the classical baseline — Spec 006.
- MONAI / PyRadiomics / XGBoost API details, especially where the API has moved. (MONAI renamed
  `pos_embed` → `proj_type` on UNETR, for instance — recall is not reliable here.)
- Prior art on reconstruction-based unsupervised anomaly detection.

## Rules

- **Look it up; do not recall it.** Library APIs drift, and a confidently-wrong API call costs more
  than the lookup. This matters most for MONAI, whose internals have shifted.
- Return: the answer, the source, and how confident it is. Flag anything version-dependent.
- **Nothing from research is implemented without a spec.** Findings inform a spec or a plan; they
  do not authorize code.
