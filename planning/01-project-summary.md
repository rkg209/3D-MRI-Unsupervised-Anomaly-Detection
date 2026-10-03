# project-summary.md

---

# 3D MRI Unsupervised Anomaly Detection — Executive Summary

## What This Project Is

This project builds a rigorous, reproducible system for detecting brain tumors and lesions in 3D MRI scans **without using any tumor labels during training**. Instead of teaching a model what a tumor looks like, the system learns what a *healthy* brain looks like — and then flags anything it cannot reconstruct well as potentially anomalous.

The core idea is straightforward: train a deep-learning reconstruction model exclusively on healthy brain scans. At inference time, feed it a scan that may contain a tumor. The model tries to reconstruct a healthy version of what it sees. Where it fails — where the original and the reconstruction diverge — is where the anomaly likely lives. That difference map becomes the detection signal.

The project is honestly framed as a **comparative study**, not a clinical tool. It systematically evaluates multiple neural network architectures (UNet, Attention-UNet, UNETR) against multiple loss functions, adds a classical machine-learning baseline for contrast, and introduces a targeted training improvement — synthetic-anomaly training — designed to attack the method's central failure mode directly.

---

## The Central Problem This Solves

Naive reconstruction models have a counterintuitive failure: they are *too good*. A sufficiently powerful model learns to reconstruct tumors faithfully alongside healthy tissue, because tumors are spatially coherent structures that the model can copy rather than erase. The result is a small residual map even where the tumor is — and the detection signal disappears.

The prior work established that UNETR, a transformer-based architecture whose patch-based processing constrains its ability to copy fine local structure, achieves the best anomaly localization (Dice ≈ 0.63) precisely *because* it reconstructs less perfectly. This project preserves and extends that finding, and adds synthetic-anomaly training as a principled way to push the model further toward erasing anomalous regions rather than reproducing them.

**The guiding principle throughout:** optimize for detection separability, not reconstruction quality. Better image fidelity made detection *worse* in the prior work. Every design decision in this rebuild respects that finding.

---

## Who It Is For

**Primary audience: technical hiring reviewers and research collaborators** evaluating the author's competence in medical imaging, deep learning, and rigorous experimental practice.

The project is designed to be defensible in a technical interview. It demonstrates:

- **Medical imaging depth** — 3D volumetric data, MRI-specific preprocessing, clinically meaningful evaluation metrics (Dice, IoU), and honest engagement with the method's limitations.
- **Classical and deep ML breadth** — a gradient-boosting baseline built from radiomic and texture features sits alongside the neural architectures, enabling a direct, honest comparison between paradigms.
- **Reproducibility discipline** — every result is regenerable from saved checkpoints with a single command (`make report`). No magic numbers in code; no manual steps; no committed data.
- **Engineering craft** — a clean, config-driven, MONAI-backed codebase built greenfield from a prior research notebook, demonstrating the ability to translate exploratory work into production-quality structure.

A secondary framing — stated explicitly in the README but not implemented in code — connects the anomaly-detection methodology conceptually to out-of-distribution detection in mobility and autonomous systems, supporting the author's broader career narrative.

---

## Why It Matters

**Methodologically**, unsupervised anomaly detection in medical imaging is a genuine open problem. Labeled tumor data is scarce, expensive to produce, and institution-specific. A system that requires only healthy scans for training is far more deployable in practice. This project contributes a clean comparative study of how architecture and loss function choices interact with the core failure mode, and a reproducible demonstration that synthetic-anomaly training measurably improves detection.

**As a portfolio artifact**, the project closes two specific gaps: it adds a classical-ML component (radiomic feature engineering, cross-validated gradient boosting) to a previously deep-learning-only body of work, and it replaces a research notebook with a structured, reproducible codebase that a collaborator could clone and run. The demo video and auto-generated results tables make the work legible to non-specialists without sacrificing technical depth for specialists.

**The honest bottom line:** this is not a system ready for clinical deployment. It is a carefully constructed, fully reproducible research artifact that demonstrates what the method can and cannot do, explains *why* it behaves as it does, and provides a credible foundation for further work — exactly what a rigorous comparative study should deliver.

---

*Data: OpenBHB (healthy T1 MRI, training) · BraTS (tumor MRI + segmentation maps, evaluation) · Framework: PyTorch + MONAI · Best result: UNETR MSE+SSIM, Dice ≈ 0.63*