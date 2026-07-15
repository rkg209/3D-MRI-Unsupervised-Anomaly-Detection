# requirements.md

## 1. Business Goals

**BG-1.** Produce a rigorous, reproducible comparative study of unsupervised anomaly detection in 3D brain MRI that is defensible in a technical interview and suitable as a public portfolio artifact.

**BG-2.** Close the classical-ML gap on the author's resume by delivering a cross-validated gradient-boosting baseline alongside the deep-learning pipeline, enabling an honest DL-vs-classical comparison.

**BG-3.** Demonstrate a measurable performance improvement over the prior work's best result (UNETR MSE+SSIM Dice ≈ 0.63) via synthetic-anomaly (anomaly-informed) training, with a controlled before/after comparison on an identical test split.

**BG-4.** Establish a clean, config-driven, greenfield codebase that replaces the prior notebook/research repo and can be reproduced by a third party from checkpoints and registered data alone.

**BG-5.** Articulate a conceptual transfer of the anomaly-detection methodology toward out-of-distribution detection in mobility/trajectory data, expressed as a README paragraph, to support the author's mobility-domain resume narrative.

---

## 2. Stakeholders

**SH-1. Author / primary developer** — sole engineer; owns all implementation, evaluation, and public presentation decisions.

**SH-2. Technical interviewers** — evaluate the project for ML engineering rigor, reproducibility, and honest comparative analysis; primary audience for the README, metric scorecard, and demo video.

**SH-3. Portfolio reviewers (non-technical)** — assess the project's scope and impact from the README and demo video without reading code.

**SH-4. Future collaborators / reproducers** — any third party who clones the repo and attempts to reproduce results from the published checkpoints and registered data; primary audience for the reproducibility instructions.

---

## 3. Users / Personas

**U-1. Reproducer** — a researcher or engineer who has registered access to OpenBHB and BraTS, clones the repo, places data at the configured path, and runs `make slice` / `make eval` / `make report` expecting to obtain the published headline numbers without manual intervention.

**U-2. Experimenter** — the author running new experiments (e.g., synthetic-anomaly fine-tuning, loss-function sweeps) via the Makefile and Hydra configs; expects full experiment logging, deterministic seeds, and no hardcoded hyperparameters.

**U-3. Spec author / SDD operator** — the author using Claude Code's SDD loop (`/specify` → `/plan` → `/tasks` → implement) to build each feature; expects the agent constitution, skills, sub-agents, and hooks to enforce workflow discipline automatically.

**U-4. Demo viewer** — a recruiter or interviewer watching the exported demo video; expects a clear visual showing original volume, reconstruction, residual map, anomaly mask, and ground-truth overlay without needing to run any code.

---

## 4. Functional Requirements

### 4.1 Repository Scaffold & Configuration

**FR-1.** The repository shall be a greenfield Python project with the directory structure defined in §2 of the build plan, including `src/mri_ad/`, `configs/`, `specs/`, `tests/`, `.claude/`, `artifacts/`, `data/`, and `checkpoints/`.

**FR-2.** All hyperparameters, file paths, model names, loss-function choices, and experiment parameters shall be expressed in Hydra/YAML configuration files under `configs/` (sub-directories: `data/`, `models/`, `losses/`, `experiment/`). No magic numbers or hardcoded paths shall appear in source code.

**FR-3.** A `Makefile` shall expose the following targets: `make slice`, `make train`, `make eval`, `make report`, `make test`. Each target shall invoke the corresponding pipeline step via the config system.

**FR-4.** A `pyproject.toml` shall declare all dependencies with pinned versions, including at minimum: PyTorch, MONAI, PyRadiomics, XGBoost, LightGBM, and Hydra.

**FR-5.** A central utility (`src/mri_ad/utils/`) shall set deterministic seeds and disable nondeterministic operations before any training or evaluation run; the git SHA and full resolved config shall be logged at the start of every run.

### 4.2 Data Layer

**FR-6.** The data layer (`src/mri_ad/data/`) shall implement MONAI-backed datasets and transform pipelines for both OpenBHB (healthy, T1) and BraTS (anomalous, multimodal + segmentation maps).

**FR-7.** The preprocessing pipeline shall apply, in order: min-max intensity normalization to [0, 1], spatial crop, and slicing to volume blocks of shape `(1, 16, 128, 128)` (channels, depth, height, width). This pipeline shall be codified in code even when preprocessed data is supplied, so it is reproducible.

**FR-8.** The data layer shall enforce a documented, deterministic train/val/test split contract. The split shall be reproducible given the same seed and shall be recorded in the experiment log.

**FR-9.** A data-validation step shall execute at dataloader initialization and shall raise a loud, descriptive error if any batch deviates from the expected shape `(B, 1, 16, 128, 128)` or if intensity values fall outside [0, 1].

**FR-10.** Dataloaders shall yield correctly shaped and normalized batches; this shall be verified by automated tests.

### 4.3 Model Registry & Checkpoint Loading

**FR-11.** The model registry (`src/mri_ad/models/`) shall expose UNet, Attention-UNet, UNETR, and a diffusion model (AnoDDPM, Spec 013) as MONAI-backed modules behind a single uniform interface. UNet and Attention-UNet are retained as prior-work reference rows; UNETR and the diffusion model are the headline deep-learning paradigms (with the classical baseline, Spec 006, they form the three-way comparison). The interface shall define at minimum: `forward(x)`, `load_checkpoint(path)`, and a `model_card` property returning architecture metadata.

**FR-12.** Every registered model shall load its corresponding pre-trained checkpoint from the path specified in config and execute a forward pass on a real batch without error.

**FR-13.** A short model card shall be maintained for each architecture documenting: parameter count, input/output shape contract, training data, and known performance characteristics.

**FR-14.** Adding a new model to the registry shall require only: implementing the uniform interface, adding a YAML config entry, and providing a checkpoint path. No changes to evaluation or reconstruction code shall be required.

### 4.4 Reconstruction & Anomaly-Map Engine

**FR-15.** The reconstruction engine (`src/mri_ad/recon/`) shall accept any registered model and an input volume and shall return three outputs: the reconstructed volume, the residual map (original − reconstruction, element-wise), and a thresholded binary anomaly mask.

**FR-16.** The thresholding strategy (e.g., fixed percentile, Otsu, adaptive) shall be configurable via the config system and shall be centralized in the engine; no thresholding logic shall be duplicated elsewhere.

**FR-17.** The engine shall operate on volumes of shape `(1, 16, 128, 128)` and shall preserve this shape in all three outputs.

### 4.5 Evaluation Harness

**FR-18.** The evaluation harness (`src/mri_ad/eval/`) shall compute the following metrics: Dice coefficient and IoU (localization, against BraTS segmentation maps); PSNR and SSIM (reconstruction quality, retained as explanatory context only, not as optimization targets).

**FR-19.** Metrics shall be computed per-volume and aggregated (mean ± std) across the test split. Both per-volume and aggregate results shall be persisted to `artifacts/`.

**FR-20.** The harness shall auto-generate results tables and plots from saved outputs (reconstructions, residual maps, anomaly masks) without requiring re-inference or re-training.

**FR-21.** Running `make eval` followed by `make report` shall reproduce the prior report's headline numbers for UNETR MSE+SSIM (Dice ≈ 0.6255, IoU ≈ 0.4551) within a tolerance of ±0.005, using the existing checkpoint and test split.

**FR-22.** PSNR and SSIM results shall be presented with explicit labeling that they are explanatory context, not performance targets, and the evaluation report shall include a written note explaining why higher reconstruction fidelity correlates with worse anomaly detection in this domain.

### 4.6 Architecture × Loss Comparison Study

**FR-23.** The fidelity-vs-detection study (`specs/005`) shall evaluate the UNETR anchor against the following loss functions: MSE, SSIM, MSE+SSIM, perceptual (ResNet50 pretrained on Med3D), and multi-scale MSE; shall include UNet and Attention-UNet as prior-work reference rows; and shall include the diffusion model (AnoDDPM) as a reconstruction-based row to test whether its generative prior escapes the fidelity-vs-detection anti-correlation. The near-identical UNet/AttUNet/UNETR headline comparison of the prior work is explicitly superseded by the three-paradigm comparison in FR-30/FR-31.

**FR-24.** Results shall be presented in a single auto-generated comparison table showing Dice and IoU for every architecture × loss combination, sourced from saved checkpoint outputs.

**FR-25.** The comparison study shall include a written analysis, authored by the `results-analyst` sub-agent, demonstrating that higher reconstruction fidelity (PSNR/SSIM) does not produce better anomaly detection (Dice/IoU), and identifying the best-performing combination.

### 4.7 Classical-ML Baseline

**FR-26.** The classical baseline (`src/mri_ad/classical/`) shall extract radiomic, intensity, and texture features from MRI volumes or regions using a PyRadiomics-style pipeline.

**FR-27.** Extracted features shall be passed to an XGBoost or LightGBM classifier trained in a k-fold cross-validated setup. The number of folds and classifier hyperparameters shall be specified in config.

**FR-28.** The baseline shall report ROC-AUC and PR-AUC as primary metrics. The operating granularity (slice-level or supervoxel/region-level) shall be explicitly chosen and documented during the `/plan` step for Spec 006, with written justification recorded in the spec.

**FR-29.** Cross-validated AUC results shall be reported with mean and standard deviation across folds.

### 4.8 DL-vs-Classical Comparison

**FR-30.** The paradigm comparison (`specs/007`) shall evaluate three paradigms — the classical baseline, the UNETR reconstruction model, and the diffusion model (AnoDDPM) — on the **identical** test split. This is the project's headline comparison.

**FR-31.** The comparison artifact shall include: a side-by-side metrics table with all three paradigms, overlaid ROC and PR curves, and a written narrative identifying where each approach wins and where each fails (including, if the data shows it, that diffusion does not beat UNETR or that the classical baseline is competitive).

**FR-32.** Any metric-comparability caveats arising from the granularity choice made in FR-28 (e.g., slice-level AUC vs. voxel-level Dice) shall be explicitly stated in the comparison narrative.

### 4.9 Synthetic-Anomaly Training

**FR-33.** The synthetic-anomaly module (`src/mri_ad/synth/`) shall implement a 3D lesion generation pipeline that produces plausible synthetic anomalies on healthy OpenBHB volumes. The generation technique (e.g., Poisson blending, FPI, 3D CutPaste-style corruption) shall be selected and documented during the `/plan` step for Spec 009.

**FR-34.** The anomaly-informed training objective shall train the model to reconstruct the healthy (pre-corruption) version of a synthetically corrupted volume, so the model learns to erase anomalous-looking regions rather than copy them.

**FR-35.** Training shall fine-tune from the existing UNETR checkpoint. The fine-tuning shall be invoked only via the `/train` skill (manual-invoke only) and shall never be launched autonomously by the agent.

**FR-36.** Synthetic-anomaly generation parameters (corruption type, intensity range, size range, blend parameters) shall be fully specified in config and logged with each training run.

**FR-37.** The spec shall produce a controlled before/after comparison table showing Dice and IoU on the identical BraTS test split, before and after synthetic-anomaly fine-tuning.

### 4.9b Diffusion-Based Anomaly Detection (AnoDDPM)

**FR-33b.** The diffusion module (Spec 013) shall implement AnoDDPM: a DDPM (MONAI `DiffusionModelUNet` + `DDPMScheduler`, `spatial_dims=3`) trained only on healthy OpenBHB volumes, exposed through the same `AnomalyDetectionModel` interface as every other registered model (FR-11), so the reconstruction engine, evaluation harness, and classical baseline require no changes.

**FR-33c.** At inference the model shall partially noise the input volume to a configurable timestep and denoise it back to a healthy estimate of shape `(1, 16, 128, 128)`; the residual against that estimate shall feed the existing reconstruction/anomaly-map engine (FR-15).

**FR-33d.** Diffusion training (noise schedule, timesteps, partial-noise level, sampler, seed) shall be fully specified in config and logged with each run, shall never be launched autonomously, and shall be reachable only via the manual-invoke `/train` skill / `make train` (C-3). Its checkpoint does not exist until trained; `load_checkpoint` shall raise a descriptive error rather than silently partial-loading.

**FR-33e.** The diffusion model shall appear as a column in the FR-30 paradigm comparison and a row in the FR-23 study, scored on the identical test split and threshold strategy. If it does not improve on UNETR, that result shall be reported, not hidden (NFR-21/honest-reporting).

### 4.10 3D Visualization & Demo Video

**FR-38.** The visualization module (`src/mri_ad/viz/`) shall render a synchronized multi-panel view showing, for a selected volume: original MRI, reconstruction, residual map, thresholded anomaly mask, and ground-truth segmentation overlay.

**FR-39.** The viewer technology (e.g., NiiVue, Gradio, Jupyter widget) shall be selected during the `/plan` step for Spec 010.

**FR-40.** The visualization pipeline shall export a short demo video clip (format: MP4 or GIF) from a reproducible script. The export shall be triggered by a Makefile target or CLI command specified in config.

**FR-41.** The demo video shall be the primary public visual artifact for the project, given that always-on hosting is out of scope.

### 4.11 Reporting & Public Artifact

**FR-42.** Running `make report` shall auto-assemble the README's metric scorecard section from saved artifacts, regenerating every headline number (Dice, IoU, ROC-AUC, PR-AUC, inference time per volume, GPU hours) without manual editing.

**FR-43.** The README shall include: an architecture diagram, the metric scorecard, reproducibility instructions (data registration links, checkpoint sources, exact commands), and the mobility-transfer paragraph.

**FR-44.** The mobility-transfer paragraph shall describe the conceptual transfer of the anomaly-detection methodology (reconstruct normal → flag deviations) toward out-of-distribution detection in mobility/trajectory data. It shall be explicitly labeled as a conceptual analogy, not a tested result, and shall contain no driving-data code or experiments.

**FR-45.** Every headline number in the README shall be traceable to a saved artifact in `artifacts/` and regenerable by `make report`.

### 4.12 SDD Workflow & Agent Configuration

**FR-46.** A `CLAUDE.md` file shall be present at the repository root containing: project description, locked decisions D1–D8, the central domain fact, repo map, Makefile commands, coding conventions, SDD workflow rule, budget/compute guardrail, data guardrail, and pointers to skills and sub-agents.

**FR-47.** The following skills shall be implemented under `.claude/skills/`: `mri-domain` (auto-invoked), `monai-patterns` (auto-invoked), `repro-discipline` (auto-invoked), `/specify`, `/plan`, `/tasks`, `/new-model`, `/eval-report`, `/run-experiment` (manual-only), `/train` (manual-only), `/lit`.

**FR-48.** The following sub-agents shall be implemented under `.claude/agents/`: `spec-reviewer`, `lit-scout`, `experiment-runner`, `results-analyst`, `torch-reviewer`.

**FR-49.** Hooks shall be configured in `.claude/settings.json` to enforce: auto-format/lint on Python file writes (`ruff format` + `ruff check --fix`); block `rm -rf` and any `git add`/`git commit` touching `data/`, `checkpoints/`, `*.nii.gz`, `*.npy`, or `*.pt`; warn before unattended training launches; validate new spec files against the template schema; emit a completion notification on long runs; inject a status summary at session start.

**FR-50.** A spec template file shall exist at `specs/_template.md` containing required sections: problem statement, data/interface contract, acceptance tests, and out-of-scope declarations.

**FR-51.** Every spec in the ordered list (Specs 000–012) shall have a corresponding `specs/NNN-name.md` file that is approved before any implementation work begins on that spec.

**FR-52.** The SDD loop shall enforce the order: `/specify` → `spec-reviewer` approval → `/plan` → `/tasks` → implement → acceptance tests pass → PR merge. No implementation code shall be written without an approved spec.

**FR-53.** The GitHub MCP server shall be configured to support opening one PR per implemented spec and tracking the spec list as issues.

---

## 5. Non-Functional Requirements

### 5.1 Reproducibility

**NFR-1.** Given the same registered data, checkpoint files, and config, `make slice`, `make eval`, and `make report` shall produce bit-identical (or within documented floating-point tolerance ±0.005 on Dice/IoU) results across runs on the same hardware.

**NFR-2.** Every experiment run shall log: the git SHA of the codebase, the full resolved Hydra config, the random seed, and the wall-clock start/end time. These logs shall be persisted to `artifacts/` alongside the metrics.

**NFR-3.** The train/val/test split shall be deterministic given the seed and shall be documented in a split-contract file so that any future run uses the identical partition.

**NFR-4.** The preprocessing pipeline shall be codified in source code (FR-7) even when preprocessed data is supplied, so the transformation from raw data to model input is fully auditable.

### 5.2 Correctness & Integrity

**NFR-5.** Dice and IoU shall be computed against BraTS ground-truth segmentation maps at the voxel level. The exact formulas (including handling of empty masks and epsilon for numerical stability) shall be defined once in `src/mri_ad/eval/` and used everywhere.

**NFR-6.** PSNR and SSIM shall be computed using standard definitions and shall never be used as a primary optimization target or headline metric; their role is strictly explanatory.

**NFR-7.** The `torch-reviewer` sub-agent shall review all PyTorch/MONAI model and training code before merge, checking for: tensor-shape mismatches, device-placement errors, `train()`/`eval()` mode correctness, gradient leaks, and nondeterministic operations.

**NFR-8.** All public functions and classes shall carry type hints and docstrings. This shall be enforced by the `ruff` hook (FR-49).

### 5.3 Performance

**NFR-9.** Inference (reconstruction + anomaly-map generation) for a single `(1, 16, 128, 128)` volume shall complete in under 5 seconds on the target GPU hardware. This shall be measured and reported in the metric scorecard.

**NFR-10.** `make eval` (full test-split evaluation from saved outputs, no training) shall complete in under 30 minutes on the target hardware.

**NFR-11.** `make report` (metrics table and README regeneration from saved artifacts) shall complete in under 2 minutes with no GPU required.

### 5.4 Security & Data Governance

**NFR-12.** The directories `data/`, `checkpoints/`, and all files matching `*.nii.gz`, `*.npy`, `*.pt` shall be listed in `.gitignore` and shall never appear in any git commit. This shall be enforced by the pre-commit hook defined in FR-49.

**NFR-13.** No raw patient identifiers, scan metadata, or imaging data shall be printed to logs, stdout, or any artifact file. Log statements shall reference only file indices, split labels, and aggregate statistics.

**NFR-14.** All data access shall comply with the OpenBHB and BraTS data-use agreements. The README shall include links to the registration pages and a statement that data must be obtained independently.

### 5.5 Maintainability & Extensibility

**NFR-15.** Adding a new model architecture shall require changes only to: the model implementation file, the model config YAML, and the checkpoint path. No changes to the reconstruction engine, evaluation harness, or comparison study code shall be required (see FR-14).

**NFR-16.** Adding a new loss function shall require changes only to the loss config YAML and a corresponding loss module. No changes to training orchestration or evaluation code shall be required.

**NFR-17.** The codebase shall achieve a minimum of 80% line coverage on `src/mri_ad/` as measured by `pytest --cov`, excluding `viz/` and generated artifact code.

**NFR-18.** All Python source files shall pass `ruff check` with zero errors. Style shall be enforced automatically by the `PostToolUse` hook on every file write (FR-49).

### 5.6 Portability

**NFR-19.** The project shall be installable and runnable on both a college GPU cluster and a developer laptop using the same `pyproject.toml` and the same Makefile targets. Data and checkpoint paths shall be the only environment-specific configuration.

**NFR-20.** The `.claude/` plugin (skills, sub-agents, hooks, MCP config) shall be version-controlled in the repository and shall be loadable on any machine where Claude Code is installed, without additional setup beyond `claude mcp add` for the GitHub server.

### 5.7 Honesty & Framing

**NFR-21.** The project shall be framed throughout as a **comparative study**, not a clinical tool or deployable detector. No claim of clinical applicability shall appear in the README, reports, or any generated artifact.

**NFR-22.** The mobility-transfer paragraph (FR-44) shall be clearly labeled as a conceptual analogy. No driving data, mobility experiments, or untested performance claims shall appear anywhere in the repository.

**NFR-23.** The central finding — that naive reconstruction models fail because they reconstruct the tumor too well, and that detection separability (not reconstruction fidelity) is the correct optimization target — shall be stated explicitly in the README, the evaluation report, and the comparison study narrative.

---

## 6. Constraints

**C-1. Greenfield rebuild.** The prior notebook/research repository is reference material only. Logic may be ported; structure, naming, and architecture must not be carried over.

**C-2. MRI modality only.** All code, configs, documentation, and framing shall be MRI-specific. CT framing from the prior work shall be dropped entirely.

**C-3. No autonomous training or GPU spending.** The agent shall never autonomously launch a training run, hyperparameter sweep, or any API call that incurs cost. These actions are gated behind manual-invoke-only skills (`/train`, `/run-experiment`).

**C-4. No data or checkpoint commits.** `data/`, `checkpoints/`, and all binary imaging/weight files are permanently git-ignored. This is both a repo-hygiene and a medical-data-licensing constraint.

**C-5. Detection separability is the optimization target.** Improving PSNR or SSIM is not a goal. Any change that improves reconstruction fidelity at the cost of Dice/IoU is a regression, not an improvement.

**C-6. Existing checkpoints and preprocessed data are the compute baseline.** Only Spec 009 (synthetic-anomaly fine-tuning) requires fresh training. All other specs build harnesses around existing saved outputs.

**C-7. Multi-scale attention is a stretch spec.** Spec 012 shall not be started until Specs 000–011 are complete and defensible. It shall not block the definition of done.

**C-8. No always-on hosting.** The project shall not deploy a live web service or API. The demo video (FR-40) is the public visual artifact.

**C-9. No clinical claims.** The project makes no claims of clinical validity, regulatory compliance, or patient-safety applicability.

**C-10. Spec-first discipline.** No implementation code shall be written without an approved spec in `specs/`. This is enforced by the SDD workflow rule in `CLAUDE.md` and the `spec-reviewer` sub-agent gate.

---

## 7. Technologies

### 7.1 Core ML Framework

**T-1. PyTorch** — primary deep-learning framework; version pinned in `pyproject.toml`.

**T-2. MONAI** — 3D medical imaging transforms, `CacheDataset`, UNet, Attention-UNet, and UNETR backbone implementations, plus diffusion components (`DiffusionModelUNet` and `DDPMScheduler`, from the MONAI Generative code merged into core) for the AnoDDPM model (Spec 013). Replaces all hand-rolled 3D model and transform code from the prior work. (The pinned MONAI version must ship these diffusion components; confirmed/bumped during the Spec 013 `/plan` step.)

### 7.2 Classical ML

**T-3. PyRadiomics** (or equivalent) — radiomic, intensity, and texture feature extraction for the classical baseline (Spec 006).

**T-4. XGBoost / LightGBM** — gradient-boosting classifiers for the classical baseline; both available, choice documented in the Spec 006 plan.

**T-5. scikit-learn** — k-fold cross-validation, ROC-AUC, PR-AUC computation.

### 7.3 Configuration & Experiment Management

**T-6. Hydra** — config composition and override system; all experiment parameters expressed as YAML configs under `configs/`.

**T-7. Weights & Biases (W&B) Python SDK** — experiment tracking (optional MCP integration; SDK alone is sufficient per §3.6 of the build plan).

### 7.4 Code Quality

**T-8. Ruff** — linting and formatting, enforced by the `PostToolUse` hook on every Python file write.

**T-9. pytest + pytest-cov** — test runner and coverage measurement; minimum 80% line coverage on `src/mri_ad/` (NFR-17).

### 7.5 Visualization

**T-10. NiiVue / Gradio / Jupyter widget** — viewer technology for the 3D visualization (Spec 010); exact choice deferred to the `/plan` step for Spec 010.

**T-11. Matplotlib / Seaborn** — 2D plots (ROC curves, PR curves, comparison tables rendered as figures).

**T-12. MP4 / GIF export** — demo video format; exact library (e.g., `imageio`, `ffmpeg`) deferred to Spec 010 plan.

### 7.6 Agent & SDD Infrastructure

**T-13. Claude Code** — primary development agent; SDD loop, skills, sub-agents, and hooks as specified in §3 of the build plan.

**T-14. GitHub MCP server** — PR-per-spec workflow and issue tracking for the spec list.

**T-15. Context7-style / arXiv MCP server** — feeds the `lit-scout` sub-agent with accurate MONAI, PyRadiomics, and synthetic-anomaly references.

### 7.7 Build & Packaging

**T-16. Make** — build orchestration via `Makefile` targets (`slice`, `train`, `eval`, `report`, `test`).

**T-17. pyproject.toml** — dependency declaration and pinning; project packaging.

---

## 8. Deployment Requirements

**DR-1.** The project shall be deployable on two target environments: (a) a college GPU cluster and (b) a developer laptop. Both environments shall use the identical `pyproject.toml`, Makefile, and config files. Only data and checkpoint paths shall differ between environments, expressed via environment-specific Hydra config overrides.

**DR-2.** The GPU cluster environment shall support `make train` (Spec 009 fine-tuning) and `make eval` at full test-split scale. Minimum GPU memory requirement shall be documented in the README after the Spec 009 plan step determines the fine-tuning batch size.

**DR-3.** The laptop environment shall support `make slice`, `make report`, and `make test` without a GPU (CPU fallback). `make eval` on a reduced subset shall also be supported for local verification.

**DR-4.** Installation shall be achievable via a single command (e.g., `pip install -e .` or `uv sync`) from the `pyproject.toml`. No manual dependency resolution steps shall be required.

**DR-5.** The `.claude/` plugin (skills, sub-agents, hooks, MCP config) shall be loadable on any machine where Claude Code is installed by running the GitHub MCP add command and loading the local plugin. This shall be documented in the README under reproducibility instructions.

**DR-6.** There is no production server, container, or cloud deployment. The project is a local research codebase with a public GitHub repository as its distribution mechanism.

**DR-7.** The public GitHub repository shall contain: all source code, specs, configs, tests, Makefile, `pyproject.toml`, `CLAUDE.md`, and the `.claude/` plugin. It shall not contain data, checkpoints, or any binary imaging files (enforced by C-4 and NFR-12).

**DR-8.** The demo video (FR-40) shall be committed to the repository or hosted as a GitHub release asset, as it is the primary public visual artifact and must be accessible without running any code.