# 3D MRI Unsupervised Anomaly Detection — Spec-Driven Build Plan

> **Purpose.** A single, self-contained build plan for rebuilding and improving the *3D MRI Unsupervised Anomaly Detection* project using **Claude Code in a spec-driven-development (SDD) workflow**. It is written so that the project can be understood end-to-end without prior context, and so that Claude Code has everything it needs to scaffold the repo, run the SDD loop, and execute the work.
>
> This document covers three things:
> 1. **Project context** — what the project is, what already exists, and the decisions that frame the rebuild.
> 2. **Claude Code SDD setup** — exactly what to put in `CLAUDE.md`, which skills, sub-agents, hooks, MCP servers, and plugins to configure.
> 3. **The ordered spec list** — every spec to build and the order to build it in, at a high level. Exact technical detail is intentionally deferred to each spec's own `/plan` step.

---

## 0. Locked decisions (read first)

These were settled before writing this plan. Claude Code should treat them as constraints, not open questions.

| # | Decision | Choice |
|---|---|---|
| D1 | Codebase starting point | **Greenfield rebuild.** The old notebook/research repo is *reference only* — port logic, not structure. |
| D2 | What "improved" optimizes for | **Rigor & reproducibility first, then performance.** A clean, defensible comparative study beats a fragile high score. |
| D3 | New modeling scope | **Headline comparison reframed to three paradigms — Classical (Spec 006) vs UNETR vs Diffusion/AnoDDPM (Spec 013)** — replacing the near-identical UNet/AttUNet/UNETR trio of the prior work (UNet/AttUNet kept as reference rows only). **Two performance novelties: synthetic-anomaly (anomaly-informed) training (Spec 009, on UNETR) and the diffusion paradigm (Spec 013).** **Multi-scale attention (Spec 012) remains a stretch spec** — written but off the critical path. |
| D4 | Modality | **MRI only.** Drop the old "CT" framing entirely. |
| D5 | Mobility-transfer framing | **README paragraph only.** No driving-data code, no experiment — stated as conceptual transfer toward the mobility resume. |
| D6 | Core framework | **Adopt MONAI** for 3D transforms, datasets, and model backbones (UNet / Attention-UNet / UNETR) instead of hand-rolling. |
| D7 | SDD flow | **Lightweight spec-kit-style** loop custom to this repo: `/specify` → `/plan` → `/tasks` → implement, with specs stored in `/specs`. |
| D8 | Data & checkpoints | **Already in hand.** Preprocessed OpenBHB + BraTS and trained checkpoints (UNETR etc.) exist. Eval/comparison/classical/viz specs are therefore *"build the harness, plug in saved outputs"*. Only synthetic-anomaly training needs fresh compute (fine-tune from the existing UNETR checkpoint). |

**The single most important domain fact** (carry it everywhere): the central finding of the prior work is that *naive reconstruction models rebuild the tumor too well, so they fail to flag it*. UNETR's patch-based features constrain that best (Dice ≈ 0.63). Do **not** let "improve reconstruction quality" become a goal — better PSNR/SSIM made detection *worse*. The goal is **detection separability**, not reconstruction fidelity.

---

## 1. What the project is

Detect anomalies (tumors / lesions) in **3D brain MRI without any tumor labels**. Train reconstruction models only on *healthy* brains; at inference, regions the model reconstructs poorly are flagged as anomalous via a residual (original − reconstruction) map. The project is honestly framed as a **comparative study** — across architectures and loss functions, plus a **classical-ML baseline** — not as a deployable clinical detector.

**Headline metrics**
- **DL:** best Dice / IoU for anomaly localization (target: reproduce/beat UNETR MSE+SSIM Dice ≈ 0.63).
- **Classical:** ROC-AUC / PR-AUC of the gradient-boosting baseline.
- **Improvement story:** Dice before vs after synthetic-anomaly training on the same test split.
- **Context metric:** PSNR / SSIM, used only to *explain why* high-fidelity models miss anomalies.

**Data**
- **OpenBHB** (healthy, T1) — training distribution of "normal" anatomy. (Prior work used the validation subset due to compute; keep that unless retraining.)
- **BraTS** (anomalous, multimodal + tumor segmentation maps) — used for testing/localization scoring. Resized to 128×128 in-plane; volumes sliced to `(1, 16, 128, 128)` blocks as in the prior pipeline.
- *Prerequisite you handle outside the specs:* BraTS + OpenBHB access requires registration. Specs assume the data is present at a configured path.

**What already exists (reference, from the prior report)**
- Reconstruction pipeline with UNet, Attention-UNet, UNETR (f-AnoGAN ruled out early on cost).
- Loss sweep: MSE, SSIM, MSE+SSIM, perceptual (ResNet50 pretrained on Med3D), multi-scale MSE.
- Best results: UNETR MSE+SSIM (Dice 0.6255 / IoU 0.4551), UNETR SSIM (Dice 0.5826).
- A 3D visualization GUI for residual / anomaly maps.
- Two future-work ideas: synthetic anomalies (now **in scope**) and multi-scale attention (now **stretch**).

**What's new in this rebuild**
- Clean, reproducible, config-driven codebase (greenfield).
- **Classical-ML baseline** (radiomic / intensity / texture features → XGBoost/LightGBM, cross-validated, ROC-AUC/PR) — closes the resume's classical-ML gap and anchors one leg of the three-paradigm comparison.
- **Diffusion model (AnoDDPM, Spec 013)** — a generative denoising prior over healthy anatomy, the third and newest paradigm; replaces the redundant UNet/AttUNet headline slots and puts a sharper test on the central failure mode.
- **Synthetic-anomaly training** to attack the core failure mode directly.
- A tightened evaluation harness and auto-generated results tables.
- Mobility-transfer framing paragraph in the README.

---

## 2. Repository layout

```
mri-anomaly-detection/
├── CLAUDE.md                      # agent constitution (see §3)
├── README.md                      # public artifact: diagram, metrics scorecard, mobility paragraph
├── pyproject.toml                 # deps pinned (PyTorch, MONAI, PyRadiomics, xgboost/lightgbm, ...)
├── Makefile                       # make slice / train / eval / report / test
├── configs/                       # Hydra/YAML: one config per experiment (no magic numbers in code)
│   ├── data/  models/  losses/  experiment/
├── specs/                         # SDD artifacts — the source of truth for WHAT to build
│   ├── _template.md               # spec template (problem, contract, acceptance tests, out-of-scope)
│   ├── 000-vertical-slice.md
│   ├── 001-data-layer.md
│   └── ...                        # one file per spec in §5
├── src/mri_ad/
│   ├── data/                      # MONAI datasets, transforms, split contract, data validation
│   ├── models/                    # UNet / AttUNet / UNETR behind one interface; checkpoint loading
│   ├── recon/                     # reconstruction + residual/anomaly-map engine
│   ├── eval/                      # Dice/IoU, PSNR/SSIM, table + plot generation
│   ├── classical/                 # feature extraction + gradient-boosting baseline
│   ├── synth/                     # synthetic-anomaly generation + anomaly-informed objective
│   ├── viz/                       # 3D viewer + demo-video export
│   └── utils/                     # seeds, logging, config, registry
├── artifacts/                     # generated tables, plots, anomaly maps (git-ignored if large)
├── data/                          # DATA — git-ignored, never committed (license!)
├── checkpoints/                   # trained weights — git-ignored, never committed
├── tests/
└── .claude/                       # skills, agents, hooks, plugin (see §3–§4)
```

**Hard rule:** `data/`, `checkpoints/`, and `*.nii.gz` / `*.npy` / `*.pt` are git-ignored and must never be committed. This is both a repo-hygiene rule and a medical-data-licensing rule, and it is enforced by a hook (§3.4).

---

## 3. Claude Code configuration

> Mechanics reflect current Claude Code (custom slash commands have merged into **Skills** at `.claude/skills/<name>/SKILL.md`; `.claude/commands/` still works but is legacy). Sub-agents live in `.claude/agents/`. Hooks are configured in `.claude/settings.json`.

### 3.1 `CLAUDE.md` (the agent constitution)

Keep it **tight** — Claude treats `CLAUDE.md` as context it *may* use, and a bloated file dilutes attention on the rules that matter. Target a short file with these sections:

1. **One-line project description** + the link to *this* plan as the canonical reference.
2. **The locked decisions (D1–D8)** — copied verbatim from §0, because they constrain every task.
3. **The central domain fact** — "models reconstruct the tumor too well; optimize detection separability, NOT reconstruction fidelity." This single sentence prevents the most likely class of wrong work.
4. **Repo map** — the §2 tree in 10 lines, so Claude knows where things live.
5. **Commands** — exact invocations: `make slice`, `make train`, `make eval`, `make report`, `make test`, how to point at a config.
6. **Conventions** — PyTorch + MONAI; everything config-driven (no hardcoded paths/hyperparameters); deterministic seeds set centrally; every model conforms to the `models/` interface; type hints + docstrings on public functions.
7. **SDD workflow rule** — *"Never write implementation code without an approved spec in `/specs`. The flow is `/specify` → `/plan` → `/tasks` → implement. Specs define WHAT and the acceptance tests; plans define HOW."*
8. **Budget & compute guardrail** — *"Training and any GPU/API-spending run is manual-invoke only. Never autonomously launch training, a sweep, or an LLM call that costs money."* (Backed by `disable-model-invocation` on those skills — §3.3.)
9. **Data guardrail** — never commit `data/` or `checkpoints/`; never print raw patient identifiers; treat all imaging data as sensitive.
10. **Pointers** — "for X use the `mri-domain` skill / the `experiment-runner` agent / the `/eval-report` command."

Everything operational (long how-tos, code patterns) belongs in **skills**, not in `CLAUDE.md` — keep the constitution lean.

### 3.2 Skills — domain knowledge (auto-invoked)

Skills with a good `description` auto-activate when Claude detects a relevant task. Use these for *expertise that should apply without being asked*.

| Skill | Folder | Purpose |
|---|---|---|
| `mri-domain` | `.claude/skills/mri-domain/SKILL.md` | MRI + anomaly-detection conventions: intensity normalization (min-max to [0,1]), the reconstruct-the-tumor failure mode, exact Dice/IoU/PSNR/SSIM definitions, slice/volume shape contract `(1,16,128,128)`, BraTS modality notes. Auto-applies so Claude never re-derives these or gets a metric wrong. |
| `monai-patterns` | `.claude/skills/monai-patterns/SKILL.md` | The repo's house style for MONAI: transform composition, `CacheDataset` usage, how UNETR is instantiated, deterministic dataloaders. Prevents drift across model modules. |
| `repro-discipline` | `.claude/skills/repro-discipline/SKILL.md` | Reproducibility checklist applied before any run: seeds set, config logged, git SHA recorded, no nondeterministic ops left on. |

### 3.3 Skills — workflows (slash-invoked)

These are the SDD loop and the operational commands. Mark the money/GPU-spending ones **manual-only** (`disable-model-invocation: true`) so Claude can't trigger them on its own — critical given the small budget.

| Command | Manual-only? | Purpose |
|---|---|---|
| `/specify` | no | Turn a feature description into a spec file following `specs/_template.md` (problem, data contract, interface, **acceptance tests**, out-of-scope). |
| `/plan` | no | Turn an approved spec into a technical plan: file list, interfaces, model/loss choices, the exact experiment matrix, risks. This is where deferred technical detail gets decided. |
| `/tasks` | no | Decompose a plan into an ordered, individually testable task list. |
| `/new-model` | no | Scaffold a new MONAI-backed model module conforming to the `models/` interface (forward, checkpoint load, model card). |
| `/eval-report` | no | Regenerate all metrics tables + plots from **saved** outputs (no training). Cheap, deterministic, re-runnable. |
| `/run-experiment` | **yes** | Standardized run: config in → metrics + artifacts out, fully logged. Manual-only because it uses the GPU. |
| `/train` | **yes** | Launch/fine-tune a model (e.g., synthetic-anomaly fine-tune from the UNETR checkpoint). Manual-only — the only spec that needs fresh compute. |
| `/lit` | no | Kick the `lit-scout` sub-agent (§3.5) for method research without polluting the main context. |

### 3.4 Hooks (`.claude/settings.json`)

Hooks enforce quality gates and guardrails automatically on lifecycle events.

| Event | Action |
|---|---|
| `PostToolUse` on `Edit`/`Write` of `*.py` | Run `ruff format` + `ruff check --fix`. Keeps style deterministic; no manual nagging. |
| `PreToolUse` on `Bash` | **Block dangerous/expensive commands**: `rm -rf`, and any `git add`/`git commit` touching `data/`, `checkpoints/`, `*.nii.gz`, `*.npy`, `*.pt`. This is the data-licensing + repo-hygiene guardrail. |
| `PreToolUse` on `Bash` | Warn/confirm before any command that looks like an unattended training launch outside `/train` or `/run-experiment`. Reinforces the budget guardrail. |
| `PostToolUse` on `Write` of `specs/*.md` | Validate the new spec against the template schema (has problem, contract, acceptance tests, out-of-scope). Reject incomplete specs early. |
| `SubagentStop` / `Stop` | Optional desktop/push notification when a long training or eval run finishes (so you're not babysitting the terminal). |
| `SessionStart` | Inject a one-line status: which specs are approved, which are implemented, current branch. Keeps long-running work oriented. |

### 3.5 Sub-agents (`.claude/agents/`)

Sub-agents get their own context window and return only a summary — use them to keep the main thread clean and to run cheaper models on grind work. (Sub-agents can't spawn sub-agents.)

| Agent | Model | Role |
|---|---|---|
| `spec-reviewer` | strong | Critiques a draft spec for completeness and **testability** before you approve it. Gatekeeper for the SDD loop. |
| `lit-scout` | mid | Web/docs research for methods (3D synthetic-lesion techniques, PyRadiomics features, MONAI APIs). Isolated so reference-reading noise never floods the main context. |
| `experiment-runner` | cheap (Haiku-class) | Drives long train/eval runs and returns just the metrics summary + artifact paths. Cheap model because it's orchestration, not reasoning. |
| `results-analyst` | strong | Reads saved metrics/outputs and writes the honest comparison narrative (fidelity-vs-detection, the three-paradigm Classical-vs-UNETR-vs-Diffusion table, before/after synthetic). |
| `torch-reviewer` | strong | Reviews PyTorch/MONAI code for the bugs research code actually hits: tensor-shape mismatches, device placement, `train()`/`eval()` mode, gradient leaks, nondeterminism. |

### 3.6 MCP servers

Be honest: an offline 3D-CV project gets **limited** value from MCP. Configure only what earns its place; don't over-engineer.

| Server | Why |
|---|---|
| **GitHub** (`claude mcp add github …`) | One PR per implemented spec, issues tracking the spec list. Real value for the SDD cadence and for the public-repo artifact. |
| **Docs / reference search** (e.g., a Context7-style or arxiv MCP) | Feeds `lit-scout` accurate MONAI / PyRadiomics / synthetic-anomaly references instead of stale recall. |
| *(Optional)* **W&B** | Only if you want experiment tracking surfaced to the agent. The W&B Python SDK alone is sufficient; skip the MCP unless you specifically want agent-queryable run history. |

Explicitly **not** using: filesystem MCP (built-in tools cover it), browser automation, database MCPs — none fit this project.

### 3.7 Plugin packaging

Bundle the skills (§3.2–3.3), sub-agents (§3.5), hooks (§3.4), and MCP config (§3.6) into **one project plugin** — e.g., `mri-sdd`. This makes the whole setup:
- **Versioned** with the repo (the constitution evolves with the code),
- **Portable** across your machines — install it once on the **college GPU cluster** and once on your **laptop** and both have the identical SDD environment,
- **Shareable** if you ever want the workflow itself to be a portfolio talking point.

Keep the plugin in-repo (under `.claude/`) and load it locally; a marketplace listing is unnecessary for a solo project.

---

## 4. The SDD operating loop

For **every** spec in §5, the cadence is the same:

1. `/specify <feature>` → drafts `specs/NNN-name.md` (problem, data/interface contract, **acceptance tests**, out-of-scope).
2. `spec-reviewer` agent critiques it → you approve.
3. `/plan` → technical plan (this is where the *exact* implementation, deferred by this document, gets decided).
4. `/tasks` → ordered, testable task list.
5. Implement task-by-task. Hooks auto-format/lint; `torch-reviewer` checks correctness.
6. Acceptance tests pass → `/eval-report` (or `/run-experiment` for GPU work) → open a PR via GitHub MCP.

**Thin-slice-first rule:** Spec 000 is a vertical slice through the *entire* stack on one model and one volume. Nothing else starts until the slice runs end to end. Then deepen.

---

## 5. The spec list (ordered)

High-level only — each becomes a full spec via `/specify`. Order respects dependencies. Because checkpoints and preprocessed data already exist (D8), most specs are *harness + plug-in-saved-outputs*; only Spec 009 needs fresh training.

### Phase A — Foundation (make one thing work end to end)

**Spec 000 · Vertical slice & reproducibility spine**
The thin end-to-end path: load the existing UNETR checkpoint → reconstruct **one** BraTS volume → compute residual/anomaly map → score Dice against the ground-truth segmentation → render one figure. Plus the spine everything hangs on: repo scaffold, pinned env, central seed/determinism, config system, experiment logging, `Makefile`. *Acceptance:* `make slice` produces a Dice number and a saved figure for one volume, reproducibly.

**Spec 001 · Data layer**
MONAI datasets + transforms for OpenBHB (healthy) and BraTS (anomalous); codify the preprocessing (min-max normalization, crop, slice to `(1,16,128,128)`) even though preprocessed data exists, so it's reproducible; the train/val/test **split contract**; a data-validation step that fails loudly on shape/intensity drift. *Acceptance:* dataloaders yield correctly-shaped, normalized batches; split is deterministic and documented.

**Spec 002 · Model registry & checkpoint loading**
UNet, Attention-UNet, UNETR as MONAI-backed modules behind **one uniform interface**; load the existing checkpoints; a short model card per architecture. *Acceptance:* every model loads its checkpoint and runs a forward pass on a real batch through the same API.

### Phase B — Core method & evaluation

**Spec 003 · Reconstruction & anomaly-map engine**
The heart of the method: given a model + volume, produce reconstruction, residual map, and a thresholded anomaly mask. Centralize the thresholding strategy (it materially affects Dice). *Acceptance:* engine emits reconstruction, residual, and mask for any registered model.

**Spec 004 · Evaluation harness**
Localization metrics (Dice, IoU) against BraTS segmentation maps; reconstruction-quality metrics (PSNR, SSIM) kept strictly as *explanatory context*; per-volume and aggregate; auto-generated results tables + plots from **saved** outputs. *Acceptance:* reproduces the prior report's headline numbers for UNETR within tolerance, from checkpoints alone.

**Spec 005 · Fidelity-vs-detection study (arch × loss + diffusion)**
Orchestrate the matrix (UNETR × MSE / SSIM / MSE+SSIM / perceptual / multi-scale; UNet/AttUNet as reference rows; a diffusion row) into one clean comparison table, with the **central-finding analysis**: show that higher reconstruction fidelity does not buy better detection, and test whether the diffusion model's generative prior escapes that anti-correlation. *Acceptance:* a single generated table + written analysis covering the matrix, authored by `results-analyst`.

### Phase C — The resume-gap closer

**Spec 006 · Classical-ML baseline**
Radiomic / intensity / texture feature extraction (PyRadiomics-style) → XGBoost/LightGBM → k-fold cross-validation → ROC-AUC / PR-AUC. *Open sub-decision for the `/plan` step:* operating granularity — **slice-level** anomaly classification (clean ROC-AUC, weak localization) vs **supervoxel/region-level** (enables crude localization comparable to Dice). Resolve in the plan; the spec should state both and pick one with justification. *Acceptance:* cross-validated AUC reported with the granularity explicitly documented.

**Spec 007 · Paradigm comparison — Classical vs UNETR vs Diffusion**
Put the three paradigms on the **same test split** and compare honestly — where each wins, where each fails, on a common footing (and note any metric-comparability caveats from Spec 006's granularity choice). This is the project's headline table. *Acceptance:* one comparison artifact (table + ROC/PR overlay + narrative).

### Phase D — Performance novelties  *(the two fresh-compute specs)*

**Spec 009 · Synthetic-anomaly (anomaly-informed) training**  *(fresh compute)*
Generate plausible synthetic lesions on healthy volumes (e.g., Poisson-blend / FPI / 3D CutPaste-style corruptions) and train the model to **restore the healthy version**, so it learns to erase anomalous-looking regions rather than copy them. Fine-tune from the existing UNETR checkpoint (cheap). Produce the **before/after** Dice story on the same test split. *Acceptance:* a controlled before/after table on the identical split, with synthetic-generation parameters logged. *Guardrail:* `/train` is manual-invoke only.

**Spec 013 · Diffusion-based anomaly detection (AnoDDPM)**  *(fresh compute — the newest paradigm)*
Train a DDPM (MONAI `DiffusionModelUNet` + `DDPMScheduler`) on **healthy** OpenBHB; at inference, partially noise the input then denoise to a healthy estimate whose residual feeds the existing `recon/` engine. Conforms to the `AnomalyDetectionModel` interface, so `recon/eval/classical` are untouched — it is a registry drop-in. Becomes the third column of the Spec 007 headline table. *Acceptance:* the model registers/loads/runs through the shared interface and appears in the 005/007 tables on the identical split; if it does not beat UNETR, that is reported. *Guardrail:* `/train` is manual-invoke only.

### Phase E — Artifact & presentation

**Spec 010 · 3D visualization & demo video**
Viewer overlaying original / reconstruction / residual / anomaly-map / ground-truth (niivue, or a notebook/Gradio surface — decide in `/plan`); export a short **demo video** (the project's public artifact, since always-on hosting isn't worth it for a large 3D model). *Acceptance:* a reproducible viewer + an exported demo clip.

**Spec 011 · Reporting & public artifact**
Auto-assembled README: architecture diagram, the consistent metric scorecard (Dice/IoU + ROC-AUC, inference time/volume, GPU hours), reproducibility instructions, and the **mobility-transfer paragraph** (D5 — conceptual transfer to OOD trajectory detection, explicitly *not* a tested result). *Acceptance:* a reviewer-ready README + a `make report` that regenerates every number from saved outputs.

### Phase F — Stretch (only if time/compute allow)

**Spec 012 · Multi-scale attention variant** *(stretch, off critical path)*
An architectural variant adding multi-scale attention to the best model, written as a full spec but clearly optional. Touch only after Phases A–E are done and defensible. *Acceptance:* either a measured result added to the comparison, or a documented decision to defer.

---

## 6. Build order at a glance

```
000 vertical slice ─► 001 data ─► 002 models ─► 003 recon engine ─► 004 eval harness
        │                                                                 │
        └──────────────── (everything below reuses these) ───────────────┘
                                      │
   ┌──────────────┬──────────────┬──────────────┬──────────────┐
   ▼              ▼              ▼              ▼              ▼
 005 fidelity   006 classical  009 synth-     013 diffusion   (004 feeds all)
 vs-detection   baseline       anomaly (TRAIN) AnoDDPM (TRAIN)
   │            │                              │
   └────────────┴──► 007 paradigm comparison ◄─┘
                     (Classical vs UNETR vs Diffusion)
                                      │
                                      ▼
                              010 visualization ─► 011 reporting/README
                                                          │
                                                          ▼
                                            012 multi-scale attention (stretch)
```

Critical path: **000 → 001 → 002 → 003 → 004**, then 005 / 006 / 009 / 013 can proceed in parallel, converging on 007 (the headline three-paradigm table) and the 010–011 artifact. 009 and 013 are the two `/train`-gated specs; 012 is optional.

---

## 7. Definition of done

- Every spec has an approved `specs/NNN-*.md`, a plan, passing acceptance tests, and a merged PR.
- `make report` regenerates **every** headline number from saved checkpoints/outputs with no manual steps.
- README carries the diagram, the metric scorecard, reproducibility instructions, and the mobility paragraph.
- A demo video exists for the 3D visualization.
- `data/` and `checkpoints/` are provably absent from git history.
- The honest comparative narrative — fidelity-vs-detection, the three-paradigm Classical-vs-UNETR-vs-Diffusion comparison, synthetic before/after — is written and defensible in an interview.
