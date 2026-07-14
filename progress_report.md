# Progress report

The narrative history of this project, in sequence. Every meaningful change gets an entry saying
**what** was done, **why**, and **how** — including problems hit and how each was resolved. Read
end to end, this file should tell the whole story of how the project was built and what it cost.

**This file is append-only.** Never rewrite or delete an entry. If a decision is reversed, that
reversal is a *new* entry explaining why — the wrong turn is part of the story and is often the
most useful part of it.

## Entry format

```markdown
## NNN · <Title>
**Date:** YYYY-MM-DD · **Spec:** <id or n/a> · **Status:** done | in progress | reverted

### What
### Why
### How
### Problems hit
### Result
```

---

## 000 · Inherited state and pre-rebuild audit
**Date:** 2026-07-14 · **Spec:** n/a · **Status:** done

### What
Audited the existing project (IE643_TensorTitan) before writing a line of new code, and recorded
what we actually inherited as opposed to what the planning documents assumed we inherited.

### Why
The build plan's locked decision **D8** states that data and checkpoints are "already in hand",
and it is load-bearing: it is what makes Specs 001–007 cheap "build the harness, plug in the saved
outputs" work rather than GPU-hungry retraining. If D8 were false, most of the plan's cost model
would be wrong. It needed checking before committing to the plan, not after.

### How
Read the SDD build plan and all of `planning/`; extracted the source cells from every notebook
(`UNET/`, `UNETR/`, `attUNET/`, `metric-uad.ipynb`); read the Tkinter GUI; inspected every `.pth`
file on disk; checked for a git repository.

### Problems hit
Six findings, in descending order of severity.

1. **The checkpoints do not exist.** All seven `.pth` files are **Git LFS pointer stubs — 133
   bytes each, containing no weights.** There is no UNETR checkpoint on disk *at all*, and UNETR
   is the best model (the reported Dice ≈ 0.63). The real weights live only in the Google Drive
   folder linked from the README. D8 survives only because the user confirmed they can fetch them.

2. **The trained UNETR weights cannot load into the class the architecture doc specifies.**
   `planning/02-architecture.md:139` says `UNETRModel` wraps `monai.networks.nets.UNETR`. The
   weights were actually produced by the custom `UNETR_Reconstruction` in
   `GUI/utilities/utility.py:15`, whose output head is an `nn.Conv3d` (state-dict keys
   `out.weight`) where MONAI uses a `UnetOutBlock` (`out.conv.conv.weight`), and which ends in a
   `Sigmoid` that MONAI's class does not have. Loading one into the other either raises — or, if
   someone reaches for `strict=False` to silence it, loads *partially* and produces a model that
   reconstructs noise while every downstream metric still computes cleanly. **That failure is
   silent, which is what makes it the most dangerous finding here.**

3. **The published Dice ≈ 0.6255 is not trustworthy.** `metric-uad.ipynb` contains four
   independent bugs, each sufficient on its own to invalidate the number:
   - `dice_scores`/`iou_scores` are re-initialized **inside** the per-subject loop, so every
     reported "average" is really the **last subject's** score;
   - the BraTS segmentation is fed in **raw** (labels `{0,1,2,4}`) rather than binarized, so the
     Dice numerator is weighted by label magnitude — a tumor-core voxel counts 4× an edema voxel.
     That is not a Dice coefficient;
   - that same segmentation is resized **trilinear**, producing fractional "labels" (0.37, 1.84…);
   - only **depth-chunk 4 of 8** is ever scored — seven-eighths of every volume is ignored.

4. **Train and eval preprocessing are different functions.** OpenBHB: crop → resize → normalize.
   BraTS: normalize → transpose → a *different* crop (depth not cropped at all, discarding 27 of
   155 slices) → resize. The model is therefore evaluated on a distribution it was not trained on.

5. **Scope creep in the planning docs.** `planning/04-database-schema.sql` (PostgreSQL 15 + pgcrypto
   + Flyway migrations + a mandated CI Postgres instance) and `planning/05-openapi.yaml` (OpenAPI
   3.0.3, `servers: localhost:8000`, global API-key auth) describe a server and database that
   `planning/02-architecture.md:9` explicitly says **do not exist** and that no spec builds. Both
   files are also truncated mid-statement and will not parse.

6. **Minor:** the perceptual loss is commented out in every notebook, so
   `UNET/3d_slices_MSE_Perceptual.pth` has no surviving training code (the live criterion in that
   notebook is actually `multi_scale_loss` — the checkpoint is probably mislabelled). PSNR and SSIM
   are imported but never computed. The README claims 100 training epochs; the notebooks say 50.

### Result
The prior work's *method* is sound and its central finding stands. Its *measurements* do not. This
reframes the rebuild: it is not merely a tidy-up, it is a **re-measurement**. Concretely,
requirement `FR-21` ("reproduce Dice 0.6255 ± 0.005") had to be rewritten — see entry 001 — because
it demands we reproduce a bug.

---

## 001 · SDD scaffolding
**Date:** 2026-07-14 · **Spec:** n/a (this builds the machinery the specs run on) · **Status:** done

### What
Built the complete spec-driven-development environment: repo restructure, git, the Python
skeleton, all 12 specs, `CLAUDE.md`, and the `.claude/` skills / agents / hooks. No ML code — that
begins with Spec 000.

### Why
The build plan calls for an SDD loop (`/specify` → `/plan` → `/tasks` → implement) with specs as
the source of truth, but **none of the machinery existed** — no `.claude/`, no `specs/`, no `src/`,
and the directory was not even a git repository. Building the guardrails *before* the code is the
whole point: the failure mode this project actually has is not crashes, it is confidently wrong
numbers, and the audit above shows that is not hypothetical.

### How
- **Restructure.** Prior code → `legacy/` (reference only, per D1), with a README cataloguing its
  defects so they are never reintroduced. The Postgres/OpenAPI docs → `planning/future/`, marked
  non-normative and *do not implement*.
- **Git.** `git init` + an initial commit. `.gitignore` excludes `data/`, `checkpoints/`, and all
  imaging/weight binaries (`*.pt`, `*.pth`, `*.nii*`, `*.npy`) — a medical-data licensing rule as
  much as repo hygiene.
- **Skeleton.** `pyproject.toml`, `Makefile`, a Hydra `configs/` tree seeded with the exact values
  recovered from the notebooks, and `src/mri_ad/` containing **interfaces and contracts only** —
  the ABC, the dataclasses, the exception hierarchy, the canonical metric signatures. Bodies get
  filled spec by spec.
- **`make check-data`.** A gate that verifies weights and data are real files and *specifically
  detects Git LFS pointer stubs*, because that is the exact trap the inherited repo lays.
- **Specs.** All 12 (000–007, 009–012; there is deliberately no 008), each with falsifiable
  acceptance criteria.
- **`CLAUDE.md`.** Deliberately lean — a bloated constitution dilutes attention on the rules that
  matter. It carries D1–D8, the central domain fact, the guardrails, and the known traps.
- **`.claude/`.** 3 auto-invoked domain skills (`mri-domain`, `monai-patterns`, `repro-discipline`),
  8 workflow skills, 5 sub-agents, and hooks.

### Problems hit
1. **The `Co-Authored-By` guard silently did not fire on first test.** Both commit-trailer cases
   returned exit 0 (allow) when they should have returned 2 (block). The cause turned out to be my
   *test fixture*, not the hook — zsh's `echo` expanded `\n` into a literal newline, producing
   invalid JSON, and the hook's `except json.JSONDecodeError: sys.exit(0)` fell open. But that
   exposed a genuine weakness: **a malformed payload was a way to bypass the guard.** Fixed by
   failing *closed* on the commit-trailer rule — if the JSON will not parse, the hook now scans the
   raw stdin text for the trailer anyway. Re-tested with correctly-encoded JSON: 15/15 cases pass,
   including the exact `Co-Authored-By: Claude Sonnet 5` line, the heredoc form, and the malformed
   payload. **Worth recording: the guard would have been decorative had I trusted it untested.**

2. **`FR-21` had to be overruled.** It requires reproducing Dice 0.6255 ± 0.005. That is
   unreachable *by construction* once Spec 001 unifies the preprocessing and Spec 004 binarizes the
   segmentation — and the target came from buggy code anyway. Spec 004 now instead: computes the
   metrics correctly; reproduces the old numbers **only** under an explicit `--legacy-bug-compat`
   mode (proving we can explain the gap); and publishes an honest re-measured baseline. **The
   corrected Dice may well be lower than 0.6255. It gets published as-is.**

3. **The architecture doc's threshold claim is wrong.** `02-architecture.md:183` calls a
   95th-percentile threshold the default "matching the prior work". The prior work used a hardcoded
   absolute `residual > 0.1`. These behave differently — a percentile threshold *fixes* the flagged-
   voxel count by construction (exactly 5% of voxels always flagged, healthy brain or not), which
   changes what Dice measures. Spec 003 now names this as a deliberate change and requires the
   operating point be chosen by a sweep on **validation**, never asserted and never tuned on test.

### Result
Scaffolding complete and verified: 17 scaffold tests pass, `make check-data` correctly **blocks**
(no weights on disk yet — which is the right answer and proves the gate works), the git history is
clean with zero `Co-Authored-By` trailers, and the hooks enforce the guardrails rather than merely
recommending them.

**Next:** the user places the real UNETR weights in `checkpoints/` and the OpenBHB/BraTS data in
`data/`. Then the SDD loop begins on **Spec 000 (vertical slice)** — the thin end-to-end path —
and proceeds spec by spec.
