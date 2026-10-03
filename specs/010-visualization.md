# Spec 010 · 3D visualization & demo video

**Status:** implemented
**Depends on:** 003, 004
**Fresh compute required:** no

---

## Problem

The project's public artifact is a video, not a hosted service (D5, C-8 — always-on hosting is not
worth it for a large 3D model). A recruiter or interviewer must be able to *see* the method work —
original, reconstruction, residual, mask, ground truth — without cloning anything.

## Contract

**Viewer: Matplotlib + `ipywidgets`,** with a separate headless export path (`Agg` backend +
`imageio[ffmpeg]`). Chosen over NiiVue (needs a browser server) and Gradio (adds a web-app
dependency) — both conflict with C-8.

```python
VolumeViewer(result: ReconResult, label: Tensor)
    .show_interactive()                       # 5-panel, depth slider
    .export_video(path: Path, fps: int = 4)   # MP4
DemoExporter.export(results_dir, volume_id, output_path)
```

Panels: original | reconstruction | residual (`cm.hot` heatmap) | anomaly mask | GT overlay.
The residual colorization is ported from `legacy/GUI/tk_app.py`.

## Acceptance tests

1. The viewer renders all five panels, synchronized on one depth slider, for any saved `ReconResult`.
2. `make demo` exports an MP4 from a **saved** `ReconResult` with no model forward pass and no GPU.
3. The export is reproducible: same input → same output frames.
4. Per-slice normalization handles an all-zero slice without dividing by zero. (The legacy GUI
   divides by `np.max(slice)` with no guard and crashes on empty slices.)
5. No patient identifier or file path is rendered into any frame (NFR-13).

## Out of scope

A hosted web app (C-8). Real-time inference in the viewer — it reads saved results only.

## Notes / deviations

`planning/02-architecture.md:288` already resolves the viewer choice to Matplotlib; we keep it.
The legacy GUI's defects (no min-subtraction in normalization, a Windows-hardcoded checkpoint path,
the model reloaded from disk on every click, a commented-out binarization) are **not** ported —
only the `cm.hot` colorization recipe is.
