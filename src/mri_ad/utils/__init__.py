"""Cross-cutting: seeds, logging, config, device. Depends on nothing in mri_ad.

**No re-exports.** ``mri_ad.utils.run_logger`` is the one submodule that must stay importable
without pulling in ``torch`` (Spec 004's "report-safe subgraph" — ``scripts/run_report.py``
imports it directly). Re-exporting ``DeviceManager``/``seed_everything`` here would import
``mri_ad.utils.device``/``mri_ad.utils.seed`` (both torch-touching) as a side effect of
importing *any* submodule of this package, since Python always runs a package's ``__init__.py``
before a submodule import completes. Import what you need from its own submodule instead:
``from mri_ad.utils.device import DeviceManager``, etc.
"""
