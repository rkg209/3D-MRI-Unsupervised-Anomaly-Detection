---
name: new-model
description: Scaffold a new MONAI-backed model module conforming to the models/ interface — forward, checkpoint loading, and a model card. Use when adding an architecture to the registry.
---

# /new-model — add an architecture

Adding a model must touch **only three things** (FR-14 / NFR-15):

1. `src/mri_ad/models/<name>.py` — a class implementing `AnomalyDetectionModel`.
2. `configs/model/<name>.yaml` — params + checkpoint path.
3. The checkpoint itself, in `checkpoints/`.

**Zero changes to `recon/`, `eval/`, `classical/`, or `viz/`.** If you find yourself editing those,
the abstraction has failed — stop and report that, because it is a more important finding than the
model.

## The interface

```python
class MyModel(AnomalyDetectionModel):
    def forward(self, x: Tensor) -> Tensor:      # (B,1,16,128,128) -> same
    def load_checkpoint(self, path: Path) -> None:
    @property
    def model_card(self) -> ModelCard:
```

## Checkpoint loading — get this right

Handle **both** on-disk layouts, load `strict=True`, and reject LFS stubs:

```python
raw = torch.load(path, map_location="cpu")
state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
model.load_state_dict(state, strict=True)
```

**Never `strict=False`.** A partially-loaded model does not crash — it produces plausible
reconstructions from partly-random weights, and every metric downstream computes cleanly on
garbage. It is the worst bug in this codebase's problem space because it is silent.

If keys do not match, the fix is a **checkpoint-compat shim with an equivalence test**, not a
loosened flag. This is exactly why UNETR is a ported class rather than `monai.networks.nets.UNETR`
— see `specs/002-model-registry.md`.

## Model card

Name, param count, input/output shape, training data, **training loss**, output activation
(only UNETR has one — a Sigmoid), and known characteristics.

## Tests to add

- Loads its real checkpoint `strict=True`, no missing/unexpected keys.
- Forward on `(2,1,16,128,128)` returns that shape.
- If it has a bounded output, assert the bound. If not, do **not** assert one.
- Registering a class missing an interface method raises `ModelRegistrationError`.
