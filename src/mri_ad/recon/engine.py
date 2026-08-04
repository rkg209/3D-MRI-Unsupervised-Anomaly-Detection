"""The reconstruction & anomaly-map engine (Spec 003).

``model + volume -> reconstruction -> residual -> binary anomaly mask``, and nothing else.
Thresholding happens here and nowhere else in the codebase (see ``recon/threshold.py``).

*Deviation from the spec's* ``run(volume) -> ReconResult`` *signature:* ``volume_id`` is a
required keyword. The spec mandates the persistence path ``.../<volume_id>.pt``, and
``BraTSDataset.__getitem__`` deliberately returns no path or subject id (NFR-13, no patient
identifiers) — so the id must be threaded in explicitly by the caller.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import Tensor

from mri_ad.data.chunking import chunk_volume
from mri_ad.data.datasets import BraTSDataset
from mri_ad.exceptions import ReconError
from mri_ad.models.base import AnomalyDetectionModel
from mri_ad.recon.types import ReconResult
from mri_ad.utils.device import DeviceManager
from mri_ad.utils.instantiate import instantiate_from_config

_CHUNK_DEPTH = 16
_SIDE = 128


class ReconstructionEngine:
    """Turns one model + one volume into a persisted :class:`ReconResult`."""

    def __init__(self, model: AnomalyDetectionModel, config: DictConfig) -> None:
        """Store the model and build the threshold strategy from ``config.threshold``."""
        self.model = model
        self.config = config
        self.device = DeviceManager.get_device(str(config.get("device", "auto")))
        self.batch_size = int(config.recon.batch_size)
        self._threshold = instantiate_from_config(config.threshold)

    def run(
        self,
        volume: Tensor,
        *,
        volume_id: str,
        ground_truth: Tensor | None = None,
    ) -> ReconResult:
        """Reconstruct, compute the residual, and threshold it — the full per-volume pipeline."""
        volume = self._validate_input(volume)

        if self.model.training:
            raise ReconError(
                "ReconstructionEngine.run() requires the model in eval mode "
                "(dropout/BN would make reconstructions nondeterministic)."
            )

        chunks = chunk_volume(volume.squeeze(0), chunk_depth=_CHUNK_DEPTH)  # (N,1,16,128,128)
        recon_chunks: list[Tensor] = []
        with torch.no_grad():
            for start in range(0, chunks.shape[0], self.batch_size):
                batch = chunks[start : start + self.batch_size].to(self.device)
                out = self.model(batch).detach().cpu()
                recon_chunks.append(out)
        reconstruction_chunks = torch.cat(recon_chunks, dim=0)  # (N,1,16,128,128)

        # Reassemble along depth (dim=1 of the per-chunk (1,16,128,128) tensors): cat the N
        # chunks -> (1, D, 128, 128). Getting this axis wrong silently reassembles the wrong way
        # (prior-work bug #2 in a new costume).
        original = torch.cat(list(chunks), dim=1)  # (1, D, 128, 128)
        reconstruction = torch.cat(list(reconstruction_chunks), dim=1)  # (1, D, 128, 128)
        residual = (original - reconstruction).abs()

        # Threshold globally over the whole reassembled volume, once — never per chunk.
        anomaly_mask = self._threshold(residual)

        return ReconResult(
            volume_id=volume_id,
            original=original.detach().cpu().float().contiguous(),
            reconstruction=reconstruction.detach().cpu().float().contiguous(),
            residual=residual.detach().cpu().float().contiguous(),
            anomaly_mask=anomaly_mask.detach().cpu().float().contiguous(),
            ground_truth=(
                ground_truth.detach().cpu().float().contiguous()
                if ground_truth is not None
                else None
            ),
        )

    def run_dataset_volume(self, dataset: BraTSDataset, volume_index: int) -> ReconResult:
        """Gather every chunk of ``volume_index``, reassemble, and delegate to :meth:`run`.

        This is where the "accumulate across all chunks" discipline lives (prior-work bug #2):
        every chunk belonging to the volume is collected before thresholding, not just one.
        """
        # dataset._offsets[i] is the first flattened chunk index of volume i (prefix sums built
        # in BraTSDataset.__init__); using it avoids scanning unrelated volumes' chunks.
        start = dataset._offsets[volume_index]
        end = dataset._offsets[volume_index + 1]

        image_chunks = []
        label_chunks = []
        for idx in range(start, end):
            item = dataset[idx]
            image_chunks.append(item["image"])
            label_chunks.append(item["label"])

        image = torch.cat(image_chunks, dim=1)  # (1, D, 128, 128)
        label = torch.cat(label_chunks, dim=1)  # (1, D, 128, 128)
        volume_id = dataset.volume_ids[volume_index]

        return self.run(image, volume_id=volume_id, ground_truth=label)

    def _validate_input(self, volume: Tensor) -> Tensor:
        """Accept ``(D,H,W)`` or ``(1,D,H,W)`` float32 with ``D % 16 == 0``, ``H == W == 128``."""
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        if volume.ndim != 4 or volume.shape[0] != 1:
            raise ReconError(
                f"Expected a (1, D, H, W) or (D, H, W) volume, got shape {tuple(volume.shape)}."
            )
        if volume.dtype != torch.float32:
            raise ReconError(f"Expected float32, got {volume.dtype}.")
        _, depth, height, width = volume.shape
        if depth % _CHUNK_DEPTH != 0:
            raise ReconError(f"Depth must be a multiple of {_CHUNK_DEPTH}, got {depth}.")
        if height != _SIDE or width != _SIDE:
            raise ReconError(f"Expected H == W == {_SIDE}, got H={height}, W={width}.")
        return volume


__all__ = ["ReconstructionEngine"]
