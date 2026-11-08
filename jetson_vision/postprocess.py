"""Decoders for RF-DETR and RT-DETRv2: raw engine outputs -> detections in pixels of the original frame.

Neither model uses NMS (they are DETR). Decoding stays on the CPU: on the Orin Nano it is faster than on
the GPU for tensors of this size (a GPU post-process measured slower).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from jetson_vision.interfaces import Decoder
from jetson_vision.manifest import ClassSpec, Manifest
from jetson_vision.types import Detection


class _Labels:
    """Maps raw engine labels to class ids and names; drops ignored and out-of-range labels."""

    def __init__(self, classes: ClassSpec) -> None:
        self.names = classes.names
        self.offset = classes.label_offset
        self.ignore = frozenset(classes.ignore_labels)

    def valid(self, raw: np.ndarray) -> np.ndarray:
        ids = raw - self.offset
        ok = (ids >= 0) & (ids < len(self.names))
        if self.ignore:
            ok &= ~np.isin(raw, list(self.ignore))
        return ok

    def detection(self, raw: int, score: float, box: np.ndarray) -> Detection:
        cid = int(raw) - self.offset
        return Detection(class_id=cid, class_name=self.names[cid], score=float(score),
                         box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])))


def _first(a: np.ndarray, ndim: int) -> np.ndarray:
    """Drop a leading batch dimension of 1 so that engines with and without it decode the same."""
    return a[0] if a.ndim == ndim + 1 else a


class RfDetrDecoder:
    """Sigmoid, top-k over queries x classes, cxcywh -> xyxy, scaled to the frame (RF-DETR PostProcess)."""

    def __init__(self, classes: ClassSpec, num_select: int) -> None:
        self.labels = _Labels(classes)
        self.num_select = num_select

    def __call__(self, outputs: Mapping[str, np.ndarray], frame_width: int, frame_height: int,
                 score_threshold: float) -> list[Detection]:
        dets = _first(np.asarray(outputs["dets"]), 2)
        logits = _first(np.asarray(outputs["labels"]), 2)
        flat = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).reshape(-1)
        k = min(self.num_select, flat.size)
        # Ties are common (quantised logits): order by score, then by flat index, so the top-k set and its order
        # are the same on every run. argpartition left the choice among equal scores to chance.
        idx = np.lexsort((np.arange(flat.size), -flat))[:k]
        query, raw = idx // logits.shape[1], idx % logits.shape[1]
        cx, cy, w, h = dets[query].astype(np.float64).T
        scale = np.array([frame_width, frame_height, frame_width, frame_height], dtype=np.float64)
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * scale
        keep = self.labels.valid(raw) & (flat[idx] >= score_threshold)
        return [self.labels.detection(r, s, b) for r, s, b in zip(raw[keep], flat[idx][keep], boxes[keep])]


class RtDetrV2Decoder:
    """RT-DETRv2 exports labels, boxes (xyxy) and scores already decoded.

    The engine scales boxes to the size given in orig_target_sizes. The backend gives it the network input
    size, so the boxes arrive in network pixels and are rescaled here to the frame.
    """

    def __init__(self, classes: ClassSpec, box_space_wh: tuple[int, int]) -> None:
        self.labels = _Labels(classes)
        self.box_space = box_space_wh

    def __call__(self, outputs: Mapping[str, np.ndarray], frame_width: int, frame_height: int,
                 score_threshold: float) -> list[Detection]:
        raw = _first(np.asarray(outputs["labels"]), 1).astype(np.int64)
        boxes = _first(np.asarray(outputs["boxes"]), 2).astype(np.float64)
        scores = _first(np.asarray(outputs["scores"]), 1).astype(np.float64)
        sx, sy = frame_width / self.box_space[0], frame_height / self.box_space[1]
        boxes = boxes * np.array([sx, sy, sx, sy], dtype=np.float64)
        keep = self.labels.valid(raw) & (scores >= score_threshold)
        order = np.flatnonzero(keep)
        order = order[np.argsort(-scores[order], kind="stable")]
        return [self.labels.detection(raw[i], scores[i], boxes[i]) for i in order]


def build_decoder(manifest: Manifest) -> Decoder:
    """The decoder named by the manifest."""
    spec = manifest.decoder
    if spec.name == "rfdetr":
        return RfDetrDecoder(manifest.classes, spec.num_select)
    if spec.name == "rtdetrv2":
        return RtDetrV2Decoder(manifest.classes, (manifest.input.width, manifest.input.height))
    raise ValueError(f"unknown decoder '{spec.name}'")
