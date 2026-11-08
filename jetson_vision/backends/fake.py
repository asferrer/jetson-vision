"""Deterministic fake backend for tests: raw outputs in the real engine format, derived from the input bytes."""
from __future__ import annotations

import time
import zlib
from typing import Callable, Mapping

import numpy as np

from jetson_vision.manifest import Manifest

_DEFAULT_SIZE = {"rfdetr": (560, 560), "rtdetrv2": (640, 640)}


class FakeBackend:
    """Same input, same output: the generator is seeded with the CRC-32 of the image bytes.

    latency_s simulates inference time; hook(call_index) runs before each inference and may raise or block.
    """

    def __init__(self, manifest: Manifest | None = None, *, kind: str = "rfdetr", latency_s: float = 0.0,
                 hook: Callable[[int], None] | None = None) -> None:
        self.kind = manifest.decoder.name if manifest else kind
        if self.kind not in _DEFAULT_SIZE:
            raise ValueError(f"unknown fake kind '{self.kind}'")
        self.model_id = manifest.model_id if manifest else f"fake_{self.kind}"
        self.size = (manifest.input.width, manifest.input.height) if manifest else _DEFAULT_SIZE[self.kind]
        self.num_classes = len(manifest.classes.names) if manifest else 8
        self.latency_s = latency_s
        self.hook = hook
        self.loaded = False
        self.calls = 0

    def load(self) -> None:
        self.loaded = True

    def infer(self, input_u8_chw_or_hwc: np.ndarray) -> Mapping[str, np.ndarray]:
        if not self.loaded:
            raise RuntimeError("FakeBackend.infer called before load()")
        call, self.calls = self.calls, self.calls + 1
        if self.hook is not None:
            self.hook(call)
        if self.latency_s:
            time.sleep(self.latency_s)
        data = np.ascontiguousarray(input_u8_chw_or_hwc)
        rng = np.random.default_rng(zlib.crc32(data.tobytes()))
        return self._rfdetr(rng) if self.kind == "rfdetr" else self._rtdetrv2(rng)

    def _rfdetr(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        dets = np.empty((1, 300, 4), np.float32)
        dets[..., :2] = rng.uniform(0.2, 0.8, (300, 2))
        dets[..., 2:] = rng.uniform(0.05, 0.3, (300, 2))
        logits = rng.normal(-5.0, 1.5, (1, 300, self.num_classes + 1)).astype(np.float32)
        hot = rng.choice(300, 5, replace=False)
        logits[0, hot, rng.integers(0, self.num_classes, 5)] = rng.uniform(1.0, 4.0, 5)
        return {"dets": dets, "labels": logits}

    def _rtdetrv2(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        w, h = self.size
        centre = rng.uniform(0.2, 0.8, (300, 2)) * [w, h]
        half = rng.uniform(0.02, 0.15, (300, 2)) * [w, h]
        boxes = np.concatenate([centre - half, centre + half], 1).astype(np.float32)
        scores = rng.uniform(0.0, 0.3, 300).astype(np.float32)
        scores[rng.choice(300, 5, replace=False)] = rng.uniform(0.7, 0.99, 5)
        labels = rng.integers(0, self.num_classes, 300).astype(np.int64)
        return {"labels": labels, "boxes": boxes, "scores": scores}

    def close(self) -> None:
        self.loaded = False
