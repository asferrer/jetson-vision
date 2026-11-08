"""Manifest-driven resize on uint8. Normalisation and transposition happen later, on the GPU.

A non-contiguous view (for example a transposed array) goes through NumPy and OpenCV without any error and
produces garbage: one evaluation recorded an AP of 0.31 for that. Both the frame and the result are therefore checked.
"""
from __future__ import annotations

import cv2
import numpy as np

from jetson_vision.manifest import InputSpec


class NonContiguousError(ValueError):
    """A buffer that must be C-contiguous is not (or has the wrong dtype or shape)."""


def require_image(arr: np.ndarray, name: str) -> np.ndarray:
    """Check that arr is a C-contiguous uint8 HxWx3 array. Raises instead of using assert (survives python -O)."""
    if not isinstance(arr, np.ndarray):
        raise NonContiguousError(f"{name}: expected a numpy array, got {type(arr).__name__}")
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise NonContiguousError(f"{name}: expected uint8 HxWx3, got {arr.dtype} {arr.shape}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise NonContiguousError(
            f"{name}: buffer is not C-contiguous (shape {arr.shape}, strides {arr.strides}). "
            "Call np.ascontiguousarray before the model sees it; a strided view silently ruins the AP."
        )
    return arr


class Preprocessor:
    """BGR uint8 frame -> uint8 HxWx3 image of the network input size, in the channel order of the manifest."""

    def __init__(self, spec: InputSpec) -> None:
        self.spec = spec

    def _interpolation(self, frame_height: int) -> int:
        mode = self.spec.interpolation
        if mode == "linear":
            return cv2.INTER_LINEAR
        if mode == "area":
            return cv2.INTER_AREA
        # area_if_downscale: the rule of build_int8_engine.preprocess (compares the height only)
        return cv2.INTER_AREA if frame_height > self.spec.height else cv2.INTER_LINEAR

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        require_image(bgr, "input frame")
        interp = self._interpolation(bgr.shape[0])
        # Resizing acts per channel, so resizing BGR and swapping afterwards is bit-identical to the reverse
        # order and converts 0.4 megapixels instead of a full frame.
        small = cv2.resize(bgr, (self.spec.width, self.spec.height), interpolation=interp)
        if self.spec.channel_order == "rgb":
            small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return require_image(small, "preprocessed image")
