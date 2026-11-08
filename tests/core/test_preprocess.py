"""Preprocessing: resize rules from the manifest and the contiguous-buffer assertion."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from jetson_vision.preprocess import NonContiguousError, Preprocessor, require_image


def noise(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def reference(bgr: np.ndarray, size: int, interp: int) -> np.ndarray:
    """RGB first, resize after: the order used by stages.py."""
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (size, size), interpolation=interp)


def test_transposed_view_is_refused_not_processed(rf_manifest):
    """A previous evaluation recorded AP 0.31 from a non-contiguous transposed view with no error at all."""
    chw = np.random.default_rng(0).integers(0, 256, (3, 48, 64), dtype=np.uint8)
    view = chw.transpose(1, 2, 0)  # right shape (48, 64, 3), wrong memory layout
    assert view.shape == (48, 64, 3) and not view.flags["C_CONTIGUOUS"]
    with pytest.raises(NonContiguousError, match="not C-contiguous"):
        Preprocessor(rf_manifest.input)(view)


def test_sliced_view_is_refused(rf_manifest):
    big = noise(100, 100)
    with pytest.raises(NonContiguousError):
        Preprocessor(rf_manifest.input)(big[:, ::2])


def test_ascontiguousarray_makes_the_same_view_acceptable(rf_manifest):
    view = np.random.default_rng(1).integers(0, 256, (3, 40, 40), dtype=np.uint8).transpose(1, 2, 0)
    out = Preprocessor(rf_manifest.input)(np.ascontiguousarray(view))
    assert out.shape == (560, 560, 3) and out.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("bad", [np.zeros((10, 10, 3), np.float32), np.zeros((10, 10), np.uint8),
                                 np.zeros((10, 10, 4), np.uint8), [[1, 2, 3]]])
def test_wrong_dtype_or_shape_is_refused(bad):
    with pytest.raises(NonContiguousError):
        require_image(bad, "x")


@pytest.mark.parametrize("h, w", [(416, 416), (1080, 1920), (300, 500)])
def test_rfdetr_resize_is_linear_rgb_and_bit_identical_to_the_reference(rf_manifest, h, w):
    bgr = noise(h, w)
    out = Preprocessor(rf_manifest.input)(bgr)
    assert out.dtype == np.uint8 and out.shape == (560, 560, 3) and out.flags["C_CONTIGUOUS"]
    assert np.array_equal(out, reference(bgr, 560, cv2.INTER_LINEAR))


@pytest.mark.parametrize("h, w, interp", [(416, 416, cv2.INTER_LINEAR), (1080, 1920, cv2.INTER_AREA),
                                          (640, 800, cv2.INTER_LINEAR), (641, 800, cv2.INTER_AREA)])
def test_rtdetrv2_uses_area_only_when_the_frame_is_taller_than_the_input(rt_manifest, h, w, interp):
    bgr = noise(h, w)
    out = Preprocessor(rt_manifest.input)(bgr)
    assert out.shape == (640, 640, 3)
    assert np.array_equal(out, reference(bgr, 640, interp))


def test_input_frame_is_not_modified(rf_manifest):
    bgr = noise(200, 300)
    before = bgr.copy()
    Preprocessor(rf_manifest.input)(bgr)
    assert np.array_equal(bgr, before)
