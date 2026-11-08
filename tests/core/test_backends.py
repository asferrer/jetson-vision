"""Fake backend determinism and the lazy imports of the TensorRT backend."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from jetson_vision.backends import tensorrt as trt_backend
from jetson_vision.backends.fake import FakeBackend
from jetson_vision.interfaces import InferenceBackend
from jetson_vision.postprocess import build_decoder

ROOT = Path(__file__).resolve().parents[2]


def image(seed: int, size: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_fake_is_deterministic_and_input_dependent(rf_manifest):
    b = FakeBackend(rf_manifest)
    b.load()
    a1, a2, other = b.infer(image(1, 560)), b.infer(image(1, 560)), b.infer(image(2, 560))
    assert all(np.array_equal(a1[k], a2[k]) for k in a1)
    assert not np.array_equal(a1["labels"], other["labels"])
    assert b.calls == 3 and isinstance(b, InferenceBackend)


@pytest.mark.parametrize("fixture, size", [("rf_manifest", 560), ("rt_manifest", 640)])
def test_fake_output_goes_through_the_real_decoder(request, fixture, size):
    man = request.getfixturevalue(fixture)
    b = FakeBackend(man)
    b.load()
    dets = build_decoder(man)(b.infer(image(3, size)), 1920, 1080, 0.5)
    assert 1 <= len(dets) <= 5
    assert all(0 <= d.class_id < 8 and 0 <= d.box_xyxy[0] <= 1920 for d in dets)


def test_fake_requires_load_and_runs_the_hook(rf_manifest):
    calls = []
    b = FakeBackend(rf_manifest, hook=calls.append)
    with pytest.raises(RuntimeError, match="before load"):
        b.infer(image(0, 560))
    b.load()
    b.infer(image(0, 560))
    assert calls == [0]


def test_importing_the_tensorrt_backend_imports_neither_tensorrt_nor_torch():
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r});"
            "import jetson_vision.backends.tensorrt, jetson_vision.pipeline, jetson_vision.preprocess, "
            "jetson_vision.postprocess, jetson_vision.manifest;"
            "bad = [m for m in ('tensorrt', 'torch') if m in sys.modules]; assert not bad, bad")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_missing_tensorrt_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", None)  # makes "import tensorrt" raise ImportError
    with pytest.raises(trt_backend.BackendUnavailableError, match="tensorrt and torch"):
        trt_backend.probe_runtime()


def test_tensorrt_infer_before_load_is_an_error(rf_manifest):
    with pytest.raises(RuntimeError, match="before load"):
        trt_backend.TensorRTBackend(rf_manifest).infer(image(0, 560))
