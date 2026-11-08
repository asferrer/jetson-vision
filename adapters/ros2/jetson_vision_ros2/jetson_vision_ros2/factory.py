"""Builds the core pipeline for the node. The node feeds frames itself and calls ``Pipeline.process_frame``.

No vision logic here: the pipeline owns preprocessing, the backend, the decoder and the tracker.
"""
from __future__ import annotations

import os

from jetson_vision.interfaces import InferenceBackend
from jetson_vision.manifest import Manifest
from jetson_vision.pipeline import Pipeline


def build_processor(manifest_path: str, backend_name: str) -> Pipeline:
    manifest = Manifest.load(manifest_path)
    if backend_name == "fake":
        from jetson_vision.backends.fake import FakeBackend
        backend: InferenceBackend = FakeBackend(manifest)
    elif backend_name == "tensorrt":
        from jetson_vision.backends.tensorrt import TensorRTBackend
        backend = TensorRTBackend(manifest, os.environ.get("JV_MODEL_DIR"))
    else:
        raise ValueError(f"unknown backend {backend_name!r}")
    pipeline = Pipeline(None, backend, manifest, [])
    pipeline.load()  # fail in on_configure, not on the first image
    return pipeline
