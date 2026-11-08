"""TensorRT backend. tensorrt and torch are imported inside load(), so the core runs without them.

Resize happens on the CPU (uint8). The uint8 image then goes to the GPU with a plain (pageable) copy and is
normalised and transposed there. Pinned memory and zero-copy inputs were measured and are slower on the Orin Nano.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from jetson_vision.manifest import EngineMismatchError, Manifest, RuntimeInfo
from jetson_vision.preprocess import require_image


log = logging.getLogger(__name__)


class BackendUnavailableError(RuntimeError):
    """tensorrt or torch cannot be imported on this machine."""


def _import_runtime() -> tuple[Any, Any]:
    try:
        import tensorrt as trt
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(f"TensorRT backend needs tensorrt and torch: {exc}") from exc
    return trt, torch


def probe_runtime() -> RuntimeInfo:
    """Read the TensorRT version and the GPU from the real libraries."""
    trt, torch = _import_runtime()
    if not torch.cuda.is_available():
        raise BackendUnavailableError("CUDA is not available to torch")
    props = torch.cuda.get_device_properties(0)
    return RuntimeInfo(tensorrt=str(trt.__version__), gpu_arch=f"sm_{props.major}{props.minor}",
                       gpu_name=str(props.name), sm_count=int(props.multi_processor_count))


class TensorRTBackend:
    def __init__(self, manifest: Manifest, model_dir: str | Path | None = None,
                 probe: Callable[[], RuntimeInfo] = probe_runtime) -> None:
        self.manifest = manifest
        self.model_id = manifest.model_id
        self._model_dir = model_dir
        self._probe = probe
        self._engine: Any = None

    def load(self) -> None:
        """Refuses (EngineMismatchError) if the engine file, the TensorRT version or the GPU do not match."""
        path = self.manifest.engine_path(self._model_dir)
        self.manifest.verify_engine_file(path)
        self.manifest.check_runtime(self._probe())
        trt, torch = _import_runtime()
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        engine = trt.Runtime(logger).deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise EngineMismatchError(f"TensorRT could not deserialize {path.name}")
        self._trt, self._torch, self._engine = trt, torch, engine
        self._ctx = engine.create_execution_context()
        self._stream = torch.cuda.Stream()
        self._alloc_buffers()
        self._warmup()
        log.info("engine %s loaded: TensorRT device memory %.1f MB, torch reserved %.1f MB", path.name,
                 self._device_memory_mb(), torch.cuda.memory_reserved() / 2**20)

    def _device_memory_mb(self) -> float:
        size = getattr(self._engine, "device_memory_size_v2", None) or self._engine.device_memory_size
        return float(size) / 2**20

    def _alloc_buffers(self) -> None:
        trt, torch, eng, spec = self._trt, self._torch, self._engine, self.manifest.input
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
                  trt.int64: torch.int64, trt.bool: torch.bool}
        self._buf = {}
        for i in range(eng.num_io_tensors):
            name = eng.get_tensor_name(i)
            self._buf[name] = torch.empty(tuple(eng.get_tensor_shape(name)),
                                          dtype=dtypes[eng.get_tensor_dtype(name)], device="cuda")
            self._ctx.set_tensor_address(name, self._buf[name].data_ptr())
        self._check_io()
        if spec.size_tensor:  # box coordinate space: the network input; the decoder rescales to the frame
            size = torch.tensor([[spec.width, spec.height]], dtype=self._buf[spec.size_tensor].dtype)
            self._buf[spec.size_tensor].copy_(size)
        self._u8 = torch.empty((spec.height, spec.width, 3), dtype=torch.uint8, device="cuda")
        norm = spec.norm
        self._mean = None if norm.mean is None else torch.tensor(norm.mean, device="cuda").view(1, 3, 1, 1)
        self._std = None if norm.std is None else torch.tensor(norm.std, device="cuda").view(1, 3, 1, 1)

    def _check_io(self) -> None:
        spec = self.manifest.input
        wanted = [spec.tensor, *self.manifest.decoder.outputs] + ([spec.size_tensor] if spec.size_tensor else [])
        missing = [n for n in wanted if n not in self._buf]
        expected = (1, 3, spec.height, spec.width)
        shape = tuple(self._buf[spec.tensor].shape) if spec.tensor in self._buf else None
        if missing or shape != expected:
            raise EngineMismatchError(f"engine I/O does not match manifest '{self.model_id}': missing tensors "
                                      f"{missing}, input shape {shape} (expected {expected})")

    def _warmup(self, runs: int = 3) -> None:
        zeros = np.zeros((self.manifest.input.height, self.manifest.input.width, 3), np.uint8)
        for _ in range(runs):
            self.infer(zeros)

    def infer(self, input_u8_chw_or_hwc: np.ndarray) -> Mapping[str, np.ndarray]:
        """Takes the contiguous HxWx3 uint8 image from Preprocessor; CHW input is not supported."""
        if self._engine is None:
            raise RuntimeError("TensorRTBackend.infer called before load()")
        spec, torch = self.manifest.input, self._torch
        x = require_image(input_u8_chw_or_hwc, "backend input")
        if x.shape[:2] != (spec.height, spec.width):
            raise ValueError(f"backend input is {x.shape[1]}x{x.shape[0]}, the manifest expects {spec.width}x{spec.height}")
        with torch.cuda.stream(self._stream):
            self._u8.copy_(torch.from_numpy(x))
            t = self._u8.permute(2, 0, 1)[None].float() * (1.0 / spec.norm.divide_by)
            if self._mean is not None:
                t = (t - self._mean) / self._std
            self._buf[spec.tensor].copy_(t)
            self._ctx.execute_async_v3(self._stream.cuda_stream)
            self._stream.synchronize()
            return {n: self._to_cpu(self._buf[n]) for n in self.manifest.decoder.outputs}

    def _to_cpu(self, t: Any) -> np.ndarray:
        t = t.float() if t.dtype == self._torch.float16 else t
        return t.cpu().numpy()

    def close(self) -> None:
        self._engine = None
        self._ctx = None
        self._buf = {}
