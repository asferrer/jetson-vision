"""Model manifest (model.yaml): everything needed to run one engine and to refuse to run the wrong one.

The manifest pins the engine (SHA-256, TensorRT version, GPU), the preprocessing (size, resize, normalisation),
the decoder, the class list and the reference AP. It contains no absolute paths: the engine directory comes
from the caller or from the JV_MODEL_DIR environment variable.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

MANIFEST_SCHEMA = 1
MODEL_DIR_ENV = "JV_MODEL_DIR"
INTERPOLATIONS = ("linear", "area", "area_if_downscale")
DECODERS = ("rfdetr", "rtdetrv2")


class ManifestError(ValueError):
    """The manifest file is malformed."""


class EngineMismatchError(RuntimeError):
    """The engine does not match the manifest, the TensorRT version or the GPU. The loader refuses to start."""


@dataclass(frozen=True)
class RuntimeInfo:
    """What the machine offers: filled by the TensorRT backend from the real libraries."""

    tensorrt: str
    gpu_arch: str  # for example "sm_87"
    gpu_name: str
    sm_count: int | None = None


@dataclass(frozen=True)
class GpuSpec:
    arch: str
    name: str | None = None
    sm_count: int | None = None


@dataclass(frozen=True)
class EngineSpec:
    file: str
    sha256: str
    tensorrt: str
    gpu: GpuSpec


@dataclass(frozen=True)
class NormSpec:
    divide_by: float
    mean: tuple[float, float, float] | None
    std: tuple[float, float, float] | None


@dataclass(frozen=True)
class InputSpec:
    tensor: str
    width: int
    height: int
    channel_order: str
    interpolation: str
    norm: NormSpec
    size_tensor: str | None = None  # engine input that receives (W, H) of the box coordinate space


@dataclass(frozen=True)
class DecoderSpec:
    name: str
    num_select: int
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class ClassSpec:
    names: tuple[str, ...]
    label_offset: int
    ignore_labels: tuple[int, ...]  # raw engine labels that are dropped (RF-DETR spurious 9th output)


@dataclass(frozen=True)
class ReferenceAp:
    value: float
    subset: str
    source: str


@dataclass(frozen=True)
class Manifest:
    model_id: str
    engine: EngineSpec
    input: InputSpec
    decoder: DecoderSpec
    classes: ClassSpec
    score_threshold: float
    reference_ap: ReferenceAp
    base_dir: Path | None = None

    @staticmethod
    def load(path: str | os.PathLike[str]) -> "Manifest":
        p = Path(path)
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ManifestError(f"cannot read manifest {p}: {exc}") from exc
        return _parse(data, p.resolve().parent)

    @staticmethod
    def from_dict(data: Mapping[str, Any], base_dir: Path | None = None) -> "Manifest":
        return _parse(data, base_dir)

    def engine_path(self, model_dir: str | os.PathLike[str] | None = None) -> Path:
        """Engine location: explicit directory, then JV_MODEL_DIR, then the manifest directory."""
        directory = model_dir or os.environ.get(MODEL_DIR_ENV) or self.base_dir
        if directory is None:
            raise ManifestError(f"no engine directory: pass one or set {MODEL_DIR_ENV}")
        return Path(directory).expanduser() / self.engine.file

    def verify_engine_file(self, path: str | os.PathLike[str]) -> None:
        """Refuse an engine whose SHA-256 differs from the manifest."""
        p = Path(path)
        if not p.is_file():
            raise EngineMismatchError(f"engine file not found: {p}")
        actual = sha256_file(p)
        if actual != self.engine.sha256:
            raise EngineMismatchError(
                f"engine {p.name} does not match manifest '{self.model_id}': SHA-256 is {actual}, "
                f"the manifest expects {self.engine.sha256}. Use the engine that the manifest describes."
            )

    def check_runtime(self, info: RuntimeInfo) -> None:
        """Refuse to start if TensorRT or the GPU differ from the ones the engine was built for."""
        want = self.engine
        problems = []
        if info.tensorrt != want.tensorrt:
            problems.append(f"TensorRT {info.tensorrt} (engine built with {want.tensorrt})")
        if info.gpu_arch != want.gpu.arch:
            problems.append(f"GPU architecture {info.gpu_arch} (engine built for {want.gpu.arch})")
        if want.gpu.name and want.gpu.name not in info.gpu_name:
            problems.append(f"GPU '{info.gpu_name}' (engine built for '{want.gpu.name}')")
        if want.gpu.sm_count and info.sm_count and want.gpu.sm_count != info.sm_count:
            problems.append(f"{info.sm_count} SMs (engine built for {want.gpu.sm_count})")
        if problems:
            raise EngineMismatchError(
                f"engine '{self.model_id}' cannot run here: " + "; ".join(problems)
                + ". Engines are not portable and are never rebuilt on the robot."
            )


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _need(data: Mapping[str, Any], key: str, where: str) -> Any:
    if not isinstance(data, Mapping) or key not in data:
        raise ManifestError(f"manifest: missing '{key}' in {where}")
    return data[key]


def _choice(value: Any, allowed: tuple[str, ...], what: str) -> str:
    if value not in allowed:
        raise ManifestError(f"manifest: {what} '{value}' is not one of {allowed}")
    return str(value)


def _triple(value: Any, what: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ManifestError(f"manifest: {what} must be a list of 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _parse_engine(d: Mapping[str, Any]) -> EngineSpec:
    gpu = _need(d, "gpu", "engine")
    sha = str(_need(d, "sha256", "engine")).lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ManifestError("manifest: engine.sha256 must be 64 hexadecimal characters")
    sm = gpu.get("sm_count")
    return EngineSpec(
        file=str(_need(d, "file", "engine")), sha256=sha, tensorrt=str(_need(d, "tensorrt", "engine")),
        gpu=GpuSpec(arch=str(_need(gpu, "arch", "engine.gpu")), name=gpu.get("name"),
                    sm_count=int(sm) if sm is not None else None),
    )


def _parse_input(d: Mapping[str, Any]) -> InputSpec:
    n = _need(d, "normalization", "input")
    norm = NormSpec(divide_by=float(_need(n, "divide_by", "input.normalization")),
                    mean=_triple(n.get("mean"), "mean"), std=_triple(n.get("std"), "std"))
    if (norm.mean is None) != (norm.std is None):
        raise ManifestError("manifest: normalization needs both mean and std, or neither")
    return InputSpec(
        tensor=str(_need(d, "tensor", "input")), width=int(_need(d, "width", "input")),
        height=int(_need(d, "height", "input")),
        channel_order=_choice(_need(d, "channel_order", "input"), ("rgb", "bgr"), "channel_order"),
        interpolation=_choice(_need(d, "interpolation", "input"), INTERPOLATIONS, "interpolation"),
        norm=norm, size_tensor=d.get("size_tensor"),
    )


def _parse(data: Any, base_dir: Path | None) -> Manifest:
    if not isinstance(data, Mapping):
        raise ManifestError("manifest: top level must be a mapping")
    if data.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"manifest: schema must be {MANIFEST_SCHEMA}, got {data.get('schema')!r}")
    dec, cls, ref = _need(data, "decoder", "root"), _need(data, "classes", "root"), _need(data, "reference_ap", "root")
    names = tuple(str(n) for n in _need(cls, "names", "classes"))
    threshold = float(_need(data, "score_threshold", "root"))
    if not 0.0 <= threshold <= 1.0:
        raise ManifestError("manifest: score_threshold must be in [0, 1]")
    return Manifest(
        model_id=str(_need(data, "model_id", "root")),
        engine=_parse_engine(_need(data, "engine", "root")),
        input=_parse_input(_need(data, "input", "root")),
        decoder=DecoderSpec(name=_choice(_need(dec, "name", "decoder"), DECODERS, "decoder"),
                            num_select=int(_need(dec, "num_select", "decoder")),
                            outputs=tuple(str(o) for o in _need(dec, "outputs", "decoder"))),
        classes=ClassSpec(names=names, label_offset=int(_need(cls, "label_offset", "classes")),
                          ignore_labels=tuple(int(i) for i in cls.get("ignore_labels", ()))),
        score_threshold=threshold,
        reference_ap=ReferenceAp(value=float(_need(ref, "value", "reference_ap")),
                                 subset=str(_need(ref, "subset", "reference_ap")),
                                 source=str(_need(ref, "source", "reference_ap"))),
        base_dir=base_dir,
    )
