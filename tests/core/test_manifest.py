"""Manifest parsing and the loader's refusal to start with the wrong engine, TensorRT or GPU."""
from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
import yaml

from jetson_vision.backends.tensorrt import TensorRTBackend
from jetson_vision.manifest import (EngineMismatchError, Manifest, ManifestError, RuntimeInfo, sha256_file)

ROOT = Path(__file__).resolve().parents[2]
GOOD = RuntimeInfo(tensorrt="10.3.0", gpu_arch="sm_87", gpu_name="Orin", sm_count=8)


def raw(name: str) -> dict:
    return yaml.safe_load((ROOT / "models" / name).read_text(encoding="utf-8"))


def test_both_manifests_load(rf_manifest, rt_manifest):
    assert rf_manifest.decoder.name == "rfdetr" and rf_manifest.input.width == 560
    assert rt_manifest.decoder.name == "rtdetrv2" and rt_manifest.input.width == 640
    assert rf_manifest.classes.ignore_labels == (8,) and len(rf_manifest.classes.names) == 8
    assert rt_manifest.input.size_tensor == "orig_target_sizes"
    for m in (rf_manifest, rt_manifest):
        assert m.engine.tensorrt == "10.3.0" and m.engine.gpu.arch == "sm_87" and len(m.engine.sha256) == 64


def test_manifests_hold_no_absolute_paths():
    for name in ("rfdetr_int8g.yaml", "rtdetrv2_int8mix.yaml"):
        text = (ROOT / "models" / name).read_text(encoding="utf-8")
        assert not re.search(r"(/home/|/Users/|[A-Za-z]:[\\/])", text), name


def test_engine_path_uses_directory_then_environment(rf_manifest, tmp_path, monkeypatch):
    monkeypatch.setenv("JV_MODEL_DIR", str(tmp_path / "env"))
    assert rf_manifest.engine_path(tmp_path / "arg").parent == tmp_path / "arg"
    assert rf_manifest.engine_path().parent == tmp_path / "env"


def test_wrong_engine_file_is_refused_before_touching_tensorrt(rf_manifest, tmp_path):
    (tmp_path / rf_manifest.engine.file).write_bytes(b"this is not the engine")
    backend = TensorRTBackend(rf_manifest, model_dir=tmp_path, probe=lambda: GOOD)
    with pytest.raises(EngineMismatchError, match="SHA-256"):
        backend.load()


def test_missing_engine_file_is_refused(rf_manifest, tmp_path):
    with pytest.raises(EngineMismatchError, match="not found"):
        TensorRTBackend(rf_manifest, model_dir=tmp_path, probe=lambda: GOOD).load()


def _manifest_for(tmp_path: Path, blob: bytes) -> tuple[Manifest, Path]:
    data = copy.deepcopy(raw("rfdetr_int8g.yaml"))
    engine = tmp_path / "e.engine"
    engine.write_bytes(blob)
    data["engine"]["file"] = "e.engine"
    data["engine"]["sha256"] = sha256_file(engine)
    return Manifest.from_dict(data, tmp_path), tmp_path


@pytest.mark.parametrize("info, expect", [
    (RuntimeInfo("10.4.0", "sm_87", "Orin", 8), "TensorRT 10.4.0"),
    (RuntimeInfo("10.3.0", "sm_86", "NVIDIA RTX", 8), "GPU architecture sm_86"),
    (RuntimeInfo("10.3.0", "sm_87", "Orin", 16), "16 SMs"),
])
def test_matching_file_but_wrong_tensorrt_or_gpu_is_refused(tmp_path, info, expect):
    man, directory = _manifest_for(tmp_path, b"engine bytes")
    backend = TensorRTBackend(man, model_dir=directory, probe=lambda: info)
    with pytest.raises(EngineMismatchError, match=expect) as err:
        backend.load()
    assert "never rebuilt" in str(err.value)


def test_runtime_check_accepts_the_right_machine(rf_manifest):
    rf_manifest.check_runtime(GOOD)


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(schema=2), "schema"),
    (lambda d: d["engine"].pop("sha256"), "sha256"),
    (lambda d: d["engine"].update(sha256="zz"), "64 hexadecimal"),
    (lambda d: d["input"].update(interpolation="bicubic"), "interpolation"),
    (lambda d: d["decoder"].update(name="yolo"), "decoder"),
    (lambda d: d.update(score_threshold=1.5), "score_threshold"),
    (lambda d: d["input"]["normalization"].pop("std"), "mean and std"),
])
def test_malformed_manifest_is_rejected_with_a_clear_message(mutate, message):
    data = copy.deepcopy(raw("rfdetr_int8g.yaml"))
    mutate(data)
    with pytest.raises(ManifestError, match=message):
        Manifest.from_dict(data)


def test_unreadable_manifest(tmp_path):
    with pytest.raises(ManifestError, match="cannot read"):
        Manifest.load(tmp_path / "missing.yaml")
