"""Pure mapping functions of the ROS 2 adapter, tested without ROS. Also checks the core stays ROS-free."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "adapters" / "ros2" / "jetson_vision_ros2"))

from jetson_vision_ros2 import mapping as m  # noqa: E402

from jetson_vision.types import DegradationLevel, FrameRef, FrameResult, HealthState  # noqa: E402

from .conftest import make_det, make_result  # noqa: E402


def test_core_never_imports_rclpy_or_ros_messages() -> None:
    forbidden = {"rclpy", "sensor_msgs", "vision_msgs", "diagnostic_msgs", "jetson_vision_ros2"}
    for path in (ROOT / "jetson_vision").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not {n.split(".")[0] for n in names} & forbidden, path


def test_mapping_module_has_no_rclpy() -> None:
    tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    roots = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)
             for a in n.names}
    roots |= {n.module.split(".")[0] for n in ast.walk(tree)
              if isinstance(n, ast.ImportFrom) and n.module}
    assert not roots & {"rclpy", "sensor_msgs", "vision_msgs", "diagnostic_msgs"}


def test_stamp_roundtrip() -> None:
    assert m.stamp_to_ns(12, 345) == 12_000_000_345
    assert m.ns_to_stamp(12_000_000_345) == (12, 345)


def rgb_payload(h: int = 2, w: int = 3, pad: int = 0) -> tuple[np.ndarray, bytes, int]:
    img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    rows = [bytes(img[r].tobytes()) + b"\xAA" * pad for r in range(h)]
    return img, b"".join(rows), w * 3 + pad


@pytest.mark.parametrize("pad", [0, 5])
def test_rgb8_to_bgr_with_row_padding(pad: int) -> None:
    img, data, step = rgb_payload(pad=pad)
    out = m.image_to_bgr("rgb8", 2, 3, step, data)
    assert out.flags["C_CONTIGUOUS"] and out.shape == (2, 3, 3) and out.dtype == np.uint8
    assert np.array_equal(out, img[:, :, ::-1])


def test_other_encodings() -> None:
    img, data, step = rgb_payload()
    assert np.array_equal(m.image_to_bgr("bgr8", 2, 3, step, data), img)
    rgba = np.concatenate([img, np.full((2, 3, 1), 255, np.uint8)], axis=2)
    assert np.array_equal(m.image_to_bgr("rgba8", 2, 3, 12, rgba.tobytes()), img[:, :, ::-1])
    assert np.array_equal(m.image_to_bgr("bgra8", 2, 3, 12, rgba.tobytes()), img)
    mono = np.arange(6, dtype=np.uint8).reshape(2, 3)
    out = m.image_to_bgr("mono8", 2, 3, 3, mono.tobytes())
    assert out.shape == (2, 3, 3) and np.array_equal(out[:, :, 0], mono)


def test_image_errors() -> None:
    with pytest.raises(m.UnsupportedEncodingError):
        m.image_to_bgr("16UC1", 2, 3, 6, bytes(12))
    with pytest.raises(ValueError):
        m.image_to_bgr("rgb8", 2, 3, 9, bytes(5))  # short buffer
    with pytest.raises(ValueError):
        m.image_to_bgr("rgb8", 2, 3, 4, bytes(100))  # step smaller than a row


def test_image_to_frame_keeps_stamp_and_frame_id() -> None:
    _, data, step = rgb_payload()
    frame = m.image_to_frame(encoding="rgb8", height=2, width=3, step=step, data=data,
                             stamp_sec=7, stamp_nanosec=9, frame_id="camera_link", frame_seq=4,
                             session_id="s", mono_ns=10, wall_ns=20)
    assert frame.ref == FrameRef("camera_link", "s", 4, 10, 20, 7_000_000_009)
    assert (frame.width, frame.height) == (3, 2)


def test_header_is_copied_from_input_not_publication_time() -> None:
    ref = FrameRef("camera_link", "s", 1, 5, 6, 7_000_000_009)
    result = FrameResult(ref, [make_det()], "m", 0.5)
    data = m.result_to_detections(result)
    assert data.header == m.HeaderData(7, 9, "camera_link")


def test_detection_box_conversion_and_ids() -> None:
    det = make_det(3, 0.8, (10.0, 20.0, 30.0, 60.0), track_id=12)
    plain = make_det(2, 0.5, (0.0, 0.0, 4.0, 2.0))
    out = m.result_to_detections(make_result(0, [det, plain])).detections
    assert out[0] == m.DetectionData("3", 0.8, 20.0, 40.0, 20.0, 40.0, "12")
    assert out[1].track_id == "" and out[1].size_x == 4.0 and out[1].center_y == 1.0


def test_empty_result_gives_empty_array_with_header() -> None:
    data = m.result_to_detections(make_result(0, []))
    assert data.detections == [] and data.header.frame_id == "cam0"


def test_meta_fields() -> None:
    r = make_result(9, level=DegradationLevel.REDUCED_FPS)
    r.dropped_before, r.timings.inference_ns = 2, 123
    meta = m.result_to_meta(r)
    assert meta["schema_version"] == 2 and meta["degradation_level"] == 1
    assert meta["frame_seq"] == 9 and meta["dropped_before"] == 2 and meta["inference_ns"] == 123
    assert meta["model_id"] == "fake-model" and meta["score_threshold"] == 0.5


@pytest.mark.parametrize("level,diag", [(0, 0), (1, 1), (2, 2), (3, 2)])
def test_diagnostics_levels(level: int, diag: int) -> None:
    state = HealthState(DegradationLevel(level), "why", 55.0, None, 14.0)
    out = m.health_to_diagnostics(state, "mid", "hw")
    assert out["level"] == diag and out["hardware_id"] == "hw"
    assert out["values"]["temperature_c"] == "55.0" and out["values"]["latency_p95_ms"] == "unknown"
    assert out["values"]["degradation_level"] == str(level)
