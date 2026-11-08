"""Pure functions between ROS 2 message fields and core types. No rclpy import, no vision logic.

Everything here takes and returns plain Python values so it can be tested without ROS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jetson_vision.types import DegradationLevel, Frame, FrameRef, FrameResult, HealthState

# encoding -> (channels, conversion to BGR)
_ENCODINGS = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}

DIAG_OK, DIAG_WARN, DIAG_ERROR = 0, 1, 2  # diagnostic_msgs/DiagnosticStatus levels


class UnsupportedEncodingError(ValueError):
    pass


def stamp_to_ns(sec: int, nanosec: int) -> int:
    return int(sec) * 1_000_000_000 + int(nanosec)


def ns_to_stamp(ns: int) -> tuple[int, int]:
    return divmod(int(ns), 1_000_000_000)


def image_to_bgr(encoding: str, height: int, width: int, step: int, data: bytes) -> np.ndarray:
    """Decode a sensor_msgs/Image payload to a contiguous HxWx3 uint8 BGR array (a copy)."""
    channels = _ENCODINGS.get(encoding)
    if channels is None:
        raise UnsupportedEncodingError(f"unsupported image encoding {encoding!r}")
    if height <= 0 or width <= 0 or step < width * channels or len(data) < step * height:
        raise ValueError("inconsistent image geometry")
    rows = np.frombuffer(data, dtype=np.uint8, count=step * height).reshape(height, step)
    img = rows[:, : width * channels].reshape(height, width, channels)
    if encoding == "bgr8":
        out = img
    elif encoding == "rgb8":
        out = img[:, :, ::-1]
    elif encoding == "bgra8":
        out = img[:, :, :3]
    elif encoding == "rgba8":
        out = img[:, :, 2::-1]
    else:  # mono8
        out = np.repeat(img, 3, axis=2)
    return np.ascontiguousarray(out)


def image_to_frame(
    *,
    encoding: str,
    height: int,
    width: int,
    step: int,
    data: bytes,
    stamp_sec: int,
    stamp_nanosec: int,
    frame_id: str,
    frame_seq: int,
    session_id: str,
    mono_ns: int,
    wall_ns: int,
) -> Frame:
    """The input stamp is kept in ``source_stamp``; ``source_id`` carries the image frame_id."""
    pixels = image_to_bgr(encoding, height, width, step, data)
    ref = FrameRef(frame_id, session_id, frame_seq, mono_ns, wall_ns,
                   stamp_to_ns(stamp_sec, stamp_nanosec))
    return Frame(ref, pixels)


@dataclass(frozen=True)
class HeaderData:
    sec: int
    nanosec: int
    frame_id: str


@dataclass(frozen=True)
class DetectionData:
    class_id: str
    score: float
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    track_id: str  # empty without tracking


@dataclass(frozen=True)
class DetectionArrayData:
    header: HeaderData
    detections: list[DetectionData] = field(default_factory=list)


def header_from_ref(ref: FrameRef) -> HeaderData:
    """Header copied from the input image, never the publication time."""
    stamp = ref.source_stamp if ref.source_stamp is not None else 0
    sec, nanosec = ns_to_stamp(stamp)
    return HeaderData(sec, nanosec, ref.source_id)


def result_to_detections(result: FrameResult) -> DetectionArrayData:
    dets = []
    for d in result.detections:
        x1, y1, x2, y2 = d.box_xyxy
        dets.append(DetectionData(
            class_id=str(d.class_id), score=float(d.score),
            center_x=(x1 + x2) / 2.0, center_y=(y1 + y2) / 2.0,
            size_x=x2 - x1, size_y=y2 - y1,
            track_id="" if d.track_id is None else str(d.track_id),
        ))
    return DetectionArrayData(header_from_ref(result.frame_ref), dets)


def result_to_meta(result: FrameResult) -> dict[str, Any]:
    """Field values of jetson_vision_msgs/FrameMeta (header excluded: use header_from_ref)."""
    t = result.timings
    return {
        "schema_version": result.schema_version,
        "model_id": result.model_id,
        "session_id": result.frame_ref.session_id,
        "frame_seq": result.frame_ref.frame_seq,
        "score_threshold": float(result.score_threshold),
        "degradation_level": int(result.degradation_level),
        "dropped_before": result.dropped_before,
        "capture_ns": t.capture_ns,
        "preprocess_ns": t.preprocess_ns,
        "inference_ns": t.inference_ns,
        "postprocess_ns": t.postprocess_ns,
        "tracking_ns": t.tracking_ns,
        "total_ns": t.total_ns,
        "queue_ns": t.queue_ns,
    }


def health_to_diagnostics(state: HealthState, model_id: str, hardware_id: str
                          ) -> dict[str, Any]:
    """One DiagnosticStatus as plain data: level, name, message, hardware_id, key/values."""
    level = {DegradationLevel.NORMAL: DIAG_OK, DegradationLevel.REDUCED_FPS: DIAG_WARN}.get(
        state.level, DIAG_ERROR)

    def fmt(v: float | None) -> str:
        return "unknown" if v is None else f"{v:.1f}"

    return {
        "level": level,
        "name": "jetson_vision",
        "message": state.reason,
        "hardware_id": hardware_id,
        "values": {
            "degradation_level": str(int(state.level)),
            "model_id": model_id,
            "temperature_c": fmt(state.temperature_c),
            "latency_p95_ms": fmt(state.latency_p95_ms),
            "fps": fmt(state.fps),
        },
    }
