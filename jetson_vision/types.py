"""Data contract of the inference core. Changing a field means bumping SCHEMA_VERSION."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

SCHEMA_VERSION = 2  # v2: StageTimings.queue_ns (time spent waiting in queues, split out of total_ns)


class DegradationLevel(IntEnum):
    NORMAL = 0
    REDUCED_FPS = 1
    ALERTS_ONLY = 2
    STOPPED = 3


@dataclass(frozen=True, slots=True)
class FrameRef:
    """Identifies a frame. session_id changes on every start, so a restart never repeats a sequence."""

    source_id: str
    session_id: str
    frame_seq: int
    t_capture_mono_ns: int
    t_capture_wall_ns: int
    source_stamp: int | None = None


@dataclass(slots=True)
class Frame:
    """A captured frame. pixels is HxWx3, uint8, BGR, and must be contiguous."""

    ref: FrameRef
    pixels: np.ndarray

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])


@dataclass(frozen=True, slots=True)
class Detection:
    """box_xyxy is in pixels of the original frame."""

    class_id: int
    class_name: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    track_id: int | None = None


@dataclass(slots=True)
class StageTimings:
    capture_ns: int = 0
    preprocess_ns: int = 0
    inference_ns: int = 0
    postprocess_ns: int = 0
    tracking_ns: int = 0
    total_ns: int = 0  # capture stamp -> publication; includes queue_ns
    queue_ns: int = 0  # waiting (input channel + publication queue), not work

    @property
    def service_ns(self) -> int:
        """total_ns without queue waits: the latency that does not depend on how fast the source pushes."""
        return max(0, self.total_ns - self.queue_ns)


@dataclass(slots=True)
class FrameResult:
    frame_ref: FrameRef
    detections: list[Detection]
    model_id: str
    score_threshold: float
    degradation_level: DegradationLevel = DegradationLevel.NORMAL
    timings: StageTimings = field(default_factory=StageTimings)
    dropped_before: int = 0
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["degradation_level"] = int(self.degradation_level)
        return data


@dataclass(frozen=True, slots=True)
class HealthState:
    level: DegradationLevel
    reason: str
    temperature_c: float | None
    latency_p95_ms: float | None
    fps: float | None
