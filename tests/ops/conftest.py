"""Shared builders and test doubles for the operations tests. No GPU, no ROS, no external service."""
from __future__ import annotations

import numpy as np
import pytest

from jetson_vision.types import (
    DegradationLevel,
    Detection,
    Frame,
    FrameRef,
    FrameResult,
)


def make_ref(seq: int = 0, t_ns: int | None = None, source_id: str = "cam0") -> FrameRef:
    t = seq * 33_000_000 if t_ns is None else t_ns
    return FrameRef(source_id, "sess", seq, t, 1_700_000_000_000_000_000 + t, None)


def make_det(class_id: int = 1, score: float = 0.9,
             box: tuple[float, float, float, float] = (10.0, 20.0, 110.0, 120.0),
             track_id: int | None = None) -> Detection:
    return Detection(class_id, f"c{class_id}", score, box, track_id)


def make_result(seq: int = 0, dets: list[Detection] | None = None, t_ns: int | None = None,
                level: DegradationLevel = DegradationLevel.NORMAL) -> FrameResult:
    return FrameResult(make_ref(seq, t_ns), dets if dets is not None else [make_det()],
                       "fake-model", 0.5, level)


def make_frame(seq: int = 0, h: int = 48, w: int = 64) -> Frame:
    return Frame(make_ref(seq), np.zeros((h, w, 3), dtype=np.uint8))


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
