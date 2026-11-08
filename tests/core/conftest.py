"""Fixtures for the core tests: manifests, a scripted source and a collecting sink. No GPU, no ROS."""
from __future__ import annotations

import dataclasses
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jetson_vision.interfaces import SourceError  # noqa: E402
from jetson_vision.manifest import Manifest  # noqa: E402
from jetson_vision.types import Frame, FrameRef, FrameResult, HealthState  # noqa: E402

DATA = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def rf_manifest() -> Manifest:
    return Manifest.load(ROOT / "models" / "rfdetr_int8g.yaml")


@pytest.fixture(scope="session")
def rt_manifest() -> Manifest:
    return Manifest.load(ROOT / "models" / "rtdetrv2_int8mix.yaml")


def make_frame(seq: int, w: int = 64, h: int = 48, session: str = "s1", fill: int | None = None) -> Frame:
    px = np.full((h, w, 3), seq % 255 if fill is None else fill, np.uint8)
    return Frame(FrameRef("test", session, seq, 1_000 + seq, 2_000 + seq, None), px)


class ScriptedSource:
    """Plays a script: a Frame, None (end), or an Exception to raise from read(). open() may fail a set number of times."""

    def __init__(self, script, live: bool = False, open_failures: int = 0, delay: float = 0.0) -> None:
        self.source_id, self.live = "scripted", live
        self.script, self.delay = list(script), delay
        self.open_failures = open_failures
        self.opens = self.closes = 0
        self.open_times: list[float] = []

    def open(self) -> None:
        self.opens += 1
        self.open_times.append(time.monotonic())
        if self.open_failures > 0:
            self.open_failures -= 1
            raise SourceError("open failed")

    def read(self):
        if self.delay:
            time.sleep(self.delay)
        if not self.script:
            return None
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        item.ref = dataclasses.replace(item.ref, t_capture_mono_ns=time.monotonic_ns())  # real capture time
        return item

    def close(self) -> None:
        self.closes += 1


class CollectSink:
    def __init__(self, delay: float = 0.0) -> None:
        self.results: list[FrameResult] = []
        self.health: list[HealthState] = []
        self.closed = False
        self.delay = delay
        self.lock = threading.Lock()

    def write(self, result: FrameResult) -> None:
        if self.delay:
                time.sleep(self.delay)
        with self.lock:
            self.results.append(result)

    def write_health(self, state: HealthState) -> None:
        self.health.append(state)

    def close(self) -> None:
        self.closed = True
