"""Shared pieces of the frame sources: identity, clocks, contiguity and pacing."""
from __future__ import annotations

import time
import uuid

import numpy as np

from jetson_vision.preprocess import require_image
from jetson_vision.types import Frame, FrameRef

MODES = ("max", "paced")


class SourceBase:
    """Gives every frame a (source_id, session_id, frame_seq) and both clocks.

    session_id is a UUID created when the source is constructed, so a restart never repeats a sequence.
    frame_seq counts delivered frames and keeps growing across reconnections of the same object.
    """

    live = False

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self.session_id = str(uuid.uuid4())
        self._seq = 0

    def make_frame(self, pixels: np.ndarray, source_stamp: int | None = None, *, t_mono_ns: int | None = None) -> Frame:
        """Stamp a captured image. Call it right after the capture returns: both clocks are read here.

        t_mono_ns: the true capture instant on the monotonic clock when the device reports it (a live camera); the wall
        clock is shifted by the same amount, so both stamps name the capture and not the moment of the read.
        """
        mono, wall = time.monotonic_ns(), time.time_ns()
        if t_mono_ns is not None:
            mono, wall = t_mono_ns, wall - (mono - t_mono_ns)
        pixels = require_image(np.ascontiguousarray(pixels), f"source '{self.source_id}'")
        ref = FrameRef(self.source_id, self.session_id, self._seq, mono, wall, source_stamp)
        self._seq += 1
        return Frame(ref, pixels)


class Pacer:
    """Releases frame i at start + i / fps (paced mode). In max mode (fps None) it never waits."""

    def __init__(self, fps: float | None) -> None:
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive")
        self.period = None if fps is None else 1.0 / fps
        self._t0: float | None = None

    def wait(self, index: int) -> None:
        if self.period is None:
            return
        if self._t0 is None:
            self._t0 = time.monotonic()
        delay = self._t0 + index * self.period - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def stamp_ns(index: int, fps: float) -> int:
    """Source stamp of a file frame: derived from its index and the nominal fps, never from the machine clock."""
    return int(round(index * 1e9 / fps))


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
    return mode
