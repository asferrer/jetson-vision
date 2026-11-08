"""Video file (OpenCV) and V4L2 camera sources."""
from __future__ import annotations

from pathlib import Path

import cv2

from jetson_vision.interfaces import SourceError
from jetson_vision.sources.base import Pacer, SourceBase, check_mode, stamp_ns
from jetson_vision.types import Frame


class VideoSource(SourceBase):
    """Video file decoded by OpenCV. Finite and non-live: 'max' for deterministic runs, 'paced' for load tests."""

    live = False

    def __init__(self, path: str | Path, *, mode: str = "max", fps: float | None = None,
                 source_id: str | None = None) -> None:
        p = Path(path)
        super().__init__(source_id or f"video:{p.name}")
        self.path, self.mode, self._fps_override = p, check_mode(mode), fps
        self._cap: cv2.VideoCapture | None = None
        self._index = 0
        self.fps = fps or 30.0
        self._pacer = Pacer(None)

    def open(self) -> None:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise SourceError(f"video source: cannot open '{self.path}'")
        declared = cap.get(cv2.CAP_PROP_FPS)
        self.fps = self._fps_override or (declared if declared and declared > 0 else 30.0)
        if self._index:  # reconnection: continue where the failed read stopped
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._index)
        self._pacer = Pacer(self.fps if self.mode == "paced" else None)
        self._cap = cap

    def read(self) -> Frame | None:
        if self._cap is None:
            raise SourceError("video source: read() before open()")
        ok, img = self._cap.read()
        if not ok:
            return None  # end of file
        index, self._index = self._index, self._index + 1
        self._pacer.wait(index)
        return self.make_frame(img, stamp_ns(index, self.fps))

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class V4L2Source(SourceBase):
    """USB or MIPI camera through V4L2. Live: the pipeline keeps only the newest frame."""

    live = True

    def __init__(self, device: int | str = 0, *, width: int | None = None, height: int | None = None,
                 fps: float | None = None, fourcc: str | None = "MJPG", source_id: str | None = None) -> None:
        super().__init__(source_id or f"v4l2:{device}")
        self.device, self.width, self.height, self.fps, self.fourcc = device, width, height, fps, fourcc
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise SourceError(f"v4l2 source: cannot open '{self.device}'")
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, self.width), (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
                            (cv2.CAP_PROP_FPS, self.fps)):
            if value:
                cap.set(prop, value)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low: the driver queue is not a frame buffer
        self._cap = cap

    def read(self) -> Frame | None:
        if self._cap is None:
            raise SourceError("v4l2 source: read() before open()")
        ok, img = self._cap.read()
        if not ok:
            raise SourceError(f"v4l2 source: '{self.device}' stopped delivering frames")
        return self.make_frame(img)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
