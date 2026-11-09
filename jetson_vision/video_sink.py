"""Annotated video sink: ``video:<path.mp4>[?size=WxH&fps=N&queue=N&encoder=auto|gst|mp4v]``.

Boxes, class, confidence and the processing rate are drawn on the frames and encoded in software (the Orin Nano has no
NVENC). The sink runs in its own thread behind a bounded queue: when encoding is slower than the pipeline it drops frames
of the *output video* (counted in ``dropped``), never frames of the inference.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

import cv2
import numpy as np

from jetson_vision.types import Detection, FrameResult, HealthState

log = logging.getLogger(__name__)

# Class colours of the Android app (hex RGB).
CLASS_COLORS_HEX = {
    "Bottle": "E69F00",
    "Can": "F0E442",
    "Fishing_Net": "D55E00",
    "Glove": "CC79A7",
    "Mask": "F2F2F2",
    "Metal_Debris": "9467BD",
    "Plastic_Debris": "FF8A80",
    "Tire": "A1887F",
}
_FALLBACK_HEX = "00B4D8"
ENCODERS = ("auto", "gst", "mp4v")
DEFAULT_SIZE = (1280, 720)
_POLL_S = 0.2
_STOP = object()


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


_BGR = {name: hex_to_bgr(h) for name, h in CLASS_COLORS_HEX.items()}


def class_color(name: str) -> tuple[int, int, int]:
    return _BGR.get(name) or hex_to_bgr(_FALLBACK_HEX)


def parse_video_spec(arg: str) -> tuple[str, dict[str, Any]]:
    """``path[?size=WxH&fps=N&queue=N&encoder=...]`` -> (path, options). Raises ValueError."""
    path, _, query = arg.partition("?")
    if not path:
        raise ValueError("video sink needs a path: video:<path.mp4>")
    opts: dict[str, Any] = {}
    for key, value in parse_qsl(query, keep_blank_values=True):
        if key == "size":
            w, sep, h = value.lower().partition("x")
            if not (sep and w.isdigit() and h.isdigit() and int(w) > 0 and int(h) > 0):
                raise ValueError(f"video size must be WxH, got '{value}'")
            opts["size"] = (int(w), int(h))
        elif key in ("fps", "queue"):
            if not value.isdigit() or int(value) <= 0:
                raise ValueError(f"video {key} must be a positive integer, got '{value}'")
            opts[key] = int(value)
        elif key == "encoder":
            if value not in ENCODERS:
                raise ValueError(f"video encoder must be one of {ENCODERS}, got '{value}'")
            opts[key] = value
        else:
            raise ValueError(f"unknown video option '{key}' (valid: size, fps, queue, encoder)")
    return path, opts


def _has_gstreamer() -> bool:
    return any("GStreamer" in line and "YES" in line for line in cv2.getBuildInformation().splitlines())


def _gst_writer(path: str, size: tuple[int, int], fps: int) -> cv2.VideoWriter:
    pipe = (
        f"appsrc ! videoconvert ! video/x-raw,format=I420 ! "
        f"x264enc speed-preset=ultrafast tune=zerolatency bitrate=4000 key-int-max={fps} ! h264parse ! "
        f"mp4mux ! filesink location={Path(path).as_posix()}"
    )
    return cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, float(fps), size, True)


def open_writer(path: str, size: tuple[int, int], fps: int, encoder: str = "auto") -> cv2.VideoWriter:
    """Software H.264 through GStreamer x264enc when OpenCV has it (encoder auto/gst), else mp4v. Raises RuntimeError."""
    if encoder in ("auto", "gst"):
        if _has_gstreamer():
            writer = _gst_writer(path, size, fps)
            if writer.isOpened():
                return writer
            writer.release()
        if encoder == "gst":
            raise RuntimeError("x264enc through GStreamer is not available in this OpenCV")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size, True)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for {path}")
    return writer


def draw_overlay(canvas: np.ndarray, detections: list[Detection], src_wh: tuple[int, int], lines: list[str]) -> None:
    """Draw boxes (scaled from the source frame to the canvas), 'class score' labels and the text lines, in place."""
    ch, cw = canvas.shape[:2]
    sx, sy = cw / src_wh[0], ch / src_wh[1]
    thick = max(2, round(cw / 640))
    scale = max(0.5, cw / 1600)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
        color = class_color(det.class_name)
        cv2.rectangle(canvas, p1, p2, color, thick)
        label = f"{det.class_name} {det.score:.2f}"
        (tw, th), base = cv2.getTextSize(label, font, scale, 1)
        top = max(0, p1[1] - th - base - 2)
        cv2.rectangle(canvas, (p1[0], top), (p1[0] + tw + 4, top + th + base + 2), color, -1)
        cv2.putText(canvas, label, (p1[0] + 2, top + th), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    y = int(24 * scale * 1.5)
    for line in lines:
        cv2.putText(canvas, line, (10, y), font, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(24 * scale * 1.3)


class VideoSink:
    """Sink that records an annotated mp4. ``wants_pixels`` makes the pipeline attach the frame to each result."""

    wants_pixels = True

    def __init__(
        self, path: str, *, size: tuple[int, int] = DEFAULT_SIZE, fps: int = 30, queue_size: int = 4, encoder: str = "auto"
    ) -> None:
        self.path, self.size, self.fps = path, size, fps
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._writer = open_writer(path, size, fps, encoder)
        self._q: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._stamps: collections.deque[float] = collections.deque(maxlen=31)  # capture times, for the on-screen rate
        self.dropped = 0
        self.written = 0
        self._thread = threading.Thread(target=self._run, name="video-sink", daemon=True)
        self._thread.start()

    def write(self, result: FrameResult) -> None:
        pixels = result.pixels
        if pixels is None:
            return
        self._stamps.append(result.frame_ref.t_capture_wall_ns / 1e9)
        stamps = self._stamps
        rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0
        t = result.timings
        lines = [
            f"{rate:4.1f} FPS  inf {t.inference_ns / 1e6:.0f} ms  lat {t.total_ns / 1e6:.0f} ms  {len(result.detections)} det",
            result.model_id,
        ]
        try:
            self._q.put_nowait((pixels, list(result.detections), lines))
        except queue.Full:
            self.dropped += 1  # the output video loses the frame; the inference never waits for the encoder

    def write_health(self, state: HealthState) -> None:
        pass

    def _render(self, pixels: np.ndarray, detections: list[Detection], lines: list[str]) -> np.ndarray:
        h, w = pixels.shape[:2]
        canvas = cv2.resize(pixels, self.size, interpolation=cv2.INTER_AREA) if (w, h) != self.size else pixels.copy()
        draw_overlay(canvas, detections, (w, h), lines)
        return canvas

    def _run(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=_POLL_S)
            except queue.Empty:
                continue
            if item is _STOP:
                return
            try:
                self._writer.write(self._render(*item))
                self.written += 1
            except Exception:
                log.exception("video sink failed on a frame")

    def close(self) -> None:
        self._q.put(_STOP)  # blocking put: close() waits for the queue to drain, then finalises the file
        self._thread.join(timeout=30)
        self._writer.release()
        log.info("video %s: %d frames written, %d dropped from the output", self.path, self.written, self.dropped)
