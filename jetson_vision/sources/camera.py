"""Live camera sources (`cam:` URIs) on top of the GStreamer appsink source.

    cam:csi:<sensor-id>[?w=&h=&fps=&flip=]        CSI sensor: nvarguscamerasrc -> NVMM -> nvvidconv
    cam:v4l2:<device or index>[?w=&h=&fps=&fmt=&hw=]   USB/V4L2: fmt yuy2 | nv12 | mjpg (default mjpg)
    cam:rtsp://host/path[?latency=&codec=&w=&h=&fps=]  IP camera: rtspsrc, h264 (default) or h265
    cam:test[?w=&h=&fps=&fmt=&frames=]            videotestsrc is-live=true. FOR TESTS ONLY (no camera needed)

All of them end in the same live appsink as the other GStreamer sources (one buffer, `drop=true`: the newest frame wins)
and BGR, which is what the preprocessing expects. For MJPEG, hw=1 (default) uses nvv4l2decoder mjpeg=1 and hw=0 uses jpegdec.

A camera never ends: an end of stream or a silent device is a SourceError, so the pipeline reconnects with backoff.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from jetson_vision.interfaces import SourceError
from jetson_vision.sources.gstreamer import GStreamerSource, _appsink, _convert_tail, load_gst

KINDS = ("csi", "v4l2", "rtsp", "test")
FORMATS = ("yuy2", "nv12", "mjpg")
CODECS = ("h264", "h265")
_PARAMS = {"w": int, "h": int, "fps": int, "flip": int, "fmt": str, "hw": int, "latency": int, "codec": str, "frames": int}
_ELEMENT_HINTS = {
    "nvarguscamerasrc": "JetPack's gst-nvarguscamerasrc (Jetson only); the nvargus-daemon service must be running",
    "nvv4l2decoder": "JetPack's NVIDIA decoder (Jetson only); use hw=0 for a software decoder",
    "nvvidconv": "JetPack's nvvidconv (Jetson only)",
}
SLICE_S = 0.5  # pull in short slices so a device error on the bus is noticed at once, not after the silence timeout


@dataclass(frozen=True)
class CameraSpec:
    kind: str
    target: str = ""
    w: int = 1920
    h: int = 1080
    fps: int = 30
    fmt: str = "mjpg"
    hw: bool = True
    flip: int = 0
    latency: int = 100
    codec: str = "h264"
    frames: int = 0  # cam:test only: end the stream after N frames (simulates a camera that goes away)

    @property
    def source_id(self) -> str:
        """Identity of the source; never carries credentials of an RTSP URL."""
        if self.kind == "rtsp":
            host = urlsplit(self.target).hostname or "?"
            return f"cam:rtsp:{host}"
        return f"cam:{self.kind}:{self.target}" if self.target else f"cam:{self.kind}"


def _parse_params(query: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in parse_qsl(query, keep_blank_values=True):
        if key not in _PARAMS:
            raise ValueError(f"unknown camera parameter '{key}' (valid: {', '.join(sorted(_PARAMS))})")
        try:
            out[key] = _PARAMS[key](value)
        except ValueError as exc:
            raise ValueError(f"camera parameter '{key}' needs {_PARAMS[key].__name__}, got '{value}'") from exc
    return out


def _check(spec: CameraSpec) -> CameraSpec:
    if min(spec.w, spec.h, spec.fps) <= 0:
        raise ValueError("camera w, h and fps must be positive")
    if spec.fmt not in FORMATS:
        raise ValueError(f"camera fmt must be one of {FORMATS}, got '{spec.fmt}'")
    if spec.codec not in CODECS:
        raise ValueError(f"camera codec must be one of {CODECS}, got '{spec.codec}'")
    if not 0 <= spec.flip <= 7:
        raise ValueError("camera flip must be 0..7 (nvvidconv flip-method)")
    if spec.kind in ("csi", "v4l2") and not spec.target:
        raise ValueError(f"cam:{spec.kind} needs a sensor id or device")
    return spec


def parse_camera_uri(uri: str) -> CameraSpec:
    """Parse `cam:<kind>[:<target>][?params]` into a CameraSpec. Raises ValueError with a clear message."""
    if not uri.startswith("cam:"):
        raise ValueError(f"not a camera URI: '{uri}'")
    body = uri[4:]
    head, _, query = body.partition("?")
    params = _parse_params(query)
    if "hw" in params:
        params["hw"] = bool(params["hw"])
    if head.startswith("rtsp://") or head.startswith("rtsps://"):
        kind, target = "rtsp", head
    else:
        kind, _, target = head.partition(":")
    if kind not in KINDS:
        raise ValueError(f"unknown camera kind '{kind}' (valid: {', '.join(KINDS)})")
    if kind == "v4l2" and target.isdigit():
        target = f"/dev/video{target}"
    if kind == "csi" and not target:
        target = "0"
    if kind == "csi" and not target.isdigit():
        raise ValueError(f"cam:csi needs a numeric sensor id, got '{target}'")
    return _check(CameraSpec(kind=kind, target=target, **params))


# ---- launch strings --------------------------------------------------------------------------------------
def _caps(spec: CameraSpec, media: str = "video/x-raw", fmt: str | None = None) -> str:
    fmt_part = f",format={fmt}" if fmt else ""
    return f"{media}{fmt_part},width={spec.w},height={spec.h},framerate={spec.fps}/1"


def _decode_tail(spec: CameraSpec) -> str:
    """Everything between the capture caps and the appsink, for a V4L2-like stream in spec.fmt."""
    if spec.fmt == "mjpg":
        dec = "nvv4l2decoder mjpeg=1" if spec.hw else "jpegdec"
        return f"{dec} ! {_convert_tail(spec.hw)}"
    return _convert_tail(spec.hw)


def _v4l2_head(spec: CameraSpec) -> str:
    if spec.fmt == "mjpg":
        caps = _caps(spec, "image/jpeg")
    else:
        caps = _caps(spec, fmt=spec.fmt.upper())
    return f"v4l2src device={spec.target} ! {caps}"


def _rtsp_head(spec: CameraSpec) -> str:
    depay, parse = ("rtph264depay", "h264parse") if spec.codec == "h264" else ("rtph265depay", "h265parse")
    dec = "nvv4l2decoder" if spec.hw else ("avdec_h264" if spec.codec == "h264" else "avdec_h265")
    return f"rtspsrc location={spec.target} latency={spec.latency} ! {depay} ! {parse} ! {dec}"


def _test_head(spec: CameraSpec) -> str:
    """videotestsrc as a live camera. mjpg is encoded so the real decode tail is exercised."""
    num = f" num-buffers={spec.frames}" if spec.frames else ""
    src = f"videotestsrc is-live=true pattern=ball{num}"
    if spec.fmt == "mjpg":
        return f"{src} ! {_caps(spec)} ! videoconvert ! jpegenc ! {_caps(spec, 'image/jpeg')}"
    return f"{src} ! {_caps(spec, fmt=spec.fmt.upper())}"


def camera_launch(spec: CameraSpec) -> str:
    """The full GStreamer launch string for a camera spec, ending in a live BGR appsink."""
    sink = _appsink(live=True, sync=False)
    if spec.kind == "csi":
        head = (
            f"nvarguscamerasrc sensor-id={spec.target} ! {_caps(spec, 'video/x-raw(memory:NVMM)', 'NV12')} ! "
            f"nvvidconv flip-method={spec.flip}"
        )
        return f"{head} ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! {sink}"
    if spec.kind == "rtsp":
        return f"{_rtsp_head(spec)} ! {_convert_tail(spec.hw)} ! {sink}"
    head = _v4l2_head(spec) if spec.kind == "v4l2" else _test_head(spec)
    return f"{head} ! {_decode_tail(spec)} ! {sink}"


def required_elements(spec: CameraSpec) -> list[str]:
    """GStreamer elements the chain needs, in order, to report a missing one by name before parsing."""
    names = {
        "csi": ["nvarguscamerasrc", "nvvidconv", "videoconvert"],
        "v4l2": ["v4l2src"],
        "rtsp": ["rtspsrc"],
        "test": ["videotestsrc"],
    }[spec.kind]
    if spec.kind == "rtsp":
        names += ["rtph264depay", "h264parse"] if spec.codec == "h264" else ["rtph265depay", "h265parse"]
        names.append("nvv4l2decoder" if spec.hw else "avdec_h264" if spec.codec == "h264" else "avdec_h265")
    if spec.kind in ("v4l2", "test"):
        if spec.fmt == "mjpg":
            names.append("nvv4l2decoder" if spec.hw else "jpegdec")
            names += ["jpegenc"] if spec.kind == "test" else []
        names += ["nvvidconv"] if spec.hw and spec.kind != "csi" else []
        names.append("videoconvert")
    return list(dict.fromkeys(names))


class CameraSource(GStreamerSource):
    """A live camera. Frames are stamped with the capture time of the sensor/driver, not with the time they were read."""

    live = True

    def __init__(self, uri: str, *, silence_s: float = 5.0) -> None:
        self.spec = parse_camera_uri(uri)
        self.silence_s = silence_s
        self._init_state(self.spec.source_id, camera_launch(self.spec))

    def open(self) -> None:
        Gst = load_gst()
        self._check_elements(Gst)
        super().open()
        self._wait_playing(Gst)

    def _check_elements(self, Gst: Any) -> None:
        for name in required_elements(self.spec):
            if Gst.ElementFactory.find(name) is None:
                hint = _ELEMENT_HINTS.get(name, "install the matching GStreamer plugin package")
                raise SourceError(f"GStreamer element '{name}' not found ({hint})")

    def _wait_playing(self, Gst: Any) -> None:
        """A live source answers ASYNC until it has produced a frame: wait a bounded time for it or for a bus error."""
        deadline = time.monotonic() + self.silence_s
        while time.monotonic() < deadline:
            self._raise_bus_error(cleanup=True)
            ret, state, _ = self._pipe.get_state(int(SLICE_S * 1e9))
            if ret == Gst.StateChangeReturn.FAILURE:
                break
            if state == Gst.State.PLAYING:
                return
        self._raise_bus_error(cleanup=True)
        self._abandon()
        raise SourceError(f"camera '{self.source_id}' did not start in {self.silence_s:.0f} s")

    def read(self):
        if self._sink is None:
            raise SourceError("camera source: read() before open()")
        deadline = time.monotonic() + self.silence_s
        while time.monotonic() < deadline:
            sample = self._sink.emit("try-pull-sample", int(SLICE_S * 1e9))
            if sample is not None:
                pixels, pts = self._to_array(sample)
                return self.make_frame(pixels, pts, t_mono_ns=self._capture_mono(pts))
            self._raise_bus_error()
            if self._sink.get_property("eos"):
                raise SourceError(f"camera '{self.source_id}' ended its stream{self._eos_hint()}")
        raise SourceError(f"camera '{self.source_id}' silent for {self.silence_s:.0f} s")

    def _eos_hint(self) -> str:
        if self.spec.kind == "csi":  # nvarguscamerasrc reports "No cameras available" on stderr and then ends the stream
            return " (no CSI sensor found or it was disconnected; check the cable and the nvargus-daemon service)"
        return ""

    def _capture_mono(self, pts: int | None) -> int | None:
        """Pipeline running time of the buffer on the monotonic clock; None if it is not plausible (use the read time)."""
        if pts is None or self._pipe is None:
            return None
        base = self._pipe.get_base_time()
        if base == self._gst.CLOCK_TIME_NONE:
            return None
        stamp, now = int(base) + int(pts), time.monotonic_ns()
        return stamp if now - 10_000_000_000 <= stamp <= now else None


__all__ = ["CameraSource", "CameraSpec", "camera_launch", "parse_camera_uri", "required_elements"]
