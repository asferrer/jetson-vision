"""GStreamer source through PyGObject and an appsink (no dependency on how OpenCV was built).

On Jetson, H.264/H.265 and MJPEG are decoded by the hardware decoder (nvv4l2decoder) and converted by nvvidconv
(VIC). The `nvjpegdec` element is not used: it gave AP 3.02 in an early test (undiagnosed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jetson_vision.interfaces import SourceError
from jetson_vision.sources.base import SourceBase, check_mode

_SINK = "sink"
PULL_TIMEOUT_S = 5.0


def _convert_tail(hw: bool) -> str:
    conv = "nvvidconv ! video/x-raw,format=BGRx ! videoconvert" if hw else "videoconvert"
    return f"{conv} ! video/x-raw,format=BGR"


def _appsink(live: bool, sync: bool) -> str:
    """live: newest frame wins (one buffer, older ones dropped). sync: the pipeline clock releases frames at the
    stream rate (a file played as a camera, or paced replay); a real camera needs no clock."""
    flag = "true" if sync else "false"
    if live:
        return f"appsink name={_SINK} max-buffers=1 drop=true sync={flag}"
    return f"appsink name={_SINK} max-buffers=4 drop=false sync={flag}"


def build_pipeline(uri: str, *, live: bool | None = None, mode: str = "max", hw_decode: bool = True,
                   width: int = 1920, height: int = 1080, fps: int = 30) -> tuple[str, bool]:
    """Launch string and the live flag for a source URI.

    csi:<sensor-id>   nvarguscamerasrc (CSI camera)
    rtsp://...        rtspsrc, H.264
    file:<path>       MP4/MOV with H.264 (hardware decoder when hw_decode, otherwise decodebin). live=True plays it
                      at its own rate and drops late frames, like a camera; otherwise it is a finite replay.
    v4l2:<device>     v4l2src, raw or MJPEG
    anything else containing '!' is taken as a launch string; a converter and an appsink are appended.
    """
    check_mode(mode)
    sync = mode == "paced"  # only finite replays use the pipeline clock; cameras and streams are already real time
    tail = _convert_tail(hw_decode)
    decode = "nvv4l2decoder" if hw_decode else "avdec_h264"
    if uri.startswith("csi:"):
        src = (f"nvarguscamerasrc sensor-id={int(uri[4:] or 0)} ! video/x-raw(memory:NVMM),format=NV12,"
               f"width={width},height={height},framerate={fps}/1 ! {tail}")
        is_live = True if live is None else live
    elif uri.startswith("rtsp://"):
        src, is_live = f"rtspsrc location={uri} latency=100 ! rtph264depay ! h264parse ! {decode} ! {tail}", True
    elif uri.startswith("file:"):
        demux = f"qtdemux ! h264parse ! {decode}" if hw_decode else "decodebin"
        src, is_live = f"filesrc location={uri[5:]} ! {demux} ! {tail}", bool(live)
        sync = sync or is_live
    elif uri.startswith("v4l2:"):
        src, is_live = f"v4l2src device={uri[5:]} ! decodebin ! {tail}", True
    elif "!" in uri:
        is_live = bool(live)
        src = uri if "appsink" in uri else f"{uri} ! {_convert_tail(False)}"
        return (src if "appsink" in uri else f"{src} ! {_appsink(is_live, sync and not is_live)}"), is_live
    else:
        raise ValueError(f"unsupported GStreamer URI '{uri}'")
    return f"{src} ! {_appsink(is_live, sync and (is_live is False or uri.startswith('file:')))}", is_live


def load_gst() -> Any:
    """Import PyGObject's Gst and initialise it; SourceError if it is not installed."""
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ImportError, ValueError) as exc:
        raise SourceError(f"GStreamer source needs PyGObject with GStreamer 1.0: {exc}") from exc
    Gst.init(None)
    return Gst


class GStreamerSource(SourceBase):
    """Frames from a GStreamer pipeline. mode only matters for files: 'paced' follows the pipeline clock."""

    def __init__(self, uri: str, *, mode: str = "max", hw_decode: bool = True, live: bool | None = None,
                 width: int = 1920, height: int = 1080, fps: int = 30, loop: bool = False,
                 source_id: str | None = None) -> None:
        launch, live_flag = build_pipeline(uri, live=live, mode=mode, hw_decode=hw_decode,
                                           width=width, height=height, fps=fps)
        self._init_state(source_id or f"gst:{Path(uri).name if uri.startswith('file:') else uri.split(':')[0]}", launch)
        self.live = live_flag
        self.loop, self.loops = loop and uri.startswith("file:"), 0  # loop: restart a file at its end

    def _init_state(self, source_id: str, launch: str) -> None:
        SourceBase.__init__(self, source_id)
        self.launch = launch
        self.loop, self.loops = False, 0
        self._pipe: Any = None
        self._sink: Any = None
        self._gst: Any = None

    def open(self) -> None:
        Gst = load_gst()
        try:
            pipe = Gst.parse_launch(self.launch)
        except Exception as exc:  # GLib.Error
            raise SourceError(f"GStreamer cannot build the pipeline: {exc}") from exc
        sink = pipe.get_by_name(_SINK)
        self._gst, self._pipe, self._sink = Gst, pipe, sink
        if sink is None:
            self._abandon()
            raise SourceError("GStreamer pipeline has no appsink")
        if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._raise_bus_error(cleanup=True)
            self._abandon()
            raise SourceError("GStreamer pipeline did not start (state change failed)")

    def read(self):
        if self._sink is None:
            raise SourceError("GStreamer source: read() before open()")
        sample = self._sink.emit("try-pull-sample", int(PULL_TIMEOUT_S * 1e9))
        if sample is not None:
            return self.make_frame(*self._to_array(sample))
        self._raise_bus_error()
        if self._sink.get_property("eos"):
            return self.read() if self._rewind() else None  # None: end of a finite source
        raise SourceError(f"GStreamer source '{self.source_id}' silent for {PULL_TIMEOUT_S:.0f} s")

    def _rewind(self) -> bool:
        if not self.loop:
            return False
        flags = self._gst.SeekFlags.FLUSH | self._gst.SeekFlags.KEY_UNIT
        if not self._pipe.seek_simple(self._gst.Format.TIME, flags, 0):
            raise SourceError("GStreamer cannot rewind the file")
        self.loops += 1
        return True

    def _raise_bus_error(self, cleanup: bool = False) -> None:
        """Raise the pending bus error, if any. cleanup: release the pipeline first (an open() that is failing)."""
        msg = self._pipe.get_bus().pop_filtered(self._gst.MessageType.ERROR)
        if msg is not None:
            err, _ = msg.parse_error()
            if cleanup:
                self._abandon()
            raise SourceError(f"GStreamer error: {err.message}")

    def _abandon(self) -> None:
        """Stop and drop a pipeline that never worked, so a retry starts from nothing."""
        if self._pipe is not None:
            self._pipe.set_state(self._gst.State.NULL)
        self._pipe = self._sink = None

    def _to_array(self, sample: Any) -> tuple[np.ndarray, int | None]:
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value("width"), caps.get_value("height")
        buf = sample.get_buffer()
        ok, info = buf.map(self._gst.MapFlags.READ)
        if not ok:
            raise SourceError("GStreamer buffer could not be mapped")
        try:
            raw = np.frombuffer(info.data, np.uint8)
            stride = raw.size // h  # rows are padded to a multiple of 4 bytes
            pixels = np.ascontiguousarray(raw.reshape(h, stride)[:, : 3 * w].reshape(h, w, 3))
        finally:
            buf.unmap(info)
        pts = buf.pts
        return pixels, (int(pts) if pts != self._gst.CLOCK_TIME_NONE else None)

    def close(self) -> None:
        if self._pipe is not None:
            self._pipe.set_state(self._gst.State.NULL)
        self._pipe = self._sink = None
