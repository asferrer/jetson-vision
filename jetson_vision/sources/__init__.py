"""Frame sources. Importing this package does not import GStreamer."""
from __future__ import annotations

from pathlib import Path

from jetson_vision.interfaces import FrameSource
from jetson_vision.sources.folder import FolderSource
from jetson_vision.sources.video import V4L2Source, VideoSource

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

__all__ = ["FolderSource", "VideoSource", "V4L2Source", "make_source"]


def make_source(spec: str, *, mode: str = "max", fps: float | None = None, live: bool | None = None,
                loop: bool = False) -> FrameSource:
    """Build a source from a string.

    gst:<uri>        GStreamer (uri as in sources.gstreamer.build_pipeline: csi:0, rtsp://..., file:x.mp4, ...)
    /dev/videoN, v4l2:<device or index>
    a directory or an image file -> FolderSource; a video file -> VideoSource
    live=True makes a gst:file: source play at its own rate and drop late frames (a camera stand-in);
    loop=True restarts a gst:file: source at its end (long runs from a short clip).
    """
    if spec.startswith("gst:"):
        from jetson_vision.sources.gstreamer import GStreamerSource
        return GStreamerSource(spec[4:], mode=mode, live=live, loop=loop)
    if spec.startswith("v4l2:") or spec.startswith("/dev/video"):
        device = spec[5:] if spec.startswith("v4l2:") else spec
        return V4L2Source(int(device) if device.isdigit() else device, fps=fps)
    path = Path(spec)
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return VideoSource(path, mode=mode, fps=fps)
    return FolderSource(path, mode=mode, fps=fps or 30.0)
