"""Frame sources: identity, clocks, contiguity, ordering, modes and GStreamer pipeline strings."""
from __future__ import annotations

import sys
import time
import uuid

import cv2
import numpy as np
import pytest

from jetson_vision.interfaces import FrameSource, SourceError
from jetson_vision.sources import FolderSource, V4L2Source, VideoSource, make_source
from jetson_vision.sources.gstreamer import GStreamerSource, build_pipeline


def write_images(folder, n: int, h: int = 30, w: int = 40) -> None:
    for i in range(n):
        img = np.full((h, w, 3), 10 * i, np.uint8)
        img[0, 0] = (255, 0, 0)  # a blue pixel in BGR: detects any channel swap
        cv2.imwrite(str(folder / f"img_{i:03d}.png"), img)


def drain(src) -> list:
    src.open()
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    src.close()
    return frames


def test_folder_frames_are_ordered_stamped_and_contiguous(tmp_path):
    write_images(tmp_path, 5)
    src = FolderSource(tmp_path, fps=10.0)
    assert isinstance(src, FrameSource) and src.live is False
    before_wall = time.time_ns()
    frames = drain(src)
    assert [f.ref.frame_seq for f in frames] == [0, 1, 2, 3, 4]
    assert [int(f.pixels[5, 5, 0]) for f in frames] == [0, 10, 20, 30, 40]  # sorted file order
    assert all(f.pixels.flags["C_CONTIGUOUS"] and f.pixels.dtype == np.uint8 for f in frames)
    assert tuple(frames[0].pixels[0, 0]) == (255, 0, 0)  # BGR preserved
    assert len({f.ref.session_id for f in frames}) == 1 and uuid.UUID(frames[0].ref.session_id)
    mono = [f.ref.t_capture_mono_ns for f in frames]
    assert mono == sorted(mono)
    assert abs(frames[0].ref.t_capture_wall_ns - before_wall) < 5e9
    assert [f.ref.source_stamp for f in frames] == [0, 100_000_000, 200_000_000, 300_000_000, 400_000_000]


def test_every_source_object_gets_its_own_session_id(tmp_path):
    write_images(tmp_path, 1)
    assert FolderSource(tmp_path).session_id != FolderSource(tmp_path).session_id


def test_paced_mode_respects_the_rate_and_max_mode_does_not(tmp_path):
    write_images(tmp_path, 4)
    t0 = time.monotonic()
    drain(FolderSource(tmp_path, mode="max", fps=10.0))
    fast = time.monotonic() - t0
    t0 = time.monotonic()
    drain(FolderSource(tmp_path, mode="paced", fps=20.0))
    paced = time.monotonic() - t0
    assert fast < 0.1 and paced >= 0.12  # 3 periods of 50 ms, minus timer slack


def test_unreadable_image_is_skipped_and_empty_or_missing_folders_fail(tmp_path):
    write_images(tmp_path, 2)
    (tmp_path / "img_001b.png").write_bytes(b"not an image")
    assert len(drain(FolderSource(tmp_path))) == 2
    with pytest.raises(SourceError):
        FolderSource(tmp_path / "nowhere").open()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SourceError, match="no images"):
        FolderSource(empty).open()


def test_single_image_file_is_a_one_frame_source(tmp_path):
    write_images(tmp_path, 1)
    assert len(drain(FolderSource(tmp_path / "img_000.png"))) == 1


def test_video_source_reads_every_frame_with_index_based_stamps(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write MJPG")
    for i in range(6):
        writer.write(np.full((48, 64, 3), 40 * i, np.uint8))
    writer.release()
    frames = drain(VideoSource(path, mode="max"))
    assert len(frames) == 6 and [f.ref.frame_seq for f in frames] == list(range(6))
    assert [f.ref.source_stamp for f in frames] == [i * 40_000_000 for i in range(6)]
    assert frames[0].pixels.shape == (48, 64, 3) and frames[0].pixels.flags["C_CONTIGUOUS"]


def test_video_source_missing_file_raises_source_error(tmp_path):
    with pytest.raises(SourceError):
        VideoSource(tmp_path / "nope.mp4").open()


def test_v4l2_source_is_live_and_fails_with_source_error_without_a_device():
    src = V4L2Source("/dev/video_that_does_not_exist")
    assert src.live is True
    with pytest.raises(SourceError):
        src.open()


def test_make_source_dispatch(tmp_path):
    assert isinstance(make_source(str(tmp_path)), FolderSource)
    assert isinstance(make_source("clip.mp4"), VideoSource)
    assert isinstance(make_source("/dev/video0"), V4L2Source) and make_source("v4l2:1").device == 1
    assert make_source("gst:csi:0").live is True


def test_gstreamer_hardware_decode_pipeline_uses_nvv4l2decoder_and_nvvidconv():
    launch, live = build_pipeline("file:clip.mp4")
    assert live is False
    assert "nvv4l2decoder" in launch and "nvvidconv" in launch and "format=BGR" in launch
    assert "nvjpegdec" not in launch and "sync=false" in launch and "drop=false" in launch
    assert "sync=true" in build_pipeline("file:clip.mp4", mode="paced")[0]
    assert "nvv4l2decoder" not in build_pipeline("file:clip.mp4", hw_decode=False)[0]


def test_gstreamer_live_pipelines_keep_only_the_newest_frame():
    launch, live = build_pipeline("csi:1", width=1280, height=720, fps=30)
    assert live and "nvarguscamerasrc sensor-id=1" in launch and "width=1280" in launch
    assert "max-buffers=1 drop=true" in launch and "nvvidconv" in launch
    rtsp, live = build_pipeline("rtsp://10.0.0.2/s")
    assert live and "nvv4l2decoder" in rtsp and "latency=100" in rtsp


def test_gstreamer_custom_launch_string_gets_a_converter_and_an_appsink():
    launch, live = build_pipeline("videotestsrc num-buffers=3 ! video/x-raw", live=False)
    assert launch.endswith("name=sink max-buffers=4 drop=false sync=false") and "videoconvert" in launch
    with pytest.raises(ValueError):
        build_pipeline("gopher://x")


def test_gstreamer_without_pygobject_raises_source_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "gi", None)
    with pytest.raises(SourceError, match="PyGObject"):
        GStreamerSource("csi:0").open()


def test_gstreamer_file_played_as_a_camera_is_live_and_follows_the_clock():
    launch, live = build_pipeline("file:clip.mp4", live=True)
    assert live is True and "max-buffers=1 drop=true sync=true" in launch
    cam, _ = build_pipeline("csi:0", mode="paced")
    assert "sync=false" in cam  # a camera needs no clock
    assert make_source("gst:file:clip.mp4", live=True, loop=True).live is True
    assert make_source("gst:file:clip.mp4", loop=True).loop is True
    assert make_source("gst:csi:0", loop=True).loop is False  # only files loop
