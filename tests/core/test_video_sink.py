"""Annotated video sink: spec parsing, colours, overlay, bounded queue and the pixels hand-off from the pipeline."""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest

from jetson_vision.backends.fake import FakeBackend
from jetson_vision.pipeline import Pipeline, PipelineConfig
from jetson_vision.sinks import make_sink
from jetson_vision.types import Detection, FrameResult
from jetson_vision.video_sink import (
    CLASS_COLORS_HEX,
    VideoSink,
    class_color,
    draw_overlay,
    hex_to_bgr,
    parse_video_spec,
)

from .conftest import CollectSink, ScriptedSource, make_frame


def make_result(seq: int, pixels: np.ndarray | None, dets: list[Detection] | None = None) -> FrameResult:
    frame = make_frame(seq, w=160, h=90)
    return FrameResult(frame.ref, dets or [], "m", 0.5, pixels=pixels)


def test_spec_parsing_defaults_and_options():
    assert parse_video_spec("out/a.mp4") == ("out/a.mp4", {})
    path, opts = parse_video_spec("C:/x/a.mp4?size=640x360&fps=15&queue=8&encoder=mp4v")
    assert path == "C:/x/a.mp4" and opts == {"size": (640, 360), "fps": 15, "queue": 8, "encoder": "mp4v"}


@pytest.mark.parametrize("bad", ["", "a.mp4?size=640", "a.mp4?size=0x10", "a.mp4?fps=0", "a.mp4?encoder=nvenc", "a.mp4?color=1"])
def test_spec_parsing_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        parse_video_spec(bad)


def test_class_colours_are_the_app_ones_in_bgr():
    assert CLASS_COLORS_HEX["Bottle"] == "E69F00" and CLASS_COLORS_HEX["Tire"] == "A1887F"
    assert hex_to_bgr("E69F00") == (0x00, 0x9F, 0xE6)
    assert len(CLASS_COLORS_HEX) == 8 and class_color("Glove") == hex_to_bgr("CC79A7")
    assert class_color("unknown") == hex_to_bgr("00B4D8")


def test_overlay_draws_the_box_in_the_class_colour_scaled_to_the_canvas():
    canvas = np.zeros((90, 160, 3), np.uint8)
    det = Detection(0, "Bottle", 0.9, (100.0, 40.0, 300.0, 160.0))  # source frame is 320x180: scale 0.5
    draw_overlay(canvas, [det], (320, 180), ["12.0 FPS"])
    assert tuple(canvas[80, 100]) == hex_to_bgr("E69F00")  # bottom edge of the scaled box
    assert canvas[:30, :80].any()  # text lines


def test_sink_writes_a_readable_video_at_the_requested_size(tmp_path):
    path = str(tmp_path / "out.mp4")
    sink = VideoSink(path, size=(320, 180), fps=10, encoder="mp4v")
    for i in range(12):
        px = np.full((90, 160, 3), 40, np.uint8)
        sink.write(make_result(i, px, [Detection(3, "Glove", 0.8, (10.0, 10.0, 80.0, 60.0))]))
        time.sleep(0.01)
    sink.close()
    cap = cv2.VideoCapture(path)
    assert cap.isOpened() and (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (320, 180)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == sink.written == 12 - sink.dropped
    cap.release()


def test_results_without_pixels_are_ignored(tmp_path):
    sink = VideoSink(str(tmp_path / "o.mp4"), size=(64, 36), encoder="mp4v")
    sink.write(make_result(0, None))
    sink.close()
    assert sink.written == 0 and sink.dropped == 0


def test_slow_encoder_drops_output_frames_and_never_blocks_the_caller(tmp_path):
    sink = VideoSink(str(tmp_path / "o.mp4"), size=(64, 36), queue_size=1, encoder="mp4v")
    real = sink._writer

    class Slow:  # encoder far slower than the producer
        def write(self, frame):
            time.sleep(0.2)
            real.write(frame)

        def release(self):
            real.release()

    sink._writer = Slow()
    px = np.zeros((90, 160, 3), np.uint8)
    t0 = time.perf_counter()
    for i in range(30):
        sink.write(make_result(i, px))
    elapsed = time.perf_counter() - t0
    sink.close()
    assert elapsed < 0.5 and sink.dropped >= 20 and sink.written + sink.dropped == 30


def test_pipeline_attaches_pixels_only_when_a_sink_wants_them(rf_manifest, tmp_path):
    seen = []

    class Spy(CollectSink):
        wants_pixels = True

        def write(self, result):
            seen.append(result.pixels is not None and result.pixels.shape == (48, 64, 3))
            super().write(result)

    for sink, expected in ((Spy(), True), (CollectSink(), False)):
        seen.clear()
        pipe = Pipeline(
            ScriptedSource([make_frame(i) for i in range(4)]), FakeBackend(rf_manifest), rf_manifest, [sink], PipelineConfig()
        )
        pipe.run()
        assert (all(seen) and len(seen) == 4) if expected else not seen
        assert all(r.pixels is None for r in sink.results)  # released after publication


def test_to_dict_never_serialises_the_pixels():
    result = make_result(0, np.zeros((90, 160, 3), np.uint8))
    data = result.to_dict()
    assert "pixels" not in data and len(json.dumps(data, default=lambda o: o.tolist())) < 2000


def test_make_sink_builds_a_video_sink(tmp_path):
    sink = make_sink(f"video:{tmp_path / 's.mp4'}?size=64x36&encoder=mp4v")
    assert isinstance(sink, VideoSink) and sink.size == (64, 36)
    sink.close()
