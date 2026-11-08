"""Camera sources: URI parsing, generated GStreamer chains, and the open/read/close logic against GStreamer doubles."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from jetson_vision.backends.fake import FakeBackend
from jetson_vision.interfaces import FrameSource, SourceError
from jetson_vision.pipeline import Pipeline, PipelineConfig
from jetson_vision.sources import camera as cam
from jetson_vision.sources import gstreamer as gst_mod
from jetson_vision.sources import make_source
from jetson_vision.sources.camera import CameraSource, camera_launch, parse_camera_uri, required_elements

from .conftest import CollectSink, ScriptedSource, make_frame

NONE = 2**64 - 1


# ---- URI parsing -----------------------------------------------------------------------------------------
def test_csi_uri_defaults_and_parameters():
    spec = parse_camera_uri("cam:csi:1?w=1280&h=720&fps=60&flip=2")
    assert (spec.kind, spec.target, spec.w, spec.h, spec.fps, spec.flip) == ("csi", "1", 1280, 720, 60, 2)
    assert parse_camera_uri("cam:csi").target == "0" and parse_camera_uri("cam:csi:0").fps == 30


def test_v4l2_uri_accepts_a_device_path_or_an_index():
    assert parse_camera_uri("cam:v4l2:/dev/video2?fmt=yuy2&hw=0").target == "/dev/video2"
    spec = parse_camera_uri("cam:v4l2:3")
    assert spec.target == "/dev/video3" and spec.fmt == "mjpg" and spec.hw is True
    assert parse_camera_uri("cam:v4l2:0?hw=0").hw is False


def test_rtsp_uri_keeps_the_url_and_hides_credentials_in_the_source_id():
    spec = parse_camera_uri("cam:rtsp://user:secret@10.0.0.9:554/stream1?latency=50&codec=h265")
    assert spec.target == "rtsp://user:secret@10.0.0.9:554/stream1" and spec.latency == 50 and spec.codec == "h265"
    assert "secret" not in spec.source_id and spec.source_id == "cam:rtsp:10.0.0.9"


@pytest.mark.parametrize(
    "uri",
    [
        "gst:csi:0",
        "cam:usb:0",
        "cam:csi:x",
        "cam:csi:0?w=big",
        "cam:csi:0?bogus=1",
        "cam:csi:0?w=0",
        "cam:v4l2",
        "cam:v4l2:0?fmt=rgb",
        "cam:csi:0?flip=9",
        "cam:rtsp://h/s?codec=vp9",
    ],
)
def test_bad_camera_uris_raise_value_error(uri):
    with pytest.raises(ValueError):
        parse_camera_uri(uri)


# ---- generated chains ------------------------------------------------------------------------------------
LIVE_SINK = "appsink name=sink max-buffers=1 drop=true sync=false"


def test_csi_chain_is_nvmm_through_nvvidconv_to_bgr():
    launch = camera_launch(parse_camera_uri("cam:csi:1?w=1280&h=720&fps=30&flip=2"))
    nvmm = "video/x-raw(memory:NVMM),format=NV12,width=1280,height=720,framerate=30/1"
    assert launch.startswith(f"nvarguscamerasrc sensor-id=1 ! {nvmm}")
    assert "nvvidconv flip-method=2 ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR" in launch
    assert launch.endswith(LIVE_SINK)


def test_v4l2_mjpeg_hardware_and_software_decoders():
    hw = camera_launch(parse_camera_uri("cam:v4l2:/dev/video0?w=1920&h=1080&fps=30"))
    head = "v4l2src device=/dev/video0 ! image/jpeg,width=1920,height=1080,framerate=30/1 ! nvv4l2decoder mjpeg=1"
    assert hw.startswith(head) and "nvvidconv" in hw and hw.endswith(LIVE_SINK)
    sw = camera_launch(parse_camera_uri("cam:v4l2:0?hw=0"))
    assert "jpegdec" in sw and "nvv4l2decoder" not in sw and "nvvidconv" not in sw
    assert "video/x-raw,format=BGR ! " + LIVE_SINK in sw


def test_v4l2_raw_formats_are_negotiated_in_the_requested_format():
    yuy2 = camera_launch(parse_camera_uri("cam:v4l2:0?fmt=yuy2&w=640&h=480&fps=15"))
    assert "video/x-raw,format=YUY2,width=640,height=480,framerate=15/1" in yuy2 and "jpeg" not in yuy2
    assert "format=NV12" in camera_launch(parse_camera_uri("cam:v4l2:0?fmt=nv12"))


def test_rtsp_chain_depayloads_and_decodes():
    h264 = camera_launch(parse_camera_uri("cam:rtsp://10.0.0.2/s"))
    assert "rtspsrc location=rtsp://10.0.0.2/s latency=100 ! rtph264depay ! h264parse ! nvv4l2decoder" in h264
    h265 = camera_launch(parse_camera_uri("cam:rtsp://10.0.0.2/s?codec=h265&hw=0&latency=20"))
    assert "latency=20 ! rtph265depay ! h265parse ! avdec_h265" in h265 and h265.endswith(LIVE_SINK)


def test_test_source_is_live_and_shares_the_real_decode_tail():
    mj = camera_launch(parse_camera_uri("cam:test?fmt=mjpg&frames=90"))
    assert "videotestsrc is-live=true pattern=ball num-buffers=90" in mj and "jpegenc" in mj
    assert mj.index("jpegenc") < mj.index("nvv4l2decoder mjpeg=1") and mj.endswith(LIVE_SINK)
    yuy2 = camera_launch(parse_camera_uri("cam:test?fmt=yuy2"))
    assert "num-buffers" not in yuy2 and "queue max-size-buffers=2 leaky=downstream" in yuy2  # generator in its own thread


def test_required_elements_name_what_each_chain_needs():
    assert required_elements(parse_camera_uri("cam:csi:0"))[0] == "nvarguscamerasrc"
    assert "nvv4l2decoder" in required_elements(parse_camera_uri("cam:v4l2:0"))
    assert "jpegdec" in required_elements(parse_camera_uri("cam:v4l2:0?hw=0"))
    assert "nvv4l2decoder" not in required_elements(parse_camera_uri("cam:v4l2:0?hw=0"))
    assert required_elements(parse_camera_uri("cam:rtsp://h/s"))[0] == "rtspsrc"


def test_make_source_builds_camera_sources():
    src = make_source("cam:v4l2:/dev/video0")
    assert isinstance(src, CameraSource) and isinstance(src, FrameSource) and src.live is True
    assert src.source_id == "cam:v4l2:/dev/video0"
    with pytest.raises(ValueError):
        make_source("cam:nope")


# ---- GStreamer doubles -----------------------------------------------------------------------------------
class FakeBus:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def pop_filtered(self, _types):
        if not self.errors:
            return None
        text = self.errors.pop(0)
        return SimpleNamespace(parse_error=lambda: (SimpleNamespace(message=text), ""))


class FakePipe:
    def __init__(self, gst, base: int = 0, play_ret: str = "SUCCESS", state: str = "PLAYING") -> None:
        self.gst, self.bus, self.base, self.states = gst, FakeBus(), base, []
        self.play_ret, self.state, self.sink = play_ret, state, None

    def set_state(self, st):
        self.states.append(st)
        return self.gst.StateChangeReturn.FAILURE if st == "PLAYING" and self.play_ret == "FAILURE" else "OK"

    def get_state(self, _timeout):
        return ("SUCCESS" if self.state == "PLAYING" else "ASYNC"), self.state, None

    def get_bus(self):
        return self.bus

    def get_base_time(self):
        return self.base

    def get_by_name(self, _name):
        return self.sink


class FakeSink:
    def __init__(self, samples) -> None:
        self.samples, self.eos = list(samples), False

    def emit(self, _name, _timeout):
        if self.samples:
            return self.samples.pop(0)
        time.sleep(0.01)
        return None

    def get_property(self, _name):
        return self.eos


def fake_gst(missing=()):
    return SimpleNamespace(
        ElementFactory=SimpleNamespace(find=lambda n: None if n in missing else object()),
        State=SimpleNamespace(PLAYING="PLAYING", NULL="NULL"),
        StateChangeReturn=SimpleNamespace(FAILURE="FAILURE"),
        MessageType=SimpleNamespace(ERROR="ERROR"),
        MapFlags=SimpleNamespace(READ=1),
        CLOCK_TIME_NONE=NONE,
        parse_launch=None,
    )


def sample(w=4, h=2, pts=0, value=7):
    data = np.full(h * w * 3, value, np.uint8)
    buf = SimpleNamespace(pts=pts, map=lambda _f: (True, SimpleNamespace(data=data)), unmap=lambda _i: None)
    caps = SimpleNamespace(get_value=lambda k: {"width": w, "height": h}[k])
    return SimpleNamespace(get_caps=lambda: SimpleNamespace(get_structure=lambda _i: caps), get_buffer=lambda: buf)


@pytest.fixture
def install(monkeypatch):
    """install(pipe, gst, samples) wires every CameraSource to the doubles and returns the pipe."""

    def _install(pipe=None, gst=None, samples=()):
        gst = gst or fake_gst()
        pipe = pipe or FakePipe(gst)
        pipe.sink = FakeSink(samples)
        gst.parse_launch = lambda _launch: pipe
        monkeypatch.setattr(cam, "load_gst", lambda: gst)
        monkeypatch.setattr(gst_mod, "load_gst", lambda: gst)
        return pipe

    return _install


def test_open_reports_a_missing_gstreamer_element_by_name(install):
    install(gst=fake_gst(missing={"nvarguscamerasrc"}))
    with pytest.raises(SourceError, match=r"element 'nvarguscamerasrc' not found.*nvargus-daemon"):
        CameraSource("cam:csi:0").open()


def test_open_without_a_device_raises_the_bus_message_and_releases_the_pipeline(install):
    gst = fake_gst()
    pipe = install(FakePipe(gst, play_ret="FAILURE"), gst)
    pipe.bus.errors.append("Cannot identify device '/dev/video9'.")
    src = CameraSource("cam:v4l2:/dev/video9")
    with pytest.raises(SourceError, match="Cannot identify device"):
        src.open()
    assert "NULL" in pipe.states and src._pipe is None
    src.close()  # safe after a failed open


def test_open_that_never_reaches_playing_times_out_instead_of_hanging(install):
    gst = fake_gst()
    pipe = install(FakePipe(gst, state="READY"), gst)
    src = CameraSource("cam:csi:0", silence_s=0.2)
    t0 = time.monotonic()
    with pytest.raises(SourceError, match="did not start"):
        src.open()
    assert time.monotonic() - t0 < 2 and "NULL" in pipe.states


def test_read_returns_bgr_frames_stamped_with_the_capture_time(install):
    gst = fake_gst()
    base = time.monotonic_ns() - 200_000_000  # pipeline started 200 ms ago; buffer captured 100 ms after that
    install(FakePipe(gst, base=base), gst, samples=[sample(pts=100_000_000)])
    src = CameraSource("cam:csi:0")
    src.open()
    frame = src.read()
    assert frame.pixels.shape == (2, 4, 3) and frame.pixels.dtype == np.uint8 and int(frame.pixels[0, 0, 0]) == 7
    age_ms = (time.monotonic_ns() - frame.ref.t_capture_mono_ns) / 1e6
    assert 95 <= age_ms <= 400  # about 100 ms old at read time, not zero
    assert frame.ref.source_stamp == 100_000_000
    assert abs(frame.ref.t_capture_wall_ns - (time.time_ns() - age_ms * 1e6)) < 50e6  # wall clock shifted the same amount


def test_implausible_device_clock_falls_back_to_the_read_time(install):
    gst = fake_gst()
    install(FakePipe(gst, base=1), gst, samples=[sample(pts=5)])
    src = CameraSource("cam:csi:0")
    src.open()
    assert (time.monotonic_ns() - src.read().ref.t_capture_mono_ns) < 50_000_000


def test_read_raises_the_bus_error_without_waiting_for_the_silence_timeout(install):
    gst = fake_gst()
    pipe = install(FakePipe(gst), gst)
    src = CameraSource("cam:v4l2:0", silence_s=30.0)
    src.open()
    pipe.bus.errors.append("Device '/dev/video0' is gone")
    t0 = time.monotonic()
    with pytest.raises(SourceError, match="is gone"):
        src.read()
    assert time.monotonic() - t0 < 2


def test_end_of_stream_is_an_error_for_a_camera_not_the_end_of_the_run(install):
    gst = fake_gst()
    pipe = install(FakePipe(gst), gst)
    src = CameraSource("cam:test?frames=3")
    src.open()
    pipe.sink.eos = True
    with pytest.raises(SourceError, match="ended its stream"):
        src.read()


def test_csi_end_of_stream_points_at_the_sensor_and_the_daemon(install):
    pipe = install()
    src = CameraSource("cam:csi:0")
    src.open()
    pipe.sink.eos = True
    with pytest.raises(SourceError, match="no CSI sensor.*nvargus-daemon"):
        src.read()


def test_silent_camera_raises_after_the_timeout(install):
    install()
    src = CameraSource("cam:csi:0", silence_s=0.15)
    src.open()
    with pytest.raises(SourceError, match="silent"):
        src.read()


def test_read_before_open_and_close_are_safe():
    src = CameraSource("cam:csi:0")
    with pytest.raises(SourceError):
        src.read()
    src.close()
    src.close()


def test_sequence_keeps_growing_across_a_reconnection(install):
    gst = fake_gst()
    install(FakePipe(gst), gst, samples=[sample()])
    src = CameraSource("cam:csi:0")
    src.open()
    first = src.read()
    src.close()
    install(FakePipe(gst), gst, samples=[sample()])
    src.open()
    second = src.read()
    assert (first.ref.frame_seq, second.ref.frame_seq) == (0, 1) and first.ref.session_id == second.ref.session_id


# ---- through the pipeline --------------------------------------------------------------------------------
def build(manifest, source, **backend):
    sink = CollectSink()
    cfg = PipelineConfig(reconnect_initial_s=0.05, reconnect_factor=2.0, reconnect_max_s=1.0)
    return Pipeline(source, FakeBackend(manifest, **backend), manifest, [sink], cfg), sink


def test_missing_camera_is_retried_with_growing_waits_and_stops_promptly(rf_manifest):
    source = ScriptedSource([], live=True, open_failures=10**6)
    pipe, _ = build(rf_manifest, source)
    pipe.start()
    time.sleep(0.9)
    t0 = time.monotonic()
    pipe.stop()
    assert time.monotonic() - t0 < 1.0 and not any(t.is_alive() for t in pipe._threads)
    gaps = np.diff(source.open_times)
    assert len(gaps) >= 3 and gaps[1] > gaps[0] * 1.5  # the wait grows
    assert pipe.metrics.counters["reconnects"] >= 3 and pipe.metrics.counters["captured"] == 0


def test_camera_that_appears_late_is_picked_up(rf_manifest):
    source = ScriptedSource([make_frame(i) for i in range(5)], live=True, open_failures=3)
    pipe, sink = build(rf_manifest, source)
    pipe.start()
    deadline = time.monotonic() + 5
    while not sink.results and time.monotonic() < deadline:
        time.sleep(0.02)
    pipe.stop()
    assert sink.results and pipe.metrics.counters["reconnects"] == 3


def test_live_camera_faster_than_the_engine_keeps_only_the_newest_and_counts_the_drops(rf_manifest):
    source = ScriptedSource([make_frame(i) for i in range(200)], live=True, delay=0.002)
    pipe, sink = build(rf_manifest, source, latency_s=0.03)
    snap = pipe.run()
    seqs = [r.frame_ref.frame_seq for r in sink.results]
    assert seqs == sorted(set(seqs)) and len(seqs) < 200 and pipe.overwritten > 0
    assert len(seqs) + sum(r.dropped_before for r in sink.results) == seqs[-1] + 1  # every drop is accounted for
    assert snap["dropped"] > 50
