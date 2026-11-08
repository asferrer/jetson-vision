"""Simulated failures against the real pipeline: camera drop, hung backend, hot board, latency drift."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from jetson_vision import cli
from jetson_vision.backends.fake import FakeBackend
from jetson_vision.health import HealthConfig, HealthMonitor, LatencyBaseline, SystemdWatchdog
from jetson_vision.interfaces import SourceError
from jetson_vision.manifest import Manifest
from jetson_vision.pipeline import Pipeline, PipelineConfig
from jetson_vision.types import DegradationLevel as L
from jetson_vision.types import Frame

from .conftest import make_frame

MANIFEST = Manifest.load(Path(__file__).resolve().parents[2] / "models" / "rfdetr_int8g.yaml")
FAST = PipelineConfig(reconnect_initial_s=0.01, reconnect_max_s=0.02)


class ScriptedSource:
    """Plays a script: 'f' = a frame, 'e' = SourceError, 'end' = finish, 'hold' = stay silent."""

    live = False
    source_id = "scripted"

    def __init__(self, script: list[str], live: bool = False) -> None:
        self.script, self.live = list(script), live
        self.opens = 0
        self.seq = 0
        self.release = threading.Event()

    def open(self) -> None:
        self.opens += 1

    def read(self) -> Frame | None:
        step = self.script.pop(0) if self.script else "end"
        if step == "e":
            raise SourceError("camera unplugged")
        if step == "hold":
            self.release.wait(5.0)
            return None
        if step == "end":
            return None
        if self.live:
            time.sleep(0.005)  # a real camera is paced; a tight loop would starve the GIL
        frame = make_frame(self.seq, 48, 64)
        self.seq += 1
        return frame

    def close(self) -> None:
        pass


class Sinkhole:
    def __init__(self) -> None:
        self.results: list = []

    def write(self, r) -> None:
        self.results.append(r)

    def write_health(self, s) -> None:
        pass

    def close(self) -> None:
        pass


def spin(seconds: float) -> None:
    """Busy wait: time.sleep is too coarse on some platforms for millisecond delays."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def wait_for(cond, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_camera_drop_reconnects_without_publishing_empty_results() -> None:
    src = ScriptedSource(["f", "f", "e", "e", "f", "f"])
    sink = Sinkhole()
    pipe = Pipeline(src, FakeBackend(MANIFEST), MANIFEST, [sink], FAST)
    snap = pipe.run()
    assert snap["reconnects"] == 2 and src.opens == 3
    # one result per real frame; the outage produced nothing (no fake "no debris" frames)
    assert len(sink.results) == 4
    assert [r.frame_ref.frame_seq for r in sink.results] == [0, 1, 2, 3]


def test_camera_gone_declares_no_source_and_keeps_watchdog_alive() -> None:
    """A controlled outage must not trigger a restart loop; the heartbeat keeps going."""
    src = ScriptedSource(["f"] + ["e"] * 200, live=True)
    pipe = Pipeline(src, FakeBackend(MANIFEST), MANIFEST, [Sinkhole()], FAST)
    sent: list[bytes] = []
    wd = SystemdWatchdog(max_result_age_s=0.05, notify=sent.append)
    monitor = HealthMonitor(thermal_reader=lambda: 40.0)
    sup = cli.Supervisor(pipe, monitor, wd, [], 30.0)
    pipe.start()
    wd.result_seen()
    assert wait_for(lambda: pipe.metrics.snapshot()["reconnects"] >= 3)
    sup.tick()  # first tick records the captured counter
    time.sleep(0.1)  # results are now older than max_result_age_s
    sup.tick()
    assert wd.beat() is True
    pipe.stop()


def test_backend_hang_stops_the_heartbeat_so_systemd_restarts() -> None:
    gate = threading.Event()
    backend = FakeBackend(MANIFEST, hook=lambda i: gate.wait(10.0) if i == 3 else None)
    src = ScriptedSource(["f"] * 50, live=False)
    sink = Sinkhole()
    pipe = Pipeline(src, backend, MANIFEST, [sink], FAST)
    sent: list[bytes] = []
    wd = SystemdWatchdog(max_result_age_s=0.2, notify=sent.append)
    ops = cli.OpsSink(None, wd)
    pipe.sinks.append(ops)
    pipe.start()
    wd.ready()
    assert wait_for(lambda: len(sink.results) >= 3)
    assert wd.beat() is True  # healthy so far
    time.sleep(0.4)  # the 4th inference is stuck: no results for longer than the limit
    assert len(sink.results) == 3
    assert wd.beat() is False  # the heartbeat is withheld, systemd would restart the service
    gate.set()
    pipe.stop()


def test_high_temperature_ends_the_run_with_exit_code_3(monkeypatch) -> None:
    monkeypatch.setattr(cli, "TICK_S", 0.05)
    src = ScriptedSource(["f"] * 100000, live=True)
    pipe = Pipeline(src, FakeBackend(MANIFEST, latency_s=0.001), MANIFEST, [Sinkhole()], FAST)
    monitor = HealthMonitor(thermal_reader=lambda: 99.0)  # above the level-3 threshold
    sup = cli.Supervisor(pipe, monitor, None, [], 30.0)
    assert cli._run_loop(pipe, sup, None) == cli.EXIT_STOPPED
    assert monitor.level == L.STOPPED


def test_high_temperature_ladder_is_applied_to_the_pipeline() -> None:
    temp = {"v": 40.0}
    monitor = HealthMonitor(thermal_reader=lambda: temp["v"])
    src = ScriptedSource(["f"] * 10, live=False)
    pipe = Pipeline(src, FakeBackend(MANIFEST), MANIFEST, [], FAST)
    sup = cli.Supervisor(pipe, monitor, None, [], 30.0)
    sup.tick()
    assert pipe._throttle_s == 0.0
    temp["v"] = 82.0
    sup.tick()
    assert pipe._level == L.REDUCED_FPS and pipe._throttle_s == pytest.approx(1 / 15)
    temp["v"] = 86.0
    sup.tick()
    assert pipe._level == L.ALERTS_ONLY and pipe._throttle_s == pytest.approx(1.0)


def test_latency_drift_is_detected_without_any_temperature_signal() -> None:
    """The hardware may throttle silently: latency alone raises the level."""
    slow = {"on": False}
    backend = FakeBackend(MANIFEST, hook=lambda i: spin(0.04 if slow["on"] else 0.001))
    cfg = HealthConfig(window=10, min_samples=5, hold_s=60.0)
    monitor = HealthMonitor(cfg, LatencyBaseline({30.0: 8.0}), thermal_reader=lambda: 45.0)
    src = ScriptedSource(["f"] * 100000, live=True)
    pipe = Pipeline(src, backend, MANIFEST, [], FAST)
    pipe.sinks.append(cli.OpsSink(monitor, None))
    sup = cli.Supervisor(pipe, monitor, None, [], 30.0)
    pipe.start()
    try:
        assert wait_for(lambda: len(monitor._lat) >= 5)
        sup.tick()
        assert monitor.level == L.NORMAL
        slow["on"] = True  # 10x slower inference, temperature unchanged
        assert wait_for(lambda: (sup.tick() or True) and monitor.level >= L.REDUCED_FPS, 10.0)
        assert pipe._level >= L.REDUCED_FPS
    finally:
        pipe.stop()


def test_low_cadence_latency_does_not_trigger_drift() -> None:
    """Same slower inference as above but the baseline for that (low) cadence says it is normal."""
    cfg = HealthConfig(window=10, min_samples=5)
    monitor = HealthMonitor(cfg, LatencyBaseline({30.0: 2.0, 15.0: 20.0}), lambda: 45.0)
    monitor.set_target_fps(15.0)
    for _ in range(10):
        monitor.observe_latency_ms(21.0)
    assert monitor.evaluate().level == L.NORMAL
    monitor.set_target_fps(30.0)
    for _ in range(10):
        monitor.observe_latency_ms(21.0)
    assert monitor.evaluate().level >= L.REDUCED_FPS


def test_run_loop_returns_after_a_stop_request_even_if_a_sink_is_blocked_forever() -> None:
    """Found on the Jetson: with a sink stuck in write() `jv run` never exited (the loop waited for every thread)."""
    release = threading.Event()

    class StuckSink:
        def write(self, result) -> None:
            release.wait(30)

        def write_health(self, state) -> None:
            pass

        def close(self) -> None:
            pass

    src = ScriptedSource(["f"] * 50, live=True)
    pipe = Pipeline(src, FakeBackend(MANIFEST), MANIFEST, [StuckSink()], FAST)
    monitor = HealthMonitor()
    sup = cli.Supervisor(pipe, monitor, None, [], None)
    threading.Timer(0.3, pipe.request_stop).start()
    done = threading.Event()
    threading.Thread(target=lambda: (cli._run_loop(pipe, sup, None), done.set()), daemon=True).start()
    try:
        assert done.wait(20), "run loop did not return"
    finally:
        release.set()
