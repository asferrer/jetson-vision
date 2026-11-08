from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

from jetson_vision.health import (
    HealthConfig,
    HealthMonitor,
    LatencyBaseline,
    SystemdWatchdog,
    banded_level,
    read_temperature_c,
)
from jetson_vision.types import DegradationLevel as L

# Illustrative baselines for the tests only; they are not measurements of any device.
BASE = LatencyBaseline({30.0: 20.0, 15.0: 30.0})
CFG = HealthConfig(window=30, min_samples=10, hold_s=30.0)


class Temp:
    def __init__(self, value: float | None = 40.0) -> None:
        self.value = value

    def __call__(self) -> float | None:
        return self.value


def make_monitor(clock, temp: Temp, fps: float = 30.0) -> HealthMonitor:
    mon = HealthMonitor(CFG, BASE, temp, clock)
    mon.set_target_fps(fps)
    return mon


def feed(mon: HealthMonitor, ms: float, n: int = 30) -> None:
    for _ in range(n):
        mon.observe_latency_ms(ms)


def test_read_temperature_picks_hottest_valid_zone(tmp_path: Path) -> None:
    for name, raw in (("thermal_zone0", "45000"), ("thermal_zone1", "61500"),
                      ("thermal_zone2", "garbage"), ("thermal_zone3", "-270000")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "temp").write_text(raw)
    assert read_temperature_c(tmp_path) == 61.5
    assert read_temperature_c(tmp_path / "missing") is None


def test_banded_level_hysteresis() -> None:
    th = (80.0, 85.0, 90.0)
    assert banded_level(86.0, th, 3.0, 0) == 2
    assert banded_level(84.0, th, 3.0, 2) == 2  # inside the band: held
    assert banded_level(81.0, th, 3.0, 2) == 1
    assert banded_level(None, th, 3.0, 2) == 2


def test_baseline_depends_on_cadence() -> None:
    assert BASE.for_fps(30.0) == 20.0
    assert BASE.for_fps(15.0) == 30.0
    assert BASE.for_fps(29.0) == 20.0
    assert BASE.for_fps(22.0) is None  # between baselines, too far from both
    assert LatencyBaseline().for_fps(30.0) is None


def test_normal_operation_stays_level_0(clock) -> None:
    mon = make_monitor(clock, Temp(50.0))
    feed(mon, 21.0)
    state = mon.evaluate()
    assert state.level == L.NORMAL and state.reason == "ok"
    assert state.temperature_c == 50.0 and state.latency_p95_ms == 21.0


def test_low_cadence_latency_is_not_throttling(clock) -> None:
    """At 15 fps latency is higher than at 30 (dynamic governor). With the 15 fps baseline it is normal."""
    mon = make_monitor(clock, Temp(50.0), fps=15.0)
    feed(mon, 31.0)
    assert mon.drift() == pytest.approx(31 / 30 - 1)
    assert mon.evaluate().level == L.NORMAL


def test_same_latency_at_30fps_is_drift(clock) -> None:
    mon = make_monitor(clock, Temp(50.0), fps=30.0)
    feed(mon, 31.0)  # +55 % over the 30 fps baseline
    state = mon.evaluate()
    assert state.level == L.REDUCED_FPS and "latency drift" in state.reason


def test_escalation_is_immediate_to_desired(clock) -> None:
    mon = make_monitor(clock, Temp(50.0))
    feed(mon, 40.0)
    assert mon.evaluate().level == L.ALERTS_ONLY
    feed(mon, 20.0)  # recovered
    assert mon.evaluate().level == L.ALERTS_ONLY  # hold time not yet elapsed
    clock.advance(31)
    assert mon.evaluate().level == L.REDUCED_FPS  # one step down at a time
    assert mon.evaluate().level == L.REDUCED_FPS  # hold restarts after the change
    clock.advance(31)
    assert mon.evaluate().level == L.NORMAL


def test_high_temperature_ladder_and_hysteresis(clock) -> None:
    temp = Temp(40.0)
    mon = make_monitor(clock, temp)
    temp.value = 81.0
    assert mon.evaluate().level == L.REDUCED_FPS
    temp.value = 86.0
    assert mon.evaluate().level == L.ALERTS_ONLY
    temp.value = 95.0
    state = mon.evaluate()
    assert state.level == L.STOPPED and "temperature" in state.reason
    temp.value = 88.0  # inside the hysteresis band of level 3 (90 - 3 = 87)
    clock.advance(100)
    assert mon.evaluate().level == L.STOPPED
    temp.value = 40.0
    levels = []
    for _ in range(4):
        clock.advance(31)
        levels.append(mon.evaluate().level)
    assert levels == [L.ALERTS_ONLY, L.REDUCED_FPS, L.NORMAL, L.NORMAL]


def test_no_flapping_around_a_threshold(clock) -> None:
    temp = Temp(81.0)
    mon = make_monitor(clock, temp)
    assert mon.evaluate().level == L.REDUCED_FPS
    seen = set()
    for i in range(40):
        temp.value = 81.0 if i % 2 else 79.0  # inside the band of level 1 (80 - 83)
        clock.advance(5)
        seen.add(mon.evaluate().level)
    assert seen == {L.REDUCED_FPS}


def test_unknown_drift_holds_level_instead_of_releasing(clock) -> None:
    """After halving the cadence there may be no baseline for it: do not conclude 'recovered'."""
    mon = make_monitor(clock, Temp(50.0), fps=30.0)
    feed(mon, 31.0)
    assert mon.evaluate().level == L.REDUCED_FPS
    mon.set_target_fps(7.0)  # no baseline within tolerance
    feed(mon, 25.0)
    clock.advance(500)
    assert mon.drift() is None
    assert mon.evaluate().level == L.REDUCED_FPS


def test_cadence_change_clears_the_window(clock) -> None:
    mon = make_monitor(clock, Temp(50.0), fps=30.0)
    feed(mon, 40.0)
    mon.set_target_fps(15.0)
    assert mon.drift() is None  # not enough samples at the new cadence
    mon.set_target_fps(15.0)  # same value: no clearing
    feed(mon, 31.0, n=12)
    assert mon.drift() is not None


def test_fps_factor_and_never_touches_nvpmodel(clock, monkeypatch) -> None:
    import subprocess
    calls: list = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: calls.append(a))
    mon = make_monitor(clock, Temp(81.0))
    mon.observe_fps(14.0)
    state = mon.evaluate()
    assert mon.fps_factor() == 0.5 and state.fps == 14.0
    import jetson_vision.health as h
    src = Path(h.__file__).read_text()
    assert "subprocess" not in src and "nvpmodel" in src  # mentioned only in docs, never run
    assert calls == []


def test_watchdog_heartbeat_requires_recent_results(clock) -> None:
    sent: list[bytes] = []
    wd = SystemdWatchdog(max_result_age_s=5.0, notify=sent.append, interval_s=1.0, clock=clock)
    wd.ready()
    assert wd.beat() is False  # nothing produced yet
    wd.result_seen()
    clock.advance(4)
    assert wd.beat() is True
    clock.advance(2)  # 6 s without results: backend hung
    assert wd.beat() is False
    assert sent == [b"READY=1", b"WATCHDOG=1"]


def test_watchdog_no_source_counts_as_alive_until_result(clock) -> None:
    sent: list[bytes] = []
    wd = SystemdWatchdog(max_result_age_s=5.0, notify=sent.append, clock=clock)
    wd.result_seen()
    clock.advance(60)
    assert wd.beat() is False
    wd.set_no_source(True)  # camera lost, reconnecting: controlled
    assert wd.beat() is True
    wd.set_no_source(False)
    assert wd.beat() is False


def test_watchdog_thread_beats_and_stops() -> None:
    import time
    sent: list[bytes] = []
    wd = SystemdWatchdog(max_result_age_s=5.0, notify=sent.append, interval_s=0.01)
    wd.result_seen()
    wd.start()
    wd.start()  # idempotent
    time.sleep(0.15)
    wd.stop()
    assert sent.count(b"WATCHDOG=1") >= 2
    n = len(sent)
    time.sleep(0.05)
    assert len(sent) == n


def test_watchdog_interval_from_env(monkeypatch) -> None:
    monkeypatch.setenv("WATCHDOG_USEC", "20000000")
    assert SystemdWatchdog(notify=lambda _m: None)._interval == 10.0
    monkeypatch.delenv("WATCHDOG_USEC")
    assert SystemdWatchdog(notify=lambda _m: None)._interval is None


@pytest.mark.skipif(sys.platform == "win32", reason="AF_UNIX datagram sockets need Linux")
def test_watchdog_real_notify_socket(tmp_path: Path, monkeypatch) -> None:
    path = str(tmp_path / "notify.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(path)
    server.settimeout(2)
    monkeypatch.setenv("NOTIFY_SOCKET", path)
    wd = SystemdWatchdog()
    wd.ready()
    assert server.recv(64) == b"READY=1"
    server.close()
    wd.ready()  # peer gone: must not raise


def test_read_temperature_survives_a_zone_that_answers_eagain(tmp_path: Path, monkeypatch) -> None:
    """Found on the Jetson: a zone answering EAGAIN made Path.read_text raise TypeError and killed `jv run`."""
    import os

    for i, milli in enumerate((90000, 52000)):
        zone = tmp_path / f"thermal_zone{i}"
        zone.mkdir()
        (zone / "temp").write_text(str(milli))
    real_read, calls = os.read, []

    def flaky(fd: int, n: int) -> bytes:
        calls.append(fd)
        if len(calls) == 1:
            raise BlockingIOError(11, "try again")
        return real_read(fd, n)

    monkeypatch.setattr(os, "read", flaky)
    assert read_temperature_c(tmp_path) == 52.0  # the zone that failed is skipped, the other one is read
