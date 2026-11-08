"""Health: temperature, latency drift against a baseline, degradation ladder, systemd watchdog.

Design rules (docs/DISENO.md, decision 7):
* Levels: 0 normal, 1 lower FPS, 2 alerts only, 3 stop. ``nvpmodel`` is never touched at runtime.
* Hardware throttling is not always reported to the OS, so it is detected by its effect: the
  rolling median of inference latency drifts above a baseline.
* The baseline depends on the input cadence. At low cadence the dynamic GPU governor lowers the
  clock and latency rises (02_detector.md, section 3), which is not throttling. Baselines are
  therefore keyed by cadence, and when the current cadence has no baseline close enough the drift
  is "unknown" instead of guessed.
* Escalation is immediate; de-escalation is one level at a time after a minimum hold time, and
  each signal has its own hysteresis band, so the level does not flap.

Threshold defaults are provisional configuration, not measurements: they must be set from the
sealed enclosure (04_arquitectura.md, 4.2).
"""
from __future__ import annotations

import math
import os
import socket
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from jetson_vision.types import DegradationLevel, HealthState

THERMAL_ROOT = Path("/sys/class/thermal")


def _read_milli_c(path: Path) -> int | None:
    """One thermal zone in millidegrees, or None if it cannot be read right now.

    Read with os.read: on the Jetson a zone sometimes answers EAGAIN, and ``Path.read_text`` then dies with
    ``TypeError: can't concat NoneType to bytes`` (found on the real board), which killed ``jv run``.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            return int(os.read(fd, 32).strip())
        finally:
            os.close(fd)
    except (OSError, ValueError):
        return None


def read_temperature_c(root: Path = THERMAL_ROOT) -> float | None:
    """Hottest thermal zone in Celsius, or None if nothing is readable (e.g. not a Jetson)."""
    best: float | None = None
    for temp_file in sorted(root.glob("thermal_zone*/temp")):
        milli = _read_milli_c(temp_file)
        if milli is None:
            continue
        value = milli / 1000.0
        if -40.0 < value < 150.0 and (best is None or value > best):
            best = value
    return best


@dataclass(frozen=True)
class LatencyBaseline:
    """Baseline inference latency (ms) per input cadence (fps), measured on the target at install."""

    by_fps: dict[float, float] = field(default_factory=dict)
    tolerance: float = 0.25

    def for_fps(self, fps: float) -> float | None:
        """Baseline of the nearest cadence, only if within ``tolerance`` (relative)."""
        if not self.by_fps or fps <= 0:
            return None
        nearest = min(self.by_fps, key=lambda f: abs(f - fps))
        if abs(nearest - fps) / nearest > self.tolerance:
            return None
        return self.by_fps[nearest]


@dataclass(frozen=True)
class HealthConfig:
    # Temperature thresholds (C) that enter levels 1, 2 and 3. Provisional, set from the real enclosure.
    temp_thresholds_c: tuple[float, float, float] = (80.0, 85.0, 90.0)
    temp_hysteresis_c: float = 3.0
    # Drift = median latency / baseline - 1. Enter level 1 and 2.
    drift_thresholds: tuple[float, float] = (0.25, 0.75)
    drift_hysteresis: float = 0.10
    window: int = 60
    min_samples: int = 20
    hold_s: float = 30.0
    fps_factor_reduced: float = 0.5


def banded_level(value: float | None, thresholds: Sequence[float], hysteresis: float,
                 current: int) -> int:
    """Count of thresholds exceeded, holding ``current`` while inside the hysteresis band."""
    if value is None:
        return current
    up = sum(value >= t for t in thresholds)
    down = sum(value + hysteresis >= t for t in thresholds)
    return max(up, min(current, down))


class HealthMonitor:
    """Turns temperature and latency observations into a degradation level."""

    def __init__(
        self,
        config: HealthConfig | None = None,
        baseline: LatencyBaseline | None = None,
        thermal_reader: Callable[[], float | None] = read_temperature_c,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = config or HealthConfig()
        self.baseline = baseline or LatencyBaseline()
        self._thermal = thermal_reader
        self._clock = clock
        self._lat: deque[float] = deque(maxlen=self.cfg.window)
        self._target_fps = 0.0
        self._fps: float | None = None
        self._temp_level = 0
        self._drift_level = 0
        self._level = DegradationLevel.NORMAL
        self._changed_at = clock()
        self._reason = "ok"
        self._lock = threading.Lock()

    @property
    def level(self) -> DegradationLevel:
        return self._level

    def set_target_fps(self, fps: float) -> None:
        """Commanded input cadence. A change invalidates the latency window (different regime)."""
        with self._lock:
            if abs(fps - self._target_fps) > 1e-9:
                self._lat.clear()
            self._target_fps = fps

    def observe_fps(self, fps: float) -> None:
        self._fps = fps

    def observe_latency_ms(self, ms: float) -> None:
        with self._lock:
            self._lat.append(ms)

    def fps_factor(self) -> float:
        """Multiplier the pipeline should apply to the input cadence for the current level."""
        return self.cfg.fps_factor_reduced if self._level == DegradationLevel.REDUCED_FPS else 1.0

    def drift(self) -> float | None:
        """Median latency over the baseline for the current cadence, minus 1. None if unknown."""
        if len(self._lat) < self.cfg.min_samples:
            return None
        base = self.baseline.for_fps(self._target_fps)
        if base is None or base <= 0:
            return None
        return statistics.median(self._lat) / base - 1.0

    def _p95(self) -> float | None:
        if not self._lat:
            return None
        ordered = sorted(self._lat)
        return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]

    def evaluate(self) -> HealthState:
        """Recompute the level. Call periodically (about once per second)."""
        with self._lock:
            temp = self._thermal()
            reasons: list[str] = []
            self._temp_level = banded_level(temp, self.cfg.temp_thresholds_c,
                                            self.cfg.temp_hysteresis_c, self._temp_level)
            self._drift_level = banded_level(self.drift(), self.cfg.drift_thresholds,
                                             self.cfg.drift_hysteresis, self._drift_level)
            if self._temp_level:
                reasons.append(f"temperature level {self._temp_level}")
            if self._drift_level:
                reasons.append(f"latency drift level {self._drift_level}")
            self._apply(max(self._temp_level, self._drift_level))
            self._reason = "; ".join(reasons) or "ok"
            return HealthState(self._level, self._reason, temp, self._p95(), self._fps)

    def _apply(self, desired: int) -> None:
        now = self._clock()
        if desired > self._level:
            self._set(desired, now)
        elif desired < self._level and now - self._changed_at >= self.cfg.hold_s:
            self._set(self._level - 1, now)

    def _set(self, level: int, now: float) -> None:
        self._level = DegradationLevel(level)
        self._changed_at = now


class SystemdWatchdog:
    """sd_notify without dependencies. The heartbeat is sent only if results are recent.

    A stuck backend or camera stops the results, so the heartbeat stops and systemd restarts the
    service. A declared "no source" state (camera lost, reconnecting) also counts as alive, so a
    controlled outage is not a restart loop; a stall with a source present is.
    """

    def __init__(
        self,
        max_result_age_s: float = 10.0,
        notify: Callable[[bytes], None] | None = None,
        interval_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_age = max_result_age_s
        self._notify = notify or _make_notifier(os.environ.get("NOTIFY_SOCKET"))
        self._interval = interval_s if interval_s is not None else _watchdog_interval()
        self._clock = clock
        self._last_result: float | None = None
        self._no_source = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def ready(self) -> None:
        self._notify(b"READY=1")

    def result_seen(self) -> None:
        self._last_result = self._clock()
        self._no_source = False

    def set_no_source(self, value: bool) -> None:
        self._no_source = value

    def is_alive(self) -> bool:
        if self._no_source:
            return True
        return self._last_result is not None and self._clock() - self._last_result <= self._max_age

    def beat(self) -> bool:
        """Send WATCHDOG=1 if alive. Returns whether it was sent."""
        if not self.is_alive():
            return False
        self._notify(b"WATCHDOG=1")
        return True

    def start(self) -> None:
        if self._interval is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="watchdog", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._interval is not None
        while not self._stop.wait(self._interval):
            self.beat()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None


def _watchdog_interval() -> float | None:
    """Half the systemd watchdog period (WATCHDOG_USEC), or None if systemd set none."""
    raw = os.environ.get("WATCHDOG_USEC")
    if not raw or not raw.isdigit() or int(raw) <= 0:
        return None
    return int(raw) / 1e6 / 2.0


def _make_notifier(path: str | None) -> Callable[[bytes], None]:
    if not path or not hasattr(socket, "AF_UNIX"):
        return lambda _msg: None
    address = "\0" + path[1:] if path.startswith("@") else path

    def send(msg: bytes) -> None:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
                sock.sendto(msg, address)
        except OSError:
            pass

    return send
