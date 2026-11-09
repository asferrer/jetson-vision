"""Output sinks: JSON Lines for local logs and a compact binary summary for acoustic links.

The compact format (schema 1) targets links of 80 to 5400 bps (WHOI Micro-Modem class), where a
message must be tens of bytes. Two message types exist.

T0 heartbeat, 8 bytes:
    [0]   0x10 | type 0 (high nibble = format version)
    [1]   degradation level
    [2]   fps, rounded (0xFF = unknown)
    [3]   temperature in C as int8 (127 = unknown)
    [4:6] latency p95 in ms, uint16 big endian (0xFFFF = unknown)
    [6:8] model tag, crc32(model_id) & 0xFFFF

T1 event summary, 5 + 3 * classes (+ 4 with box) bytes:
    [0]   0x10 | type 1
    [1]   degradation level
    [2:4] frame_seq & 0xFFFF of the most relevant frame
    [4]   number of classes, bit 7 set when a box follows
    per class: class_id, count, best score * 255
    optional box: x1, y1, x2, y2 normalised to 0..255 (box of the best detection)
"""
from __future__ import annotations

import gzip
import json
import os
import struct
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Callable

from jetson_vision.interfaces import Sink
from jetson_vision.types import Detection, FrameResult, HealthState

COMPACT_VERSION = 1
LOCK_TIMEOUT_S = 1.0
T0_HEARTBEAT = 0
T1_EVENT = 1
_UNKNOWN_U8 = 0xFF
_UNKNOWN_I8 = 127
_UNKNOWN_U16 = 0xFFFF


def _json_default(obj: Any) -> Any:
    """Make numpy scalars and arrays JSON serialisable."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, default=_json_default, separators=(",", ":"), allow_nan=False)


def health_to_dict(state: HealthState) -> dict[str, Any]:
    return {
        "level": int(state.level),
        "reason": state.reason,
        "temperature_c": state.temperature_c,
        "latency_p95_ms": state.latency_p95_ms,
        "fps": state.fps,
    }


class JsonlSink:
    """One JSON object per line, flushed per record, with size-based rotation.

    A failing disk never raises into the pipeline: errors are counted in ``write_errors``.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        max_bytes: int = 0,
        backups: int = 3,
        compress: bool = False,
    ) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.backups = backups
        self.compress = compress
        self.write_errors = 0
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        # Own byte counter: tell() fails on FIFOs and other non-seekable files, and every write was then dropped
        self._size = self.path.stat().st_size if self.path.is_file() else 0

    def write(self, result: FrameResult) -> None:
        self._append({"record": "result", **result.to_dict()})

    def write_health(self, state: HealthState) -> None:
        self._append({"record": "health", **health_to_dict(state)})

    def write_event(self, kind: str, **fields: Any) -> None:
        """Free-form event, e.g. start with the restart cause."""
        self._append({"record": "event", "kind": kind, "t_wall_ns": time.time_ns(), **fields})

    def _append(self, record: dict[str, Any]) -> None:
        line = _dumps(record) + "\n"
        # A writer stuck in a blocked file (a FIFO nobody reads) holds the lock: others give up instead of
        # blocking too (the supervisor writes health from the main thread and must keep running).
        if not self._lock.acquire(timeout=LOCK_TIMEOUT_S):
            self.write_errors += 1
            return
        try:
            self._rotate_if_needed(len(line))
            self._fh.write(line)
            self._fh.flush()
            self._size += len(line.encode("utf-8"))
        except (OSError, ValueError):
            self.write_errors += 1
        finally:
            self._lock.release()

    def _rotate_if_needed(self, incoming: int) -> None:
        if self.max_bytes <= 0 or not self.path.is_file() or self._size + incoming <= self.max_bytes:
            return
        self._fh.close()
        self._shift_backups()
        self._fh = open(self.path, "a", encoding="utf-8")
        self._size = 0

    def _backup(self, n: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{n}" + (".gz" if self.compress else ""))

    def _shift_backups(self) -> None:
        if self.backups <= 0:
            self.path.unlink(missing_ok=True)
            return
        self._backup(self.backups).unlink(missing_ok=True)
        for n in range(self.backups - 1, 0, -1):
            if self._backup(n).exists():
                self._backup(n).replace(self._backup(n + 1))
        if self.compress:
            self._backup(1).write_bytes(gzip.compress(self.path.read_bytes()))
            self.path.unlink()
        else:
            self.path.replace(self._backup(1))

    def close(self) -> None:
        """Never raises: a full disk fails the final flush, and a write blocked forever must not block shutdown."""
        if not self._lock.acquire(timeout=LOCK_TIMEOUT_S):
            self.write_errors += 1  # a write is stuck holding the lock; leave the file to the OS
            return
        try:
            if not self._fh.closed:
                self._fh.close()
        except (OSError, ValueError):
            self.write_errors += 1
        finally:
            self._lock.release()


def model_tag(model_id: str) -> int:
    return zlib.crc32(model_id.encode("utf-8")) & 0xFFFF


def _clamp(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def encode_heartbeat(state: HealthState, model_id: str) -> bytes:
    """T0: 8 bytes with level, fps, temperature, latency p95 and model tag."""
    fps = _UNKNOWN_U8 if state.fps is None else _clamp(state.fps, 0, 254)
    temp = _UNKNOWN_I8 if state.temperature_c is None else _clamp(state.temperature_c, -128, 126)
    lat = _UNKNOWN_U16 if state.latency_p95_ms is None else _clamp(state.latency_p95_ms, 0, 65534)
    head = (COMPACT_VERSION << 4) | T0_HEARTBEAT
    return struct.pack(">BBBbHH", head, int(state.level), fps, temp, lat, model_tag(model_id))


def _check_head(head: int, expected_type: int) -> None:
    if head != (COMPACT_VERSION << 4) | expected_type:
        raise ValueError(f"unexpected compact header 0x{head:02x}")


def decode_heartbeat(data: bytes) -> dict[str, Any]:
    head, level, fps, temp, lat, tag = struct.unpack(">BBBbHH", data)
    _check_head(head, T0_HEARTBEAT)
    return {
        "level": level,
        "fps": None if fps == _UNKNOWN_U8 else fps,
        "temperature_c": None if temp == _UNKNOWN_I8 else temp,
        "latency_p95_ms": None if lat == _UNKNOWN_U16 else lat,
        "model_tag": tag,
    }


def _quantise_box(box: tuple[float, float, float, float], width: int, height: int) -> bytes:
    x1, y1, x2, y2 = box
    vals = (x1 / width, y1 / height, x2 / width, y2 / height)
    return bytes(_clamp(v * 255, 0, 255) for v in vals)


def encode_event(
    summary: dict[int, tuple[int, float]],
    level: int,
    frame_seq: int,
    best: Detection | None = None,
    frame_size: tuple[int, int] | None = None,
) -> bytes:
    """T1: per class (count, best score) and an optional quantised box of the best detection."""
    with_box = best is not None and frame_size is not None
    n = len(summary)
    if n > 127:
        raise ValueError("too many classes for the compact format")
    head = (COMPACT_VERSION << 4) | T1_EVENT
    out = bytearray(struct.pack(">BBHB", head, level, frame_seq & 0xFFFF,
                                n | (0x80 if with_box else 0)))
    for class_id in sorted(summary):
        count, score = summary[class_id]
        out += bytes((class_id & 0xFF, min(count, 255), _clamp(score * 255, 0, 255)))
    if with_box and best is not None and frame_size is not None:
        out += _quantise_box(best.box_xyxy, frame_size[0], frame_size[1])
    return bytes(out)


def decode_event(data: bytes) -> dict[str, Any]:
    head, level, seq, n_flag = struct.unpack(">BBHB", data[:5])
    _check_head(head, T1_EVENT)
    n = n_flag & 0x7F
    classes = {}
    for i in range(n):
        class_id, count, score = data[5 + 3 * i : 8 + 3 * i]
        classes[class_id] = {"count": count, "best_score": score / 255}
    box = tuple(b / 255 for b in data[5 + 3 * n : 9 + 3 * n]) if n_flag & 0x80 else None
    return {"level": level, "frame_seq": seq, "classes": classes, "box_norm": box}


class CompactSink:
    """Windowed summary for a low-rate link. Bytes go to ``emit``; the link driver is not here.

    Counts per class are the maximum seen in one frame inside the window (or distinct track ids
    when the tracker is on), so a static object is not counted once per frame.
    """

    def __init__(
        self,
        emit: Callable[[bytes], None],
        model_id: str,
        frame_size: tuple[int, int] | None = None,
        window_s: float = 5.0,
        heartbeat_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._emit = emit
        self._model_id = model_id
        self._frame_size = frame_size
        self._window_ns = int(window_s * 1e9)
        self._heartbeat_s = heartbeat_s
        self._clock = clock
        self._last_beat: float | None = None
        self._window_start: int | None = None
        self._reset_window()

    def _reset_window(self) -> None:
        self._max_count: dict[int, int] = {}
        self._tracks: dict[int, set[int]] = {}
        self._best_score: dict[int, float] = {}
        self._best: tuple[Detection, int] | None = None
        self._level = 0

    def write(self, result: FrameResult) -> None:
        t = result.frame_ref.t_capture_mono_ns
        if self._window_start is None:
            self._window_start = t
        elif t - self._window_start >= self._window_ns:
            self._flush()
            self._window_start = t
        self._accumulate(result)

    def _accumulate(self, result: FrameResult) -> None:
        self._level = max(self._level, int(result.degradation_level))
        per_frame: dict[int, int] = {}
        for det in result.detections:
            per_frame[det.class_id] = per_frame.get(det.class_id, 0) + 1
            if det.track_id is not None:
                self._tracks.setdefault(det.class_id, set()).add(det.track_id)
            if det.score > self._best_score.get(det.class_id, -1.0):
                self._best_score[det.class_id] = det.score
            if self._best is None or det.score > self._best[0].score:
                self._best = (det, result.frame_ref.frame_seq)
        for cid, n in per_frame.items():
            self._max_count[cid] = max(self._max_count.get(cid, 0), n)

    def _flush(self) -> None:
        if self._best is None:
            self._reset_window()
            return
        summary = {
            cid: (max(self._max_count[cid], len(self._tracks.get(cid, ()))), score)
            for cid, score in self._best_score.items()
        }
        best, seq = self._best
        self._emit(encode_event(summary, self._level, seq, best, self._frame_size))
        self._reset_window()

    def write_health(self, state: HealthState) -> None:
        now = self._clock()
        if self._last_beat is None or now - self._last_beat >= self._heartbeat_s:
            self._last_beat = now
            self._emit(encode_heartbeat(state, self._model_id))

    def close(self) -> None:
        self._flush()


class HexFileSink(CompactSink):
    """CompactSink that appends one hex line per message to a file (inspectable, replayable)."""

    def __init__(self, path: str | os.PathLike[str], model_id: str, **kwargs: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(p, "a", encoding="ascii")
        super().__init__(self._write_hex, model_id, **kwargs)

    def _write_hex(self, data: bytes) -> None:
        self._fh.write(data.hex() + "\n")
        self._fh.flush()

    def close(self) -> None:
        super().close()
        self._fh.close()


class MultiSink:
    """Fan out to several sinks; one failing sink does not stop the others."""

    def __init__(self, sinks: list[Sink]) -> None:
        self.sinks = sinks
        self.errors = 0

    def _each(self, fn: Callable[[Sink], None]) -> None:
        for sink in self.sinks:
            try:
                fn(sink)
            except Exception:  # noqa: BLE001 - a sink must never break the pipeline
                self.errors += 1

    def write(self, result: FrameResult) -> None:
        self._each(lambda s: s.write(result))

    def write_health(self, state: HealthState) -> None:
        self._each(lambda s: s.write_health(state))

    def close(self) -> None:
        self._each(lambda s: s.close())


class StdoutSink:
    def write(self, result: FrameResult) -> None:
        print(_dumps({"record": "result", **result.to_dict()}), flush=True)

    def write_health(self, state: HealthState) -> None:
        print(_dumps({"record": "health", **health_to_dict(state)}), flush=True)

    def close(self) -> None:
        pass


def make_sink(spec: str, model_id: str = "unknown", *, max_bytes: int = 0, backups: int = 3) -> Sink:
    """Build a sink from ``jsonl:<path>``, ``compact:<path>``, ``video:<path.mp4>[?options]`` or ``stdout``.

    max_bytes > 0 rotates a JSONL file at that size, keeping ``backups`` older files.
    """
    kind, _, arg = spec.partition(":")
    if kind == "jsonl" and arg:
        return JsonlSink(arg, max_bytes=max_bytes, backups=backups)
    if kind == "compact" and arg:
        return HexFileSink(arg, model_id)
    if kind == "stdout" and not arg:
        return StdoutSink()
    if kind == "video" and arg:
        from jetson_vision.video_sink import VideoSink, parse_video_spec

        path, opts = parse_video_spec(arg)
        return VideoSink(path, size=opts.get("size", (1280, 720)), fps=opts.get("fps", 30),
                         queue_size=opts.get("queue", 4), encoder=opts.get("encoder", "auto"))
    raise ValueError(f"invalid sink spec {spec!r}; use jsonl:<path>, compact:<path>, video:<path.mp4> or stdout")
