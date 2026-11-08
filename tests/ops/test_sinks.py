from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from jetson_vision.sinks import (
    CompactSink,
    JsonlSink,
    MultiSink,
    decode_event,
    decode_heartbeat,
    encode_event,
    encode_heartbeat,
    make_sink,
)
from jetson_vision.types import DegradationLevel, HealthState

from .conftest import make_det, make_result

LINK_MIN_BYTES_PER_S = 80 / 8  # lowest rate of the WHOI Micro-Modem range, in bytes per second


def read_lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_jsonl_roundtrip_and_numpy(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "a" / "out.jsonl")
    det = make_det(score=np.float32(0.75))  # type: ignore[arg-type]
    sink.write(make_result(3, [det]))
    sink.write_health(HealthState(DegradationLevel.REDUCED_FPS, "hot", 81.5, 40.0, 14.0))
    sink.write_event("start", restart_cause="watchdog")
    sink.close()
    rows = read_lines(tmp_path / "a" / "out.jsonl")
    assert [r["record"] for r in rows] == ["result", "health", "event"]
    assert rows[0]["frame_ref"]["frame_seq"] == 3
    assert rows[0]["detections"][0]["score"] == pytest.approx(0.75)
    assert rows[0]["schema_version"] == 2
    assert rows[1]["level"] == 1
    assert rows[2]["restart_cause"] == "watchdog"


def test_jsonl_rotation(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "o.jsonl", max_bytes=600, backups=2)
    for i in range(30):
        sink.write(make_result(i))
    sink.close()
    assert (tmp_path / "o.jsonl.1").exists() and (tmp_path / "o.jsonl.2").exists()
    assert not (tmp_path / "o.jsonl.3").exists()
    assert (tmp_path / "o.jsonl").stat().st_size <= 600
    assert sink.write_errors == 0


def test_jsonl_rotation_compressed(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "o.jsonl", max_bytes=500, backups=1, compress=True)
    for i in range(10):
        sink.write(make_result(i))
    sink.close()
    packed = tmp_path / "o.jsonl.1.gz"
    assert json.loads(gzip.decompress(packed.read_bytes()).splitlines()[0])["record"] == "result"


def test_jsonl_write_error_is_counted_not_raised(tmp_path: Path) -> None:
    """Simulated full disk: the pipeline must not see the exception."""
    sink = JsonlSink(tmp_path / "o.jsonl")

    class Full:
        def tell(self) -> int:
            return 0

        def write(self, _s: str) -> None:
            raise OSError(28, "No space left on device")

        def close(self) -> None:
            pass

        closed = False

    sink._fh = Full()  # type: ignore[assignment]
    sink.write(make_result())
    assert sink.write_errors == 1


def test_heartbeat_size_and_roundtrip() -> None:
    state = HealthState(DegradationLevel.ALERTS_ONLY, "x", 83.4, 52.6, 14.6)
    data = encode_heartbeat(state, "rfdetr_exp07")
    assert len(data) == 8
    out = decode_heartbeat(data)
    assert out == {"level": 2, "fps": 15, "temperature_c": 83, "latency_p95_ms": 53,
                   "model_tag": out["model_tag"]}


def test_heartbeat_unknowns_and_clamping() -> None:
    data = encode_heartbeat(HealthState(DegradationLevel.NORMAL, "", None, None, None), "m")
    assert decode_heartbeat(data)["temperature_c"] is None
    big = encode_heartbeat(HealthState(DegradationLevel.NORMAL, "", 500.0, 1e9, 1e9), "m")
    out = decode_heartbeat(big)
    assert out["temperature_c"] == 126 and out["latency_p95_ms"] == 65534 and out["fps"] == 254


def test_event_size_8_classes_with_box_is_tens_of_bytes() -> None:
    summary = {c: (3, 0.5 + c / 20) for c in range(8)}
    best = make_det(7, 0.85, (100.0, 50.0, 300.0, 250.0))
    data = encode_event(summary, 1, 70_000, best, (640, 480))
    assert len(data) == 5 + 3 * 8 + 4 == 33
    assert len(data) <= 48
    out = decode_event(data)
    assert out["frame_seq"] == 70_000 & 0xFFFF
    assert out["classes"][7]["count"] == 3
    assert out["classes"][7]["best_score"] == pytest.approx(0.85, abs=1 / 255)
    assert out["box_norm"] == pytest.approx((100 / 640, 50 / 480, 300 / 640, 250 / 480), abs=1 / 255)


def test_event_without_box_and_bad_header() -> None:
    data = encode_event({2: (1, 0.9)}, 0, 5)
    assert len(data) == 8 and decode_event(data)["box_norm"] is None
    with pytest.raises(ValueError):
        decode_event(b"\x99" + data[1:])
    with pytest.raises(ValueError):
        decode_heartbeat(bytes([0x11]) + bytes(7))


def test_message_sizes_fit_the_slowest_link_in_a_few_seconds() -> None:
    """At 80 bps (10 B/s) a full 8-class event must take well under a minute of raw airtime."""
    size = len(encode_event({c: (1, 0.9) for c in range(8)}, 0, 1, make_det(), (640, 480)))
    assert size / LINK_MIN_BYTES_PER_S < 5.0


def test_compact_sink_windows_counts_and_heartbeat(clock) -> None:
    sent: list[bytes] = []
    sink = CompactSink(sent.append, "m", frame_size=(640, 480), window_s=1.0,
                       heartbeat_s=10.0, clock=clock)
    state = HealthState(DegradationLevel.NORMAL, "ok", 50.0, 30.0, 15.0)
    sink.write_health(state)
    sink.write_health(state)  # inside the heartbeat period: not sent again
    clock.advance(11)
    sink.write_health(state)
    assert [len(m) for m in sent] == [8, 8]
    sent.clear()
    for i in range(5):  # 5 frames in the first window, same two objects of class 1 and one of class 3
        sink.write(make_result(i, [make_det(1, 0.6), make_det(1, 0.7), make_det(3, 0.9)],
                               t_ns=i * 100_000_000))
    assert sent == []
    sink.write(make_result(30, [], t_ns=1_500_000_000))  # closes the window
    assert len(sent) == 1
    event = decode_event(sent[0])
    assert event["classes"][1]["count"] == 2  # max per frame, not 5 frames * 2
    assert event["classes"][3]["best_score"] == pytest.approx(0.9, abs=1 / 255)


def test_compact_sink_counts_distinct_tracks_and_skips_empty_windows() -> None:
    sent: list[bytes] = []
    sink = CompactSink(sent.append, "m", window_s=1.0)
    sink.write(make_result(0, [make_det(1, 0.8, track_id=5)], t_ns=0))
    sink.write(make_result(1, [make_det(1, 0.8, track_id=6)], t_ns=100))
    sink.write(make_result(2, [], t_ns=2_000_000_000))  # flush window 1
    sink.write(make_result(3, [], t_ns=4_000_000_000))  # empty window: nothing sent
    sink.close()
    assert len(sent) == 1
    assert decode_event(sent[0])["classes"][1]["count"] == 2


def test_multisink_isolates_failures() -> None:
    class Bad:
        def write(self, r) -> None: raise RuntimeError("boom")
        def write_health(self, s) -> None: raise RuntimeError("boom")
        def close(self) -> None: raise RuntimeError("boom")

    got: list[int] = []

    class Good(Bad):
        def write(self, r) -> None: got.append(r.frame_ref.frame_seq)
        def write_health(self, s) -> None: got.append(-1)
        def close(self) -> None: got.append(-2)

    multi = MultiSink([Bad(), Good()])
    multi.write(make_result(4))
    multi.write_health(HealthState(DegradationLevel.NORMAL, "", None, None, None))
    multi.close()
    assert got == [4, -1, -2] and multi.errors == 3


def test_make_sink(tmp_path: Path, capsys) -> None:
    j = make_sink(f"jsonl:{tmp_path / 'x.jsonl'}")
    j.write(make_result())
    j.close()
    c = make_sink(f"compact:{tmp_path / 'x.hex'}", "m")
    c.write_health(HealthState(DegradationLevel.NORMAL, "", None, None, None))
    c.close()
    assert len(bytes.fromhex((tmp_path / "x.hex").read_text().strip())) == 8
    s = make_sink("stdout")
    s.write(make_result())
    s.write_health(HealthState(DegradationLevel.NORMAL, "", None, None, None))
    s.close()
    assert capsys.readouterr().out.count("\n") == 2
    for bad in ("jsonl:", "foo:bar", "stdout:x", ""):
        with pytest.raises(ValueError):
            make_sink(bad)


def test_jsonl_sink_writes_to_a_fifo_with_rotation_enabled(tmp_path) -> None:
    """tell() fails on a FIFO; with rotation on, every record used to be dropped silently."""
    import os
    import threading

    if not hasattr(os, "mkfifo"):
        return  # Windows
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    got: list[str] = []
    reader = threading.Thread(target=lambda: got.extend(open(fifo).read().splitlines()))
    reader.start()
    sink = JsonlSink(fifo, max_bytes=10_000)
    sink.write_event("hello")
    sink.close()
    reader.join(5)
    assert sink.write_errors == 0 and len(got) == 1 and json.loads(got[0])["kind"] == "hello"


def test_jsonl_sink_close_never_raises_on_a_full_disk() -> None:
    import sys

    if not sys.platform.startswith("linux"):
        return  # /dev/full is Linux only
    sink = JsonlSink("/dev/full")
    sink.write_event("x")
    sink.close()  # the final flush fails with ENOSPC
    assert sink.write_errors >= 1


def test_jsonl_sink_does_not_block_other_threads_while_a_write_is_stuck(tmp_path, monkeypatch) -> None:
    import threading
    import time

    from jetson_vision import sinks

    monkeypatch.setattr(sinks, "LOCK_TIMEOUT_S", 0.1)
    sink = JsonlSink(tmp_path / "o.jsonl")
    sink._lock.acquire()  # stands for a writer blocked inside write() on a FIFO
    t0 = time.monotonic()
    done = threading.Event()
    threading.Thread(target=lambda: (sink.write_event("health"), sink.close(), done.set()), daemon=True).start()
    assert done.wait(3) and time.monotonic() - t0 < 2
    assert sink.write_errors == 2
    sink._lock.release()
