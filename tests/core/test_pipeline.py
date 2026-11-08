"""Pipeline: mailbox and queue semantics, ordering, drops by frame_seq, reconnection, errors and clean stop."""
from __future__ import annotations

import threading
import time

import numpy as np

from jetson_vision.backends.fake import FakeBackend
from jetson_vision.interfaces import SourceError
from jetson_vision.pipeline import END, EMPTY, BoundedChannel, LatestMailbox, Pipeline, PipelineConfig
from jetson_vision.postprocess import build_decoder
from jetson_vision.preprocess import Preprocessor
from jetson_vision.types import DegradationLevel

from .conftest import CollectSink, ScriptedSource, make_frame


# ---- mailbox and bounded channel ------------------------------------------------------------------------------
def test_mailbox_newest_wins_and_counts_the_overwritten():
    box = LatestMailbox()
    for i in (1, 2, 3):
        assert box.put(i)
    assert box.get(0.1) == 3 and box.overwritten == 2
    assert box.get(0.01) is EMPTY


def test_mailbox_delivers_the_last_item_before_end_and_abort_discards_it():
    box = LatestMailbox()
    box.put("last")
    box.finish()
    assert box.get(0.1) == "last" and box.get(0.1) is END and box.put("late") is False
    aborted = LatestMailbox()
    aborted.put("x")
    aborted.abort()
    assert aborted.get(0.1) is END


def test_mailbox_consumer_sees_strictly_increasing_items_under_contention():
    box, seen = LatestMailbox(), []

    def consume():
        while (item := box.get(0.05)) is not END:
            if item is not EMPTY:
                seen.append(item)

    t = threading.Thread(target=consume)
    t.start()
    for i in range(2000):
        box.put(i)
    box.finish()
    t.join(5)
    assert seen == sorted(seen) and len(set(seen)) == len(seen) and seen[-1] == 1999
    assert len(seen) + box.overwritten == 2000  # every item was either delivered or counted as replaced


def test_bounded_channel_blocks_when_full_and_keeps_fifo_order():
    ch, done = BoundedChannel(2), threading.Event()
    ch.put(0)
    ch.put(1)

    def producer():
        ch.put(2)  # blocks: the channel is full
        done.set()

    threading.Thread(target=producer, daemon=True).start()
    assert not done.wait(0.3)
    assert ch.get(0.1) == 0 and done.wait(2)
    assert [ch.get(0.1), ch.get(0.1)] == [1, 2]
    assert ch.get(0.01) is EMPTY


def test_bounded_channel_abort_releases_a_blocked_producer():
    ch, result = BoundedChannel(1), []
    ch.put("a")
    t = threading.Thread(target=lambda: result.append(ch.put("b")))
    t.start()
    time.sleep(0.2)
    ch.abort()
    t.join(2)
    assert result == [False]


# ---- helpers -------------------------------------------------------------------------------------------------
def build(manifest, source, sink=None, backend=None, **cfg):
    sink = sink or CollectSink()
    config = PipelineConfig(reconnect_initial_s=0.05, reconnect_max_s=0.4, **cfg)
    return Pipeline(source, backend or FakeBackend(manifest), manifest, [sink], config), sink


# ---- replay mode ---------------------------------------------------------------------------------------------
def test_replay_publishes_every_frame_in_order_with_no_drops(rf_manifest):
    frames = [make_frame(i, fill=i * 7) for i in range(30)]
    pipe, sink = build(rf_manifest, ScriptedSource(frames), queue_size=2)
    snap = pipe.run()
    assert [r.frame_ref.frame_seq for r in sink.results] == list(range(30))
    assert all(r.dropped_before == 0 for r in sink.results)
    assert pipe.overwritten == 0 and snap["published"] == 30 and snap["dropped"] == 0
    assert all(r.model_id == "rfdetr_exp07_qG_int8" and r.score_threshold == rf_manifest.score_threshold for r in sink.results)


def test_replay_results_equal_a_manual_run_of_the_same_stages(rf_manifest):
    frame = make_frame(3, w=200, h=100, fill=90)
    pipe, sink = build(rf_manifest, ScriptedSource([frame]), score_threshold=0.0)
    pipe.run()
    backend = FakeBackend(rf_manifest)
    backend.load()
    out = backend.infer(Preprocessor(rf_manifest.input)(frame.pixels))
    assert sink.results[0].detections == build_decoder(rf_manifest)(out, 200, 100, 0.0)


def test_replay_waits_for_a_slow_backend_instead_of_dropping(rf_manifest):
    pipe, sink = build(rf_manifest, ScriptedSource([make_frame(i) for i in range(12)]),
                       backend=FakeBackend(rf_manifest, latency_s=0.01), queue_size=1)
    pipe.run()
    assert len(sink.results) == 12 and pipe.overwritten == 0


def test_timings_and_metrics_are_filled_per_stage(rf_manifest):
    pipe, sink = build(rf_manifest, ScriptedSource([make_frame(i) for i in range(10)]),
                       backend=FakeBackend(rf_manifest, latency_s=0.005))
    snap = pipe.run()
    t = sink.results[-1].timings
    assert t.preprocess_ns > 0 and t.inference_ns >= 5_000_000 and t.postprocess_ns > 0 and t.tracking_ns == 0
    assert t.total_ns >= t.preprocess_ns + t.inference_ns + t.postprocess_ns
    assert snap["stages"]["inference"]["p50_ms"] >= 5 and snap["stages"]["total"]["p95_ms"] > 0
    assert snap["captured"] == snap["processed"] == snap["published"] == 10


def test_dropped_before_comes_from_jumps_in_frame_seq_and_resets_per_session(rf_manifest):
    frames = [make_frame(0), make_frame(1), make_frame(4), make_frame(5), make_frame(0, session="s2"),
              make_frame(3, session="s2")]
    pipe, sink = build(rf_manifest, ScriptedSource(frames))
    snap = pipe.run()
    assert [r.dropped_before for r in sink.results] == [0, 0, 2, 0, 0, 2]
    assert snap["dropped"] == 4


def test_tracker_hook_runs_after_decoding_and_is_timed(rf_manifest):
    pipe, sink = build(rf_manifest, ScriptedSource([make_frame(0)]), score_threshold=0.0)
    pipe.tracker = lambda dets: dets[:1]
    pipe.run()
    assert len(sink.results[0].detections) == 1 and sink.results[0].timings.tracking_ns > 0


def test_threshold_override_is_applied_and_reported(rf_manifest):
    low, sink_low = build(rf_manifest, ScriptedSource([make_frame(1)]), score_threshold=0.0)
    low.run()
    high, sink_high = build(rf_manifest, ScriptedSource([make_frame(1)]), score_threshold=0.9999)
    high.run()
    assert sink_low.results[0].score_threshold == 0.0 and sink_high.results[0].score_threshold == 0.9999
    assert len(sink_low.results[0].detections) > len(sink_high.results[0].detections) == 0


# ---- live mode -----------------------------------------------------------------------------------------------
def test_live_mode_drops_old_frames_and_accounts_for_every_one(rf_manifest):
    n = 80
    src = ScriptedSource([make_frame(i) for i in range(n)], live=True, delay=0.001)
    pipe, sink = build(rf_manifest, src, backend=FakeBackend(rf_manifest, latency_s=0.02))
    assert isinstance(pipe._chan, LatestMailbox)
    pipe.run()
    seqs = [r.frame_ref.frame_seq for r in sink.results]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert len(seqs) < n and seqs[-1] == n - 1  # the newest frame always gets through
    assert sum(r.dropped_before for r in sink.results) + len(seqs) == n  # jumps in frame_seq add up exactly
    assert pipe.overwritten == n - len(seqs)


def test_live_mode_latency_stays_bounded_with_a_slow_backend(rf_manifest):
    src = ScriptedSource([make_frame(i) for i in range(60)], live=True, delay=0.002)
    pipe, sink = build(rf_manifest, src, backend=FakeBackend(rf_manifest, latency_s=0.03))
    pipe.run()
    waits_ms = [(r.timings.total_ns - r.timings.preprocess_ns - r.timings.inference_ns
                 - r.timings.postprocess_ns) / 1e6 for r in sink.results]
    assert max(waits_ms) < 150  # without the mailbox this would grow with the backlog (60 x 30 ms)


# ---- reconnection --------------------------------------------------------------------------------------------
def test_source_error_triggers_reconnection_with_growing_wait(rf_manifest):
    script = [make_frame(0), make_frame(1), SourceError("lost"), make_frame(2), SourceError("lost"),
              SourceError("lost"), make_frame(3)]
    src = ScriptedSource(script)
    pipe, sink = build(rf_manifest, src)
    snap = pipe.run()
    assert [r.frame_ref.frame_seq for r in sink.results] == [0, 1, 2, 3]
    assert snap["reconnects"] == 3 and src.opens == 4 and src.closes >= 4
    gaps = np.diff(src.open_times)
    # 1st failure waits 0.05; frame 2 resets the wait, so the 2nd waits 0.05 again; the 3rd (consecutive) waits 0.1
    assert gaps[0] >= 0.04 and gaps[1] >= 0.04 and gaps[2] >= 0.09  # minus timer slack
    assert gaps[2] > gaps[1] * 1.3


def test_failing_open_is_retried_and_never_ends_the_process(rf_manifest):
    src = ScriptedSource([make_frame(0)], open_failures=3)
    pipe, sink = build(rf_manifest, src)
    snap = pipe.run()
    assert len(sink.results) == 1 and src.opens == 4 and snap["reconnects"] == 3


def test_backoff_grows_and_is_capped(rf_manifest):
    src = ScriptedSource([make_frame(0)], open_failures=6)
    pipe, _ = build(rf_manifest, src)
    pipe.run()
    gaps = np.diff(src.open_times)  # waits: 0.05, 0.1, 0.2, 0.4, 0.4, 0.4
    assert gaps[3] > gaps[0] * 4 and gaps.max() < 0.4 + 0.2


# ---- errors, stop and control --------------------------------------------------------------------------------
def test_backend_error_skips_the_frame_and_the_run_continues(rf_manifest):
    def hook(call):
        if call == 2:
            raise RuntimeError("cuda hiccup")

    pipe, sink = build(rf_manifest, ScriptedSource([make_frame(i) for i in range(6)]),
                       backend=FakeBackend(rf_manifest, hook=hook))
    snap = pipe.run()
    assert [r.frame_ref.frame_seq for r in sink.results] == [0, 1, 3, 4, 5]
    assert snap["backend_errors"] == 1 and pipe.error is None
    assert [r.dropped_before for r in sink.results][2] == 1  # the skipped frame shows up as a jump


def test_repeated_backend_failure_stops_the_pipeline_and_records_the_error(rf_manifest):
    def hook(call):
        raise RuntimeError("gpu lost")

    src = ScriptedSource([make_frame(i) for i in range(100)], delay=0.001)
    pipe, sink = build(rf_manifest, src, backend=FakeBackend(rf_manifest, hook=hook), max_consecutive_errors=5)
    pipe.run()
    assert isinstance(pipe.error, RuntimeError) and sink.results == []


def test_non_contiguous_frame_is_rejected_not_inferred(rf_manifest):
    bad = make_frame(0)
    bad.pixels = np.zeros((3, 48, 64), np.uint8).transpose(1, 2, 0)
    pipe, sink = build(rf_manifest, ScriptedSource([bad, make_frame(1)]))
    snap = pipe.run()
    assert [r.frame_ref.frame_seq for r in sink.results] == [1] and snap["backend_errors"] == 1


def test_a_failing_sink_does_not_stop_the_others(rf_manifest):
    class Broken(CollectSink):
        def write(self, result):
            raise OSError("disk full")

    good = CollectSink()
    pipe = Pipeline(ScriptedSource([make_frame(i) for i in range(5)]), FakeBackend(rf_manifest), rf_manifest,
                    [Broken(), good], PipelineConfig())
    snap = pipe.run()
    assert len(good.results) == 5 and snap["sink_errors"] == 5


def test_stop_is_clean_closes_source_and_backend_but_not_the_sinks(rf_manifest):
    src = ScriptedSource([make_frame(i) for i in range(100_000)], live=True, delay=0.002)
    backend = FakeBackend(rf_manifest, latency_s=0.005)
    pipe, sink = build(rf_manifest, src, backend=backend)
    pipe.start()
    time.sleep(0.3)
    t0 = time.monotonic()
    pipe.stop()
    assert time.monotonic() - t0 < 3 and pipe.join(0) is True
    assert src.closes >= 1 and backend.loaded is False and sink.closed is False
    assert len(sink.results) > 0
    pipe.stop()  # safe to call twice


def test_stop_while_replay_is_blocked_on_a_full_queue(rf_manifest):
    sink = CollectSink(delay=0.2)  # slow sink: back-pressure fills every queue
    src = ScriptedSource([make_frame(i) for i in range(1000)])
    pipe, _ = build(rf_manifest, src, sink=sink, queue_size=1, publish_queue_size=1)
    pipe.start()
    time.sleep(0.5)
    t0 = time.monotonic()
    pipe.stop(timeout=3)
    assert time.monotonic() - t0 < 6 and pipe.join(0) is True


def test_degradation_level_is_stamped_and_stopped_ends_the_run(rf_manifest):
    pipe, sink = build(rf_manifest, ScriptedSource([make_frame(i) for i in range(5)]))
    pipe.set_degradation(DegradationLevel.ALERTS_ONLY)
    pipe.run()
    assert {r.degradation_level for r in sink.results} == {DegradationLevel.ALERTS_ONLY}
    src = ScriptedSource([make_frame(i) for i in range(100_000)], live=True, delay=0.002)
    pipe2, _ = build(rf_manifest, src)
    pipe2.start()
    pipe2.set_degradation(DegradationLevel.STOPPED)
    assert pipe2.join(5)
    pipe2.stop()


def test_throttle_limits_the_processing_rate(rf_manifest):
    pipe, sink = build(rf_manifest, ScriptedSource([make_frame(i) for i in range(6)]))
    pipe.set_throttle(0.05)
    t0 = time.monotonic()
    pipe.run()
    assert time.monotonic() - t0 >= 0.25 and len(sink.results) == 6


def test_live_flag_comes_from_the_source_unless_configured(rf_manifest):
    assert Pipeline(ScriptedSource([], live=True), FakeBackend(rf_manifest), rf_manifest, []).live is True
    assert Pipeline(ScriptedSource([], live=False), FakeBackend(rf_manifest), rf_manifest, []).live is False
    forced = Pipeline(ScriptedSource([], live=False), FakeBackend(rf_manifest), rf_manifest, [],
                      PipelineConfig(live=True))
    assert isinstance(forced._chan, LatestMailbox)


# ---- single frame and queue waiting --------------------------------------------------------------------------
def test_process_frame_runs_one_frame_without_threads_or_source(rf_manifest):
    backend = FakeBackend(rf_manifest)
    pipe = Pipeline(None, backend, rf_manifest, [])
    r0 = pipe.process_frame(make_frame(0))
    r3 = pipe.process_frame(make_frame(3), queue_ns=7)
    assert r0.model_id == pipe.model_id == backend.model_id and r0.dropped_before == 0
    assert r3.dropped_before == 2 and r3.timings.queue_ns == 7  # frames 1 and 2 never arrived
    assert r0.timings.preprocess_ns > 0 and r0.timings.inference_ns > 0 and r0.timings.total_ns > 0
    assert not pipe._threads
    pipe.close()
    pipe.close()  # safe twice


def test_process_frame_and_threaded_run_give_the_same_detections(rf_manifest):
    frames = [make_frame(i) for i in range(6)]
    pipe, sink = build(rf_manifest, ScriptedSource(list(frames)))
    pipe.run()
    single = Pipeline(None, FakeBackend(rf_manifest), rf_manifest, [])
    assert [r.detections for r in sink.results] == [single.process_frame(f).detections for f in frames]


def test_start_without_source_is_refused(rf_manifest):
    import pytest
    with pytest.raises(RuntimeError):
        Pipeline(None, FakeBackend(rf_manifest), rf_manifest, []).start()


def test_replay_at_max_speed_reports_the_queue_wait_apart_from_the_service_time(rf_manifest):
    src = ScriptedSource([make_frame(i) for i in range(30)])
    pipe, sink = build(rf_manifest, src, backend=FakeBackend(rf_manifest, latency_s=0.01), queue_size=4)
    snap = pipe.run()
    last = sink.results[-1].timings
    assert last.queue_ns > 0 and last.total_ns >= last.queue_ns
    assert last.service_ns == last.total_ns - last.queue_ns
    # frames wait behind a full queue, so total is several service times while service stays near the work done
    q, s, t = (snap["stages"][k]["p50_ms"] for k in ("queue", "service", "total"))
    assert q > 10 and s < t
