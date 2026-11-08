"""Staged pipeline: overlap of preprocess/inference/postprocess, order, back-pressure, drop policy, errors, shutdown."""
from __future__ import annotations

import random
import threading
import time

from jetson_vision.backends.fake import FakeBackend
from jetson_vision.pipeline import Pipeline, PipelineConfig
from jetson_vision.postprocess import build_decoder
from jetson_vision.preprocess import Preprocessor

from .conftest import CollectSink, ScriptedSource, make_frame

STAGE_THREADS = ("jv-capture", "jv-preprocess", "jv-inference", "jv-postprocess", "jv-publish")


def stage_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("jv-")]


class CountingPre:
    """Preprocessor wrapper: counts calls, may sleep, may fail on chosen calls."""

    def __init__(self, manifest, sleep=0.0, fail_on=()):
        self.inner, self.sleep, self.fail_on, self.calls = Preprocessor(manifest.input), sleep, set(fail_on), 0

    def __call__(self, bgr):
        n, self.calls = self.calls, self.calls + 1
        time.sleep(self.sleep)
        if n in self.fail_on:
            raise ValueError(f"preprocess failed on call {n}")
        return self.inner(bgr)


class CountingDec:
    def __init__(self, manifest, sleep=0.0, fail_on=(), fail_all=False):
        self.inner, self.sleep, self.fail_on, self.fail_all = build_decoder(manifest), sleep, set(fail_on), fail_all
        self.calls = 0

    def __call__(self, outputs, w, h, thr):
        n, self.calls = self.calls, self.calls + 1
        time.sleep(self.sleep)
        if self.fail_all or n in self.fail_on:
            raise ValueError(f"decode failed on call {n}")
        return self.inner(outputs, w, h, thr)


def build(manifest, source, *, sink=None, backend=None, pre=None, dec=None, **cfg):
    sink = sink or CollectSink()
    config = PipelineConfig(reconnect_initial_s=0.05, reconnect_max_s=0.4, **cfg)
    pipe = Pipeline(source, backend or FakeBackend(manifest), manifest, [sink], config, preprocessor=pre, decoder=dec)
    return pipe, sink


def seqs(sink) -> list[int]:
    return [r.frame_ref.frame_seq for r in sink.results]


# ---- overlap ---------------------------------------------------------------------------------------------------
def test_stages_overlap_with_the_engine(rf_manifest):
    """Deterministic: inference of frame n can only finish if frame n+1 was already preprocessed and frame n-1
    already decoded. A serial pipeline would hit the wait timeout on every call."""
    n_frames, misses = 12, []
    pre, dec = CountingPre(rf_manifest), CountingDec(rf_manifest)

    def hook(call):
        ok_pre = call >= n_frames - 1 or _wait(lambda: pre.calls >= call + 2)
        ok_dec = call < 2 or _wait(lambda: dec.calls >= call)
        if not (ok_pre and ok_dec):
            misses.append(call)

    pipe, sink = build(rf_manifest, ScriptedSource([make_frame(i) for i in range(n_frames)]),
                       backend=FakeBackend(rf_manifest, hook=hook), pre=pre, dec=dec, stage_queue_size=2)
    pipe.run()
    assert misses == [] and seqs(sink) == list(range(n_frames))


def _wait(cond, timeout=2.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.001)
    return False


def test_overlap_shortens_wall_time_versus_serial(rf_manifest):
    n, s = 40, 0.01  # three stages of 10 ms: serial >= 1.2 s, staged ~ 0.45 s
    pipe, sink = build(rf_manifest, ScriptedSource([make_frame(i) for i in range(n)]),
                       backend=FakeBackend(rf_manifest, latency_s=s), pre=CountingPre(rf_manifest, s),
                       dec=CountingDec(rf_manifest, s))
    t0 = time.monotonic()
    pipe.run()
    assert len(sink.results) == n and time.monotonic() - t0 < 0.8 * n * 3 * s


# ---- order -----------------------------------------------------------------------------------------------------
def test_order_is_preserved_with_jittery_stage_latencies(rf_manifest):
    rng = random.Random(7)
    lat = lambda: rng.uniform(0, 0.004)  # noqa: E731

    class Jitter(CountingDec):
        def __call__(self, *a):
            time.sleep(lat())
            return super().__call__(*a)

    class JitterPre(CountingPre):
        def __call__(self, bgr):
            time.sleep(lat())
            return super().__call__(bgr)

    backend = FakeBackend(rf_manifest, hook=lambda call: time.sleep(lat()))
    pipe, sink = build(rf_manifest, ScriptedSource([make_frame(i) for i in range(120)]), backend=backend,
                       pre=JitterPre(rf_manifest), dec=Jitter(rf_manifest), stage_queue_size=1)
    snap = pipe.run()
    assert seqs(sink) == list(range(120)) and snap["dropped"] == 0 and pipe.overwritten == 0


# ---- back-pressure and drop policy -----------------------------------------------------------------------------
def test_replay_backpressure_bounds_what_capture_reads_and_loses_nothing(rf_manifest):
    gate = threading.Event()

    class Gated(CollectSink):
        def write(self, result):
            gate.wait(5)
            super().write(result)

    reads = []
    frames = [make_frame(i) for i in range(200)]
    src = ScriptedSource(frames)
    original = src.read
    src.read = lambda: (reads.append(1), original())[1]
    pipe, sink = build(rf_manifest, src, sink=Gated(), queue_size=1, stage_queue_size=1, publish_queue_size=1)
    pipe.start()
    time.sleep(0.5)
    # in flight at most: one per queue plus one inside each stage and the publisher
    assert len(reads) < 20
    gate.set()
    assert pipe.join(10)
    pipe.stop()
    assert seqs(sink) == list(range(200))


def test_live_mode_serves_increasing_frames_and_accounts_for_every_drop(rf_manifest):
    n = 150
    src = ScriptedSource([make_frame(i) for i in range(n)], live=True, delay=0.001)
    pipe, sink = build(rf_manifest, src, backend=FakeBackend(rf_manifest, latency_s=0.01))
    snap = pipe.run()
    s = seqs(sink)
    assert s == sorted(set(s)) and len(s) < n  # strictly increasing, and the slow engine really caused drops
    assert pipe.overwritten > 0
    assert len(s) + sum(r.dropped_before for r in sink.results) == s[-1] + 1
    assert snap["published"] == len(s)


def test_live_inference_takes_the_freshest_preprocessed_frame(rf_manifest):
    """With a slow engine the frame it takes next is never older than the one it just finished by more than 1."""
    src = ScriptedSource([make_frame(i) for i in range(100)], live=True, delay=0.002)
    pipe, sink = build(rf_manifest, src, backend=FakeBackend(rf_manifest, latency_s=0.02))
    pipe.run()
    gaps = [r.dropped_before for r in sink.results[1:]]
    # 20 ms engine over a 2 ms source: it should skip ahead (~9 frames), not walk through stale frames one by one
    assert max(gaps) >= 5


# ---- errors ----------------------------------------------------------------------------------------------------
def test_error_in_preprocess_skips_the_frame_and_the_run_continues(rf_manifest):
    pipe, sink = build(rf_manifest, ScriptedSource([make_frame(i) for i in range(8)]),
                       pre=CountingPre(rf_manifest, fail_on=(2,)))
    snap = pipe.run()
    assert seqs(sink) == [0, 1, 3, 4, 5, 6, 7] and snap["backend_errors"] == 1 and pipe.error is None
    assert sink.results[2].dropped_before == 1


def test_error_in_postprocess_skips_the_frame_and_the_run_continues(rf_manifest):
    pipe, sink = build(rf_manifest, ScriptedSource([make_frame(i) for i in range(8)]),
                       dec=CountingDec(rf_manifest, fail_on=(4,)))
    snap = pipe.run()
    assert seqs(sink) == [0, 1, 2, 3, 5, 6, 7] and snap["backend_errors"] == 1 and pipe.error is None


def test_error_in_the_tracker_is_isolated_to_that_frame(rf_manifest):
    calls = []

    def tracker(dets):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("tracker blew up")
        return dets

    sink = CollectSink()
    pipe = Pipeline(ScriptedSource([make_frame(i) for i in range(6)]), FakeBackend(rf_manifest), rf_manifest, [sink],
                    PipelineConfig(), tracker=tracker)
    pipe.run()
    assert seqs(sink) == [0, 1, 3, 4, 5]


def test_persistent_failure_in_a_downstream_stage_stops_everything_cleanly(rf_manifest):
    src = ScriptedSource([make_frame(i) for i in range(10_000)], live=True, delay=0.001)
    pipe, sink = build(rf_manifest, src, dec=CountingDec(rf_manifest, fail_all=True), max_consecutive_errors=5)
    pipe.start()
    assert pipe.join(5), "pipeline did not stop by itself after repeated postprocess failures"
    pipe.stop()
    assert isinstance(pipe.error, ValueError) and sink.results == [] and stage_threads() == []


def test_errors_in_different_stages_do_not_hide_each_other(rf_manifest):
    pipe, sink = build(rf_manifest, ScriptedSource([make_frame(i) for i in range(10)]),
                       pre=CountingPre(rf_manifest, fail_on=(1,)),
                       backend=FakeBackend(rf_manifest, hook=lambda c: (_ for _ in ()).throw(RuntimeError("x"))
                                           if c == 4 else None),
                       dec=CountingDec(rf_manifest, fail_on=(6,)))
    snap = pipe.run()
    assert snap["backend_errors"] == 3 and len(sink.results) == 7 and seqs(sink) == sorted(seqs(sink))


# ---- shutdown --------------------------------------------------------------------------------------------------
def test_stop_leaves_no_stage_thread_behind_live_and_replay(rf_manifest):
    for live in (True, False):
        src = ScriptedSource([make_frame(i) for i in range(100_000)], live=live, delay=0.001)
        pipe, sink = build(rf_manifest, src, backend=FakeBackend(rf_manifest, latency_s=0.003))
        pipe.start()
        time.sleep(0.2)
        assert sorted(stage_threads()) == sorted(STAGE_THREADS)
        t0 = time.monotonic()
        pipe.stop()
        assert time.monotonic() - t0 < 3 and stage_threads() == [] and len(sink.results) > 0


def test_stop_with_every_queue_full_and_a_blocked_sink_cuts_all_stages(rf_manifest):
    gate = threading.Event()

    class Stuck(CollectSink):
        def write(self, result):
            gate.wait(30)

    src = ScriptedSource([make_frame(i) for i in range(1000)])
    pipe, _ = build(rf_manifest, src, sink=Stuck(), queue_size=1, stage_queue_size=1, publish_queue_size=1)
    pipe.start()
    time.sleep(0.4)
    t0 = time.monotonic()
    pipe.stop(timeout=1.0)
    gate.set()  # release the sink thread only after stop(): stop must not depend on it
    assert time.monotonic() - t0 < 6
    assert _wait(lambda: stage_threads() == [], 3)


def test_finite_source_end_flushes_every_stage_before_closing(rf_manifest):
    """The last frame must come out even though END travels behind it through four hand-offs."""
    pipe, sink = build(rf_manifest, ScriptedSource([make_frame(i) for i in range(3)]),
                       backend=FakeBackend(rf_manifest, latency_s=0.02), pre=CountingPre(rf_manifest, 0.02),
                       dec=CountingDec(rf_manifest, 0.02))
    pipe.run()
    assert seqs(sink) == [0, 1, 2] and stage_threads() == []
