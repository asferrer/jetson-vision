"""Five-thread pipeline: capture -> preprocess -> inference -> postprocess -> publication.

Preprocessing and decoding run in their own threads, overlapped with the engine (cv2 and numpy release the GIL), so
the engine thread only does the GPU copy, the execution and the read-back. Order is preserved: one thread per stage
and FIFO hand-offs.

Hand-off policy (explicit):
- Live sources: capture -> preprocess and preprocess -> inference are one-slot mailboxes where the newest item wins,
  so inference always takes the freshest frame and old ones are dropped (counted by `overwritten`). inference ->
  postprocess -> publication are bounded blocking queues: work that reached the engine is never discarded.
- Finite (replay) sources: every hand-off is a bounded blocking queue, so nothing is lost and runs are reproducible.
Drops are not counted per stage: they are the jumps in frame_seq seen by the postprocess thread
(FrameResult.dropped_before).

Ownership: the pipeline loads the backend, opens the source and closes both on stop(). Sinks belong to the caller:
the pipeline writes to them until the last result is out, and the caller closes them afterwards.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from jetson_vision.interfaces import Decoder, FrameSource, InferenceBackend, Sink, SourceError
from jetson_vision.manifest import Manifest
from jetson_vision.postprocess import build_decoder
from jetson_vision.preprocess import Preprocessor
from jetson_vision.types import DegradationLevel, Detection, Frame, FrameResult, StageTimings

log = logging.getLogger(__name__)
POLL_S = 0.1
# queue = time waiting in queues; service = total - queue (latency without waiting for the consumer)
STAGES = ("capture", "preprocess", "inference", "postprocess", "tracking", "queue", "service", "total")


class _Marker:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


EMPTY = _Marker("EMPTY")  # get() timed out
END = _Marker("END")  # the channel was finished or aborted and holds nothing more


class LatestMailbox:
    """One slot, the newest item wins. put() never blocks; overwritten counts the items that were replaced."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: Any = None
        self._closed = False
        self.overwritten = 0

    def put(self, item: Any) -> bool:
        with self._cond:
            if self._closed:
                return False
            if self._item is not None:
                self.overwritten += 1
            self._item = item
            self._cond.notify()
            return True

    def get(self, timeout: float) -> Any:
        with self._cond:
            if self._item is None and not self._closed:
                self._cond.wait(timeout)
            if self._item is not None:
                item, self._item = self._item, None
                return item
            return END if self._closed else EMPTY

    def finish(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def abort(self) -> None:
        with self._cond:
            self._item, self._closed = None, True
            self._cond.notify_all()


class BoundedChannel:
    """Bounded FIFO whose put() blocks while it is full: nothing is dropped (replay mode)."""

    def __init__(self, maxsize: int) -> None:
        self._q: queue.Queue = queue.Queue(maxsize)
        self._aborted = threading.Event()
        self.overwritten = 0

    def put(self, item: Any) -> bool:
        while not self._aborted.is_set():
            try:
                self._q.put(item, timeout=POLL_S)
                return True
            except queue.Full:
                continue
        return False

    def get(self, timeout: float) -> Any:
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            return END if self._aborted.is_set() else EMPTY
        return item

    def finish(self) -> None:
        self.put(END)

    def abort(self) -> None:
        self._aborted.set()


@dataclass
class PipelineConfig:
    live: bool | None = None  # None: take it from the source
    queue_size: int = 4  # replay: frames waiting between capture and inference
    publish_queue_size: int = 16
    stage_queue_size: int = 2  # items waiting between preprocess/inference/postprocess (blocking queues)
    score_threshold: float | None = None  # None: the manifest default
    reconnect_initial_s: float = 0.5
    reconnect_factor: float = 2.0
    reconnect_max_s: float = 10.0
    max_consecutive_errors: int = 20  # inference failures in a row before the pipeline stops itself
    metrics_window: int = 300


@dataclass
class _Work:
    """One frame travelling through the stages. `enqueued` is when it entered the queue it now sits in."""
    frame: Frame
    capture_ns: int
    enqueued: int
    queue_ns: int = 0
    image: np.ndarray | None = None
    outputs: Any = None
    preprocess_ns: int = 0
    inference_ns: int = 0


class PipelineMetrics:
    """Counters and per-stage timing windows; safe to read from any thread."""

    def __init__(self, window: int = 300) -> None:
        self._lock = threading.Lock()
        self._stage = {s: deque(maxlen=window) for s in STAGES}
        self._times: deque[float] = deque(maxlen=window)
        self.counters = {"captured": 0, "processed": 0, "published": 0, "dropped": 0, "reconnects": 0,
                         "backend_errors": 0, "sink_errors": 0}

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def record(self, t: StageTimings, dropped: int) -> None:
        with self._lock:
            for s in STAGES:
                self._stage[s].append(getattr(t, f"{s}_ns"))
            self._times.append(time.monotonic())
            self.counters["published"] += 1
            self.counters["dropped"] += dropped

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = {}
            for s, vals in self._stage.items():
                a = np.asarray(vals, dtype=np.float64) / 1e6
                stages[s] = ({"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95))}
                             if a.size else {"p50_ms": None, "p95_ms": None})
            times = list(self._times)
            span = times[-1] - times[0] if len(times) > 1 else 0.0
            fps = (len(times) - 1) / span if span > 0 else None
            return {**self.counters, "fps": fps, "stages": stages}


class Pipeline:
    def __init__(self, source: FrameSource | None, backend: InferenceBackend, manifest: Manifest,
                 sinks: Sequence[Sink], config: PipelineConfig | None = None, *,
                 decoder: Decoder | None = None, preprocessor: Preprocessor | None = None,
                 tracker: Callable[[list[Detection]], list[Detection]] | None = None) -> None:
        self.source, self.backend, self.manifest, self.sinks = source, backend, manifest, list(sinks)
        self.config = config or PipelineConfig()
        # source=None: single-frame use through process_frame (the ROS 2 node feeds frames itself)
        self.live = (source.live if source is not None else False) if self.config.live is None else self.config.live
        self.decoder = decoder or build_decoder(manifest)
        self.preprocessor = preprocessor or Preprocessor(manifest.input)
        self.tracker = tracker
        self.threshold = (manifest.score_threshold if self.config.score_threshold is None
                          else self.config.score_threshold)
        self.metrics = PipelineMetrics(self.config.metrics_window)
        self.error: BaseException | None = None
        self._level = DegradationLevel.NORMAL
        self._throttle_s = 0.0
        self._stop = threading.Event()
        self._chan: LatestMailbox | BoundedChannel = (
            LatestMailbox() if self.live else BoundedChannel(self.config.queue_size))
        stage_q = self.config.stage_queue_size
        self._pre: LatestMailbox | BoundedChannel = LatestMailbox() if self.live else BoundedChannel(stage_q)
        self._post = BoundedChannel(stage_q)
        self._out = BoundedChannel(self.config.publish_queue_size)
        self._threads: list[threading.Thread] = []
        self._last_seq: dict[str, int] = {}
        self._loaded = False
        self._want_pixels = any(getattr(s, "wants_pixels", False) for s in self.sinks)

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    # ---- control -------------------------------------------------------------------------------------------
    def set_degradation(self, level: DegradationLevel) -> None:
        """Stamp the level on every result from now on. STOPPED stops the pipeline."""
        self._level = DegradationLevel(level)
        if self._level == DegradationLevel.STOPPED:
            self.request_stop()

    def set_throttle(self, min_interval_s: float) -> None:
        """Process at most one frame every min_interval_s (0 removes the limit). In live mode the rest are dropped."""
        self._throttle_s = max(0.0, float(min_interval_s))

    @property
    def overwritten(self) -> int:
        """Live frames replaced in a mailbox before inference took them."""
        return self._chan.overwritten + self._pre.overwritten

    def load(self) -> None:
        """Load the backend once (start() and process_frame() call it; calling it again does nothing)."""
        if not self._loaded:
            self.backend.load()
            self._loaded = True

    def start(self) -> None:
        if self.source is None:
            raise RuntimeError("Pipeline.start needs a source; use process_frame for single frames")
        self.load()
        names = (("capture", self._capture_loop), ("preprocess", self._preprocess_loop),
                 ("inference", self._inference_loop), ("postprocess", self._postprocess_loop),
                 ("publish", self._publish_loop))
        self._threads = [threading.Thread(target=fn, name=f"jv-{n}", daemon=True) for n, fn in names]
        for t in self._threads:
            t.start()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def request_stop(self) -> None:
        self._stop.set()
        self._chan.abort()

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the end of a finite source (every result published). True if all threads finished."""
        end = None if timeout is None else time.monotonic() + timeout
        for t in self._threads:
            t.join(None if end is None else max(0.0, end - time.monotonic()))
        return not any(t.is_alive() for t in self._threads)

    def stop(self, timeout: float = 10.0) -> None:
        """Clean stop: threads end, source and backend are closed. Safe to call twice."""
        self.request_stop()
        if not self.join(timeout):  # something is stuck (a blocked sink, a silent camera): cut every queue
            for chan in (self._pre, self._post, self._out):
                chan.abort()
            self.join(2.0)
        self.close()

    def close(self) -> None:
        """Close the source (if any) and the backend. Safe to call twice."""
        closers = ([self.source.close] if self.source is not None else []) + [self.backend.close]
        for closer in closers:
            try:
                closer()
            except Exception:
                log.exception("error while closing %s", closer)
        self._loaded = False

    def run(self) -> dict[str, Any]:
        """Start, wait for the source to end (or for a stop request) and stop. Returns the metrics snapshot."""
        self.start()
        try:
            while not self.join(0.2):
                if self._stop.is_set():
                    break
        finally:
            self.stop()
        return self.metrics.snapshot()

    # ---- capture -------------------------------------------------------------------------------------------
    def _capture_loop(self) -> None:
        backoff = self.config.reconnect_initial_s
        opened = False
        while not self._stop.is_set():
            try:
                if not opened:
                    self.source.open()
                    opened = True
                t0 = time.perf_counter_ns()
                frame = self.source.read()
                if frame is None:
                    break
                backoff = self.config.reconnect_initial_s
                self.metrics.count("captured")
                now = time.perf_counter_ns()
                if not self._chan.put(_Work(frame, now - t0, now)):
                    break
            except SourceError as exc:
                opened, backoff = False, self._reconnect(exc, backoff)
            except Exception as exc:
                log.exception("capture thread failed")
                self.error = exc
                self._stop.set()
        self._chan.finish()

    def _reconnect(self, exc: SourceError, backoff: float) -> float:
        log.warning("source '%s' failed (%s); reconnecting in %.1f s", self.source.source_id, exc, backoff)
        self.metrics.count("reconnects")
        try:
            self.source.close()
        except Exception:
            log.exception("error closing the failed source")
        self._stop.wait(backoff)
        return min(backoff * self.config.reconnect_factor, self.config.reconnect_max_s)

    # ---- stages --------------------------------------------------------------------------------------------
    def _stage_loop(self, inbox: Any, outbox: Any, work: Callable[[Any], Any],
                    after: Callable[[], None] | None = None) -> None:
        """Take from inbox, run work, put the result in outbox; always finish outbox at the end.

        A failing frame is skipped and counted; max_consecutive_errors failures in a row (per stage) stop the pipeline.
        """
        errors = 0
        while True:
            item = inbox.get(POLL_S)
            if item is END:
                break
            if item is EMPTY:
                continue
            try:
                out = work(item)
                errors = 0
            except Exception as exc:
                errors += 1
                self.metrics.count("backend_errors")
                log.exception("%s failed on frame %d", work.__name__, item.frame.ref.frame_seq)
                if errors >= self.config.max_consecutive_errors:
                    self.error = exc
                    self.request_stop()
                continue
            if not outbox.put(out):
                break
            if after is not None:
                after()
        outbox.finish()

    def _preprocess_loop(self) -> None:
        self._stage_loop(self._chan, self._pre, self._do_preprocess)

    def _inference_loop(self) -> None:
        self._stage_loop(self._pre, self._post, self._do_inference, self._wait_throttle)

    def _postprocess_loop(self) -> None:
        self._stage_loop(self._post, self._out, self._do_postprocess)

    def _do_preprocess(self, w: _Work) -> _Work:
        t0 = time.perf_counter_ns()
        w.queue_ns += t0 - w.enqueued
        w.image = self.preprocessor(w.frame.pixels)
        w.enqueued = time.perf_counter_ns()
        w.preprocess_ns = w.enqueued - t0
        return w

    def _do_inference(self, w: _Work) -> _Work:
        t0 = time.perf_counter_ns()
        w.queue_ns += t0 - w.enqueued
        w.outputs = self.backend.infer(w.image)
        w.image = None
        w.enqueued = time.perf_counter_ns()
        w.inference_ns = w.enqueued - t0
        return w

    def _do_postprocess(self, w: _Work) -> tuple[FrameResult, int]:
        t0 = time.perf_counter_ns()
        w.queue_ns += t0 - w.enqueued
        result = self._finish(w.frame, w.outputs, w.preprocess_ns, w.inference_ns, w.capture_ns, w.queue_ns)
        if self._want_pixels:
            result.pixels = w.frame.pixels
        self.metrics.count("processed")
        return result, time.perf_counter_ns()

    def _wait_throttle(self) -> None:
        if self._throttle_s > 0:
            self._stop.wait(self._throttle_s)

    def _finish(self, frame: Frame, outputs: Any, pre_ns: int, inf_ns: int, capture_ns: int,
                queue_ns: int) -> FrameResult:
        """Decode, track and build the result. total_ns here is capture stamp to now (publication overwrites it)."""
        clock = time.perf_counter_ns
        t0 = clock()
        dets = self.decoder(outputs, frame.width, frame.height, self.threshold)
        t1 = clock()
        if self.tracker is not None:
            dets = self.tracker(dets)
        t2 = clock()
        timings = StageTimings(capture_ns=capture_ns, preprocess_ns=pre_ns, inference_ns=inf_ns,
                               postprocess_ns=t1 - t0, tracking_ns=t2 - t1 if self.tracker else 0,
                               total_ns=max(0, time.monotonic_ns() - frame.ref.t_capture_mono_ns),
                               queue_ns=queue_ns)
        return FrameResult(frame_ref=frame.ref, detections=dets, model_id=self.backend.model_id,
                           score_threshold=self.threshold, degradation_level=self._level, timings=timings,
                           dropped_before=self._dropped_before(frame))

    def process_frame(self, frame: Frame, *, capture_ns: int = 0, queue_ns: int = 0) -> FrameResult:
        """Run one frame through preprocess, backend, decoder and tracker on the calling thread.

        Same stage code the threaded pipeline runs, without the overlap. total_ns is measured from the frame's
        capture stamp up to the end of this call; queue_ns is what the caller waited before calling. Raises whatever
        the stages raise. Not thread-safe against itself, nor against a running start().
        """
        self.load()
        w = _Work(frame, capture_ns, time.perf_counter_ns(), queue_ns)
        self._do_preprocess(w)
        self._do_inference(w)
        return self._finish(frame, w.outputs, w.preprocess_ns, w.inference_ns, capture_ns, queue_ns)

    def _dropped_before(self, frame: Frame) -> int:
        """Frames lost since the previous processed frame of the same session, from the jump in frame_seq."""
        ref = frame.ref
        previous = self._last_seq.get(ref.session_id, -1)
        self._last_seq = {ref.session_id: ref.frame_seq}
        return max(0, ref.frame_seq - previous - 1)

    # ---- publication ---------------------------------------------------------------------------------------
    def _publish_loop(self) -> None:
        while True:
            item = self._out.get(POLL_S)
            if item is END:
                break
            if item is EMPTY:
                continue
            result, enqueued = item
            result.timings.queue_ns += time.perf_counter_ns() - enqueued
            result.timings.total_ns = time.monotonic_ns() - result.frame_ref.t_capture_mono_ns
            for sink in self.sinks:
                try:
                    sink.write(result)
                except Exception:
                    self.metrics.count("sink_errors")
                    log.exception("sink %s failed", type(sink).__name__)
            result.pixels = None
            self.metrics.record(result.timings, result.dropped_before)
