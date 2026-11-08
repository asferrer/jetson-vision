"""Command line: ``jv run``, ``jv bench`` and ``jv replay``. First client of the core, no ROS.

Exit codes: 0 ok, 1 pipeline or comparison failure, 2 usage error, 3 the health monitor ordered
an orderly stop (systemd unit: RestartPreventExitStatus=3).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from jetson_vision.health import HealthConfig, HealthMonitor, LatencyBaseline, SystemdWatchdog
from jetson_vision.interfaces import InferenceBackend, Sink
from jetson_vision.manifest import EngineMismatchError, Manifest
from jetson_vision.pipeline import Pipeline, PipelineConfig
from jetson_vision.sinks import JsonlSink, make_sink
from jetson_vision.tracking import IoUTracker
from jetson_vision.types import DegradationLevel, Detection, FrameResult, HealthState

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_STOPPED = 0, 1, 2, 3
TICK_S = 1.0
HEALTH_LOG_S = 10.0
ALERT_PERIOD_S = 1.0  # level 2: one frame per second, alerts only

log = logging.getLogger("jv")


class CliError(Exception):
    """A usage or configuration problem; reported without a traceback."""


# ---- sinks used by the CLI itself ---------------------------------------------------------------
class OpsSink:
    """Feeds the health monitor and the watchdog from published results, and counts for --max-frames."""

    def __init__(self, monitor: HealthMonitor | None, watchdog: SystemdWatchdog | None,
                 on_limit: Any = None, limit: int = 0) -> None:
        self.monitor, self.watchdog = monitor, watchdog
        self.on_limit, self.limit, self.count = on_limit, limit, 0

    def write(self, result: FrameResult) -> None:
        self.count += 1
        if self.monitor is not None:
            self.monitor.observe_latency_ms(result.timings.inference_ns / 1e6)
        if self.watchdog is not None:
            self.watchdog.result_seen()
        if self.limit and self.count >= self.limit and self.on_limit:
            self.on_limit()

    def write_health(self, state: HealthState) -> None:
        pass

    def close(self) -> None:
        pass


class CollectSink:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def write(self, result: FrameResult) -> None:
        self.results.append(result.to_dict())

    def write_health(self, state: HealthState) -> None:
        pass

    def close(self) -> None:
        pass


# ---- construction -------------------------------------------------------------------------------
def build_backend(manifest: Manifest, kind: str, model_dir: str | None) -> InferenceBackend:
    if kind == "fake":
        from jetson_vision.backends.fake import FakeBackend
        return FakeBackend(manifest)
    if kind == "tensorrt":
        from jetson_vision.backends.tensorrt import TensorRTBackend
        return TensorRTBackend(manifest, model_dir)
    raise CliError(f"unknown backend {kind!r}")


def parse_baseline(text: str | None) -> LatencyBaseline:
    """``30:20,15:30`` -> {30 fps: 20 ms, 15 fps: 30 ms}, measured on the target at install time."""
    pairs: dict[float, float] = {}
    for item in filter(None, (text or "").split(",")):
        try:
            fps, ms = item.split(":")
            pairs[float(fps)] = float(ms)
        except ValueError as exc:
            raise CliError(f"invalid --baseline entry {item!r}; use FPS:MS,FPS:MS") from exc
    return LatencyBaseline(pairs)


class _TrackerAdapter:
    """Adapts IoUTracker to the pipeline hook, which passes detections only: counts calls as frames."""

    def __init__(self) -> None:
        self._tracker, self._n = IoUTracker(), 0

    def __call__(self, dets: list[Detection]) -> list[Detection]:
        self._n += 1
        return self._tracker.update(dets, self._n)


def build_pipeline(args: argparse.Namespace, sinks: Sequence[Sink], mode: str) -> Pipeline:
    from jetson_vision.sources import make_source

    manifest = Manifest.load(args.model)
    live, loop = getattr(args, "live", False) or None, getattr(args, "loop", False)
    source = make_source(args.source, mode=mode, fps=args.fps, live=live, loop=loop)
    backend = build_backend(manifest, args.backend, args.model_dir)
    cfg = PipelineConfig(live=live, score_threshold=args.threshold)
    tracker = _TrackerAdapter() if getattr(args, "track", False) else None
    return Pipeline(source, backend, manifest, sinks, cfg, tracker=tracker)


# ---- run ----------------------------------------------------------------------------------------
class Supervisor:
    """Once per tick: evaluate health, apply the level to the pipeline, drive the watchdog."""

    def __init__(self, pipeline: Pipeline, monitor: HealthMonitor, watchdog: SystemdWatchdog | None,
                 sinks: Sequence[Sink], nominal_fps: float | None) -> None:
        self.pipeline, self.monitor, self.watchdog, self.sinks = pipeline, monitor, watchdog, sinks
        self.nominal_fps = nominal_fps
        if nominal_fps:
            monitor.set_target_fps(nominal_fps)
        self._last_level: DegradationLevel | None = None
        self._last_log = 0.0
        self._logged: DegradationLevel | None = None
        self._captured = 0

    def tick(self) -> None:
        snap = self.pipeline.metrics.snapshot()
        if self.nominal_fps is None and snap["fps"]:
            self.nominal_fps = float(snap["fps"])  # latched at the first measured cadence
            self.monitor.set_target_fps(self.nominal_fps)
        self.monitor.observe_fps(snap["fps"] or 0.0)
        state = self.monitor.evaluate()
        self._apply(state.level)
        self._update_watchdog(snap)
        self._log_health(state)

    def _apply(self, level: DegradationLevel) -> None:
        if level != self._last_level:
            self._last_level = level
            self.pipeline.set_degradation(level)
            self.pipeline.set_throttle(self._throttle_for(level))
            self.monitor.set_target_fps(self._cadence_for(level))

    def _cadence_for(self, level: DegradationLevel) -> float:
        if level == DegradationLevel.ALERTS_ONLY:
            return 1.0 / ALERT_PERIOD_S
        return (self.nominal_fps or 0.0) * self.monitor.fps_factor()

    def _throttle_for(self, level: DegradationLevel) -> float:
        cadence = self._cadence_for(level)
        if level == DegradationLevel.NORMAL or cadence <= 0:
            return 0.0
        return 1.0 / cadence

    def _update_watchdog(self, snap: dict[str, Any]) -> None:
        if self.watchdog is None:
            return
        waiting = snap["reconnects"] > 0 and snap["captured"] == self._captured
        self._captured = snap["captured"]
        self.watchdog.set_no_source(waiting)

    def _log_health(self, state: HealthState) -> None:
        now = time.monotonic()
        if state.level != self._logged or now - self._last_log >= HEALTH_LOG_S:
            self._last_log = now
            self._logged = state.level
            for sink in self.sinks:
                sink.write_health(state)


def restart_cause() -> str:
    path = os.environ.get("JV_LAST_STOP_FILE")
    try:
        return Path(path).read_text().strip() if path else "unknown"
    except OSError:
        return "unknown"


def _install_signals(pipeline: Pipeline) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: pipeline.request_stop())
        except ValueError:  # not in the main thread
            pass


def _make_sinks(args: argparse.Namespace, model_id: str) -> list[Sink]:
    max_bytes = int(getattr(args, "log_max_mb", 0) * 1e6)
    sinks = [make_sink(spec, model_id, max_bytes=max_bytes) for spec in (args.sink or [])]
    for sink in sinks:
        if isinstance(sink, JsonlSink):
            sink.write_event("start", restart_cause=restart_cause(), model=model_id,
                             source=args.source)
    return sinks


def cmd_run(args: argparse.Namespace) -> int:
    model_id = Manifest.load(args.model).model_id
    sinks = _make_sinks(args, model_id)
    monitor = HealthMonitor(HealthConfig(), parse_baseline(args.baseline))
    watchdog = SystemdWatchdog() if args.watchdog else None
    ops = OpsSink(monitor, watchdog)
    pipeline = build_pipeline(args, [*sinks, ops], "paced" if args.fps else "max")
    ops.on_limit, ops.limit = pipeline.request_stop, args.max_frames or 0
    sup = Supervisor(pipeline, monitor, watchdog, sinks, args.fps)
    _install_signals(pipeline)
    if args.max_seconds > 0:
        timer = threading.Timer(args.max_seconds, pipeline.request_stop)
        timer.daemon = True
        timer.start()
    code = _run_loop(pipeline, sup, watchdog)
    _close_sinks(sinks)
    return code


def _safe_tick(sup: Supervisor) -> None:
    """A health-monitoring error is logged and retried next tick: it must not take the detector down."""
    try:
        sup.tick()
    except Exception:
        log.exception("health tick failed; detection continues")


def _close_sinks(sinks: Sequence[Sink]) -> None:
    for sink in sinks:
        try:
            sink.close()
        except Exception:
            log.exception("error closing %s", type(sink).__name__)


def _run_loop(pipeline: Pipeline, sup: Supervisor, watchdog: SystemdWatchdog | None) -> int:
    try:
        pipeline.start()
        if watchdog:
            watchdog.ready()
            watchdog.start()
        while not pipeline.join(TICK_S):
            if pipeline.stop_requested:
                break  # stop() below cuts the output after its timeout, so a blocked sink cannot hold the exit
            _safe_tick(sup)
    finally:
        pipeline.stop()
        if watchdog:
            watchdog.stop()
    if sup.monitor.level == DegradationLevel.STOPPED:
        log.error("stopped by the health monitor (level 3)")
        return EXIT_STOPPED
    return EXIT_FAIL if pipeline.error else EXIT_OK


# ---- bench --------------------------------------------------------------------------------------
def cmd_bench(args: argparse.Namespace) -> int:
    ops = OpsSink(None, None, limit=args.frames or 0)
    pipeline = build_pipeline(args, [ops], "max")
    ops.on_limit = pipeline.request_stop
    snapshot = pipeline.run()
    print(json.dumps({"backend": args.backend, "model": pipeline.backend.model_id, **snapshot},
                     indent=2))
    return EXIT_FAIL if pipeline.error else EXIT_OK


# ---- replay -------------------------------------------------------------------------------------
def load_reference(path: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("record", "result") == "result"]


def _same_detection(a: dict[str, Any], b: dict[str, Any], score_tol: float, box_tol: float) -> bool:
    return (a["class_id"] == b["class_id"] and abs(a["score"] - b["score"]) <= score_tol
            and all(abs(x - y) <= box_tol for x, y in zip(a["box_xyxy"], b["box_xyxy"])))


def _canonical(dets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order that does not depend on how an implementation breaks ties between equal scores."""
    return sorted(dets, key=lambda d: (-d["score"], d["class_id"], tuple(d["box_xyxy"])))


def compare_results(actual: list[dict[str, Any]], reference: list[dict[str, Any]],
                    score_tol: float = 1e-4, box_tol: float = 0.5) -> list[str]:
    """Differences between two runs, matched by frame_seq (detections in score order, ties by class and box)."""
    problems: list[str] = []
    if len(actual) != len(reference):
        problems.append(f"frame count {len(actual)} != reference {len(reference)}")
    for a, r in zip(actual, reference):
        seq = a["frame_ref"]["frame_seq"]
        if seq != r["frame_ref"]["frame_seq"]:
            problems.append(f"frame_seq {seq} != reference {r['frame_ref']['frame_seq']}")
        elif len(a["detections"]) != len(r["detections"]):
            problems.append(f"frame {seq}: {len(a['detections'])} detections, "
                            f"reference {len(r['detections'])}")
        elif not all(_same_detection(x, y, score_tol, box_tol)
                     for x, y in zip(_canonical(a["detections"]), _canonical(r["detections"]))):
            problems.append(f"frame {seq}: detections differ beyond tolerance")
    return problems


def cmd_replay(args: argparse.Namespace) -> int:
    collect = CollectSink()
    model_id = Manifest.load(args.model).model_id
    sinks: list[Sink] = [collect, *_make_sinks(args, model_id)]
    pipeline = build_pipeline(args, sinks, "max")
    pipeline.run()
    _close_sinks(sinks)
    if pipeline.error:
        return EXIT_FAIL
    if not args.reference:
        print(f"replayed {len(collect.results)} frames")
        return EXIT_OK
    problems = compare_results(collect.results, load_reference(args.reference),
                               args.score_tol, args.box_tol)
    for p in problems[:20]:
        print("DIFF", p)
    print("replay identical" if not problems else f"replay differs: {len(problems)} problems")
    return EXIT_FAIL if problems else EXIT_OK


# ---- parser -------------------------------------------------------------------------------------
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", required=True, help="folder, image, video file, v4l2:N, gst:<uri>, cam:<csi|v4l2|rtsp|test>...")
    p.add_argument("--model", required=True, help="path of the model manifest (yaml)")
    p.add_argument("--backend", choices=("tensorrt", "fake"), default="tensorrt")
    p.add_argument("--model-dir", default=None, help="engine directory (or JV_MODEL_DIR)")
    p.add_argument("--threshold", type=float, default=None, help="score threshold override")
    p.add_argument("--fps", type=float, default=None, help="nominal input cadence (paced sources)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jv", description="jetson-vision command line")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the detector on a source")
    _common(run)
    run.add_argument("--sink", action="append", help="jsonl:<path>, compact:<path> or stdout")
    run.add_argument("--track", action="store_true", help="enable the optional IoU tracker")
    run.add_argument("--watchdog", action="store_true", help="talk to the systemd watchdog")
    run.add_argument("--baseline", help="latency baselines FPS:MS,FPS:MS measured on the target")
    run.add_argument("--max-frames", type=int, default=0)
    run.add_argument("--log-max-mb", type=float, default=100.0,
                     help="rotate each JSONL sink at this size, keeping 3 older files (0 = never; the default keeps "
                          "a long run from filling the disk)")
    run.add_argument("--max-seconds", type=float, default=0, help="stop cleanly after this many seconds")
    run.add_argument("--live", action="store_true",
                     help="treat the source as a camera: newest frame wins; a gst:file: plays at its own rate")
    run.add_argument("--loop", action="store_true", help="restart a gst:file: source at its end")
    run.set_defaults(func=cmd_run)
    bench = sub.add_parser("bench", help="measure throughput and per-stage latency")
    _common(bench)
    bench.add_argument("--frames", type=int, default=0, help="stop after N results")
    bench.set_defaults(func=cmd_bench, track=False)
    replay = sub.add_parser("replay", help="deterministic replay, optionally compared to a reference")
    _common(replay)
    replay.add_argument("--sink", action="append")
    replay.add_argument("--reference", help="JSONL of a previous run to compare against")
    replay.add_argument("--score-tol", type=float, default=1e-4)
    replay.add_argument("--box-tol", type=float, default=0.5, help="pixels")
    replay.set_defaults(func=cmd_replay, track=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except EngineMismatchError as exc:
        print(f"jv: engine refused: {exc}", file=sys.stderr)
        return EXIT_FAIL
    except (CliError, ValueError, OSError) as exc:
        print(f"jv: error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
