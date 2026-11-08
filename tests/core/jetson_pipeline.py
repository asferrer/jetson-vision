"""On-device smoke test (not collected by pytest): the full three-thread pipeline with the real TensorRT backend.

Runs a source (folder of images, video file or gst:<uri>) through Pipeline and prints the metrics snapshot. With a
finite source in max mode it also checks that the threaded results equal a direct, single-threaded run.
  JV_MODEL_DIR=<engines> $PY tests/core/jetson_pipeline.py --manifest models/rfdetr_int8g.yaml --source <folder|video|gst:uri>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jetson_vision.backends.tensorrt import TensorRTBackend  # noqa: E402
from jetson_vision.manifest import Manifest  # noqa: E402
from jetson_vision.pipeline import Pipeline, PipelineConfig  # noqa: E402
from jetson_vision.postprocess import build_decoder  # noqa: E402
from jetson_vision.preprocess import Preprocessor  # noqa: E402
from jetson_vision.sources import make_source  # noqa: E402


class ListSink:
    def __init__(self) -> None:
        self.results = []

    def write(self, result) -> None:
        self.results.append(result)

    def write_health(self, state) -> None:
        pass

    def close(self) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--mode", default="max", choices=["max", "paced"])
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--live", action="store_true", help="force the newest-frame-wins mailbox")
    a = ap.parse_args()
    man = Manifest.load(a.manifest)
    sink = ListSink()
    backend = TensorRTBackend(man)
    pipe = Pipeline(make_source(a.source, mode=a.mode, fps=a.fps), backend, man, [sink],
                    PipelineConfig(live=True if a.live else None))
    snap = pipe.run()
    print(json.dumps(snap, indent=1, default=str))
    if a.live or not sink.results:
        return
    direct = TensorRTBackend(man)
    direct.load()
    pre, dec = Preprocessor(man.input), build_decoder(man)
    src = make_source(a.source, mode="max", fps=a.fps)
    src.open()
    same = checked = 0
    for res in sink.results:
        frame = src.read()
        dets = dec(direct.infer(pre(frame.pixels)), frame.width, frame.height, man.score_threshold)
        checked += 1
        same += dets == res.detections and frame.ref.frame_seq == res.frame_ref.frame_seq
    print(f"threaded results equal to a direct run: {same} of {checked}")


if __name__ == "__main__":
    main()
