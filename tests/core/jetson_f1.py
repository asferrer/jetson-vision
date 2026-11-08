"""F1-versus-threshold curve of the real engine on VAL (not collected by pytest). VAL only, never TEST.

Runs Pipeline.process_frame once per image at the lowest threshold, then counts matches per threshold: a detection
is a true positive if it matches an unused ground-truth box of its class with IoU >= 0.5 (greedy, by score).
Micro F1 pools all detections; macro F1 averages the classes that have ground truth.
  JV_MODEL_DIR=<engines> $PY tests/core/jetson_f1.py --manifest M --val V --data-root R --out f1.json [--n N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GRID = [round(0.05 * i, 2) for i in range(1, 20)]
IOU = 0.5


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    def area(r: np.ndarray) -> np.ndarray:
        return (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])

    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def match(dets: list, gts: list, threshold: float, counts: dict) -> None:
    """Add TP/FP/FN of one image at one threshold to counts[class] = [tp, fp, fn]."""
    for cls in {d.class_id for d in dets} | {g[0] for g in gts}:
        d = sorted((x for x in dets if x.class_id == cls and x.score >= threshold), key=lambda x: -x.score)
        g = np.array([x[1:] for x in gts if x[0] == cls], dtype=np.float64).reshape(-1, 4)
        used: set[int] = set()
        if len(d) and len(g):
            ious = iou_matrix(np.array([x.box_xyxy for x in d]), g)
            for i in range(len(d)):
                j = next((j for j in np.argsort(-ious[i]) if ious[i, j] >= IOU and j not in used), None)
                if j is not None:
                    used.add(int(j))
        c = counts[cls]
        c[0] += len(used)
        c[1] += len(d) - len(used)
        c[2] += len(g) - len(used)


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / max(2 * tp + fp + fn, 1)


def curve_row(t: float, per: dict) -> dict:
    tp, fp, fn = (sum(c[i] for c in per.values()) for i in range(3))
    macro = [f1(*c) for c in per.values() if c[0] + c[2] > 0]
    return {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "f1_micro": f1(tp, fp, fn), "f1_macro": float(np.mean(macro))}


def run(a: argparse.Namespace) -> dict:
    import cv2
    from jetson_vision.backends.tensorrt import TensorRTBackend
    from jetson_vision.manifest import Manifest
    from jetson_vision.pipeline import Pipeline, PipelineConfig
    from jetson_vision.types import Frame, FrameRef

    man = Manifest.load(a.manifest)
    low = 0.001 if man.decoder.name == "rtdetrv2" else 0.0
    pipe = Pipeline(None, TensorRTBackend(man), man, [], PipelineConfig(score_threshold=low))
    pipe.load()
    val = json.load(open(a.val, encoding="utf8"))
    boxes: dict[int, list] = defaultdict(list)
    for an in val["annotations"]:
        x, y, w, h = an["bbox"]
        boxes[an["image_id"]].append((an["category_id"], x, y, x + w, y + h))
    images = sorted(val["images"], key=lambda d: d["id"])[: a.n or None]
    counts = {t: defaultdict(lambda: [0, 0, 0]) for t in GRID}
    t0 = time.time()
    for seq, im in enumerate(images):
        bgr = cv2.imread(str(Path(a.data_root) / im["file_name"]), cv2.IMREAD_COLOR)
        ref = FrameRef("val", "f1", seq, time.monotonic_ns(), time.time_ns())
        dets = pipe.process_frame(Frame(ref, bgr)).detections
        for t in GRID:
            match(dets, boxes[im["id"]], t, counts[t])
    curve = [curve_row(t, counts[t]) for t in GRID]
    return {"model": man.model_id, "n_images": len(images), "iou": IOU, "seconds": time.time() - t0,
            "best_micro": max(curve, key=lambda r: r["f1_micro"]),
            "best_macro": max(curve, key=lambda r: r["f1_macro"]), "curve": curve}


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("--manifest", "--val", "--data-root", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--n", type=int, default=0)
    out = run(ap.parse_args())
    json.dump(out, open(ap.parse_args().out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("model", "n_images", "best_micro", "best_macro")}, indent=1))


if __name__ == "__main__":
    main()
