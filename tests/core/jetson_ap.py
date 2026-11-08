"""On-device AP of the new stack (manifest + Preprocessor + TensorRTBackend + decoder) on VAL. Never TEST.

Run on the Jetson like jetson_compare.py. Compare the AP with `reference_ap` in the manifest.
  JV_MODEL_DIR=<engines> $PY tests/core/jetson_ap.py --manifest models/rfdetr_int8g.yaml --val <val.json> --data-root <root>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jetson_vision.backends.tensorrt import TensorRTBackend  # noqa: E402
from jetson_vision.manifest import Manifest  # noqa: E402
from jetson_vision.postprocess import build_decoder  # noqa: E402
from jetson_vision.preprocess import Preprocessor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all of VAL")
    a = ap.parse_args()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    man = Manifest.load(a.manifest)
    backend = TensorRTBackend(man)
    backend.load()
    pre, dec = Preprocessor(man.input), build_decoder(man)
    gt = COCO(a.val)
    images = sorted(gt.dataset["images"], key=lambda d: d["id"])[: a.n or None]
    preds, t_inf = [], []
    for im in images:
        bgr = cv2.imread(str(Path(a.data_root) / im["file_name"]), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        out = backend.infer(pre(bgr))
        t_inf.append(time.perf_counter() - t0)
        for d in dec(out, bgr.shape[1], bgr.shape[0], 0.001 if man.decoder.name == "rtdetrv2" else 0.0):
            x1, y1, x2, y2 = d.box_xyxy
            preds.append({"image_id": im["id"], "category_id": d.class_id, "bbox": [x1, y1, x2 - x1, y2 - y1],
                          "score": d.score})
    ev = COCOeval(gt, gt.loadRes(preds), "bbox")
    ev.params.imgIds = [i["id"] for i in images]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    print(json.dumps({"model": man.model_id, "n_images": len(images), "AP": 100 * ev.stats[0],
                      "AP50": 100 * ev.stats[1], "reference_ap": man.reference_ap.value,
                      "preprocess_infer_ms_median": 1000 * float(np.median(t_inf[20:]))}))


if __name__ == "__main__":
    main()
