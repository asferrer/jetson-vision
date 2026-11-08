"""On-device check (not collected by pytest): the real TensorRT backend against the validated reference `stages.py`.

For N VAL images it runs both paths and compares the preprocessed uint8 image (bit by bit), the class, score and box
of every detection. Reads VAL only, never TEST. It can also save one image's raw outputs as a small test fixture.

Run on the Jetson with the TensorRT/torch environment of the Jetson (tools/reference/stages.py needs it):
  JV_MODEL_DIR=<engines> $PY tests/core/jetson_compare.py --manifest models/rfdetr_int8g.yaml \
      --stages-dir <dir with stages.py> --val <val.json> --data-root <image root> [--save-fixture out.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jetson_vision.backends.tensorrt import TensorRTBackend  # noqa: E402
from jetson_vision.manifest import Manifest  # noqa: E402
from jetson_vision.postprocess import build_decoder  # noqa: E402
from jetson_vision.preprocess import Preprocessor  # noqa: E402

REF_THRESHOLD = {"rfdetr": 0.0, "rtdetrv2": 0.001}  # RT-DETRv2 reference keeps scores > 0.001; RF-DETR keeps top-300


def pick_images(images: list[dict], n_small: int, n_1080p: int, n_large: int) -> list[dict]:
    images = sorted(images, key=lambda d: d["id"])
    small = [i for i in images if (i["width"], i["height"]) == (416, 416)]
    hd = [i for i in images if (i["width"], i["height"]) == (1920, 1080)]
    large = [i for i in images if i["width"] > 1920 or i["height"] > 1080]
    pick = lambda pool, n: pool[:: max(1, len(pool) // n)][:n] if n else []  # noqa: E731
    return pick(small, n_small) + pick(hd, n_1080p) + pick(large, n_large)


def reference_rows(ref, res, rt: bool) -> np.ndarray:
    """stages.py result -> rows (class, score, x1, y1, x2, y2), RF-DETR spurious class removed."""
    if rt:
        rows = np.concatenate([res[:, :1], res[:, 5:6], res[:, 1:5]], 1)
    else:
        scores, labels, boxes = res
        rows = np.concatenate([labels[:, None], scores[:, None], boxes], 1)
    return rows[rows[:, 0] < 8]


def mine_rows(dets) -> np.ndarray:
    return np.array([[d.class_id, d.score, *d.box_xyxy] for d in dets], dtype=np.float64).reshape(-1, 6)


def canonical(rows: np.ndarray) -> np.ndarray:
    return rows[np.lexsort((rows[:, 2], rows[:, 0], -rows[:, 1]))]


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = canonical(a), canonical(b)
    if len(a) != len(b):
        return {"same_count": False, "n_ref": len(a), "n_mine": len(b)}
    return {"same_count": True, "n": len(a), "class_equal": bool((a[:, 0] == b[:, 0]).all()),
            "max_score_diff": float(np.abs(a[:, 1] - b[:, 1]).max(initial=0.0)),
            "max_box_diff_px": float(np.abs(a[:, 2:] - b[:, 2:]).max(initial=0.0))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stages-dir", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--n-small", type=int, default=30)
    ap.add_argument("--n-1080p", type=int, default=12)
    ap.add_argument("--n-large", type=int, default=8)
    ap.add_argument("--save-fixture", help="npz path; uses the first 1080p image selected")
    ap.add_argument("--out", help="json with the per-image comparison")
    a = ap.parse_args()

    sys.path.insert(0, a.stages_dir)
    import stages  # noqa: E402  (reference implementation, validated against AP 43.61 / 43.32)

    man = Manifest.load(a.manifest)
    rt = man.decoder.name == "rtdetrv2"
    backend = TensorRTBackend(man)
    backend.load()
    engine = str(man.engine_path())
    ref = stages.Pipe(SimpleNamespace(model="rt" if rt else "rf", engine=engine, dec="cv2", pre="u8gpu", copy="pageable",
                                      post="cpu", d2h="sync"))
    pre, dec = Preprocessor(man.input), build_decoder(man)
    val = json.load(open(a.val, encoding="utf8"))
    images = pick_images(val["images"], a.n_small, a.n_1080p, a.n_large)
    report, fixture_done = [], False
    for im in images:
        data = open(Path(a.data_root) / im["file_name"], "rb").read()
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        image = pre(bgr)
        outputs = backend.infer(image)
        mine = mine_rows(dec(outputs, w, h, REF_THRESHOLD[man.decoder.name]))
        res, _ = ref.one(data)
        ref_rows = reference_rows(ref, res, rt)
        row = {"image_id": im["id"], "size": [w, h], "input_identical": bool(np.array_equal(image, ref.p_u8n)),
               **compare(ref_rows, mine)}
        report.append(row)
        if a.save_fixture and not fixture_done and (w, h) == (1920, 1080):
            keep = canonical(ref_rows)
            keep = keep[keep[:, 1] >= 0.05]
            np.savez_compressed(a.save_fixture, frame_wh=np.array([w, h]), expected=keep,
                                **{k: v for k, v in outputs.items()})
            fixture_done = True
    summarise(report, man.model_id, a.out)


def summarise(report: list[dict], model_id: str, out: str | None) -> None:
    ok = [r for r in report if r["same_count"]]
    print(f"model {model_id}: {len(report)} images, same detection count in {len(ok)}")
    print("  preprocessed input bit-identical:", sum(r["input_identical"] for r in report), "of", len(report))
    print("  class ids equal:", sum(r.get("class_equal", False) for r in ok), "of", len(ok))
    print("  max score diff:", max((r["max_score_diff"] for r in ok), default=None))
    print("  max box diff (px):", max((r["max_box_diff_px"] for r in ok), default=None))
    print("  detections compared:", sum(r["n"] for r in ok))
    if out:
        json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
