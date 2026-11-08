"""On-device AP regression helpers (not collected by pytest). VAL only, never TEST.

  prepare    folder of numbered symlinks to VAL images (a fixed subset or all of VAL) + list.json (seq -> image id)
  reference  JSONL of the independent reference path (stages.py, validated against AP 43.61 / 43.32), same order
  filter     keep the detections with score >= --min-score (compare at the operating threshold, where no tie
             at the top-k boundary exists; at threshold 0 two implementations may keep different tied entries)
  ap         COCO AP of a `jv replay --sink jsonl:...` run, against the reference AP of the manifest

  $PY tests/core/jetson_regression.py prepare --val V --data-root R --out-dir D [--all]
  $PY tests/core/jetson_regression.py reference --manifest M --stages-dir S --list D/list.json --out ref.jsonl
  jv replay --source D --model M --threshold <0 | 0.001> --sink jsonl:run.jsonl --reference ref.jsonl
  $PY tests/core/jetson_regression.py ap --manifest M --val V --list D/list.json --jsonl run.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jetson_compare import REF_THRESHOLD, canonical, pick_images, reference_rows  # noqa: E402
from jetson_vision.manifest import Manifest  # noqa: E402


def cmd_prepare(a: argparse.Namespace) -> None:
    val = json.load(open(a.val, encoding="utf8"))
    images = sorted(val["images"], key=lambda d: d["id"]) if a.all else pick_images(val["images"], 100, 40, 20)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    listing = []
    for seq, im in enumerate(images):
        src = Path(a.data_root) / im["file_name"]
        link = out / f"{seq:05d}_{im['id']}{src.suffix.lower()}"
        if not link.exists():
            os.symlink(src, link)
        listing.append({"seq": seq, "image_id": im["id"], "file": str(src), "size": [im["width"], im["height"]]})
    json.dump(listing, open(out / "list.json", "w"))
    print(f"{len(listing)} images in {out}")


def cmd_reference(a: argparse.Namespace) -> None:
    import stages  # reference implementation, tools/reference/stages.py (--stages-dir)

    man = Manifest.load(a.manifest)
    rt = man.decoder.name == "rtdetrv2"
    engine = str(man.engine_path(os.environ.get("JV_MODEL_DIR")))
    ref = stages.Pipe(SimpleNamespace(model="rt" if rt else "rf", engine=engine, dec="cv2", pre="u8gpu",
                                      copy="pageable", post="cpu", d2h="sync"))
    thr = REF_THRESHOLD[man.decoder.name]
    with open(a.out, "w") as fh:
        for item in json.load(open(a.list)):
            res, _ = ref.one(open(item["file"], "rb").read())
            rows = canonical(reference_rows(ref, res, rt))
            rows = rows[rows[:, 1] >= thr]
            dets = [{"class_id": int(r[0]), "score": float(r[1]), "box_xyxy": [float(x) for x in r[2:]]}
                    for r in rows]
            fh.write(json.dumps({"record": "result", "frame_ref": {"frame_seq": item["seq"]},
                                 "detections": dets}) + "\n")


def cmd_filter(a: argparse.Namespace) -> None:
    with open(a.out, "w") as out:
        for line in open(a.jsonl):
            row = json.loads(line)
            row["detections"] = [d for d in row["detections"] if d["score"] >= a.min_score]
            out.write(json.dumps(row) + chr(10))


def cmd_ap(a: argparse.Namespace) -> None:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    man = Manifest.load(a.manifest)
    ids = {i["seq"]: i["image_id"] for i in json.load(open(a.list))}
    preds = []
    for line in open(a.jsonl):
        row = json.loads(line)
        if row.get("record", "result") != "result":
            continue
        image_id = ids[row["frame_ref"]["frame_seq"]]
        for d in row["detections"]:
            x1, y1, x2, y2 = d["box_xyxy"]
            preds.append({"image_id": image_id, "category_id": d["class_id"],
                          "bbox": [x1, y1, x2 - x1, y2 - y1], "score": d["score"]})
    gt = COCO(a.val)
    ev = COCOeval(gt, gt.loadRes(preds), "bbox")
    ev.params.imgIds = sorted(ids.values())
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    ap = 100 * ev.stats[0]
    print(json.dumps({"model": man.model_id, "n_images": len(ids), "n_predictions": len(preds), "AP": ap,
                      "AP50": 100 * ev.stats[1], "reference_ap": man.reference_ap.value,
                      "diff_vs_reference": ap - man.reference_ap.value}))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--val", required=True)
    pr.add_argument("--data-root", required=True)
    pr.add_argument("--out-dir", required=True)
    pr.add_argument("--all", action="store_true")
    rf = sub.add_parser("reference")
    rf.add_argument("--manifest", required=True)
    rf.add_argument("--stages-dir", required=True)
    rf.add_argument("--list", required=True)
    rf.add_argument("--out", required=True)
    fl = sub.add_parser("filter")
    fl.add_argument("--jsonl", required=True)
    fl.add_argument("--min-score", type=float, required=True)
    fl.add_argument("--out", required=True)
    ap = sub.add_parser("ap")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--jsonl", required=True)
    return p


def main() -> None:
    a = build_parser().parse_args()
    if a.cmd == "reference":
        sys.path.insert(0, a.stages_dir)
    {"prepare": cmd_prepare, "reference": cmd_reference, "filter": cmd_filter, "ap": cmd_ap}[a.cmd](a)


if __name__ == "__main__":
    main()
