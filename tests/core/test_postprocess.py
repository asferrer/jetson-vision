"""Decoders: real engine outputs saved from the Jetson (tests/core/fixtures) and hand-made cases."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pathlib import Path

from jetson_vision.interfaces import Decoder
from jetson_vision.postprocess import RfDetrDecoder, RtDetrV2Decoder, build_decoder


DATA = Path(__file__).parent / "fixtures"


def as_rows(dets) -> np.ndarray:
    return np.array([[d.class_id, d.score, *d.box_xyxy] for d in dets], dtype=np.float64).reshape(-1, 6)


def canonical(rows: np.ndarray) -> np.ndarray:
    return rows[np.lexsort((rows[:, 2], rows[:, 0], -rows[:, 1]))]


def load(name: str):
    z = np.load(DATA / name)
    outputs = {k: z[k] for k in z.files if k not in ("frame_wh", "expected")}
    return outputs, tuple(int(v) for v in z["frame_wh"]), z["expected"]


def test_rfdetr_real_outputs_match_the_validated_reference(rf_manifest):
    """Expected rows come from stages.py (AP-validated) on the same engine output, scores >= 0.05."""
    outputs, (w, h), expected = load("rfdetr_real.npz")
    got = canonical(as_rows(build_decoder(rf_manifest)(outputs, w, h, 0.05)))
    want = canonical(expected)
    assert len(got) == len(want) > 0
    assert (got[:, 0] == want[:, 0]).all()
    np.testing.assert_allclose(got[:, 1:], want[:, 1:], rtol=0, atol=1e-9)
    assert got[:, 0].max() <= 7  # the spurious 9th output never appears


def test_rtdetrv2_real_outputs_match_the_validated_reference(rt_manifest):
    """The engine saw (640, 640) as target size; the reference engine saw the real frame size.

    Boxes are fp16 at the engine output, so the reference is quantised more at large coordinates: up to 1 px on 1080p.
    """
    outputs, (w, h), expected = load("rtdetrv2_real.npz")
    got = canonical(as_rows(build_decoder(rt_manifest)(outputs, w, h, 0.05)))
    want = canonical(expected)
    assert len(got) == len(want) > 0
    assert (got[:, 0] == want[:, 0]).all()
    np.testing.assert_allclose(got[:, 1], want[:, 1], rtol=0, atol=1e-6)
    assert np.abs(got[:, 2:] - want[:, 2:]).max() <= 1.5


def test_decoders_satisfy_the_protocol(rf_manifest, rt_manifest):
    assert isinstance(build_decoder(rf_manifest), Decoder) and isinstance(build_decoder(rt_manifest), Decoder)


def rf_outputs(logit_row: list[float], box=(0.5, 0.5, 0.2, 0.4)) -> dict[str, np.ndarray]:
    dets = np.tile(np.array([[0.1, 0.1, 0.05, 0.05]], np.float32), (300, 1))
    logits = np.full((300, 9), -20.0, np.float32)
    dets[0], logits[0] = box, logit_row
    return {"dets": dets[None], "labels": logits[None]}


def test_rfdetr_box_math_and_class_mapping(rf_manifest):
    row = [-20.0] * 9
    row[3] = 5.0  # Glove
    dets = RfDetrDecoder(rf_manifest.classes, 300)(rf_outputs(row), 1000, 500, 0.5)
    assert len(dets) == 1
    d = dets[0]
    assert (d.class_id, d.class_name) == (3, "Glove")
    assert d.score == pytest.approx(1 / (1 + np.exp(-5.0)))
    assert d.box_xyxy == pytest.approx((400.0, 150.0, 600.0, 350.0))  # cx 0.5, cy 0.5, w 0.2, h 0.4 of 1000x500


def test_rfdetr_spurious_ninth_output_is_dropped_by_configuration(rf_manifest):
    row = [-20.0] * 9
    row[8] = 9.0  # the extra output of the head
    assert RfDetrDecoder(rf_manifest.classes, 300)(rf_outputs(row), 640, 480, 0.01) == []
    keep = dataclasses.replace(rf_manifest.classes, ignore_labels=())
    assert RfDetrDecoder(keep, 300)(rf_outputs(row), 640, 480, 0.01) == []  # still outside the 8 class names


def test_rfdetr_threshold_and_batchless_input(rf_manifest):
    row = [-20.0] * 9
    row[0] = 0.0  # score 0.5
    out = rf_outputs(row)
    dec = RfDetrDecoder(rf_manifest.classes, 300)
    assert len(dec(out, 100, 100, 0.5)) == 1 and dec(out, 100, 100, 0.51) == []
    batchless = {k: v[0] for k, v in out.items()}
    assert dec(batchless, 100, 100, 0.5) == dec(out, 100, 100, 0.5)


def rt_outputs(label=2, score=0.9, box=(64.0, 64.0, 320.0, 320.0)) -> dict[str, np.ndarray]:
    labels = np.zeros(300, np.int64)
    scores = np.zeros(300, np.float32)
    boxes = np.zeros((300, 4), np.float32)
    labels[0], scores[0], boxes[0] = label, score, box
    return {"labels": labels, "boxes": boxes, "scores": scores}


def test_rtdetrv2_rescales_from_network_space_to_the_frame(rt_manifest):
    dets = RtDetrV2Decoder(rt_manifest.classes, (640, 640))(rt_outputs(), 1280, 320, 0.5)
    assert len(dets) == 1
    assert dets[0].class_name == "Fishing_Net"
    assert dets[0].box_xyxy == pytest.approx((128.0, 32.0, 640.0, 160.0))  # x scale 2, y scale 0.5


def test_rtdetrv2_label_offset_and_out_of_range(rt_manifest):
    shifted = dataclasses.replace(rt_manifest.classes, label_offset=1)
    dec = RtDetrV2Decoder(shifted, (640, 640))
    assert dec(rt_outputs(label=1), 640, 640, 0.5)[0].class_id == 0
    assert dec(rt_outputs(label=0), 640, 640, 0.5) == []  # below the offset
    assert dec(rt_outputs(label=9), 640, 640, 0.5) == []  # past the last class


def test_rtdetrv2_threshold_and_ordering(rt_manifest):
    out = rt_outputs()
    out["scores"][1], out["labels"][1] = 0.95, 1
    out["boxes"][1] = (1.0, 2.0, 3.0, 4.0)
    dets = RtDetrV2Decoder(rt_manifest.classes, (640, 640))(out, 640, 640, 0.5)
    assert [d.class_id for d in dets] == [1, 2]  # by descending score
    assert len(RtDetrV2Decoder(rt_manifest.classes, (640, 640))(out, 640, 640, 0.92)) == 1


def test_rfdetr_ties_are_broken_by_flat_index_so_runs_repeat(rf_manifest):
    dec = build_decoder(rf_manifest)
    logits = np.full((1, 300, 9), -9.0, np.float32)
    logits[0, [7, 3, 200], 2] = 0.5  # three queries with exactly the same score
    boxes = np.tile(np.array([0.5, 0.5, 0.2, 0.2], np.float32), (1, 300, 1))
    boxes[0, 3, 0], boxes[0, 7, 0], boxes[0, 200, 0] = 0.3, 0.6, 0.9
    out = dec({"dets": boxes, "labels": logits}, 100, 100, 0.5)
    assert [round((d.box_xyxy[0] + d.box_xyxy[2]) / 2) for d in out] == [30, 60, 90]  # queries 3, 7, 200
