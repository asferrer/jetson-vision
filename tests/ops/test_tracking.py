from __future__ import annotations

from jetson_vision.tracking import IoUTracker, TrackerConfig, iou

from .conftest import make_det


def box(x: float, y: float = 0.0, s: float = 100.0) -> tuple[float, float, float, float]:
    return (x, y, x + s, y + s)


def test_iou_values() -> None:
    assert iou(box(0), box(0)) == 1.0
    assert iou(box(0), box(200)) == 0.0
    assert abs(iou(box(0), box(50)) - (50 * 100) / (2 * 10000 - 5000)) < 1e-9
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_stable_ids_across_frames_and_new_object() -> None:
    tr = IoUTracker()
    a = tr.update([make_det(1, 0.9, box(0))], 0)
    b = tr.update([make_det(1, 0.9, box(5)), make_det(1, 0.8, box(400))], 1)
    assert a[0].track_id == 1
    assert b[0].track_id == 1 and b[1].track_id == 2


def test_class_aware() -> None:
    tr = IoUTracker()
    tr.update([make_det(1, 0.9, box(0))], 0)
    out = tr.update([make_det(2, 0.9, box(0))], 1)
    assert out[0].track_id == 2


def test_low_score_keeps_track_but_cannot_start_one() -> None:
    tr = IoUTracker()
    tr.update([make_det(1, 0.9, box(0))], 0)
    out = tr.update([make_det(1, 0.3, box(3))], 1)  # below high, matches the track
    assert [d.track_id for d in out] == [1]
    assert tr.update([make_det(1, 0.3, box(500))], 2) == []  # would-be new track is ignored
    assert tr.update([make_det(1, 0.05, box(3))], 3) == []  # below low: ignored


def test_track_expires_and_gap_resets() -> None:
    tr = IoUTracker(TrackerConfig(max_age_frames=2, max_gap_frames=10))
    tr.update([make_det(1, 0.9, box(0))], 0)
    assert tr.update([make_det(1, 0.9, box(0))], 4)[0].track_id == 2  # expired
    tr.update([], 5)
    assert tr.update([make_det(1, 0.9, box(0))], 100)[0].track_id == 3  # gap: all dropped


def test_min_hits_and_reset() -> None:
    tr = IoUTracker(TrackerConfig(min_hits=2))
    assert tr.update([make_det(1, 0.9, box(0))], 0)[0].track_id is None
    assert tr.update([make_det(1, 0.9, box(0))], 1)[0].track_id == 1
    tr.reset()
    assert tr.update([make_det(1, 0.9, box(0))], 2)[0].track_id is None


def test_greedy_matching_prefers_best_iou() -> None:
    tr = IoUTracker()
    tr.update([make_det(1, 0.9, box(0)), make_det(1, 0.9, box(300))], 0)
    out = tr.update([make_det(1, 0.9, box(305)), make_det(1, 0.9, box(2))], 1)
    assert [d.track_id for d in out] == [2, 1]
