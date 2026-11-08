"""Optional IoU tracker (off by default). Own code, no third-party tracking library.

Two-threshold association in the spirit of ByteTrack, without appearance and without a motion
model: high-score detections are matched to tracks first, then low-score ones are matched to the
tracks that remain. Only high-score detections can start a track. Matching is greedy by IoU and
class-aware. The goal is counting distinct objects and smoothing flicker, not re-identification.
It has not been evaluated on real footage, so counts are not to be trusted yet (04_arquitectura 3.5).

The core must hand this stage the detections below the publication threshold too.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from jetson_vision.types import Detection

Box = tuple[float, float, float, float]


class Tracker(Protocol):
    def update(self, detections: Sequence[Detection], frame_seq: int) -> list[Detection]: ...

    def reset(self) -> None: ...


def iou(a: Box, b: Box) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    class_id: int
    box: Box
    last_seq: int
    hits: int = 1


@dataclass(frozen=True)
class TrackerConfig:
    high_score: float = 0.6
    low_score: float = 0.1
    iou_high: float = 0.3
    iou_low: float = 0.5
    max_age_frames: int = 15
    max_gap_frames: int = 30  # a larger frame_seq jump is a gap: tracks are dropped
    min_hits: int = 1  # hits before a track id is reported


class IoUTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.cfg = config or TrackerConfig()
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._last_seq: int | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._last_seq = None

    def update(self, detections: Sequence[Detection], frame_seq: int) -> list[Detection]:
        """Assign track ids. Returns the detections at or above ``high_score`` (and matched low ones)."""
        if self._last_seq is not None and frame_seq - self._last_seq > self.cfg.max_gap_frames:
            self._tracks.clear()
        self._last_seq = frame_seq
        self._tracks = [t for t in self._tracks
                        if frame_seq - t.last_seq <= self.cfg.max_age_frames]
        high = [d for d in detections if d.score >= self.cfg.high_score]
        low = [d for d in detections if self.cfg.low_score <= d.score < self.cfg.high_score]
        free = list(self._tracks)
        matched, free = self._match(high, free, self.cfg.iou_high, frame_seq)
        matched_low, free = self._match(low, free, self.cfg.iou_low, frame_seq)
        out = self._finish(high, matched, frame_seq)
        out += [replace(d, track_id=t.track_id) for d, t in matched_low.items_list()]
        return out

    def _match(self, dets: list[Detection], tracks: list[_Track], thr: float, seq: int
               ) -> tuple["_Matches", list[_Track]]:
        pairs = sorted(
            ((iou(d.box_xyxy, t.box), i, j)
             for i, d in enumerate(dets) for j, t in enumerate(tracks)
             if d.class_id == t.class_id),
            reverse=True,
        )
        used_d: set[int] = set()
        used_t: set[int] = set()
        result = _Matches(dets)
        for score, i, j in pairs:
            if score < thr:
                break
            if i in used_d or j in used_t:
                continue
            used_d.add(i)
            used_t.add(j)
            trk = tracks[j]
            trk.box, trk.last_seq, trk.hits = dets[i].box_xyxy, seq, trk.hits + 1
            result.add(i, trk)
        return result, [t for j, t in enumerate(tracks) if j not in used_t]

    def _finish(self, high: list[Detection], matched: "_Matches", seq: int) -> list[Detection]:
        out: list[Detection] = []
        for i, det in enumerate(high):
            trk = matched.by_index.get(i)
            if trk is None:
                trk = _Track(self._next_id, det.class_id, det.box_xyxy, seq)
                self._next_id += 1
                self._tracks.append(trk)
            out.append(replace(det, track_id=trk.track_id if trk.hits >= self.cfg.min_hits else None))
        return out


class _Matches:
    def __init__(self, dets: list[Detection]) -> None:
        self._dets = dets
        self.by_index: dict[int, _Track] = {}

    def add(self, index: int, track: _Track) -> None:
        self.by_index[index] = track

    def items_list(self) -> list[tuple[Detection, _Track]]:
        return [(self._dets[i], t) for i, t in self.by_index.items()]
