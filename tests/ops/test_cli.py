"""CLI end to end with the fake backend and a folder of synthetic images."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from jetson_vision import cli

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = str(ROOT / "models" / "rfdetr_int8g.yaml")


@pytest.fixture
def images(tmp_path: Path) -> Path:
    d = tmp_path / "imgs"
    d.mkdir()
    rng = np.random.default_rng(1)
    for i in range(5):
        cv2.imwrite(str(d / f"i{i}.png"), rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))
    return d


def run_cli(*argv: str) -> int:
    return cli.main(list(argv))


def test_run_writes_jsonl_and_compact(images: Path, tmp_path: Path) -> None:
    out, hexf = tmp_path / "o.jsonl", tmp_path / "o.hex"
    code = run_cli("run", "--source", str(images), "--model", MANIFEST, "--backend", "fake",
                   "--sink", f"jsonl:{out}", "--sink", f"compact:{hexf}")
    assert code == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert rows[0]["record"] == "event" and rows[0]["kind"] == "start"
    results = [r for r in rows if r["record"] == "result"]
    assert [r["frame_ref"]["frame_seq"] for r in results] == [0, 1, 2, 3, 4]
    assert results[0]["model_id"] == "rfdetr_exp07_qG_int8" and results[0]["schema_version"] == 2
    assert all(len(bytes.fromhex(line)) <= 48 for line in hexf.read_text().split())


def test_restart_cause_is_written(images: Path, tmp_path: Path, monkeypatch) -> None:
    cause = tmp_path / "cause"
    cause.write_text("watchdog/exit-code\n")
    monkeypatch.setenv("JV_LAST_STOP_FILE", str(cause))
    out = tmp_path / "o.jsonl"
    run_cli("run", "--source", str(images), "--model", MANIFEST, "--backend", "fake",
            "--sink", f"jsonl:{out}", "--max-frames", "2")
    assert json.loads(out.read_text().splitlines()[0])["restart_cause"] == "watchdog/exit-code"


def test_run_with_tracker_assigns_ids(images: Path, tmp_path: Path) -> None:
    out = tmp_path / "o.jsonl"
    assert run_cli("run", "--source", str(images), "--model", MANIFEST, "--backend", "fake",
                   "--sink", f"jsonl:{out}", "--track", "--threshold", "0.3") == 0
    dets = [d for r in map(json.loads, out.read_text().splitlines()) if r["record"] == "result"
            for d in r["detections"]]
    assert dets and all(d["track_id"] is not None for d in dets)


def test_bench_prints_metrics(images: Path, capsys) -> None:
    assert run_cli("bench", "--source", str(images), "--model", MANIFEST, "--backend", "fake") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["processed"] == 5 and data["stages"]["inference"]["p50_ms"] is not None


def test_replay_is_deterministic_against_reference(images: Path, tmp_path: Path, capsys) -> None:
    ref = tmp_path / "ref.jsonl"
    args = ("--source", str(images), "--model", MANIFEST, "--backend", "fake")
    assert run_cli("replay", *args, "--sink", f"jsonl:{ref}") == 0
    assert run_cli("replay", *args, "--reference", str(ref)) == 0
    assert "replay identical" in capsys.readouterr().out


def test_replay_detects_a_changed_threshold(images: Path, tmp_path: Path) -> None:
    ref = tmp_path / "ref.jsonl"
    args = ("--source", str(images), "--model", MANIFEST, "--backend", "fake")
    run_cli("replay", *args, "--sink", f"jsonl:{ref}")
    assert run_cli("replay", *args, "--threshold", "0.99", "--reference", str(ref)) == 1


def test_compare_results_reports_each_kind_of_difference() -> None:
    def row(seq: int, score: float = 0.9) -> dict:
        return {"frame_ref": {"frame_seq": seq},
                "detections": [{"class_id": 1, "score": score, "box_xyxy": [0, 0, 10, 10]}]}

    assert cli.compare_results([row(0)], [row(0)]) == []
    assert "frame count" in cli.compare_results([row(0)], [row(0), row(1)])[0]
    assert "frame_seq" in cli.compare_results([row(1)], [row(0)])[0]
    assert "differ" in cli.compare_results([row(0, 0.5)], [row(0)])[0]
    other = row(0)
    other["detections"] = []
    assert "detections" in cli.compare_results([other], [row(0)])[0]


def test_usage_errors(images: Path, capsys) -> None:
    assert run_cli("run", "--source", str(images), "--model", MANIFEST, "--backend", "fake",
                   "--sink", "nonsense:x") == 2
    assert run_cli("run", "--source", str(images), "--model", MANIFEST, "--backend", "fake",
                   "--baseline", "abc") == 2
    assert run_cli("run", "--source", str(images), "--model", str(images / "none.yaml"),
                   "--backend", "fake") == 2
    assert "jv: error" in capsys.readouterr().err


def test_parse_baseline() -> None:
    assert cli.parse_baseline("30:20,15:30").by_fps == {30.0: 20.0, 15.0: 30.0}
    assert cli.parse_baseline(None).by_fps == {}


def test_compare_results_ignores_the_order_of_detections_with_equal_scores() -> None:
    def det(cls: int, x: float) -> dict:
        return {"class_id": cls, "score": 0.25, "box_xyxy": [x, 0.0, x + 5, 5.0]}

    def row(dets: list) -> dict:
        return {"frame_ref": {"frame_seq": 0}, "detections": dets}

    assert cli.compare_results([row([det(1, 0), det(2, 9)])], [row([det(2, 9), det(1, 0)])]) == []


def test_run_rotates_the_jsonl_log_at_the_configured_size(images: Path, tmp_path: Path) -> None:
    out = tmp_path / "o.jsonl"
    assert run_cli("run", "--source", str(images), "--model", MANIFEST, "--backend", "fake",
                   "--sink", f"jsonl:{out}", "--log-max-mb", "0.0005") == 0  # 500 bytes
    assert (tmp_path / "o.jsonl.1").exists()


def test_a_failing_health_tick_does_not_stop_the_detector(images: Path, tmp_path: Path, monkeypatch) -> None:
    def boom(self) -> None:
        raise RuntimeError("thermal sensor exploded")

    monkeypatch.setattr(cli.Supervisor, "tick", boom)
    monkeypatch.setattr(cli, "TICK_S", 0.05)
    out = tmp_path / "o.jsonl"
    assert run_cli("run", "--source", str(images), "--model", MANIFEST, "--backend", "fake",
                   "--sink", f"jsonl:{out}", "--fps", "20") == 0
    assert sum('"record":"result"' in line for line in out.read_text().splitlines()) == 5


def test_max_seconds_stops_a_run_cleanly(images: Path, tmp_path: Path) -> None:
    out = tmp_path / "o.jsonl"
    assert run_cli("run", "--source", str(images), "--model", MANIFEST, "--backend", "fake",
                   "--sink", f"jsonl:{out}", "--fps", "5", "--max-seconds", "0.5") == 0
    n = sum('"record":"result"' in line for line in out.read_text().splitlines())
    assert 1 <= n < 5  # five images at 5 fps would take a second
