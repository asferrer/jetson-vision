"""Summarises one `jv run` on the Jetson from its three logs (not collected by pytest; plain numpy, no GPU).

  python3 tests/core/jetson_summary.py --jsonl run.jsonl [--monitor monitor.csv] [--tegra tegrastats.log]
        [--skip 30] [--window-s 60] [--out summary.json]

Latency is total_ns of every result: from the capture stamp (right after the frame left the source) to the
publication. queue_ns is the waiting inside it and service_ns the rest. Dropped frames are the gap between the
frames the source delivered (last frame_seq + 1) and the results published.
"""
from __future__ import annotations

import argparse
import csv
import json
import re

import numpy as np

STAGES = ("capture", "preprocess", "inference", "postprocess", "queue", "service", "total")


def load_results(path: str) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf8"):
        row = json.loads(line)
        if row.get("record", "result") == "result":
            rows.append(row)
    return rows


def pct(values: list[float] | np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if len(values) else None


def stage_ms(row: dict, stage: str) -> float:
    t = row["timings"]
    if stage == "service":
        return max(0, t["total_ns"] - t.get("queue_ns", 0)) / 1e6
    return t[f"{stage}_ns"] / 1e6


def latency_table(rows: list[dict]) -> dict:
    out = {}
    for s in STAGES:
        v = [stage_ms(r, s) for r in rows]
        out[s] = {"p50_ms": pct(v, 50), "p95_ms": pct(v, 95), "p99_ms": pct(v, 99), "max_ms": max(v, default=None)}
    return out


def summarize_results(rows: list[dict], skip: int) -> dict:
    wall = [r["frame_ref"]["t_capture_wall_ns"] for r in rows]
    span = (wall[-1] - wall[0]) / 1e9 if len(rows) > 1 else 0.0
    delivered = rows[-1]["frame_ref"]["frame_seq"] + 1
    steady = rows[skip:]
    return {"results": len(rows), "delivered_by_source": delivered, "dropped": delivered - len(rows),
            "dropped_pct": 100 * (delivered - len(rows)) / max(delivered, 1), "span_s": span,
            "fps": (len(rows) - 1) / span if span > 0 else None, "skipped_warmup": skip,
            "mean_detections": float(np.mean([len(r["detections"]) for r in rows])),
            "schema_version": rows[0]["schema_version"], "latency": latency_table(steady),
            "degradation_levels": sorted({r["degradation_level"] for r in rows})}


def windows(rows: list[dict], seconds: float) -> list[dict]:
    t0 = rows[0]["frame_ref"]["t_capture_wall_ns"]
    groups: dict[int, list[dict]] = {}
    for r in rows:
        groups.setdefault(int((r["frame_ref"]["t_capture_wall_ns"] - t0) / 1e9 // seconds), []).append(r)
    table = []
    for k, g in sorted(groups.items()):
        total = [stage_ms(r, "total") for r in g]
        inf = [stage_ms(r, "inference") for r in g]
        table.append({"window": k, "start_s": k * seconds, "frames": len(g), "total_p50_ms": pct(total, 50),
                      "total_p95_ms": pct(total, 95), "inference_p50_ms": pct(inf, 50),
                      "dropped": g[-1]["frame_ref"]["frame_seq"] - g[0]["frame_ref"]["frame_seq"] + 1 - len(g)})
    return table


def drift(table: list[dict], key: str) -> dict:
    """Least-squares slope of a per-window metric, in ms per hour, over the full windows (the last one is partial)."""
    full = table[:-1] if len(table) > 3 else table
    x = np.array([w["start_s"] for w in full], dtype=float) / 3600
    y = np.array([w[key] for w in full], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(full) > 2 else None
    return {"slope_ms_per_hour": slope, "first_ms": float(y[0]), "last_ms": float(y[-1]),
            "min_ms": float(y.min()), "max_ms": float(y.max())}


def summarize_monitor(path: str, skip_s: float = 30) -> dict:
    rows = [r for r in csv.DictReader(open(path)) if float(r["t_s"]) >= skip_s]
    col = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}
    if not col:
        return {}
    return {"samples": len(rows), "proc_cores_mean": float(col["proc_cores"].mean()),
            "proc_cores_max": float(col["proc_cores"].max()), "rss_mb_mean": float(col["rss_mb"].mean()),
            "rss_mb_max": float(col["rss_mb"].max()), "rss_mb_first": float(col["rss_mb"][0]),
            "rss_mb_last": float(col["rss_mb"][-1]), "ram_used_mb_max": float(col["ram_used_mb"].max()),
            "temp_max_c": float(col["temp_max_c"].max()), "temp_mean_c": float(col["temp_max_c"].mean()),
            "gpu_mhz_mean": float(col["gpu_mhz"].mean()), "gpu_mhz_min": float(col["gpu_mhz"].min()),
            "gpu_mhz_max": float(col["gpu_mhz"].max()), "cpu_mhz_mean": float(col["cpu_mhz"].mean())}


def summarize_tegra(path: str, skip: int = 30) -> dict:
    lines = [ln for ln in open(path) if "RAM" in ln][skip:]
    keys = {"ram_mb": r"RAM (\d+)/", "gr3d_pct": r"GR3D_FREQ (\d+)%", "vdd_in_mw": r"VDD_IN (\d+)mW",
            "tj_c": r"tj@([\d.]+)C", "gpu_mhz": r"GR3D_FREQ \d+%@\[?(\d+)", "emc_pct": r"EMC_FREQ (\d+)%"}
    vals = {k: [float(m.group(1)) for ln in lines if (m := re.search(p, ln))] for k, p in keys.items()}
    cpu = [np.mean([int(x) for x in re.findall(r"(\d+)%@", re.search(r"CPU \[(.*?)\]", ln).group(1))]) for ln in lines
           if re.search(r"CPU \[", ln)]
    if not lines:
        return {}
    nvdec = sum(1 for ln in lines if re.search(r"NVDEC (?!off)", ln))
    vic = sum(1 for ln in lines if re.search(r"VIC (?!off)", ln))
    f = lambda k, fn: float(fn(vals[k])) if vals[k] else None  # noqa: E731
    return {"samples": len(lines), "ram_mb_mean": f("ram_mb", np.mean), "ram_mb_max": f("ram_mb", np.max),
            "cpu_pct_mean_per_core": float(np.mean(cpu)) if cpu else None, "gr3d_pct_mean": f("gr3d_pct", np.mean),
            "gr3d_mhz_mean": f("gpu_mhz", np.mean), "vdd_in_w_mean": (f("vdd_in_mw", np.mean) or 0) / 1000,
            "vdd_in_w_max": (f("vdd_in_mw", np.max) or 0) / 1000, "tj_max_c": f("tj_c", np.max),
            "emc_pct_mean": f("emc_pct", np.mean), "nvdec_active_samples": nvdec, "vic_active_samples": vic}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--monitor")
    ap.add_argument("--tegra")
    ap.add_argument("--skip", type=int, default=30)
    ap.add_argument("--window-s", type=float, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = load_results(a.jsonl)
    out = {"jsonl": a.jsonl, **summarize_results(rows, a.skip)}
    if a.monitor:
        out["monitor"] = summarize_monitor(a.monitor)
    if a.tegra:
        out["tegrastats"] = summarize_tegra(a.tegra)
    if a.window_s:
        out["windows"] = windows(rows, a.window_s)
        out["drift_total_p50"] = drift(out["windows"], "total_p50_ms")
        out["drift_inference_p50"] = drift(out["windows"], "inference_p50_ms")
    text = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w").write(text)
    print(json.dumps({k: v for k, v in out.items() if k != "windows"}, indent=1))


if __name__ == "__main__":
    main()
