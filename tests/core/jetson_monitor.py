"""Samples one process and the board once per second into a CSV (not collected by pytest). Needs no root.

  python3 tests/core/jetson_monitor.py --pid <pid> --out monitor.csv [--interval 1.0]

Columns: t_s, proc_cores (CPU of the process in cores), rss_mb, threads, ram_used_mb (MemTotal - MemAvailable),
temp_max_c (hottest thermal zone), gpu_mhz, cpu_mhz (cpu0). It stops by itself when the process ends.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

TICK = os.sysconf("SC_CLK_TCK")
GPU_FREQ = glob.glob("/sys/devices/platform/17000000.gpu/devfreq/*/cur_freq")


def read(path: str) -> str:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            return b"".join(iter(lambda: os.read(fd, 65536), b"")).decode(errors="replace")
        finally:
            os.close(fd)
    except OSError:  # includes EAGAIN, which some thermal zones answer on the Jetson
        return ""


def proc_ticks(pid: int) -> int | None:
    text = read(f"/proc/{pid}/stat")
    if not text:
        return None
    fields = text.rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])  # utime + stime


def proc_status(pid: int) -> tuple[float, int]:
    rss, threads = 0.0, 0
    for line in read(f"/proc/{pid}/status").splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) / 1024
        elif line.startswith("Threads:"):
            threads = int(line.split()[1])
    return rss, threads


def meminfo_used_mb() -> float:
    kv = {ln.split(":")[0]: int(ln.split()[1]) for ln in read("/proc/meminfo").splitlines() if ":" in ln}
    return (kv["MemTotal"] - kv["MemAvailable"]) / 1024


def temp_max_c() -> float:
    temps = [int(t) for t in (read(p).strip() for p in glob.glob("/sys/class/thermal/thermal_zone*/temp")) if t.isdigit()]
    return max(temps, default=0) / 1000


def number(path: str) -> float:
    text = read(path).strip()
    return float(text) if text.isdigit() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()
    t0 = time.monotonic()
    last_ticks, last_t = proc_ticks(a.pid), t0
    with open(a.out, "w") as fh:
        fh.write("t_s,proc_cores,rss_mb,threads,ram_used_mb,temp_max_c,gpu_mhz,cpu_mhz\n")
        while True:
            time.sleep(max(0.0, a.interval - (time.monotonic() - last_t)))
            now, ticks = time.monotonic(), proc_ticks(a.pid)
            if ticks is None or last_ticks is None:
                break
            cores = (ticks - last_ticks) / TICK / (now - last_t)
            rss, threads = proc_status(a.pid)
            gpu = number(GPU_FREQ[0]) / 1e6 if GPU_FREQ else 0.0  # devfreq reports Hz
            cpu = number("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") / 1e3  # kHz
            fh.write(f"{now - t0:.1f},{cores:.3f},{rss:.1f},{threads},{meminfo_used_mb():.1f},"
                     f"{temp_max_c():.1f},{gpu:.0f},{cpu:.0f}\n")
            fh.flush()
            last_ticks, last_t = ticks, now


if __name__ == "__main__":
    main()
