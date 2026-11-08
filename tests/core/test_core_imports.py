"""The core must not depend on ROS (design decision 5), nor on GPU libraries at import time."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "jetson_vision"
ROS_IMPORT = re.compile(r"^\s*(import|from)\s+(rclpy|vision_msgs|sensor_msgs|diagnostic_msgs|std_msgs|geometry_msgs|rosbag2\w*|cv_bridge)\b", re.M)  # noqa: E501


def test_no_core_source_imports_ros():
    offenders = [str(p.relative_to(ROOT)) for p in CORE.rglob("*.py") if ROS_IMPORT.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_importing_every_core_module_does_not_load_rclpy_or_gpu_libraries():
    code = (
        f"import sys, pkgutil, importlib; sys.path.insert(0, {str(ROOT)!r}); import jetson_vision;"
        "[importlib.import_module(m.name) for m in pkgutil.walk_packages(jetson_vision.__path__, 'jetson_vision.')];"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('rclpy', 'rclpy_', 'tensorrt', 'torch', 'gi')];"
        "assert not bad, bad"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_core_runs_when_rclpy_is_unimportable():
    code = (f"import sys; sys.modules['rclpy'] = None; sys.path.insert(0, {str(ROOT)!r});"
            "import jetson_vision.pipeline, jetson_vision.sources, jetson_vision.backends.fake")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
