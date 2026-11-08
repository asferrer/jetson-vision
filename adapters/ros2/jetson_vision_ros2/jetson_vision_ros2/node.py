"""LifecycleNode: sensor_msgs/Image in, vision_msgs/Detection2DArray out. No vision logic here.

The node converts messages (mapping.py), calls the core processor and publishes. Level policy
comes from the core HealthMonitor: level 1 processes one of every two frames, level 2 one frame
per ``alert_period_s``, level 3 processes nothing and reports an error in diagnostics.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Callable, Protocol

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from jetson_vision_msgs.msg import FrameMeta
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from jetson_vision.health import HealthConfig, HealthMonitor, LatencyBaseline
from jetson_vision.types import DegradationLevel, Frame, FrameResult

from . import mapping


class Processor(Protocol):
    """What the node needs from the core (jetson_vision.pipeline.Pipeline): one frame in, one result out."""

    model_id: str

    def process_frame(self, frame: Frame) -> FrameResult: ...

    def close(self) -> None: ...


ProcessorFactory = Callable[[str, str], Processor]


def _fill_header(header: Any, data: mapping.HeaderData) -> None:
    header.stamp.sec = data.sec
    header.stamp.nanosec = data.nanosec
    header.frame_id = data.frame_id


def build_detection_array(data: mapping.DetectionArrayData) -> Detection2DArray:
    msg = Detection2DArray()
    _fill_header(msg.header, data.header)
    for d in data.detections:
        det = Detection2D()
        _fill_header(det.header, data.header)
        det.id = d.track_id
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = d.class_id
        hyp.hypothesis.score = d.score
        det.results.append(hyp)
        det.bbox.center.position.x = d.center_x
        det.bbox.center.position.y = d.center_y
        det.bbox.size_x = d.size_x
        det.bbox.size_y = d.size_y
        msg.detections.append(det)
    return msg


def build_meta(result: FrameResult, header: mapping.HeaderData) -> FrameMeta:
    msg = FrameMeta()
    _fill_header(msg.header, header)
    for key, value in mapping.result_to_meta(result).items():
        setattr(msg, key, value)
    return msg


def build_diagnostics(plain: dict[str, Any], stamp: Any) -> DiagnosticArray:
    status = DiagnosticStatus()
    status.level = bytes([plain["level"]])
    status.name, status.message = plain["name"], plain["message"]
    status.hardware_id = plain["hardware_id"]
    status.values = [KeyValue(key=k, value=v) for k, v in plain["values"].items()]
    msg = DiagnosticArray()
    msg.header.stamp = stamp
    msg.status.append(status)
    return msg


class VisionNode(LifecycleNode):
    def __init__(self, processor_factory: ProcessorFactory | None = None) -> None:
        super().__init__("jetson_vision")
        self._factory = processor_factory
        self.declare_parameter("manifest", "")
        self.declare_parameter("backend", "tensorrt")
        self.declare_parameter("image_topic", "image_raw")
        self.declare_parameter("input_fps", 30.0)
        self.declare_parameter("alert_period_s", 1.0)
        self.declare_parameter("baseline_fps", [30.0])
        self.declare_parameter("baseline_ms", [0.0])
        self._proc: Processor | None = None
        self._sub: Any = None
        self._seq = 0
        self._session = uuid.uuid4().hex
        self._last_processed = 0.0
        self._skip = 0
        self._monitor = HealthMonitor()

    def on_configure(self, _state: Any) -> TransitionCallbackReturn:
        factory = self._factory or _default_factory()
        manifest = self.get_parameter("manifest").value
        backend = self.get_parameter("backend").value
        try:
            self._proc = factory(manifest, backend)
        except Exception as exc:  # noqa: BLE001 - report and stay unconfigured
            self.get_logger().error(f"cannot build the core processor: {exc}")
            return TransitionCallbackReturn.FAILURE
        self._pub_det = self.create_lifecycle_publisher(Detection2DArray, "~/detections", 10)
        self._pub_meta = self.create_lifecycle_publisher(FrameMeta, "~/frame_meta", 10)
        self._pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self._monitor = self._make_monitor()
        self._timer = self.create_timer(1.0, self._on_timer)
        return TransitionCallbackReturn.SUCCESS

    def _make_monitor(self) -> HealthMonitor:
        fps = list(self.get_parameter("baseline_fps").value)
        ms = list(self.get_parameter("baseline_ms").value)
        pairs = {f: m for f, m in zip(fps, ms) if m > 0}  # 0 = no baseline measured
        monitor = HealthMonitor(HealthConfig(), LatencyBaseline(pairs))
        monitor.set_target_fps(float(self.get_parameter("input_fps").value))
        return monitor

    def on_activate(self, state: Any) -> TransitionCallbackReturn:
        self._sub = self.create_subscription(
            Image, self.get_parameter("image_topic").value, self._on_image,
            qos_profile_sensor_data)
        return super().on_activate(state)

    def on_deactivate(self, state: Any) -> TransitionCallbackReturn:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        return super().on_deactivate(state)

    def on_cleanup(self, _state: Any) -> TransitionCallbackReturn:
        self.destroy_timer(self._timer)
        if self._proc is not None:
            self._proc.close()
            self._proc = None
        return TransitionCallbackReturn.SUCCESS

    def _should_process(self) -> bool:
        level = self._monitor.level
        if level == DegradationLevel.STOPPED:
            return False
        if level == DegradationLevel.ALERTS_ONLY:
            period = float(self.get_parameter("alert_period_s").value)
            return time.monotonic() - self._last_processed >= period
        if level == DegradationLevel.REDUCED_FPS:
            self._skip ^= 1
            return self._skip == 0
        return True

    def _on_image(self, msg: Image) -> None:
        if self._proc is None or not self._should_process():
            return
        now = time.monotonic()
        if self._last_processed:
            self._monitor.observe_fps(1.0 / max(now - self._last_processed, 1e-6))
        self._last_processed = now
        seq, self._seq = self._seq, self._seq + 1  # frame_seq starts at 0, like every core source
        try:
            frame = mapping.image_to_frame(
                encoding=msg.encoding, height=msg.height, width=msg.width, step=msg.step,
                data=bytes(msg.data), stamp_sec=msg.header.stamp.sec,
                stamp_nanosec=msg.header.stamp.nanosec, frame_id=msg.header.frame_id,
                frame_seq=seq, session_id=self._session,
                mono_ns=time.monotonic_ns(), wall_ns=time.time_ns())
            result = self._proc.process_frame(frame)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the node
            self.get_logger().error(f"frame {seq} failed: {exc}")
            return
        self._monitor.observe_latency_ms(result.timings.inference_ns / 1e6)
        result.degradation_level = self._monitor.level
        self._publish(result)

    def _publish(self, result: FrameResult) -> None:
        header = mapping.header_from_ref(result.frame_ref)
        self._pub_det.publish(build_detection_array(mapping.result_to_detections(result)))
        self._pub_meta.publish(build_meta(result, header))

    def _effective_fps(self) -> float:
        """Cadence the node actually processes at, so the latency baseline matches it."""
        if self._monitor.level == DegradationLevel.ALERTS_ONLY:
            return 1.0 / max(float(self.get_parameter("alert_period_s").value), 1e-3)
        return float(self.get_parameter("input_fps").value) * self._monitor.fps_factor()

    def _on_timer(self) -> None:
        state = self._monitor.evaluate()
        self._monitor.set_target_fps(self._effective_fps())
        model_id = getattr(self._proc, "model_id", "unknown")
        plain = mapping.health_to_diagnostics(state, model_id, os.uname().nodename)
        self._pub_diag.publish(build_diagnostics(plain, self.get_clock().now().to_msg()))


def _default_factory() -> ProcessorFactory:
    from .factory import build_processor  # imports the core pipeline lazily

    return build_processor


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
