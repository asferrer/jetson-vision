# jetson-vision v1 design

Vision inference module for a small underwater robot on a Jetson Orin Nano Super (JetPack 6.2.1, TensorRT 10.3, 15 W mode).
It detects marine debris in 8 classes. It does not depend on a specific robot or camera. Decisions taken on 2026-11-06.

## Decisions

1. **Default detector: RF-DETR INT8 G** (`rfdetr_exp07`, AP 43.31 on VAL). It is the only one measured that delivers 30 fps without
   dropping frames: 45 ms from a 416×416 JPEG and 68 ms from 1080p. **RT-DETRv2 mixed INT8** (`exp08_real_synth_minor_s3`,
   AP 43.60) remains as a second manifest for when AP matters more than 30 fps (20 fps at 48 ms).
2. **Gemma stays out of the real-time path.** On this Jetson no route preserves the fine-tuned vision at a useful speed.
3. **NVIDIA stack in use:** TensorRT, GStreamer with NVDEC and VIC for video, jtop/tegrastats for health, `nsys` for profiling.
   No DeepStream, Isaac ROS, Holoscan or TensorRT-LLM. VPI stays out of the detector path: measured, it is slower
   than torch and gives AP −0.22. The Orin Nano has no DLA, PVA or NVENC.
4. **Preprocessing fixed by manifest.** Resizing on CPU over u8; normalization and transposition on GPU. RF-DETR uses
   `cv2 INTER_LINEAR` (measured: −48 % time, AP +0.13 versus PIL); RT-DETRv2 uses `cv2 INTER_AREA`. Prohibited without measuring AP:
   pinned or zero-copy memory (both worsen it), GPU postprocessing (worsens it) and `nvjpegdec` (AP 3.02, undiagnosed).
5. **ROS-free core.** `jetson_vision/` does not import `rclpy`; an automated test checks this. Adapters contain no vision logic.
6. **Three-thread pipeline** (capture, inference, publishing). Live, the latest frame wins; in replay, a bounded blocking queue is
   used. Drops are computed from jumps in `frame_seq`.
7. **Health:** levels without changing `nvpmodel` at runtime: 0 normal, 1 lower FPS, 2 alerts only, 3 stop. Throttling is
   detected by latency drift against a baseline, because the hardware does not always notify the operating system.
8. **Underwater preprocessing off.** The model was trained without it. It could only be enabled by measuring AP on VAL with the 99 % rule.
9. **Optional tracking, off by default** (simple IoU, with own or MIT-licensed code; no AGPL).

10. **Validated on the Jetson (2026-11-06)** (`docs/VALIDATION.md`): RF-DETR threshold 0.45 (F1 maximum on VAL with its engine; 0.6
   was RT-DETRv2's), `SCHEMA_VERSION` 2 (`StageTimings.queue_ns`) and `Pipeline.process_frame` as the single-frame entry point. For the
   robot, `jetson_clocks` at 15 W is recommended (a deployment decision, not the module's).

## Goals

These are goals, not measurements: end-to-end p95 ≤ 100 ms and ≥ 15 FPS sustained at 15 W. AP on a fixed VAL subset
must not drop below the measured reference. TEST is not used.

## Structure

```
jetson_vision/          core, no ROS
  types.py              data contract (fixed)
  interfaces.py         protocols between components (fixed)
  manifest.py           model.yaml: hash, TensorRT version, GPU, input, decoder, classes
  sources/              file, folder, video, v4l2, gstreamer
  preprocess.py         manifest-driven resizing; asserts a contiguous buffer
  backends/             tensorrt, fake
  postprocess.py        RF-DETR and RT-DETRv2 decoders
  pipeline.py           threads, mailbox, metrics
  health.py             temperature, latency drift, levels
  sinks.py              JSON Lines and compact summary
  tracking.py           optional
  cli.py                jv run | bench | replay
adapters/ros2/          colcon package, LifecycleNode node
deploy/                 systemd with watchdog, Jetson env file, installation
models/                 manifests (no weights)
tests/                  unit tests with a fake backend, replay, AP regression
```

## Rules

Engines, ONNX files and data are never committed. The loader refuses to start if the engine does not match the TensorRT version or
the GPU, and it does not rebuild on the robot. Every figure in a README comes from a measurement with its results file. Code and
comments in English.
