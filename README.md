# jetson-vision

Real-time inference module for a small underwater robot (Jetson Orin Nano Super, JetPack 6.2.1): it detects marine debris in
8 classes. It does not depend on a particular robot or camera. Design decisions are in `docs/DESIGN.md`; every figure in
this README comes from a file in `docs/validation_data/`, and the conditions are in `docs/VALIDATION.md`.

License: Apache-2.0 (`LICENSE`). Third-party components and data: `NOTICE`.

## Status

Validated on a real Jetson Orin Nano Super (15 W power mode, dynamic clocks) with **synthetic H.264 video** made of
validation photos and decoded by NVDEC. Model weights and TensorRT engines are **not** in this repository (see
"Models and engines").

| | RF-DETR INT8 G (default) | RT-DETRv2 INT8 mixed |
|---|---|---|
| AP on full VAL (2,185 images) | 43.445 (prototype reference 43.45) | 43.605 (prototype reference 43.61) |
| End to end at 30 fps input, 1080p and 720p | 29.9 to 30.0 fps, 0.2 % drops, p95 latency 36 to 41 ms | 22 to 24 fps, 18 to 26 % drops, p95 74 to 76 ms |
| End to end at 15 fps input | 15.0 fps, no drops, p95 43 to 44 ms | 15.0 fps, no drops, p95 50 to 55 ms |
| 32 min at 30 fps, 1080p | 29.998 fps, p95 41.7 ms, stable RSS after warm-up, 60 C, no throttling | not measured |
| Maximum throughput (no input limit) | 32 fps | 22 to 24 fps |
| Board power (`VDD_IN`) at 30 fps / 15 fps | 9.9 to 10.2 W / 7.0 to 7.2 W | 9.4 to 9.5 W / 7.5 to 7.6 W |

The table above was measured with the serial (single inference thread) pipeline. The staged pipeline described in
"Performance" replaced it afterwards; the before/after measurements are in `docs/VALIDATION.md`, section 13 (RF-DETR: capacity 31 to 33 fps, p95 latency at 30 fps 47 to 38 ms).

With `jetson_clocks` (at 15 W) the RF-DETR latency at 15 fps drops from 41 to 29 ms (p95 from 43 to 30) for +0.1 W
(`docs/VALIDATION.md`, section 10). Targets (`docs/DESIGN.md`): p95 <= 100 ms and >= 15 FPS at 15 W.

**Power modes.** All figures in this README are at **15 W**. In `docs/VALIDATION.md`, section 13.1, the staged pipeline was
also measured at 25 W (RF-DETR capacity 48 fps, RT-DETRv2 35 fps, both hold 30 fps input) and MAXN_SUPER (51 and 37 fps) with
dynamic clocks; 7 W could not be measured because `nvpmodel` requires a reboot for it.

![Validation AP against GPU latency and energy per image of the TensorRT engines on the Jetson Orin Nano Super](docs/images/ap-latency-energy.png)

*Figure from the accompanying article. (a) Validation AP against GPU latency at 15 W of the TensorRT engines (gray: FP32;
teal: FP16; light teal: mixed; orange: INT8). (b) Energy per image of the main configuration of each detector in each power
mode. The figure also shows DiffusionDet, which this repository does not deploy. Its RF-DETR AP (43.31) uses a PIL resize, as
in the training-side evaluation, so it differs from the 43.445 above (see "About the AP figures").*

### About the AP figures

The **43.445 AP** of RF-DETR is measured **with this pipeline's preprocessing** (cv2 `INTER_LINEAR` resize on uint8, then
normalisation on the GPU). It is a **regression of this module against the author's prototype** (same decoding code, same
VAL images), not an independent evaluation. The same engine evaluated with a PIL resize, as in the training-side evaluation,
gives **43.31 to 43.32**; the +0.13 comes from the preprocessing, not from a better model. The 0.45 score threshold of
RF-DETR was chosen on VAL (F1 peak) and the AP is also measured on VAL; the test split was not used. RT-DETRv2 boxes differ
from the prototype by up to 4.1 px on very large frames (cause not verified, `docs/VALIDATION.md` section 4); its AP
difference is 0.009.

### What has NOT been tested

- **A physical camera** (CSI, USB/V4L2, RTSP): the `cam:` sources are implemented and were exercised only with a simulated
  live source (`cam:test`, `videotestsrc`) and with missing devices, never with real hardware. Before that, only `gst:file:`
  and RTP over UDP, with NVDEC, had been measured (`docs/VALIDATION.md`, section 14).
- **ROS 2 on the Jetson**: the adapter is tested only in a container on a PC (CPU, fake backend).
- **Real water**: no turbidity, no moving underwater scenes. The VAL images are 92.5 % 416x416 photos, and the video is
  slides of still photos, so decoding is lighter than with real footage.
- **Long runs**: 32 min (staged pipeline: see section 13), not 8 to 24 h; no sealed enclosure, so the temperature
  thresholds of the health monitor are provisional.
- Latency starts when the frame leaves the decoder; sensor exposure, ISP and transport are not included.
- The system-wide systemd unit (needs `sudo`) was not tested, only the user unit (`systemctl --user`).
- Tracking (`--track`) was not evaluated.
- Power is the whole board (`VDD_IN`), without propulsion or lighting.

## Models and engines

From a trained detector to a running engine. This workflow comes from the accompanying article; this repository covers the
last two boxes, from the engine file on:

![Deployment workflow: PyTorch, ONNX, TensorRT engine, real-time inference](docs/images/deployment-pipeline.png)

*Figure from the accompanying article.*

No weights, ONNX files or TensorRT engines are distributed here (`.gitignore` excludes them). A manifest in `models/`
describes one engine: file name, SHA-256, TensorRT version, GPU architecture, input size and normalisation, decoder,
classes, threshold. `jv` refuses to load an engine whose hash, TensorRT version or GPU does not match its manifest.

How `jv` checks an engine before it runs a single frame (`jetson_vision/manifest.py`, `jetson_vision/backends/tensorrt.py`):

```mermaid
flowchart TD
    M["Manifest YAML<br/>engine file, SHA-256, TensorRT version, GPU,<br/>input, decoder, classes, threshold"] --> P["Engine path<br/>--model-dir, then JV_MODEL_DIR,<br/>then the manifest folder"]
    P --> H{"SHA-256 of the file<br/>equals the manifest?"}
    H -- no --> X["EngineMismatchError<br/>jv exits with code 1"]
    H -- yes --> R{"TensorRT version, GPU architecture,<br/>name and SM count match?"}
    R -- no --> X
    R -- yes --> D{"TensorRT deserializes the engine<br/>and its tensors match the manifest?"}
    D -- no --> X
    D -- yes --> W["Allocate buffers, 3 warm-up runs"]
    W --> OK["Engine ready<br/>the manifest also fixes the preprocessing and the decoder"]
```

An engine is never rebuilt on the robot: if any check fails, `jv` stops.

To use the module you need to bring your own engine:

1. Train or obtain an RF-DETR or RT-DETRv2 detector with the 8 classes (Bottle, Can, Fishing_Net, Glove, Mask,
   Metal_Debris, Plastic_Debris, Tire) and export it to ONNX (RF-DETR outputs `dets` and `labels`; RT-DETRv2 outputs
   `labels`, `boxes`, `scores`).
2. Build the engine **on the target Jetson** with `trtexec` or the TensorRT Python API (the manifests use INT8 with
   quantization applied to the attention and `fc1` layers for RF-DETR, and mixed precision for the RT-DETRv2 stem).
3. Copy a manifest, set `engine.file`, `engine.sha256` (`sha256sum`) and the thresholds, and point `JV_MODEL_DIR` at the
   directory that holds the engine.

The shipped manifests record the hashes of the engines that were measured, so they will not match an engine you build
yourself.

## Quick start (PC, no GPU)

```
pip install -e .[dev]
pytest tests
```

The tests use a fake backend that produces outputs in the real engine format; no GPU or external service is needed.
On the Jetson the real backend needs `tensorrt` and `torch` from JetPack (see `deploy/INSTALL.md`).

### `jv run`

```
jv run --source <uri> --model models/rfdetr_int8g.yaml --sink jsonl:out/results.jsonl
```

- `--source`: folder or image, video file, `v4l2:N` or `/dev/videoN`, `gst:<uri>`, or a live camera `cam:...` (next section).
- `--backend tensorrt|fake` (default `tensorrt`); `--model-dir` or `JV_MODEL_DIR` for the engine.
- `--sink` can be repeated: `jsonl:<path>`, `compact:<path>` or `stdout`.
- `--track` enables IoU tracking (off by default). `--threshold` overrides the score threshold (manifest default: RF-DETR
  0.45, RT-DETRv2 0.6). `--max-frames N` and `--max-seconds S` stop the run.
- `--live` treats the source as a camera (newest frame wins); with `gst:file:` it plays the file at its own rate and drops
  late frames. `--loop` restarts the file at its end (long tests).
- `--log-max-mb` (default 100) rotates each JSONL keeping 3 copies; 0 disables rotation (about 1.2 GB per hour at 30 fps).
- `--baseline 30:26.5,15:37.2` sets the inference-latency baseline per cadence (`FPS:MS`), to be measured on the robot at
  install time. Without a baseline there is no latency-based throttling detection, only temperature.
- `--watchdog` talks to systemd (`READY=1`, `WATCHDOG=1`).
- Exit codes: 0 ok, 1 failure, 2 bad usage, 3 orderly stop because of temperature (level 3).

### Camera sources (`cam:`)

```
jv run --source "cam:csi:0?w=1920&h=1080&fps=30" --model models/rfdetr_int8g.yaml --sink jsonl:out/results.jsonl
jv run --source "cam:v4l2:/dev/video0?w=1920&h=1080&fps=30&fmt=mjpg" --model ...
jv run --source "cam:rtsp://user:pass@camera.local/stream1?codec=h264&latency=100" --model ...
```

| URI | Chain |
|---|---|
| `cam:csi:<id>[?w=&h=&fps=&flip=]` | `nvarguscamerasrc` (NVMM, NV12) -> `nvvidconv` -> BGR. `flip` is the `nvvidconv` flip-method (0 to 7). |
| `cam:v4l2:<device or N>[?w=&h=&fps=&fmt=&hw=]` | `v4l2src` with `fmt` = `mjpg` (default), `yuy2` or `nv12`. MJPEG: `nvv4l2decoder mjpeg=1` (`hw=1`, default) or `jpegdec` (`hw=0`). Raw: `nvvidconv` (`hw=1`) or `videoconvert` (`hw=0`) to BGR. |
| `cam:rtsp://host/path[?codec=&latency=&hw=]` | `rtspsrc` -> depayloader -> `nvv4l2decoder` (`hw=1`) or `avdec_*` (`hw=0`). `codec` = `h264` (default) or `h265`; `latency` in ms (default 100). The URL cannot carry its own query string; use a `gst:` launch string for that. |
| `cam:test[?w=&h=&fps=&fmt=&frames=]` | `videotestsrc is-live=true` through the same conversion tail. **For tests only**: it needs no camera. `frames=N` ends the stream after N frames (simulates a camera that goes away). |

Defaults: `w=1920 h=1080 fps=30`. An unknown parameter or value is an error, not ignored.

Live semantics (all `cam:` sources):

- The `appsink` keeps one buffer and drops the older ones (`max-buffers=1 drop=true sync=false`), so latency never accumulates;
  the pipeline's "newest frame wins" hand-offs do the rest. Frames the pipeline replaces are counted in `dropped_before`.
  Frames dropped by the driver or by the `appsink` before the read are not visible in `frame_seq`.
- `t_capture_mono_ns` is the capture time reported by the device (buffer timestamp plus the pipeline base time on the monotonic
  clock), so `total` latency includes the time a frame waited in the camera chain. If that time is not plausible the read time
  is used. `source_stamp` is the buffer timestamp.
- A missing device, a failed start, a silent device (5 s) or an end of stream raise `SourceError`; the pipeline reconnects with
  waits of 0.5, 1, 2, 4, 8 and then 10 s, the health monitor declares "no source" and a stop request interrupts the wait.
  A missing GStreamer element is reported by name.
- `source_id` is `cam:<kind>:<target>`; for RTSP only the host is kept, so credentials do not reach the JSONL.

### `jv bench` and `jv replay`

```
jv bench  --source <folder> --model <manifest> [--frames N]          # JSON with counters and p50/p95 per stage
jv replay --source <folder> --model <manifest> --sink jsonl:ref.jsonl      # record a reference
jv replay --source <folder> --model <manifest> --reference ref.jsonl       # compare: exit 0 if identical
```

`replay` uses a bounded blocking queue (no frame is lost) and compares by `frame_seq` with a score tolerance
(`--score-tol`) and a box tolerance in pixels (`--box-tol`). Ties in score are compared without order, and the comparison
should be done at the operating threshold: at minimum score 0 two implementations pick different members of a tie at the
300-best cut. RT-DETRv2 on large frames needs a 5 px tolerance (`docs/VALIDATION.md`, section 4). `bench` separates queue
wait (`queue`) from latency without waiting (`service`).

### Single frame

`Pipeline.process_frame(frame)` runs preprocessing, engine, decoder and tracker on the calling thread, with the same stage
code as the threaded pipeline (without the overlap). A `Pipeline(None, backend, manifest, [])` works without a source (the
ROS 2 node uses it). `StageTimings.total_ns` runs from the capture stamp to publication, `queue_ns` is the time spent waiting
in queues and `service_ns` is the difference; `SCHEMA_VERSION` is 2.

## Performance: the staged pipeline

`Pipeline` runs five threads: capture, preprocess, inference, postprocess and publication. Preprocessing (cv2 resize,
about 3 ms) and decoding (about 2 ms) overlap with the engine instead of running in series with it. Hand-offs are explicit:

- **Live sources**: capture -> preprocess and preprocess -> inference are one-slot mailboxes where the newest item wins, so
  the engine always takes the freshest frame and stale ones are dropped (counted in `Pipeline.overwritten` and visible as
  jumps in `frame_seq`). Inference -> postprocess -> publication are bounded blocking queues: work that reached the engine
  is never discarded.
- **Replay sources**: every hand-off is a bounded blocking queue; nothing is lost.
- Order is preserved (one thread per stage, FIFO hand-offs). A failing frame is skipped and counted per stage;
  `max_consecutive_errors` failures in a row in one stage stop the pipeline. `stop()` ends every thread and closes source
  and backend; sinks belong to the caller.

```mermaid
flowchart LR
    SRC["Source<br/>folder, video, v4l2, gst, RTP"] --> CAP
    subgraph T["Five threads, one per stage, FIFO order"]
        direction LR
        CAP["capture<br/>read, reconnect with backoff"]
        PRE["preprocess<br/>cv2 resize on uint8"]
        INF["inference<br/>GPU normalisation + TensorRT engine"]
        POST["postprocess<br/>decoder, optional tracker"]
        PUB["publish<br/>write to every sink"]
        CAP -- "live: mailbox, newest wins<br/>replay: blocking queue (4)" --> PRE
        PRE -- "live: mailbox, newest wins<br/>replay: blocking queue (2)" --> INF
        INF -- "blocking queue (2)" --> POST
        POST -- "blocking queue (16)" --> PUB
    end
    PUB --> SNK["Sinks<br/>JSON Lines, compact summary, stdout"]
```

The queue sizes are the defaults of `PipelineConfig`. A frame that overwrote an older one in a mailbox is counted in
`Pipeline.overwritten`; downstream of the engine nothing is dropped, so a slow sink slows the pipeline instead of losing
results.

Measured on the Jetson against the serial version (same engines, clips, power mode and flags; details and caveats in
`docs/VALIDATION.md`, section 13):

| RF-DETR, 1080p | Serial | Staged |
|---|---|---|
| Capacity (`jv bench`) | 31.0 to 31.3 fps | 33.0 to 33.1 fps (+6 %) |
| 30 fps input, 300 s: drops | 5 of 8,883 | 1 of 8,883 |
| 30 fps input, 300 s: latency p50 / p95 / p99 | 37.8 / 47.0 / 48.8 ms | 34.5 / 38.3 / 40.8 ms |
| 30 fps input, 12 min: drops, p95 | 4 of 21,482, 44.4 ms | 1 of 21,483, 38.4 ms |

The throughput gain is **6 %, not the roughly 15 % the stage times suggested**: with the stages overlapped the engine call
itself slows from 26.6 to 30.2 ms because the other threads compete for CPU cores and memory bandwidth. The clear gain is in
tail latency and drops at 30 fps. RT-DETRv2, with its more expensive preprocessing, gains more (21.3 to 25.3 fps delivered at
30 fps input, drops from 29 % to 16 %) but still cannot sustain 30 fps. Output is identical to the serial pipeline (replay of
160 images at threshold 0.0).

## Sinks (`jetson_vision/sinks.py`)

- **JSON Lines.** One line per record, with `"record"` equal to `result` (a `FrameResult.to_dict()`), `health` or `event`
  (for example `start`, which carries the previous stop cause read from `JV_LAST_STOP_FILE`). Optional size rotation and
  compression. A full disk does not stop the pipeline: the failure is counted in `write_errors`.
- **Compact summary** (`compact:<path>` writes one hexadecimal message per line) for acoustic links of 80 to 5,400 bps.
  Two messages, with the format documented in the module header:
  - T0, heartbeat: 8 bytes (level, FPS, temperature, latency p95, model tag).
  - T1, event: 5 bytes plus 3 per class, plus 4 if it carries a quantised box; with 8 classes and a box it is 33 bytes.
  Sizes are fixed by `tests/ops/test_sinks.py`. A concrete modem driver is out of scope: a function `emit(bytes)` is enough.
  The per-class count is the simultaneous maximum in the window (or distinct tracks when tracking is on), not a per-frame sum.

## Health (`jetson_vision/health.py`)

Levels: 0 normal, 1 half FPS, 2 alerts only (one frame per second), 3 stop. `nvpmodel` is never touched.

- **Temperature** from `/sys/class/thermal/thermal_zone*/temp` (the hottest zone). The default thresholds
  (`HealthConfig.temp_thresholds_c`) are provisional, not measured: set them with the real sealed enclosure.
- **Latency drift.** Hardware does not always tell the OS about throttling, so the moving median of inference latency is
  compared with a baseline. The baseline depends on the cadence: at low cadences the dynamic GPU governor raises latency,
  and that is not throttling. `LatencyBaseline` therefore stores one baseline per cadence and only compares when the current
  cadence is close to a measured one; otherwise drift is "unknown" and the level is held.
- **Hysteresis.** It rises immediately and falls one step at a time after a minimum hold time; each signal has its band.
- **Watchdog heartbeat.** `SystemdWatchdog` sends `WATCHDOG=1` only if there is a recent result or the pipeline declared
  "no source" (a dropped camera being reconnected). A hung backend stops results, and with them the heartbeat.

```mermaid
flowchart LR
    Z["Thermal zones<br/>hottest zone, Celsius"] --> HM
    PUB["Published results"] --> OPS["OpsSink"]
    OPS -- "inference latency" --> HM["HealthMonitor<br/>temperature level, drift level<br/>level = the higher of the two"]
    OPS -- "result seen" --> WD["SystemdWatchdog<br/>WATCHDOG=1 only if results are recent<br/>or the source is reconnecting"]
    SUP["Supervisor<br/>one tick per second"] -- "evaluate" --> HM
    HM -- "level" --> SUP
    SUP -- "set_degradation, set_throttle" --> PIPE["Pipeline"]
    SUP -- "no source" --> WD
    WD --> SD["systemd<br/>restarts the service if the beat stops"]
```

```mermaid
stateDiagram-v2
    direction LR
    L0: 0 normal
    L1: 1 half FPS
    L2: 2 alerts only
    L3: 3 stop
    L0 --> L1: signal above the level 1 threshold
    L1 --> L2: signal above the level 2 threshold
    L2 --> L3: signal above the level 3 threshold
    L3 --> L2: one step down
    L2 --> L1: one step down
    L1 --> L0: one step down
    L3 --> [*]: jv exits with code 3, no restart
```

A level rises as soon as a signal crosses its threshold, and falls one step at a time after the minimum hold time
(`HealthConfig.hold_s`, 30 s by default) once the signal is back inside its hysteresis band. The temperature thresholds
(80, 85 and 90 C) and the drift thresholds (25 % and 75 % above the baseline) are provisional defaults, as explained above.

Limitation: when FPS is halved a baseline for that cadence is needed (for example `30:..,15:..`); without it level 1 is
released only by temperature, not by latency.

## Tracking (`jetson_vision/tracking.py`)

Greedy IoU with two score thresholds in the style of ByteTrack, no appearance model or Kalman filter. It counts distinct
objects and smooths flicker. It is not evaluated on real recordings: do not trust the counts yet.

## ROS 2 Humble (`adapters/ros2`)

Two colcon packages: `jetson_vision_msgs` (`FrameMeta` message) and `jetson_vision_ros2` (lifecycle node `jetson_vision`).
No vision logic: the mapping is in `mapping.py`, pure functions without `rclpy`, tested in `tests/ops/test_ros_mapping.py`.
The core never imports `rclpy` (a test checks it).

- Subscribes to `sensor_msgs/Image` with sensor QoS (`image_topic`, default `image_raw`; encodings `bgr8`, `rgb8`, `bgra8`,
  `rgba8`, `mono8`).
- Publishes `~/detections` (`vision_msgs/Detection2DArray`) with `header.stamp` and `frame_id` copied from the input image,
  `~/frame_meta` (`jetson_vision_msgs/FrameMeta`: `schema_version`, model, threshold, level, drops and timings) and
  `/diagnostics`. `Detection2D.id` carries the `track_id` or is empty.
- Parameters: `manifest`, `backend` (`tensorrt` or `fake`), `image_topic`, `input_fps`, `alert_period_s`, `baseline_fps`,
  `baseline_ms`.

```mermaid
flowchart LR
    CAM["Camera driver<br/>sensor_msgs/Image"] -- "image_raw, sensor QoS" --> N
    subgraph N["jetson_vision lifecycle node (jetson_vision_ros2)"]
        direction TB
        MAP["mapping.py<br/>pure functions, no rclpy"]
        CORE["Core Pipeline.process_frame<br/>preprocess, engine, decoder, tracker"]
        HM["HealthMonitor<br/>timer, once per second"]
        MAP --> CORE
        HM -. "level: skip frames<br/>(1 of 2, one per alert_period_s, or none)" .-> MAP
    end
    N -- "~/detections<br/>vision_msgs/Detection2DArray" --> OUT1["Consumers"]
    N -- "~/frame_meta<br/>jetson_vision_msgs/FrameMeta" --> OUT1
    N -- "/diagnostics" --> OUT2["Monitoring"]
```

The node copies `header.stamp` and `frame_id` from the input image to every output, and the transitions are
`configure` (builds the core and the publishers) and `activate` (subscribes to the image topic).
`jetson_vision_ros2/launch/vision.launch.py` brings the node to the active state.

```
cd <workspace> && cp -r <repo>/adapters/ros2/* src/ && colcon build
export PYTHONPATH=<repo>:$PYTHONPATH            # the core is imported as is
ros2 run jetson_vision_ros2 vision_node --ros-args -p manifest:=<repo>/models/rfdetr_int8g.yaml -p backend:=fake
ros2 lifecycle set /jetson_vision configure && ros2 lifecycle set /jetson_vision activate
```

PC test with Docker (CPU, fake backend): `adapters/ros2/jetson_vision_ros2/test/docker_smoke.sh` (inside a `ros:humble`
image, repository mounted at `/repo` and `fake_camera.py` at `/pub.py`); log in
`docs/validation_data/faults/ros_docker_smoke.log`. It checks the colcon build, lifecycle transitions, topics, `BEST_EFFORT`
QoS and that the output header is the input header. **Not tested on the Jetson**, which has no ROS installed. Limitation:
`frame_seq` is assigned by the node, so frames DDS drops before the callback do not show in `dropped_before`.

## Deployment (`deploy/`)

systemd unit with `Type=notify` and `WatchdogSec`, an `env` file and the steps in `deploy/INSTALL.md`. The heartbeat is
conditioned on recent results and the temperature stop (exit code 3) is not restarted. **Tested on the Jetson as a user
service** (`deploy/jetson-vision.user.service`, `systemctl --user`, no sudo): start with `READY=1`, `SIGSTOP` (the watchdog
restarts it after 10 s with `WatchdogSec=10`, new `session_id` and cause `watchdog/ABRT` in the JSONL), dead source (service
stays alive and recovers), wrong engine hash (exit 1; the restart limit leaves it `failed`) and a blocked sink (the watchdog
restarts it). **The system unit** (`jetson-vision.service`, with `User=`) **was not tested**.

## Tests

`pytest tests` covers sinks, health, tracking, CLI, ROS mapping, simulated faults (`tests/ops/test_faults.py`: a camera that
drops and reconnects, a hung backend, high temperature, drifting latency) and the pipeline, including the staged threading
(`tests/core/test_pipeline_stages.py`: overlap, order, back-pressure, drop policy, errors per stage, clean shutdown)), and the camera sources
(`tests/core/test_camera.py`: URI parsing, generated GStreamer chains, reconnection and live drop).
Tests use no GPU or external services. On-device measurements are in `tests/core/jetson_*.py` and `jetson_e2e.sh` (not
collected by pytest). `tools/reference/stages.py` is the author's prototype pipeline, used as the independent reference of
the AP regression; it needs the Jetson stack.
