# Validation on the real Jetson

Date: 2026-11-06. Jetson Orin Nano Super, JetPack 6.2.1, TensorRT 10.3, **15 W** (`nvpmodel` untouched).
Code: commit `eedfb12` plus the changes of section 1, already included in commit `8fab937`. All of it under
`$DEPLOY_DIR/jetson_vision/repo/` on the Jetson. VAL only; never TEST. Every figure points to a file in
`docs/validation_data/` (text only). The engines were not rebuilt: they are those in `$JV_MODEL_DIR`, with the
hash fixed by each manifest.

## 0. Result against the goals

Goals (from `DESIGN.md`; they are not measurements): end-to-end p95 ≤ 100 ms and ≥ 15 FPS sustained at 15 W; AP on a fixed
VAL subset does not drop below the reference.

| Goal | RF-DETR INT8 G (default) | RT-DETRv2 mixed INT8 | Source |
|---|---|---|---|
| p95 ≤ 100 ms | **Met**: 36 to 44 ms in the 4 combinations, 41.7 ms over 32 min at 30 fps | **Met**: 50 to 76 ms | sections 5 and 6 |
| ≥ 15 FPS | **Met**: 30.0 fps at 30 fps input (0.2 % drops) and 15.0 at 15 | **Met with drops**: 22 to 24 fps at 30 fps input (18 to 26 % dropped), 15.0 at 15 | section 5 |
| AP not lower | **Met**: 43.445 on full VAL versus 43.45 | **Met**: 43.605 versus 43.61 | section 4 |

This was measured with **synthetic H.264 video** made from VAL photos (section 5), with no camera, no CSI or V4L2, no real water and
outside a sealed enclosure. The result holds for those conditions, not for the robot.

## 1. What changed (the three planned improvements and what turned up)

Requested improvements:

1. **Public `Pipeline.process_frame(frame)`**, single-frame and threadless. The three-thread pipeline calls this same method.
   `factory.py` no longer chains components: it builds a `Pipeline` without a source (`Pipeline(None, backend, manifest, [])`) and the ROS 2 node
   calls `process_frame`. Tests: `tests/core/test_pipeline.py`.
2. **Queue wait separated from latency.** `StageTimings` gains `queue_ns` (wait in the mailbox or input queue plus the
   publishing queue) and the property `service_ns = total_ns - queue_ns`. The `PipelineMetrics` metrics add the stages `queue` and
   `service`. **`SCHEMA_VERSION` goes from 1 to 2** (new field in `types.py`); the ROS message `FrameMeta` gains `queue_ns`.
   `total_ns` keeps its meaning (capture stamp to publication). Measured in full-speed replay
   (`jv bench` on H.264, section 11): `total` p50 185 ms versus `service` 31 ms and `queue` 154 ms for RF-DETR; without the
   separation, those 185 ms would look like latency.
3. **RF-DETR threshold: 0.45** (previously 0.6, which came from the Android app with RT-DETRv2). Section 3.

Changes that turned out to be necessary (all with a new test in `tests/`; the cause and the evidence file are in the sections indicated):

| # | Failure | Where it was seen | Fix |
|---|---|---|---|
| 1 | `read_temperature_c` raised `TypeError` when reading a thermal zone that answers `EAGAIN` and **killed `jv run`** (exit 1) | first real run of `jv run` | read with `os.read`, zone skipped; in addition, a failure of the health tick is logged and does not stop the detector (`cli._safe_tick`) |
| 2 | A full disk (`/dev/full`) made `JsonlSink.close()` fail and `jv run` exited with **code 2** after the work was done | section 8, test T3 | `close()` does not raise; sinks are closed one by one with error capture |
| 3 | A blocked sink (FIFO with no reader) **hung the exit** (the main loop waited for all threads and the health thread stayed on the sink lock): it had to be killed after 90 s | section 8, test T4 | the loop exits when a stop is requested (`stop()` cuts the exit after its deadline); the sink locks use a 1 s timeout |
| 4 | The JSONL log **grew without bound** (rotation existed but the CLI could not enable it): 675 bytes per result, about 1.2 GB per hour at 30 fps | section 6 | `jv run --log-max-mb` (default 100 MB, 3 copies); the byte counter is custom because `tell()` fails on FIFOs and silently dropped every line |
| 5 | `jv replay --reference` reported differences in every frame even though the results were equal: score ties ordered randomly by `argpartition` | section 4 | the RF-DETR decoder sorts by score and then by index (deterministic); the comparison ignores the order within a tie |
| 6 | The ROS 2 node numbered `frame_seq` from 1, which with `process_frame` would have counted a false drop | review of change 1 | starts at 0, like all sources |

Added to make testing possible: `jv run --live` (a `gst:file:` plays at its own rate and drops like a camera), `--loop`
(restarts the file when it reaches the end), `--max-seconds`, and the engine logs its memory on load. Measurement scripts in
`tests/core/jetson_*.py` and `jetson_e2e.sh`. Pytest: 173 tests green on the PC (1 skipped, Linux only) and 174 on the Jetson
(`.vendor/` with pytest; the one skipped on the PC does run).

## 2. How it was measured

- **Video.** There is no real video. 4 H.264 clips were created (main profile, x264 `ultrafast`, one keyframe per second) with
  `gst-launch` from the **82 1920×1080 photos of VAL**, each repeated for 3 frames (slideshow): 1080p and 720p, at
  30 fps (60 s, 1,800 frames) and at 15 fps (40 s, 600). 720p comes from rescaling the same photos. It is piecewise static content: the decoder does less work than with real video, and there is no motion or turbidity. Script: `validation_data/mkvideo.sh`.
  The clips are not copied (text only); they are on the Jetson, in `$VIDEO_DIR`.
- **Execution.** `jv run --source gst:file:<clip> --live --fps <F>`: decoding by NVDEC (`nvv4l2decoder`) and conversion by
  VIC (`nvvidconv`), one-slot mailbox where the latest frame wins. `tegrastats` confirms NVDEC and VIC active in almost all
  samples (for example 1,862 of 1,865 in the 32 min test; `nvdec_active_samples`, `vic_active_samples` in each `summary.json`).
- **Latency = `total_ns`**: from the capture stamp (right when the frame leaves the `appsink`, already decoded and converted)
  to publication. **It does not include sensor exposure, transmission or decoding.** The first 30 results are discarded
  (warm-up).
- **Drops** = frames delivered by the source (last `frame_seq` + 1) minus published results.
- Process CPU and RSS from `/proc` (`monitor.csv`, every second). Board RAM, GPU, `VDD_IN` power and `tj` temperature from
  `tegrastats` (`tegrastats.log`, `sudo` only for that). Power = **whole board**, without propulsion or lighting.
- GPU memory: the Orin shares RAM, so there is no separate `nvidia-smi` figure (it reports N/A). The engine logs on load the
  TensorRT execution memory and the memory reserved by torch (`jv.log` of each run): **RF-DETR 22.2 MB + 42.0 MB, RT-DETRv2
  96.1 MB + 42.0 MB**. The rest comes from the total `tegrastats` RAM.
- Summary of each run: `summary.json` (from `tests/core/jetson_summary.py`).

## 3. RF-DETR threshold (F1 curve on VAL with the real engine)

`tests/core/jetson_f1.py`: `Pipeline.process_frame` over the 2,185 VAL images at minimum score, and for each threshold each
detection is matched to an unused box of its class with IoU ≥ 0.5 (greedy, by score). Micro F1 pools all
detections; macro F1 averages the classes with ground truth. Data: `regression/f1_rfdetr.json`, `regression/f1_rtdetrv2.json`.

| Threshold | RF-DETR precision | recall | F1 micro | F1 macro | RT-DETRv2 F1 micro | F1 macro |
|---|---|---|---|---|---|---|
| 0.40 | 0.760 | 0.735 | 0.7472 | **0.6296** | 0.7225 | 0.6266 |
| **0.45** | 0.799 | 0.710 | **0.7521** | 0.6285 | 0.7372 | 0.6332 |
| 0.50 | 0.833 | 0.681 | 0.7494 | 0.6223 | 0.7439 | 0.6390 |
| 0.60 | 0.882 | 0.621 | 0.7285 | 0.5968 | **0.7513** | **0.6391** |

- **RF-DETR: 0.45**, the micro F1 maximum (the 0.40 to 0.50 plateau varies by less than 0.005; macro F1 is highest at 0.40 by
  0.001). At 0.6 F1 drops by 0.024. It is set in `models/rfdetr_int8g.yaml` with the source in the comment.
- **RT-DETRv2: 0.6 is confirmed** (maximum of micro and macro F1), so its manifest does not change.
- Limits: 0.05 grid; own F1 criterion (the app uses its own, it could not be compared); VAL is 92.5 % 416×416 photos
  from a fixed camera. With real water the value may differ.

## 4. AP regression (`jv replay --reference`)

Independent reference: the validated reference path (`tools/reference/stages.py`, different preprocessing, copy and postprocessing code) generates a
JSONL per manifest with the same images in the same order. Fixed subset: 160 VAL images (100 of 416×416, 40 of
1920×1080, 20 larger; `pick_images` over VAL sorted by id, list in `regression/subset/list.json`). Full VAL: 2,185.
Scripts: `tests/core/jetson_regression.py`, `regression/run_regression*.sh`.

**AP** (minimum score 0 for RF-DETR and 0.001 for RT-DETRv2, like the reference):

| Set | Manifest | Module AP | Reference path AP | `reference_ap` AP (manifest) | Files |
|---|---|---|---|---|---|
| Subset (160) | RF-DETR | 48.2574 | 48.2570 | n/a | `ap2_rfdetr_int8g_subset.log`, `ap_ref_rfdetr_int8g_subset.log` |
| Subset (160) | RT-DETRv2 | 42.9705 | 43.0353 (−0.065) | n/a | `ap2_rtdetrv2_int8mix_subset.log`, `ap_ref_rtdetrv2_int8mix_subset.log` |
| Full VAL | RF-DETR | **43.4454** | 43.4454 | 43.45 (−0.005) | `ap2_rfdetr_int8g_full.log`, `ap_ref_rfdetr_int8g_full.log` |
| Full VAL | RT-DETRv2 | **43.6052** | 43.6146 (−0.009) | 43.61 (−0.005) | `ap2_rtdetrv2_int8mix_full.log`, `ap_ref_rtdetrv2_int8mix_full.log` |

The subset has no prior reference value; the reference is the independent path. On full VAL
the AP matches the `reference_ap` stored in each manifest (differences of 0.005, from rounding in the stored value).

**Strict `jv replay --reference`, at the operating threshold** (RF-DETR 0.45; RT-DETRv2 0.6; the reference JSONL filtered to the same
threshold): **identical in all 4 cases** (`replaythr_*.log`). Default tolerances (score 1e-4, box 0.5 px) except
RT-DETRv2, where **5 px** were needed:

- **RT-DETRv2: boxes differ by up to 4.1 px in large frames** (5616×3744; 2.4 px in 4608×3456); in the 1080p ones
  they also exceed 0.5 px (40 of 40) and in the 416×416 ones they do not differ. With 0.5 px all 40 of 1080p and the 20 larger ones of the subset fail (60 of 160). The most likely cause: the
  manifest passes the network size (640×640) to the engine in `orig_target_sizes` and the module rescales afterwards, while the
  reference passes the real frame size, and the engine computes the boxes at reduced precision. **This has not been verified.**
  Effect on AP: −0.009 on full VAL, within the margin. RF-DETR does not have the problem.
- **At minimum score 0 the comparison is not identical** (RF-DETR: 63 of 160 frames, 759 of 2,185; in RT-DETRv2 the 60 and 173
  differences are those of the large boxes): at the cut of the top 300 there are exact score ties (the logits come out quantized) and two implementations
  keep different members of the tie, in the tail (in the samples seen, scores of 0.003 to 0.008; 61 of the 63 subset frames
  differ only in the last positions). AP is not affected (figures above). That is why the strict regression is done at the operating threshold. The logs
  of that pass (`replay2_*.log`, 63/759/60/173 differences) are kept as evidence; the `replay_*.log` files are from an earlier
  pass with the order-sensitive comparison (failure 5 of section 1).

## 5. End to end with the real `jv run` (dynamic clocks, 15 W)

Clips from section 2, 60 s (30 fps) or 40 s (15 fps) each, one clip per run, one run per cell, engine freshly loaded.
`e2e_dynamic_clocks/<manifest>_<resolution>_<fps>fps/`. Latency in ms, `p50 / p95`, from the capture stamp.
Power and energy: mean board `VDD_IN` (includes NVDEC and VIC); **mJ/frame = VDD_IN / FPS**; **net** subtracts the measured
idle power (4.26 W, section 10).

| Model | Res. | Input | FPS | Drops | Latency p50 / p95 | Inf. p50 | CPU (cores) | RSS MB | Board RAM MB | VDD_IN W | mJ/frame | net mJ/frame | Max T °C | GPU MHz |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RF-DETR | 1080p | 30 | 29.9 | 4 (0.2 %) | 31.3 / 41.4 | 26.5 | 1.02 | 1046 | 2699 | 10.17 | 340 | 198 | 56.3 | 587 |
| RF-DETR | 1080p | 15 | 15.0 | 0 | 41.3 / 43.0 | 37.0 | 0.53 | 1034 | 2648 | 7.18 | 478 | 194 | 54.8 | 390 |
| RF-DETR | 720p | 30 | 30.0 | 3 (0.2 %) | 34.6 / 36.2 | 26.3 | 0.77 | 1004 | 2658 | 9.93 | 332 | 189 | 57.0 | 587 |
| RF-DETR | 720p | 15 | 15.0 | 0 | 41.9 / 43.9 | 37.4 | 0.41 | 1000 | 2622 | 7.04 | 469 | 185 | 54.9 | 413 |
| RT-DETRv2 | 1080p | 30 | 22.3 | 463 (25.7 %) | 60.3 / 76.2 | 37.1 | 1.06 | 1098 | 2748 | 9.49 | 426 | 235 | 56.6 | 587 |
| RT-DETRv2 | 1080p | 15 | 15.0 | 0 | 48.7 / 54.6 | 37.8 | 0.75 | 1092 | 2703 | 7.62 | 508 | 224 | 55.5 | 549 |
| RT-DETRv2 | 720p | 30 | 23.5 | 391 (21.7 %) | 59.2 / 74.4 | 37.3 | 0.80 | 1059 | 2710 | 9.43 | 402 | 220 | 56.8 | 587 |
| RT-DETRv2 | 720p | 15 | 15.0 | 0 | 44.6 / 50.3 | 37.5 | 0.52 | 1053 | 2658 | 7.47 | 498 | 214 | 55.3 | 532 |

(Same numbers in each `summary.json`; the table comes from `tests/core/jetson_summary.py` and from `tegrastats.log`.)

Reading:
- RF-DETR sustains 30 fps with 3 or 4 drops in 1,800 and p95 of 36 to 41 ms. RT-DETRv2 does not reach 30 fps (22 to 24) and drops
  a fifth; at 15 fps it drops nothing.
- **At 15 fps RF-DETR latency rises from 31 to 41 ms**: it is the dynamic GPU governor (390 to 413 MHz versus 587 to 612),
  which stretches inference from 26.5 to 37.0 ms. With fixed clocks it disappears (section 10).
- The p50 latency at 30 fps depends on the phase between the arrival of the frame and the end of the previous one: in the 32 min run
  with the same configuration the p50 was 39.9 ms and in the 60 s one, 31.3 ms (inference, 26.4 ms, was identical). The p95 is quoted.
- Compared with the author's prototype (RF-DETR 45 ms at 30 fps from a 416² JPEG, 68 ms at 1080p with JPEG): the module gives
  31 to 35 ms p50 with H.264 video at 30 fps. They are not the same input (JPEG versus H.264 with NVDEC) nor the same content.
- This latency starts when the frame has already left NVDEC and VIC; **it does not include the latency of the real camera**.

## 6. Endurance: 32 minutes, RF-DETR at 30 fps, 1080p

`jv run --live --loop --max-seconds 1920 --baseline 30:26.5,15:37.2` on the 30 fps 1080p clip in a loop.
`e2e_dynamic_clocks/endurance_rfdetr_1080p30/` (`summary.json` with 32 windows of 60 s, `monitor.csv`, `tegrastats.log`).

| Measure | Result |
|---|---|
| Duration, results | 1,916 s, 57,477 results; **29.998 fps** |
| Drops | 3 of 57,480 (0.005 %), all 3 in the first minute; **0 in the following 31 windows** |
| Total latency p50 / p95 / p99 / max | 39.9 / 41.7 / 45.8 / 56.0 ms; maximum per-window p95 43.4 ms |
| Inference per window | 26.39 to 26.43 ms; drift **−0.04 ms/h** (linear fit) |
| Total latency p50 per window | from 31.2 to 40.6 ms; it changes phase and does not grow (the −6.5 ms/h slope is the first minute) |
| Process memory (RSS) | 1,050 MB in the first 2 min, 1,109 MB after 10 min, then **+6 MB/h** |
| Board RAM used | max 2,660 MB (monitor), 2,820 MB (tegrastats); after 10 min **−5 MB/h** |
| Max temperature of the zones / `tj` | 60.2 °C / 60.2 °C; mean from 58.0 °C (first 5 min) to 59.3 °C (last 5); slope −0.8 °C/h after 10 min |
| Throttling | GPU at 612 MHz except 1 sample of 1,890 (306 MHz at startup); the CPU goes up and down with the governor |
| `VDD_IN` power | 10.65 W mean, 11.26 W max |
| Health | 190 `health` records, all level 0 `ok` |
| JSONL size | 38.8 MB (675 bytes per result, about 1.2 GB per hour at 30 fps) |

No measurable leaks in 32 minutes (RSS rises during warm-up and stabilizes), no inference drift, no
throttling. **Thirty-two minutes do not demonstrate absence of leaks over 8 to 24 h**, nor is
there a sealed enclosure.

## 7. Latency baselines for `health.py` and false alarms

Baseline = median of inference (`inference_ns`, what the monitor watches) per cadence, from the runs of section 5.
Inference latency **does not depend on the input resolution** (the rescaling happens before the engine).

| Manifest | Dynamic clocks | Fixed clocks (section 10) |
|---|---|---|
| RF-DETR | `30:26.5,15:37.2` | `30:26.1,15:26.1` |
| RT-DETRv2 | `30:37.2,15:37.7` | `30:36.9,15:36.8` |

(`jv run --baseline 30:26.5,15:37.2`. Each value is the mean of the 1080p and 720p medians: 26.5 and 26.3; 37.0 and 37.4; 37.1 and 37.3;
37.8 and 37.5 in dynamic; 26.2 and 25.9; 26.1 and 26.1; 37.0 and 36.7; 36.8 and 36.7 in fixed.)

**False alarms** with the dynamic baselines, `--baseline` active, 4 runs (`e2e_dynamic_clocks/endurance_rfdetr_1080p30`,
`baseline_rfdetr_720p15`, `baseline_rtdetrv2_1080p30`, `baseline_rtdetrv2_720p15`; the last three of 5 min):

| Run | Cadence | Measured inf. p50 | Levels seen | `health` records |
|---|---|---|---|---|
| RF-DETR 1080p, 32 min | 30 | 26.4 | only 0 | 190, all `ok` |
| RF-DETR 720p, 5 min | 15 | 37.3 | only 0 | 30, all `ok` |
| RT-DETRv2 1080p, 5 min | 30 | 37.2 | only 0 | 30, all `ok` |
| RT-DETRv2 720p, 5 min | 15 | 40.9 | only 0 | 30, all `ok` |

**Zero false alarms.** Note: RT-DETRv2 at 15 fps measured 40.9 ms in this run and 37.5 ms in the 40 s one: 9 % variation
between runs with the same configuration (the drift threshold is 25 %). **The fixed-clock baselines were not
validated with long runs**: they come from runs of 40 to 60 s. **It was not tested that the alarm does trigger** (for example with added GPU
load): only the absence of false alarms; the positive case is covered by the tests with a fake backend (`tests/ops/test_faults.py`).
The temperature thresholds (80/85/90 °C) remain provisional: the maximum temperature measured was 60.2 °C and there is no sealed
enclosure.

## 8. Real faults

`faults/` (logs `fail/log_t1.txt`, `fail/log_t4.txt`, `fail/log.txt`, `fail_tests.sh`).
**The user unit does work without sudo**: `systemctl --user` is available; the service reaches `active (running)`
(`Type=notify` accepts `READY=1`). `deploy/jetson-vision.user.service` is used (new; same behavior without `User=`/`StateDirectory=`)
with `WatchdogSec=10` to shorten the tests, the user manager without `enable-linger` (an ssh session was kept open) and
the unit link in the user systemd directory (`.config/systemd/user/`, removed at the end). **The system unit
(`deploy/jetson-vision.service`, with `User=<service-user>`) was not tested** (it would need `sudo`).

| Test | What was done | Result |
|---|---|---|
| T1a Dead source | RTP/H.264 over UDP from an external `gst-launch` to `udpsrc` → NVDEC, under the service; the sender is killed and restarted 35 s later | After the death: 571 → 573 results in 35 s, 5 warnings «silent for 5 s», **no empty detections published**, the service stays alive with `NRestarts=0` (the heartbeat continues with «no source»). When the sender returns: 573 → 1,139 results in 25 s without restarting |
| T1b `SIGSTOP` | `kill -STOP` on the main process | The watchdog acts at 10 s (`stop-watchdog`, `Result=watchdog`, `ExecMainStatus=6`, SIGABRT); 5 s later there is a new process, `NRestarts=1`, **new `session_id`** and `restart_cause: watchdog/ABRT` in the JSONL `start` record |
| T2 Wrong hash | manifest with a different hexadecimal string | `jv run` exits with **code 1** and «engine refused: SHA-256 is f7e4… the manifest expects a7e4…»; under the service: 5 restarts in about 35 s and state `failed` (the `StartLimitBurst=5` limit works). A hash of 64 zeros was a test error: YAML reads it as the integer 0 and the manifest is rejected with code 2 |
| T3 Full disk | sink `jsonl:/dev/full` next to a good one | **Before the fix** it exited with code 2 (failure 2). After: **code 0**, the good sink receives all 480 results and the failing one counts `write_errors` |
| T4 Blocked sink | FIFO with a reader that does not read, `jv run` directly | **Before the fix** it did not exit (it had to be killed at 90 s; twice). After: it exits with code 0 at 36 s after starting with 20 s requested; **later sinks get no results** (88 in the good one): one blocked sink stops publishing altogether |
| T4b Blocked under the service | `results.jsonl` turned into a FIFO | Results stop, the heartbeat is cut and systemd restarts by watchdog (`NRestarts` 1 and 2 in 55 s), as designed. It was done before the 1 s timeout was put on the sink locks |
| ROS 2 | `docker_smoke.sh` in `ros:humble`, CPU, fake backend, after the change to `process_frame` | build, lifecycle, topics, `header` copied, `queue_ns` in `frame_meta`: fine (`faults/ros_docker_smoke.log`) |

The logs of the first pass (`faults/fail_first/log_first_pass.txt`, `faults/fail/log.txt` includes T2 to T4b) have an
invalid T1: the `kill` killed the `timeout` and not the sender, and an earlier service was reused. T1 was repeated cleanly
(`log_t1.txt`). The following remain untested: a camera that hangs over V4L2/CSI, `nvargus-daemon`, a frozen frame, added
CPU and GPU load, and the restart via `Restart=on-failure` with a real CUDA failure.

## 9. `nsys` over 60 s of `jv run`

`nsys profile --duration=60 --delay=8 --trace=cuda,nvtx` under `timeout -k 15 150` (because of the hang that occurred with VPI), on RF-DETR
1080p at 30 fps (`nsys/`, version 2024.5.4). **It did not hang**: it finished with code 0 after 2 min. 36 MB trace (`trace.nsys-rep`,
not copied; it is on the Jetson). Data: `nsys/gpu_summary.txt`, `nsys/stats.csv`, `nsys/jv_summary.json`.

- GPU busy with kernels **70 %** of the 60 s (42.1 s; 405,560 launches); memory copies 0.12 s (**0.2 %**).
- 1,741 inferences in the window (29 per second): **24.2 ms of kernels per inference** versus 26.4 ms of `inference_ns`; the
  remaining 2 ms are copy, GPU normalization, synchronization and outputs.
- Longest-running kernels: INT8 GEMM (17.9 %), f16 GEMM (13.7 %), i8→f32 GEMM (11.6 %), attention `_gemm_mha` (3 kernels of
  3.3 %), f16 convolutions and `__myl` normalization kernels.
- `jv run` under `nsys`: 28.9 fps, 80 drops; preprocessing 3.8 ms, inference 27.1 ms, postprocessing 1.8 ms, p50 latency 42.5 ms. The
  profiler costs some performance.
- **Conclusion: the time goes to the GPU (INT8 engine)**, not to copies or Python. With preprocessing (2.8 ms) and
  postprocessing (1.7 ms) in the same thread as inference, the inference thread takes 31 ms per frame (section 11).

## 10. Fixed clocks with `jetson_clocks`

Procedure (`clocks/clocks_run.sh`): the state was saved with `sudo jetson_clocks --store` in `clocks/clocks/clocks_orig.conf`,
fixed with `sudo jetson_clocks` **at 15 W (without touching `nvpmodel`)**, the matrix was repeated and it was restored with
`--restore` (with `trap`, also if something fails). After restoring, `jetson_clocks --show` matches the initial one except for
`CurrentFreq` noise and the fan speed, which is dynamic (`clocks/clocks/show_before.txt` versus `show_now.txt`;
`nvpmodel -q` is still at 15 W). With fixed clocks: CPU at 1.50 GHz (the maximum at 15 W), GPU at 612 MHz (`show_fixed.txt`).

Idle power (board with nothing running, 60 s, first 10 s discarded): **dynamic 4.26 W; fixed 4.54 W (+0.28 W)**
(`clocks/clocks/idle_dynamic.log`, `idle_fixed.log`).

| Model | Res. | Input | Fixed FPS | Fixed drops | **Fixed** latency p50 / p95 | (dynamic) | Fixed inf. p50 | Fixed VDD_IN W | (dynamic) | Fixed mJ/frame | (dynamic) | Fixed net mJ | (dynamic) | Fixed max T °C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RF-DETR | 1080p | 30 | 30.0 | 0 | **28.9 / 30.0** | 31.3 / 41.4 | 26.2 | 10.30 | 10.17 | 343 | 340 | 192 | 198 | 55.3 |
| RF-DETR | 1080p | 15 | 15.0 | 0 | **28.9 / 30.0** | 41.3 / 43.0 | 26.1 | 7.30 | 7.18 | 486 | 478 | 184 | 194 | 54.2 |
| RF-DETR | 720p | 30 | 30.0 | 0 | **28.5 / 29.4** | 34.6 / 36.2 | 25.9 | 9.98 | 9.93 | 333 | 332 | 181 | 189 | 56.7 |
| RF-DETR | 720p | 15 | 15.0 | 0 | **28.6 / 29.3** | 41.9 / 43.9 | 26.1 | 7.15 | 7.04 | 476 | 469 | 174 | 185 | 54.8 |
| RT-DETRv2 | 1080p | 30 | 23.2 | 406 (22.6 %) | 58.3 / 73.0 | 60.3 / 76.2 | 37.0 | 9.74 | 9.49 | 419 | 426 | 224 | 235 | 56.8 |
| RT-DETRv2 | 1080p | 15 | 15.0 | 0 | **42.5 / 42.9** | 48.7 / 54.6 | 36.8 | 7.66 | 7.62 | 511 | 508 | 208 | 224 | 55.7 |
| RT-DETRv2 | 720p | 30 | 24.5 | 329 (18.3 %) | 56.8 / 71.9 | 59.2 / 74.4 | 36.7 | 9.69 | 9.43 | 395 | 402 | 210 | 220 | 57.2 |
| RT-DETRv2 | 720p | 15 | 15.0 | 0 | **40.5 / 41.0** | 44.6 / 50.3 | 36.7 | 7.53 | 7.47 | 502 | 498 | 199 | 214 | 55.7 |

Data: `e2e_fixed_clocks/<...>/summary.json`. Each fixed run lasts 40 to 60 s; **there is no long test with fixed clocks**, so
the steady-state temperature with them is not measured (at 55 to 57 °C in 60 s, the same as with dynamic ones).

Comparison:
- **Latency**: at 15 fps RF-DETR goes from 41 to **29 ms** p50 and from 43 to 30 p95; at 30 fps the p95 drops from 36 to 41 to 29 to 30 ms. The
  dispersion almost disappears (p95 − p50 of 1 ms). RT-DETRv2 at 15 fps gains 4 to 6 ms; at 30 fps it still does not reach it (23 to 24 fps) because
  it is limited by the engine (37 ms), not the clock.
- **Power**: RF-DETR **+0.05 to +0.13 W with inference** (+0.5 to +1.8 %): at 30 fps, 340 → 343 mJ per frame (+1 %); at 15
  fps, 478 → 486 (+1.7 %). RT-DETRv2 up to +0.26 W (+2.7 %) at 30 fps. **+0.28 W at idle** (+6.6 %).
- **Net energy per frame** (without the idle power of each mode): with fixed clocks it drops from 185 to 198 mJ to 174 to 192 mJ
  in RF-DETR, because the GPU works less time per frame; but idle costs more. That subtraction is an approximation: each
  mode uses its own idle power.
- **Baselines**: with fixed clocks inference does not depend on the cadence (26.1 ms at 15 and at 30 fps in RF-DETR), so one
  baseline is enough and the case "the governor stretches latency at low cadence", which forced one baseline per cadence, disappears.
  With dynamic ones two are needed (`26.5` and `37.2`). **Both sets are documented** (section 7).
- Temperature: no appreciable difference in runs of 40 to 60 s (54 to 57 °C in both modes).

**Recommendation for the robot: fixed clocks (`jetson_clocks`) at 15 W**, with these figures: it costs about 0.1 W (less than 2 %) while
detecting and 0.28 W at idle, and gives a stable p95 latency of about 30 ms at 15 and at 30 fps instead of 43 to 44 ms at 15 fps, with a single
latency baseline (and therefore more reliable throttling detection). 1.7 % of sustained board power is
little battery compared with propulsion and lighting (not measured). **Conditions**: if the robot spends most of the time
waiting without detecting, the +0.28 W at idle (+6.6 % of the board at idle) may weigh more than the gain; and the
temperature with fixed clocks in a sealed enclosure and over the long term **is not measured**. Fixed clocks are a deployment decision
(`jetson_clocks` needs `sudo` on every boot; see `deploy/INSTALL.md`); the module code works the same with both.

## 11. Maximum throughput (`jv bench`, replay with no input limit)

`jv bench --source gst:file:<30 fps clip>` (`bench/`), dynamic clocks, 1,800 frames, input queue of 4, no
drops. `queue` is the queue wait (the one that used to inflate `total_ns`); `service` = `total` − `queue`.

| Model | Res. | Max FPS | Inf. p50 | `queue` p50 | `service` p50 / p95 | `total` p50 |
|---|---|---|---|---|---|---|
| RF-DETR | 1080p | 32.1 | 26.4 | 153.6 | 31.0 / 32.2 | 184.7 |
| RF-DETR | 720p | 32.2 | 26.2 | 154.5 | 31.1 / 31.3 | 185.6 |
| RT-DETRv2 | 1080p | 22.3 | 37.1 | 222.3 | 44.6 / 46.1 | 267.0 |
| RT-DETRv2 | 720p | 23.5 | 37.3 | 211.4 | 42.3 / 44.4 | 253.8 |

- **The module gives at most 32 fps (RF-DETR) and 22 to 24 fps (RT-DETRv2)**, below the author's prototype (37 and 27 fps).
  Cause: in `Pipeline`, preprocessing (2.8 ms) and postprocessing (1.7 ms) run in the inference thread, in series with the engine
  (26.4 ms), whereas the prototype overlapped them with capture threads. At 30 fps input the thread works at 94 % and
  drops 3 or 4 frames per minute. This does not violate the goal (≥ 15 fps), but there is **no margin above 32 fps**. Overlapping
  preprocessing and decoding with the engine gives margin; it was done afterwards and measured in section 13.
- A first RT-DETRv2 pass (12 fps) was discarded: it overlapped with another run of the validation itself. The files in
  `bench/` are the clean repetition.
- A `queue` p50 of 154 ms versus a `service` of 31 ms shows why the separation in point 2 of section 1 was needed.

## 12. What has not been measured

- **Real camera, CSI and V4L2**: untested (no camera). Only `gst:file:` and UDP/RTP through NVDEC.
- **ROS 2 only in a container** (PC, CPU, fake backend): the Jetson has no ROS 2.
- **No real water**: no turbidity or moving underwater content; the VAL AP (92.5 % 416×416 photos) does not detect it.
- **systemd system unit** (with `sudo`): untested; the user variant was.
- **Camera latency** (exposure, ISP, transmission): outside what was measured; latency starts at the decoder output.
- **Endurance**: 32 min with dynamic clocks; no 8 to 24 h, no fixed clocks, no sealed enclosure.
- **Temperature thresholds** of `HealthConfig`: provisional.
- **Tracking** (`--track`): not evaluated.
- **Energy**: board `VDD_IN`; without propulsion or lighting.
- **TEST**: not used.
- **Synthetic video**: slideshows of VAL photos, not real motion.

## 13. Staged pipeline: preprocessing and decoding overlapped with the engine

Sections 1 to 11 were measured with the serial pipeline (preprocess, engine and decode on one thread). The pipeline now has
five threads (capture, preprocess, inference, postprocess, publication) with explicit hand-offs (see the README, "Performance").
Output is unchanged: `jv replay` of the 160-image subset at score threshold 0.0 against the output of the serial version is
**identical** (`staged_pipeline/replay_equivalence.txt`), so the AP of section 4 is unchanged by construction.

**Method.** Same Jetson, same power mode (15 W, dynamic clocks, read and not changed), same engines, same 1080p/30 fps H.264
clips (section 2), same flags as sections 5 and 6, one run at a time. "Before" is the commit before the change, "after" the
staged pipeline, both copied into separate directories and run interleaved (before, after, before, after) to cancel drift.
Latency is `total_ns` (capture stamp to publication, includes queue waits); the first 30 results are discarded.
Files: `docs/validation_data/staged_pipeline/`.

**Capacity (`jv bench`, 1,800 frames, no input limit):**

| Model | Before (fps) | After (fps) | Gain | Per-stage p50 before / after (ms): preprocess, engine, decode |
|---|---|---|---|---|
| RF-DETR 1080p | 31.31, 31.02 | 33.07, 33.00 | +6 % | 3.2, 26.6, 1.7 / 4.4, 30.2, 3.9 |
| RT-DETRv2 1080p | 21.63 | 23.39 | +8 % | 7.1, 37.8, 0.4 / 10.1, 43.0, 0.6 |

**End to end at 30 fps input (`jv run --live --loop`, 1080p):**

| Run | Results | fps | Drops | `total` p50 / p95 / p99 (ms) | Board W | Tj max C |
|---|---|---|---|---|---|---|
| RF-DETR, 300 s, before | 8,878 | 29.981 | 5 (0.06 %) | 37.8 / 47.0 / 48.8 | 9.54 | 57.5 |
| RF-DETR, 300 s, after | 8,882 | 29.994 | 1 (0.01 %) | 34.5 / 38.3 / 40.8 | 9.51 | 57.6 |
| RF-DETR, 12 min, before | 21,478 | 29.992 | 4 (0.02 %) | 36.1 / 44.4 / 48.2 | 9.59 | 57.8 |
| RF-DETR, 12 min, after | 21,482 | 29.996 | 1 (0.005 %) | 34.7 / 38.4 / 41.0 | 9.56 | 57.6 |
| RT-DETRv2, 300 s, before | 6,309 | 21.30 | 2,575 (29.0 %) | 62.3 / 76.9 / 79.8 | 8.67 | 56.6 |
| RT-DETRv2, 300 s, after | 7,485 | 25.27 | 1,399 (15.8 %) | 64.6 / 81.2 / 83.9 | 9.49 | 57.5 |

Reading:
- **The gain is smaller than the 15 % estimated from the stage times.** Capacity of RF-DETR rises 6 % (31.2 to 33.0 fps), not
  15 %: with the stages overlapped, the engine thread takes 30.2 ms instead of 26.6 ms. The other threads compete for the same
  CPU cores and LPDDR memory bandwidth, which slows the GPU-bound engine call. A shorter GIL switch interval (0.05 to 5 ms) was
  tried and changed nothing (33.1 +- 0.3 fps), so it is not GIL hand-over latency; the remaining cause was not isolated.
- At the 30 fps operating point RF-DETR already had little margin (94 % engine occupancy); now it has 91 %. Drops fall from
  0.02-0.06 % to 0.005-0.01 % and the tail latency improves clearly (p95 47.0 to 38.3 ms, p99 48.8 to 40.8 ms) because frames wait
  less in front of the engine. The p50 improves by 3 ms.
- RT-DETRv2 (more costly preprocessing, 7 ms) benefits most: +19 % delivered fps at 30 fps input and drops halve, but it is
  still below 30 fps. Its p95 latency is 4 ms higher because it now processes more frames at saturation.
- Power is unchanged for RF-DETR (9.5 W) and about +0.8 W for RT-DETRv2, which does more work per second.
- 12-minute stability run with the staged pipeline: 21,482 results, no errors, RSS 1,040 to 980 MB (no growth), 57.6 C maximum,
  degradation level 0 throughout, `jv` exit code 0. This is 12 minutes, not the 32 of section 6.
- Not re-measured with the staged pipeline: 720p, 15 fps, fixed clocks, the full-VAL AP (the replay equivalence above stands in
  for it), and the 32 min endurance run.

### 13.1 Other power modes (staged pipeline)

All figures above are at **15 W**, the reference mode. This subsection repeats the measurements in the other modes with the
final staged pipeline. Method: mode set with `nvpmodel`, verified with `nvpmodel -q`, 60 s settle, dynamic clocks
(`jetson_clocks` not used), same engines and 1080p/30 fps clips as above, `jv bench` twice and a 300 s `jv run --live --loop`
at 30 fps input per model. The runs here omit `--baseline`: the latency baseline was measured at 15 W and would raise false
throttling alarms in another mode. Tj stayed far below the 85 C abort limit. Data: `docs/validation_data/power_modes/`.

| Mode | Model | Capacity (`jv bench`, 2 runs) | 30 fps input, 300 s: fps | Drops | `total` p50 / p95 / p99 (ms) | Board W | Tj max C |
|---|---|---|---|---|---|---|---|
| 15 W (reference, section 13) | RF-DETR | 33.07, 33.00 | 29.994 | 1 (0.01 %) | 34.5 / 38.3 / 40.8 | 9.51 | 57.6 |
| 25 W | RF-DETR | 48.34, 48.17 | 29.995 | 1 (0.01 %) | 27.1 / 28.1 / 31.8 | 11.20 | 59.8 |
| MAXN_SUPER | RF-DETR | 51.02, 51.44 | 29.999 | 0 | 27.0 / 28.0 / 31.5 | 11.48 | 62.0 |
| 15 W (reference, section 13) | RT-DETRv2 | 23.39 (1 run) | 25.27 | 1,399 (15.8 %) | 64.6 / 81.2 / 83.9 | 9.49 | 57.5 |
| 25 W | RT-DETRv2 | 34.58, 34.56 | 29.992 | 2 (0.02 %) | 36.9 / 40.4 / 41.9 | 13.36 | 62.8 |
| MAXN_SUPER | RT-DETRv2 | 37.07, 36.97 | 29.987 | 3 (0.03 %) | 34.8 / 40.1 / 40.8 | 14.08 | 64.5 |

- **7 W (mode id 3) was not measured**: `nvpmodel` reports that this mode requires a reboot. No reboot was requested or
  forced, so the mode was skipped.
- Higher modes raise capacity (RF-DETR 33 to 48 to 51 fps; RT-DETRv2 23 to 35 to 37 fps) and RT-DETRv2 sustains 30 fps input
  from 25 W upward. Board power at 30 fps input rises by 1.7 to 2.0 W (RF-DETR) and 3.9 to 4.6 W (RT-DETRv2) relative to 15 W;
  these are board power (`VDD_IN`) while processing, not energy per frame at equal throughput, and the 15 W RT-DETRv2 row
  processes fewer frames.
- 25 W to MAXN_SUPER adds little (+6 % RF-DETR, +7 % RT-DETRv2) for +0.3 to 0.7 W.
- Each mode was a single 300 s run per model with temperatures measured in open air (no enclosure). The 15 W RT-DETRv2
  reference is a single bench run. The sweep did not re-measure 720p, 15 fps, endurance or AP in other modes.
