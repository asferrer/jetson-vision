# Installation on the Jetson (JetPack 6.2.1)

The system unit (`/etc/systemd/system`) has not been tested: it would require `sudo`. The user variant
(`jetson-vision.user.service`, see below) was tested with `systemctl --user`; results are in `docs/VALIDATION.md`. Do not touch `nvpmodel`: the
power mode is a deployment decision, not a runtime one.

1. User and directories:

   ```
   sudo useradd --system --home /opt/jetson-vision --groups video <service-user>
   sudo mkdir -p /opt/jetson-vision/{models,engines} /etc/jetson-vision
   sudo chown -R <service-user>: /opt/jetson-vision
   ```

2. Python 3.10 environment with access to the system packages (TensorRT, CUDA) and installation of the core:

   ```
   sudo -u <service-user> python3 -m venv --system-site-packages /opt/jetson-vision/venv
   sudo -u <service-user> /opt/jetson-vision/venv/bin/pip install /path/to/jetson-vision
   ```

3. Manifest in `/opt/jetson-vision/models/` and engine in `/opt/jetson-vision/engines/`. The engine is built on the Jetson
   itself (the loader refuses to start if it does not match the TensorRT version or the GPU, and it does not rebuild).

4. Configuration and unit:

   ```
   sudo install -m 644 deploy/env /etc/jetson-vision/env     # edit JV_SOURCE and JV_MODEL
   sudo install -m 644 deploy/jetson-vision.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now jetson-vision
   ```

5. Check:

   ```
   systemctl status jetson-vision          # must be "active (running)"
   journalctl -u jetson-vision -f
   tail -f /var/log/jetson-vision/results.jsonl
   ```

## Cameras

Sources are described in the README ("Camera sources"). Prerequisites on the Jetson:

- **Permissions.** The service user must be in the `video` group (step 1 does it for a system user; for your own user,
  `sudo usermod -aG video $USER` and log in again). `ls -l /dev/video*` shows the group that owns the devices. A user that cannot
  open the device gets `Permission denied` from `v4l2src`, reported as a `SourceError` and retried.
- **CSI.** `nvarguscamerasrc` talks to the `nvargus-daemon` service: `systemctl is-active nvargus-daemon` must say `active`
  (`sudo systemctl restart nvargus-daemon` after hot-plugging or if it is stuck). The sensor must be enabled by the device tree
  or `sudo /opt/nvidia/jetson-io/jetson-io.py` for the camera in use. With no sensor `nvarguscamerasrc` prints
  "No cameras available" and the source retries with backoff.
- **USB.** `v4l2-ctl --list-devices` and `v4l2-ctl -d /dev/video0 --list-formats-ext` show the device nodes, formats and rates
  (`sudo apt install v4l-utils` if missing). Ask for a format and size the camera lists: `fmt=mjpg` for 1080p30 on most USB
  cameras (raw YUY2 is limited by USB bandwidth). Webcams often expose a second node for metadata; use the first one.
- **GStreamer elements.** `gst-inspect-1.0 nvarguscamerasrc nvv4l2decoder nvvidconv v4l2src rtspsrc` must find all of them
  (JetPack provides them). On this JetPack the software `jpegdec` could not decode the output of `jpegenc` even with
  `gst-launch-1.0`, so keep `hw=1` for MJPEG on the Jetson.
- **Quick check without the detector:**
  `gst-launch-1.0 v4l2src device=/dev/video0 ! image/jpeg,width=1920,height=1080,framerate=30/1 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! fakesink`
  (CSI: `nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! fakesink`).

## How the watchdog works

`Type=notify`: the process sends `READY=1` when the engine is loaded and `WATCHDOG=1` at half of `WatchdogSec`, but only if
it has produced a result in the last few seconds (or has declared "no source" in a controlled way). If the engine or the camera
hangs, the heartbeat stops and systemd restarts the service (`Restart=on-failure`, at most 5 restarts per 10 minutes).
If health orders a stop (level 3) the process exits with code 3 and systemd does not restart it (`RestartPreventExitStatus=3`).

Watchdog test without a reboot: `sudo kill -STOP $(systemctl show -p MainPID --value jetson-vision)`; after
`WatchdogSec` seconds systemd must restart the service and the JSONL must record a new `session_id` and the cause
(`watchdog`).

## User variant (no sudo)

`deploy/jetson-vision.user.service` is the same unit without `User=`, `Group=` or `StateDirectory=`, with paths under `%h/jetson-vision`
(`env`, `logs/`, `last_stop_cause`). Installation:

```
mkdir -p ~/.config/systemd/user ~/jetson-vision/logs && cp deploy/jetson-vision.user.service ~/.config/systemd/user/
cp deploy/env ~/jetson-vision/env            # edit JV_SOURCE, JV_MODEL and JV_MODEL_DIR
systemctl --user daemon-reload && systemctl --user start jetson-vision
```

Without `loginctl enable-linger <user>` the user manager stops when the last session closes (and the service with it).

## Logging and disk

`jv run` rotates each JSONL at 100 MB (`--log-max-mb`, 3 old copies; 0 disables rotation). Without rotation the log grows without
bound (measured figure in `docs/VALIDATION.md`).

## No ROS

The Jetson has no ROS 2. The adapter (`adapters/ros2`) is a separate package for a PC or a container with Humble; the
systemd unit only starts the core.
