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
