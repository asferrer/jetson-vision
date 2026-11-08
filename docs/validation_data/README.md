# Validation data

Raw outputs of the measurements described in `docs/VALIDATION.md`: logs, `summary.json` per run, benchmark JSON, AP
logs and the scripts that produced them.

The scripts are the commands that were run on the Jetson, with the site-specific paths replaced by variables:

| Variable | Meaning |
|---|---|
| `REPO_DIR` | checkout of this repository on the Jetson |
| `DEPLOY_DIR` | working directory that holds the engines, videos and data |
| `JV_MODEL_DIR` | directory with the TensorRT engines |
| `DATA_DIR` | root of the validation images and `val.json` (COCO format) |
| `REF_DIR` | directory with the Jetson Python environment pieces used by the reference code |
| `PY` | Python interpreter with `tensorrt` and `torch` |

`out/val/env_run.sh` (not versioned) exported `PY`, `JV_MODEL_DIR` and `PYTHONPATH` before each script. Hostnames, user
names and absolute paths were redacted from the recorded logs; no numbers were changed. The video clips, images and
engines are not included.
