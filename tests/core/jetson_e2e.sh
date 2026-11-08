#!/bin/bash
# One end-to-end run of `jv run` on the Jetson with logs: tests/core/jetson_e2e.sh TAG MANIFEST VIDEO FPS FRAMES [extra jv args...]
# Writes out/e2e/TAG/{run.jsonl,jv.log,monitor.csv,tegrastats.log,summary.json}. Needs sudo for tegrastats only.
TAG=$1; MANIFEST=$2; VIDEO=$3; FPS=$4; FRAMES=$5; shift 5
source "$REPO_DIR/out/val/env_run.sh"
O=out/e2e/$TAG; rm -rf $O; mkdir -p $O
sudo -n tegrastats --interval 1000 --logfile $O/tegrastats.log > /dev/null 2>&1 &
sleep 2
$PY -m jetson_vision.cli run --source gst:file:$VIDEO --live --fps $FPS --model $MANIFEST \
    --sink jsonl:$O/run.jsonl --max-frames $FRAMES "$@" > $O/jv.log 2>&1 &
JV=$!
$PY tests/core/jetson_monitor.py --pid $JV --out $O/monitor.csv &
wait $JV; echo "jv exit $?" >> $O/jv.log
sleep 2; sudo -n tegrastats --stop > /dev/null 2>&1
$PY tests/core/jetson_summary.py --jsonl $O/run.jsonl --monitor $O/monitor.csv --tegra $O/tegrastats.log \
    --out $O/summary.json > /dev/null 2>&1
echo "$TAG done"
