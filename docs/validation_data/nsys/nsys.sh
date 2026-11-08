source "$REPO_DIR/out/val/env_run.sh"
N=out/nsys; rm -rf $N; mkdir -p $N
nsys --version > $N/version.txt 2>&1
date +%T > $N/start.txt
timeout -k 15 150 nsys profile --duration=60 --delay=8 --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true -o $N/trace \
  $PY -m jetson_vision.cli run --source gst:file:$PWD/out/video/val_1920x1080_30fps.mp4 --live --fps 30 --loop --model models/rfdetr_int8g.yaml \
  --sink jsonl:$N/run.jsonl --max-seconds 80 > $N/jv.log 2>&1
echo "nsys exit $? at $(date +%T)" > $N/exit.txt
ls -la $N >> $N/exit.txt
