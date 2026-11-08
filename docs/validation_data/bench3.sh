source "$REPO_DIR/out/val/env_run.sh"
B=out/bench
for M in rtdetrv2_int8mix; do for RES in 1920x1080 1280x720; do
  $PY -m jetson_vision.cli bench --source gst:file:$PWD/out/video/val_${RES}_30fps.mp4 --model models/$M.yaml > $B/bench_${M}_${RES}.json 2> $B/bench_${M}_${RES}.log
done; done
echo BENCHDONE > $B/done.txt
