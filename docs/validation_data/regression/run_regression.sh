# AP regression: subset (fixed 160 VAL images) and all of VAL, both manifests. VAL only.
source "$REPO_DIR/out/val/env_run.sh"
V=$DATA_DIR/val.json; R=$DATA_DIR; O=out/val
ST=$REF_DIR
R_PY="$PY tests/core/jetson_regression.py"
JV="$PY -m jetson_vision.cli"
$R_PY prepare --val $V --data-root $R --out-dir $O/subset > $O/prepare_subset.log 2>&1
$R_PY prepare --val $V --data-root $R --out-dir $O/full --all > $O/prepare_full.log 2>&1
for M in rfdetr_int8g:0.0 rtdetrv2_int8mix:0.001; do
  NAME=${M%:*}; THR=${M#*:}
  for SET in subset full; do
    TAG=${NAME}_$SET
    echo "== $TAG $(date +%T)"
    $R_PY reference --manifest models/$NAME.yaml --stages-dir $ST --list $O/$SET/list.json --out $O/ref_$TAG.jsonl > $O/ref_$TAG.log 2>&1
    $JV replay --source $O/$SET --model models/$NAME.yaml --threshold $THR --sink jsonl:$O/run_$TAG.jsonl --reference $O/ref_$TAG.jsonl > $O/replay_$TAG.log 2>&1
    echo "replay exit $?" >> $O/replay_$TAG.log
    $R_PY ap --manifest models/$NAME.yaml --val $V --list $O/$SET/list.json --jsonl $O/run_$TAG.jsonl > $O/ap_$TAG.log 2>&1
    tail -1 $O/ap_$TAG.log
    tail -2 $O/replay_$TAG.log
  done
done
echo ALLDONE
