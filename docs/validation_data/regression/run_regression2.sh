# Second pass with the tie-robust comparison: replay against the independent reference, plus the AP of the reference itself.
source "$REPO_DIR/out/val/env_run.sh"
V=$DATA_DIR/val.json; O=out/val
R_PY="$PY tests/core/jetson_regression.py"; JV="$PY -m jetson_vision.cli"
for M in rfdetr_int8g:0.0 rtdetrv2_int8mix:0.001; do
  NAME=${M%:*}; THR=${M#*:}
  for SET in subset full; do
    TAG=${NAME}_$SET
    echo "== $TAG $(date +%T)"
    $JV replay --source $O/$SET --model models/$NAME.yaml --threshold $THR --sink jsonl:$O/run2_$TAG.jsonl --reference $O/ref_$TAG.jsonl > $O/replay2_$TAG.log 2>&1
    echo "replay exit $?" >> $O/replay2_$TAG.log
    $R_PY ap --manifest models/$NAME.yaml --val $V --list $O/$SET/list.json --jsonl $O/run2_$TAG.jsonl > $O/ap2_$TAG.log 2>&1
    $R_PY ap --manifest models/$NAME.yaml --val $V --list $O/$SET/list.json --jsonl $O/ref_$TAG.jsonl > $O/ap_ref_$TAG.log 2>&1
    tail -3 $O/replay2_$TAG.log; tail -1 $O/ap2_$TAG.log; tail -1 $O/ap_ref_$TAG.log
  done
done
echo ALLDONE
