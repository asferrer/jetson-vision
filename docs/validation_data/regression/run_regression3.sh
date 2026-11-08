# Third pass: strict comparison at the OPERATING threshold of each manifest (no tie at the top-k boundary).
source "$REPO_DIR/out/val/env_run.sh"
O=out/val; R_PY="$PY tests/core/jetson_regression.py"; JV="$PY -m jetson_vision.cli"
for M in rfdetr_int8g:0.45:0.5 rtdetrv2_int8mix:0.6:5; do
  IFS=: read NAME THR BOXTOL <<< "$M"
  for SET in subset full; do
    TAG=${NAME}_$SET
    echo "== $TAG thr=$THR box_tol=$BOXTOL $(date +%T)"
    $R_PY filter --jsonl $O/ref_$TAG.jsonl --min-score $THR --out $O/refthr_$TAG.jsonl
    $JV replay --source $O/$SET --model models/$NAME.yaml --sink jsonl:$O/runthr_$TAG.jsonl --reference $O/refthr_$TAG.jsonl --box-tol $BOXTOL > $O/replaythr_$TAG.log 2>&1
    echo "replay exit $?" >> $O/replaythr_$TAG.log
    grep -v "^DIFF\|^20\|TRT" $O/replaythr_$TAG.log | tail -3
  done
done
echo ALLDONE
