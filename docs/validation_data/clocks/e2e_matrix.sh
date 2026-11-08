# End-to-end matrix: both manifests x {1080p,720p} x {30,15} fps, live H.264 file through NVDEC. Optional 1st arg: baseline string.
cd $REPO_DIR
BASE=${1:-}; SUF=${2:-}
EXTRA=""; [ -n "$BASE" ] && EXTRA="--baseline $BASE"
for M in rfdetr_int8g rtdetrv2_int8mix; do
  for RES in 1920x1080 1280x720; do
    for FPS in 30 15; do
      bash tests/core/jetson_e2e.sh ${M}_${RES}_${FPS}fps$SUF models/$M.yaml out/video/val_${RES}_${FPS}fps.mp4 $FPS 0 $EXTRA
    done
  done
done
echo ALLDONE
