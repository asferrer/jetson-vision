cd $REPO_DIR
bash tests/core/jetson_e2e.sh endurance_rfdetr_1080p30 models/rfdetr_int8g.yaml out/video/val_1920x1080_30fps.mp4 30 0 --loop --max-seconds 1920 --log-max-mb 0 --baseline 30:26.5,15:37.2
bash tests/core/jetson_e2e.sh baseline_rfdetr_720p15 models/rfdetr_int8g.yaml out/video/val_1280x720_15fps.mp4 15 0 --loop --max-seconds 300 --baseline 30:26.5,15:37.2
bash tests/core/jetson_e2e.sh baseline_rtdetrv2_1080p30 models/rtdetrv2_int8mix.yaml out/video/val_1920x1080_30fps.mp4 30 0 --loop --max-seconds 300 --baseline 30:37.2,15:37.7
bash tests/core/jetson_e2e.sh baseline_rtdetrv2_720p15 models/rtdetrv2_int8mix.yaml out/video/val_1280x720_15fps.mp4 15 0 --loop --max-seconds 300 --baseline 30:37.2,15:37.7
echo ALLDONE
