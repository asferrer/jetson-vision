# Builds the user-service test layout under repo/out/svc from deploy/jetson-vision.user.service (paths and WatchdogSec adapted).
R=$REPO_DIR; D=$R/out/svc
mkdir -p $D/logs
PYBIN=$PY
sed -e "s#%h/jetson-vision/venv/bin/jv#$PYBIN -m jetson_vision.cli#" -e "s#%h/jetson-vision#$D#g" \
    -e "s#^WatchdogSec=30#WatchdogSec=${WD:-10}#" $R/deploy/jetson-vision.user.service > $D/jetson-vision.service
grep -q "^WorkingDirectory" $D/jetson-vision.service || sed -i "s#^NotifyAccess=main#NotifyAccess=main\nWorkingDirectory=$R#" $D/jetson-vision.service
cat > $D/env <<EOF
JV_SOURCE=gst:file:$R/out/video/val_1280x720_30fps.mp4
JV_MODEL=$R/models/rfdetr_int8g.yaml
JV_MODEL_DIR=$JV_MODEL_DIR
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6
PYTHONPATH=$R:/usr/lib/python3.10/dist-packages:$REF_DIR/pylib
LD_LIBRARY_PATH=/opt/nvidia/vpi3/lib/aarch64-linux-gnu
EOF
systemctl --user link $D/jetson-vision.service 2>&1 | tail -1
systemctl --user daemon-reload
cat $D/jetson-vision.service
