# Real failures on the Jetson, with the user-service variant of the unit (WatchdogSec=10 for speed). Log: out/fail/log.txt
R=$REPO_DIR; D=$R/out/svc; F=$R/out/fail; mkdir -p $F
source $R/out/val/env_run.sh
LOG=$F/log_t1.txt; : > $LOG
say() { echo "[$(date +%T)] $*" | tee -a $LOG; }
st() { systemctl --user show jetson-vision -p ActiveState,SubState,MainPID,NRestarts,Result,ExecMainStatus | tr '\n' ' ' | tee -a $LOG; echo >> $LOG; echo; }
rows() { python3 - "$1" <<EOF
import json,sys
r=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
res=[x for x in r if x.get("record")=="result"]
print("results",len(res),"sessions",sorted({x["frame_ref"]["session_id"][:8] for x in res}),"starts",[(x.get("restart_cause")) for x in r if x.get("record")=="event"])
EOF
}
killsrc() { pkill -9 -f "gst-launch-1.0 -q filesrc"; }
sender() { # sender SECONDS: RTP/H.264 over UDP from a file, real time
  timeout -s KILL $1 gst-launch-1.0 -q filesrc location=$R/out/video/val_1280x720_30fps.mp4 ! qtdemux ! h264parse config-interval=-1 ! \
    rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=5600 sync=true > /dev/null 2>&1 &
  echo $!
}
UDPSRC='gst:udpsrc port=5600 caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink max-buffers=1 drop=true sync=false'
say "T1 source killed, SIGSTOP and recovery under systemctl --user"
sed -i "s#^JV_SOURCE=.*#JV_SOURCE=$UDPSRC#" $D/env
sed -i "s#^JV_MODEL=.*#JV_MODEL=$R/models/rfdetr_int8g.yaml#" $D/env
rm -f $D/logs/results.jsonl $D/last_stop_cause $D/logs/service.log; grep -q JV_LAST_STOP_FILE $D/env || echo JV_LAST_STOP_FILE=$D/last_stop_cause >> $D/env
grep -q "^StandardOutput" $D/jetson-vision.service || sed -i "s#^Restart=on-failure#StandardOutput=append:$D/logs/service.log\nStandardError=append:$D/logs/service.log\nRestart=on-failure#" $D/jetson-vision.service
sed -i 's#--live##' $D/jetson-vision.service; sed -i 's#--watchdog$#--watchdog --live --fps 30#' $D/jetson-vision.service
systemctl --user daemon-reload
S1=$(sender 60); say "sender A pid $S1"
systemctl --user start jetson-vision; sleep 20; say "service started"; st
MP=$(systemctl --user show jetson-vision -p MainPID --value); rows $D/logs/results.jsonl | tee -a $LOG
killsrc; say "sender KILLED (source silent); waiting 35 s, WatchdogSec=10"; sleep 35; st
grep -c "silent for" $D/logs/service.log | sed 's/^/silent-warnings: /' | tee -a $LOG; rows $D/logs/results.jsonl | tee -a $LOG
S2=$(sender 150); say "sender B pid $S2"; sleep 25; st; rows $D/logs/results.jsonl | tee -a $LOG
say "SIGSTOP on MainPID $MP"; kill -STOP $MP; for i in 5 10 15 20 25; do sleep 5; say "t+$i s"; st; done
sleep 20; st; rows $D/logs/results.jsonl | tee -a $LOG; cat $D/last_stop_cause | tee -a $LOG
journalctl --user -u jetson-vision --no-pager 2>/dev/null | tail -15 >> $LOG
say "stop service"; systemctl --user stop jetson-vision; st; killsrc
