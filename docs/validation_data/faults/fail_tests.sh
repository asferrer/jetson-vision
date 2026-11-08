# Real failures on the Jetson, with the user-service variant of the unit (WatchdogSec=10 for speed). Log: out/fail/log.txt
R=$REPO_DIR; D=$R/out/svc; F=$R/out/fail; mkdir -p $F
source $R/out/val/env_run.sh
LOG=$F/log.txt; : > $LOG
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

say "T2 manifest with a wrong sha256"
sed 's/sha256: f7/sha256: a7/' $R/models/rfdetr_int8g.yaml > $F/bad_hash.yaml
sed -i "s#^JV_MODEL=.*#JV_MODEL=$F/bad_hash.yaml#" $D/env
systemctl --user reset-failed jetson-vision 2>/dev/null; systemctl --user start jetson-vision; for i in 1 2 3 4 5 6; do sleep 10; st; done
grep -m2 "refused" $D/logs/service.log >> $LOG; systemctl --user stop jetson-vision; systemctl --user reset-failed jetson-vision 2>/dev/null
$PY -m jetson_vision.cli run --source $R/out/video/val_1280x720_30fps.mp4 --model $F/bad_hash.yaml --max-frames 5 > $F/hash_direct.log 2>&1; say "direct run with bad hash: exit $? (log: hash_direct.log)"

say "T3 full disk: sink on /dev/full, with a good sink next to it"
$PY -m jetson_vision.cli run --source gst:file:$R/out/video/val_1280x720_30fps.mp4 --live --fps 30 --model models/rfdetr_int8g.yaml \
   --sink jsonl:/dev/full --sink jsonl:$F/good.jsonl --max-seconds 20 > $F/full_disk.log 2>&1; say "exit $?"; rows $F/good.jsonl | tee -a $LOG

say "T4 blocked sink (FIFO nobody reads), direct run, SIGTERM-like stop at 20 s"
rm -f $F/fifo; mkfifo $F/fifo; (exec 3<$F/fifo; sleep 300) & HOLD=$!
T0=$(date +%s)
timeout -s KILL 90 $PY -m jetson_vision.cli run --source gst:file:$R/out/video/val_1280x720_30fps.mp4 --live --fps 30 --model models/rfdetr_int8g.yaml \
   --sink jsonl:$F/fifo --sink jsonl:$F/good2.jsonl --max-seconds 20 > $F/blocked.log 2>&1; say "exit $? after $(( $(date +%s) - T0 )) s (20 s requested)"; rows $F/good2.jsonl | tee -a $LOG
kill $HOLD 2>/dev/null

say "T4b blocked sink under the service: results.jsonl is a FIFO"
sed -i "s#^JV_MODEL=.*#JV_MODEL=$R/models/rfdetr_int8g.yaml#" $D/env
sed -i "s#^JV_SOURCE=.*#JV_SOURCE=gst:file:$R/out/video/val_1280x720_30fps.mp4#" $D/env
sed -i 's#--watchdog --live --fps 30#--watchdog --live --fps 30#' $D/jetson-vision.service
rm -f $D/logs/results.jsonl; mkfifo $D/logs/results.jsonl; (exec 3<$D/logs/results.jsonl; sleep 300) & HOLD=$!
systemctl --user start jetson-vision; sleep 15; st
for i in 1 2 3 4 5 6; do sleep 8; st; done
journalctl --user -u jetson-vision --no-pager 2>/dev/null | tail -6 >> $LOG
systemctl --user stop jetson-vision; kill $HOLD 2>/dev/null; rm -f $D/logs/results.jsonl
say "DONE"
