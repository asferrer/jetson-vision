# Real failures on the Jetson, with the user-service variant of the unit (WatchdogSec=10 for speed). Log: out/fail/log.txt
R=$REPO_DIR; D=$R/out/svc; F=$R/out/fail; mkdir -p $F
source $R/out/val/env_run.sh
LOG=$F/log_t4.txt; : > $LOG
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
say "T4 blocked sink (FIFO nobody reads), direct run, SIGTERM-like stop at 20 s"
rm -f $F/fifo; mkfifo $F/fifo; (exec 3<$F/fifo; sleep 300) & HOLD=$!
T0=$(date +%s)
timeout -s KILL 90 $PY -m jetson_vision.cli run --source gst:file:$R/out/video/val_1280x720_30fps.mp4 --live --fps 30 --model models/rfdetr_int8g.yaml \
   --sink jsonl:$F/fifo --sink jsonl:$F/good2.jsonl --max-seconds 20 > $F/blocked.log 2>&1; say "exit $? after $(( $(date +%s) - T0 )) s (20 s requested)"; rows $F/good2.jsonl | tee -a $LOG
kill $HOLD 2>/dev/null
