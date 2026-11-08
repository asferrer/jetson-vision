# Slideshow clips from VAL 1920x1080 images (no real footage exists). Usage: mkvideo.sh WxH FPS SECONDS
set -e
RES=$1; FPS=$2; SECS=$3; W=${RES%x*}; H=${RES#*x}
D=$REPO_DIR/out/video
python - <<PY
import json, os
val = json.load(open(os.path.join(os.environ["DATA_DIR"], "val.json")))
hd = sorted((i for i in val["images"] if (i["width"], i["height"]) == (1920, 1080)), key=lambda i: i["id"])
d = "$D/seq_${RES}_${FPS}"; os.makedirs(d, exist_ok=True)
n = $FPS * $SECS
for f in range(n):
    im = hd[(f // 3) % len(hd)]
    l = f"{d}/f_{f:05d}.jpg"
    if os.path.lexists(l): os.remove(l)
    os.symlink(os.environ["DATA_DIR"] + "/" + im["file_name"], l)
print(len(hd), "source images,", n, "frames")
PY
BR=$(( W >= 1920 ? 6000 : 3000 ))
gst-launch-1.0 -q multifilesrc location=$D/seq_${RES}_${FPS}/f_%05d.jpg index=0 caps="image/jpeg,framerate=$FPS/1" ! jpegdec ! videoscale ! video/x-raw,width=$W,height=$H ! videoconvert ! x264enc speed-preset=ultrafast bitrate=$BR key-int-max=$FPS ! video/x-h264,profile=main ! h264parse ! mp4mux ! filesink location=$D/val_${RES}_${FPS}fps.mp4
rm -rf $D/seq_${RES}_${FPS}
ls -la $D/val_${RES}_${FPS}fps.mp4
