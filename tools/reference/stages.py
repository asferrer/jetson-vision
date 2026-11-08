"""Stage-by-stage breakdown of the detection pipeline on the Jetson (the author's prototype, used as the independent
reference of the AP regression). VAL only.
Stages (serial, synchronised between them): read (file bytes), decode, pre_cpu, h2d, pre_gpu, infer, post.
Usage: python stages.py --model rt|rf --engine X --dec cv2|gst --pre cpu|u8gpu|u8gpu_area|gpu|vpi
                      --copy pageable|pinned|zerocopy --post cpu|gpu --d2h sync|pinned --out res/name [--n 0] [--noap]
"""
import argparse, json, os, sys, time, threading
import numpy as np, cv2, torch, torch.nn.functional as F, tensorrt as trt

ROOT = os.environ.get("DEPLOY_DIR", ".")
DATA = os.environ.get("DATA_DIR", ROOT + "/data") + "/"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
GPUF = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq"


def temp_max():
    m = 0
    for i in range(8):
        try:
            m = max(m, int(open(f"/sys/class/thermal/thermal_zone{i}/temp").read()) / 1000)
        except Exception:
            pass
    return m


class Sampler(threading.Thread):
    """GPU clock and temperature every 200 ms; aborts the process if the temperature exceeds TMAX."""
    TMAX = 80.0

    def __init__(self):
        super().__init__(daemon=True)
        self.f, self.t, self.on = [], [], True

    def run(self):
        while self.on:
            try:
                self.f.append(int(open(GPUF).read()) / 1e6)
            except Exception:
                pass
            t = temp_max()
            self.t.append(t)
            if t > self.TMAX:
                print("TEMPERATURA", t, "ABORTO", flush=True)
                os._exit(3)
            time.sleep(0.2)


def summ(a):
    a = 1000 * np.asarray(a, dtype=np.float64)
    if len(a) == 0:
        return {"med": 0.0, "p95": 0.0, "mean": 0.0}
    return {"med": float(np.median(a)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean())}


class Eng:
    def __init__(self, path):
        lg = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(lg, "")
        self.engine = trt.Runtime(lg).deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.buf, self.io = {}, {}
        dt = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
              trt.int64: torch.int64, trt.bool: torch.bool}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            self.io[n] = self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
            self.buf[n] = torch.empty(tuple(self.engine.get_tensor_shape(n)), dtype=dt[self.engine.get_tensor_dtype(n)], device="cuda")
            self.ctx.set_tensor_address(n, self.buf[n].data_ptr())

    def run(self):
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)


def sync():
    torch.cuda.synchronize()


def pin(shape, dtype):
    t = torch.empty(shape, dtype=dtype).pin_memory()
    return t, t.numpy()


class Pipe:
    def __init__(s, a):
        s.a, s.rt = a, a.model == "rt"
        s.S = 640 if s.rt else 560
        S = s.S
        s.eng = Eng(a.engine)
        s.in_name = "images" if s.rt else "input"
        s.inp = s.eng.buf[s.in_name]
        s.p_f32, s.p_f32n = pin((1, 3, S, S), torch.float32)   # pinned fp32 input (async copy or zero copy)
        s.p_u8, s.p_u8n = pin((S, S, 3), torch.uint8)
        s.sz_t = torch.zeros(1, 2, dtype=torch.int32).pin_memory()
        if a.copy == "zerocopy" and a.pre == "cpu":
            s.eng.ctx.set_tensor_address(s.in_name, s.p_f32.data_ptr())
        if a.dec == "gst":
            from gstdec import GstDec
            s.gst = GstDec("videoconvert ! video/x-raw,format=BGR")
        s.mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
        s.std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
        s.u8g = torch.empty((S, S, 3), dtype=torch.uint8, device="cuda")
        outs = ("labels", "boxes", "scores") if s.rt else ("dets", "labels")
        s.hp = {n: torch.empty(s.eng.buf[n].shape, dtype=s.eng.buf[n].dtype).pin_memory() for n in outs}
        if a.pre == "vpi":
            import vpi
            s.vpi = vpi

    def decode(s, data):
        if s.a.dec == "gst":
            arr, w, h, f = s.gst.decode(data)
            stride = arr.size // h   # GStreamer pads every row to a multiple of 4 bytes
            return np.ascontiguousarray(arr.reshape(h, stride)[:, : 3 * w].reshape(h, w, 3))
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def norm_gpu(s, x_u8_rgb_gpu):
        """uint8 HWC RGB (GPU) -> normalised fp32 NCHW, written into the engine input."""
        x = x_u8_rgb_gpu.permute(2, 0, 1)[None].float() * (1.0 / 255.0)
        if not s.rt:
            x = (x - s.mean) / s.std
        s.inp.copy_(x)

    def pre_cpu(s, bgr):
        a, S = s.a, s.S
        h = bgr.shape[0]
        if a.pre == "cpu":
            if s.rt:   # preprocess() of build_int8_engine.py
                interp = cv2.INTER_AREA if h > S else cv2.INTER_LINEAR
                img = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (S, S), interpolation=interp)
                if a.copy == "pageable":
                    return np.ascontiguousarray((img.astype(np.float32) * (1.0 / 255.0)).transpose(2, 0, 1)[None])
                np.multiply(img, np.float32(1.0 / 255.0), out=s.p_f32n[0].transpose(1, 2, 0), dtype=np.float32)
                return None
            from PIL import Image   # eval_rfdetr_val.prepare: PIL bilineal
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((S, S), Image.BILINEAR)
            x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
            if a.copy == "pageable":
                return np.ascontiguousarray(x.transpose(2, 0, 1)[None])
            s.p_f32n[0] = x.transpose(2, 0, 1)
            return None
        if a.pre in ("u8gpu", "u8gpu_area", "pil_u8gpu"):   # resize on the CPU in uint8, normalisation on the GPU
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if a.pre == "pil_u8gpu":   # the same PIL resize as the reference, in uint8; only the normalisation runs on the GPU
                from PIL import Image
                np.copyto(s.p_u8n, np.asarray(Image.fromarray(rgb).resize((S, S), Image.BILINEAR)))
                return None
            if s.rt:
                interp = cv2.INTER_AREA if h > S else cv2.INTER_LINEAR
            else:
                interp = cv2.INTER_AREA if a.pre == "u8gpu_area" else cv2.INTER_LINEAR
            cv2.resize(rgb, (S, S), dst=s.p_u8n, interpolation=interp)
            return None
        return bgr   # gpu / vpi: the raw frame is uploaded as is

    def h2d(s, pay, wh):
        a = s.a
        if s.rt:
            s.sz_t[0, 0], s.sz_t[0, 1] = wh
            s.eng.buf["orig_target_sizes"].copy_(s.sz_t, non_blocking=True)
        if a.pre == "cpu":
            if a.copy == "pageable":
                s.inp.copy_(torch.from_numpy(pay))
            elif a.copy == "pinned":
                s.inp.copy_(s.p_f32, non_blocking=True)
            return None
        if a.pre in ("u8gpu", "u8gpu_area", "pil_u8gpu"):
            s.u8g.copy_(s.p_u8, non_blocking=True)
            return None
        if a.pre == "gpu":
            return torch.from_numpy(pay).cuda()
        return s.vpi.asimage(pay, s.vpi.Format.BGR8)

    def pre_gpu(s, ph, bgr):
        a, S = s.a, s.S
        if a.pre == "cpu":
            return
        if a.pre in ("u8gpu", "u8gpu_area", "pil_u8gpu"):
            s.norm_gpu(s.u8g)
            return
        if a.pre == "gpu":
            x = ph.permute(2, 0, 1).flip(0)[None].float()
            if s.rt and bgr.shape[0] > S:
                x = F.interpolate(x, (S, S), mode="area")
            elif s.rt:
                x = F.interpolate(x, (S, S), mode="bilinear", align_corners=False)
            else:
                x = F.interpolate(x, (S, S), mode="bilinear", align_corners=False, antialias=True)
            s.norm_gpu(x.round().clamp_(0, 255).to(torch.uint8)[0].permute(1, 2, 0))   # redondeo a uint8 como cv2/PIL
            return
        vp = s.vpi
        with vp.Backend.CUDA:
            im = ph.convert(vp.Format.RGB8).rescale((S, S), interp=vp.Interp.LINEAR)
        with im.rlock_cuda() as d:
            t = torch.as_tensor(d, device="cuda").clone()
        s.norm_gpu(t)

    def post(s, wh):
        e = s.eng.buf
        if s.rt:
            keys, outs = ("labels", "boxes", "scores"), [e["labels"], e["boxes"], e["scores"]]
        else:
            keys, outs = ("dets", "labels"), [e["dets"], e["labels"]]
            if s.a.post == "gpu":
                lg = e["labels"][0].float()
                C = lg.shape[1]
                v, i = torch.sigmoid(lg).flatten().topk(300)
                q, lab = i // C, i % C
                cx, cy, w, h = e["dets"][0].float()[q].unbind(1)
                bx = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
                v, lab, bx = v.cpu().numpy(), lab.cpu().numpy(), bx.cpu().numpy()
                return v, lab, bx.astype(np.float64) * np.tile(np.array(wh, np.float64), 2)
        if s.a.d2h == "pinned":
            for k, o in zip(keys, outs):
                s.hp[k].copy_(o, non_blocking=True)
            sync()
            hs = [s.hp[k].float().numpy() if s.hp[k].dtype == torch.float16 else s.hp[k].numpy() for k in keys]
        else:
            hs = [o.float().cpu().numpy() if o.dtype == torch.float16 else o.cpu().numpy() for o in outs]
        if s.rt:
            lab, bx, sc = hs
            k = sc > 0.001
            return np.concatenate([lab[k, None], bx[k], sc[k, None]], 1)
        dets, logits = hs[0][0], hs[1][0]
        flat = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).reshape(-1)
        idx = np.argpartition(-flat, 299)[:300]
        idx = idx[np.argsort(-flat[idx], kind="stable")]
        q, lab = idx // logits.shape[1], idx % logits.shape[1]
        cx, cy, w, h = dets[q].astype(np.float64).T
        bx = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1) * np.tile(np.array(wh, np.float64), 2)
        return flat[idx], lab, bx

    def one(s, data, rec=None):
        T = time.perf_counter
        t = [T()]
        bgr = s.decode(data)
        t.append(T())
        wh = (bgr.shape[1], bgr.shape[0])
        pay = s.pre_cpu(bgr)
        t.append(T())
        ph = s.h2d(pay, wh)
        sync()
        t.append(T())
        s.pre_gpu(ph, bgr)
        sync()
        t.append(T())
        s.eng.run()
        sync()
        t.append(T())
        res = s.post(wh)
        t.append(T())
        if rec is not None:
            rec.append(t)
        return res, wh


NAMES = ["decode", "pre_cpu", "h2d", "pre_gpu", "infer", "post"]


def coco_eval(p, ims, preds, cats):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    P = []
    for im, r in zip(ims, preds):
        rows = [(int(x[0]), x[1:5], x[5]) for x in r] if p.rt else [(int(l), b, s_) for s_, l, b in zip(*r)]
        for l, b, s_ in rows:
            if l in cats:
                P.append({"image_id": im["id"], "category_id": l, "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], "score": float(s_)})
    gt = COCO(DATA + "val.json")
    ev = COCOeval(gt, gt.loadRes(P), "bbox")
    ev.params.imgIds = [i["id"] for i in ims]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"AP": 100 * ev.stats[0], "AP50": 100 * ev.stats[1], "AP75": 100 * ev.stats[2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--dec", default="cv2")
    ap.add_argument("--pre", default="cpu")
    ap.add_argument("--copy", default="pageable")
    ap.add_argument("--post", default="cpu")
    ap.add_argument("--d2h", default="sync")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--noap", action="store_true")
    ap.add_argument("--warm", type=int, default=40)
    a = ap.parse_args()
    val = json.load(open(DATA + "val.json"))
    ims = sorted(val["images"], key=lambda d: d["id"])[: a.n or None]
    cats = {c["id"] for c in val["categories"]}
    sm = Sampler()
    sm.start()
    datas = [open(DATA + im["file_name"], "rb").read() for im in ims]   # first pass: warms the page cache
    torch.cuda.set_stream(torch.cuda.Stream())   # non-default stream (TensorRT warns about stream 0)
    p = Pipe(a)
    tr = []
    for im in ims:   # read timed with a warm page cache
        t = time.perf_counter()
        open(DATA + im["file_name"], "rb").read()
        tr.append(time.perf_counter() - t)
    for i in range(a.warm):
        p.one(datas[i % len(datas)])
    sync()
    torch.cuda.reset_peak_memory_stats()
    recs, preds, whs = [], [], []
    c0, w0 = time.process_time(), time.perf_counter()
    for d in datas:
        r, wh = p.one(d, recs)
        preds.append(r)
        whs.append(wh)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    sm.on = False
    R = np.diff(np.array(recs), axis=1)
    tr = np.array(tr)
    cls = np.array(["416" if wh == (416, 416) else "1080p" if wh == (1920, 1080) else "grande" for wh in whs])
    out = {"args": vars(a), "n": len(ims), "read": summ(tr)}
    for k, nm in enumerate(NAMES):
        out[nm] = summ(R[:, k])
    out["total_sin_lectura"] = summ(R.sum(1))
    out["total_con_lectura"] = summ(R.sum(1) + tr)
    out["por_clase"] = {}
    for c in ("416", "1080p", "grande"):
        m = cls == c
        out["por_clase"][c] = {"n": int(m.sum()), "read": summ(tr[m]), "total_con_lectura": summ(R[m].sum(1) + tr[m]),
                               **{nm: summ(R[m, k]) for k, nm in enumerate(NAMES)}}
    out["cpu_pct_un_nucleo"] = 100 * cpu / wall
    out["wall_s"] = wall
    out["gpu_mem_torch_max_MB"] = torch.cuda.max_memory_allocated() / 2**20
    out["trt_device_mem_MB"] = p.eng.engine.device_memory_size_v2 / 2**20
    out["rss_MB"] = int([l for l in open("/proc/self/status") if l.startswith("VmHWM")][0].split()[1]) / 1024
    out["gpu_freq_MHz"] = {"med": float(np.median(sm.f)) if sm.f else 0, "min": min(sm.f) if sm.f else 0, "max": max(sm.f) if sm.f else 0}
    out["temp_max_C"] = max(sm.t) if sm.t else 0
    if not a.noap:
        out.update(coco_eval(p, ims, preds, cats))
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    keys = NAMES + ["total_con_lectura"]
    print(a.out, {k: round(out[k]["med"], 2) for k in keys}, "cpu%", round(out["cpu_pct_un_nucleo"]), "gpuMHz", out["gpu_freq_MHz"],
          "T", out["temp_max_C"], *( [("AP", round(out["AP"], 2))] if not a.noap else []), flush=True)


if __name__ == "__main__":
    main()
