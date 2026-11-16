"""Live run of the v6 vision chain at the decision rate (plan 15 Nov, HISTORY 62): every 100 ms grab the MuMu window ->
det ONNX + seg ONNX on DirectML -> adapter_v5 -> per-stage timings, and every frame saved (960 x 540 png) for
live_game_check.py / shadow_run.py. Windows venv of out/v5/onnx_timing (onnxruntime-directml, opencv-python, mss):
  ..\\out\\v5\\onnx_timing\\.venv\\Scripts\\python live_run.py --out ..\\out\\v5\\live_v6\\trio [--minutes 5]
  ... live_run.py --image FRAME.png --out DIR [--seconds 20]     (self-check: the same still frame in a loop)
timing.csv columns: i, t (s since start), late (ms behind the 100 ms clock), grab, det, seg, adapt, total (ms), battle,
entities. One summary line per 10 s to stdout. detect / segment / load are imported by bench_chain.py."""
import argparse, ast, ctypes, os, queue, sys, threading, time
from ctypes import wintypes
import cv2, numpy as np, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, '//wsl.localhost/Ubuntu-24.04/home/randy/nb68/bulletlabels/vendor']   # nbpc (obs_v5 needs it)
import adapter_v5 as A

OT = os.path.join(HERE, '..', 'out', 'v5', 'onnx_timing', 'trained_v6')
W, H, HP = 960, 540, 544; PAD = (HP - H) // 2   # det: ultralytics letterbox, 2 px top and bottom
PROV = ['DmlExecutionProvider', 'CPUExecutionProvider']
det_s = seg_s = None
LB, DS, DPAD = np.full((HP, W, 3), 114, np.uint8), 1.0, PAD   # det letterbox, reused (not thread-safe; the chain is sequential)


def load(det, seg):   # sessions as module globals, shared by detect / segment
    global det_s, seg_s
    det_s, seg_s = ort.InferenceSession(det, providers=PROV), ort.InferenceSession(seg, providers=PROV)
    global LB, DS, DPAD   # det input size from the model: 960 x 544 (v6) or 1280 x 736 (Step F pilot, frame upsampled)
    dh, dw = det_s.get_inputs()[0].shape[2:]; DS = dw / W; DPAD = (dh - round(H * DS)) // 2
    LB = np.full((dh, dw, 3), 114, np.uint8)
    names = list(ast.literal_eval(det_s.get_modelmeta().custom_metadata_map['names']).values())
    assert names == list(A.DET), 'det classes differ from adapter_v5.DET'


def detect(f, conf):
    """960 x 540 BGR -> (N, 5) rows cls, x1, y1, x2, y2 (960 x 540 coords). Decode + NMS as onnx_timing/live_det.py."""
    LB[DPAD:DPAD + round(H * DS)] = f if DS == 1 else cv2.resize(f, (LB.shape[1], round(H * DS)), interpolation=cv2.INTER_LINEAR)   # grey pad rows stay
    o = det_s.run(None, {'images': cv2.dnn.blobFromImage(LB, 1 / 255, swapRB=True)})[0][0]
    sc = o[4:]; c = sc.argmax(0); s = sc[c, np.arange(sc.shape[1])]; k = s >= conf
    if not k.any(): return np.zeros((0, 5))
    cx, cy, w, h = o[:4, k]; c, s = c[k], s[k]
    b = np.stack([cx - w / 2, cy - h / 2 - DPAD, cx + w / 2, cy + h / 2 - DPAD], 1) / DS
    keep = np.array(cv2.dnn.NMSBoxesBatched(np.c_[b[:, :2], b[:, 2:] - b[:, :2]].tolist(), s.tolist(), c.tolist(), conf, 0.6)).ravel()
    return np.c_[c[keep], b[keep]]


def segment(rgb):
    """960 x 540 RGB -> (540, 960) class ids: train_seg2 input (480 x 272 / 255, 16 zero rows below), argmax, nearest up
    (as rl/measure_noise.py). seg_argmax.onnx gives ids (argmax in the graph); plain seg.onnx gives logits."""
    x = np.zeros((1, 3, 288, 480), np.float32); x[0, :, :272] = cv2.resize(rgb, (480, 272), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1) / 255
    y = seg_s.run(None, {'images': x})[0][0]
    y = (y if y.ndim == 2 else y.argmax(0))[:272].astype(np.uint8)
    return cv2.resize(y, (W, H), interpolation=cv2.INTER_NEAREST)


def _timed(fn, *a):
    t = time.perf_counter(); r = fn(*a); return r, (time.perf_counter() - t) * 1e3


def vision(f, conf):
    """960 x 540 BGR -> det, rgb, seg, det ms, seg ms. Sequential: det and seg in two threads crash DirectML
    (DmlCommandRecorder / DmlGraphFusionHelper 80004005 within ~100 frames, HISTORY 63)"""
    det, ms_det = _timed(detect, f, conf)
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB); seg, ms_seg = _timed(segment, rgb)
    return det, rgb, seg, ms_det, ms_seg


def shrink(img):   # grabbed window (any 16:9 size) -> 960 x 540
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


ctypes.windll.shcore.SetProcessDpiAwareness(2); U = ctypes.windll.user32


def find_window(title):   # as live_det.py: the largest visible window with title in its name
    best = []
    def cb(hwnd, _):
        n = U.GetWindowTextLengthW(hwnd)
        if n and U.IsWindowVisible(hwnd):
            bf = ctypes.create_unicode_buffer(n + 1); U.GetWindowTextW(hwnd, bf, n + 1)
            if bf.value.lower().startswith(title.lower()):   # prefix only: "in" matched a browser window once
                r = wintypes.RECT(); U.GetClientRect(hwnd, ctypes.byref(r)); best.append((r.right * r.bottom, hwnd, bf.value))
        return True
    U.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)(cb), 0)
    if best: print('window title:', max(best)[2], flush=True)
    return max(best)[1] if best else None


def region(hwnd):   # as live_det.py: client area trimmed to 16:9 (toolbar on the right / strip on top)
    r = wintypes.RECT(); U.GetClientRect(hwnd, ctypes.byref(r)); p = wintypes.POINT(0, 0); U.ClientToScreen(hwnd, ctypes.byref(p))
    w, h = r.right, r.bottom
    if w * 9 > h * 16: w = h * 16 // 9
    else: top = h - w * 9 // 16; p.y += top; h -= top
    return {'left': p.x, 'top': p.y, 'width': w, 'height': h}


def open_grab(title, kind='dxcam'):
    """-> hwnd, grab(region) -> BGR image or None, in_front() (the game window is the foreground one)"""
    import mss   # also the fallback grab
    hwnd = find_window(title)
    if not hwnd: raise SystemExit('no visible "%s" window (start MuMu first, or pass --title)' % title)
    def pid_of(w):
        p = wintypes.DWORD(); U.GetWindowThreadProcessId(w, ctypes.byref(p)); return p.value
    # MuMu draws in MuMuNxDevice (one process) while MuMuPlayer (another process) holds the focus: accept any process
    # that owns a window titled like --title
    game_pids = set()
    def cb2(w, _):
        n = U.GetWindowTextLengthW(w)
        if n:
            bf = ctypes.create_unicode_buffer(n + 1); U.GetWindowTextW(w, bf, n + 1)
            if bf.value.lower().startswith(title.lower()): game_pids.add(pid_of(w))
        return True
    U.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)(cb2), 0)
    if kind == 'dxcam':   # DXGI desktop duplication: 2.4 vs 15 ms (mss, idle desktop) for 2496 x 1404 (HISTORY 63)
        import dxcam   # shortcut: primary monitor only (output 0), pass --grab mss for a game on another screen
        cap, last = dxcam.create(output_color='BGR'), [None]
        def grab(rg):
            x = cap.grab(region=(rg['left'], rg['top'], rg['left'] + rg['width'], rg['top'] + rg['height']))
            if x is not None: last[0] = x   # None = screen unchanged since the last grab: reuse that frame
            return last[0]
    else:
        m = mss.mss(); grab = lambda rg: np.asarray(m.grab(rg))[:, :, :3]
    print('window', region(hwnd), 'grab', kind, flush=True)
    return hwnd, grab, lambda: pid_of(U.GetForegroundWindow()) in game_pids


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', required=True); a.add_argument('--minutes', type=float, default=5)
    a.add_argument('--det', default=os.path.join(OT, 'det.onnx')); a.add_argument('--seg', default=os.path.join(OT, 'seg_argmax.onnx'))
    a.add_argument('--title', default='MuMu'); a.add_argument('--conf', type=float, default=0.3)   # live_game_check conf
    a.add_argument('--period', type=float, default=0.1); a.add_argument('--image', default=''); a.add_argument('--seconds', type=float, default=20)
    a.add_argument('--grab', choices=('dxcam', 'mss'), default='dxcam')
    a = a.parse_args()
    load(a.det, a.seg)
    os.makedirs(a.out, exist_ok=True)
    wq = queue.Queue(maxsize=200)
    def writer():   # png encode off the timed loop
        while (it := wq.get()) is not None: cv2.imwrite(*it)
    wt = threading.Thread(target=writer, daemon=True); wt.start()

    if a.image:
        still = cv2.resize(cv2.imread(a.image), (W, H), interpolation=cv2.INTER_AREA); grab = None; dur = a.seconds
    else:
        hwnd, grab, in_front = open_grab(a.title, a.grab); dur = a.minutes * 60

    ad, rows, skipped = A.Adapter(), [], 0
    detect(np.zeros((H, W, 3), np.uint8), a.conf); segment(np.zeros((H, W, 3), np.uint8))   # DirectML warm-up, not timed
    csv = open(os.path.join(a.out, 'timing.csv'), 'w', buffering=1); csv.write('i,t,late,grab,det,seg,adapt,total,battle,entities\n')
    t0 = time.perf_counter(); nxt = t0; last_print = t0; i = 0
    while time.perf_counter() - t0 < dur:
        now = time.perf_counter()
        if now < nxt: time.sleep(nxt - now)
        tick = nxt; nxt += a.period
        t1 = time.perf_counter(); late = (t1 - tick) * 1e3
        if grab is None: f = still.copy()
        else:
            rg = region(hwnd)
            # minimized, or not the foreground window (mss grabs the screen area: a covered game = the user's other windows)
            if rg['width'] < 100 or rg['height'] < 100 or not in_front(): continue
            try: img = grab(rg)
            except ValueError: continue   # dxcam: window partly off the primary screen
            if img is None: continue
            f = shrink(img)
        t2 = time.perf_counter(); det, rgb, seg, ms_det, ms_seg = vision(f, a.conf)
        t4 = time.perf_counter(); o = ad(seg, det, rgb); t5 = time.perf_counter()
        ms = [(t2 - t1) * 1e3, ms_det, ms_seg, (t5 - t4) * 1e3, (t5 - t1) * 1e3]
        bat, ne = int(bool(o.get('battle', False))), int(o['ent'][0, :, 0].sum())
        csv.write('%d,%.3f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%d\n' % (i, tick - t0, late, *ms, bat, ne)); rows.append(ms[-1])
        try: wq.put_nowait((os.path.join(a.out, 'f%06d.png' % i), f))
        except queue.Full: skipped += 1   # disk slower than 10 fps: drop the frame file, never the timing row
        while time.perf_counter() > nxt: nxt += a.period   # missed slots: skip them (late column shows it)
        if t5 - last_print >= 10:
            r = np.array(rows[-100:]); print('%4.0fs frames %d  total p50 %.1f p95 %.1f ms  battle %d  ent %d  png dropped %d'
                                           % (t5 - t0, i + 1, np.median(r), np.percentile(r, 95), bat, ne, skipped), flush=True)
            last_print = t5
        i += 1
    csv.close(); wq.put(None); wt.join()
    if not rows: raise SystemExit('no frames: the "%s" window was never in front' % a.title)
    r = np.array(rows); print('done: %d frames in %.0f s (%.1f / s), total ms p50 %.1f p95 %.1f max %.1f, png dropped %d -> %s'
                              % (len(r), time.perf_counter() - t0, len(r) / (time.perf_counter() - t0), np.median(r),
                                 np.percentile(r, 95), r.max(), skipped, a.out))


if __name__ == '__main__':
    main()
