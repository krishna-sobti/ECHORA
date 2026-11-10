#!/usr/bin/env python3
"""
Live predictor + dashboard server.

LIVE (needs sudo, same as recording):
  sudo $(which python3) scripts/live_predict.py --model models/booth1.joblib
  then open  http://localhost:8000

REPLAY an old recording as if it were live (no hardware, no sudo):
  python3 scripts/live_predict.py --model models/booth1.joblib --replay "data/sessions/booth1/*/*.jsonl" --loop
  (the dashboard shows a big REPLAY badge so nobody mistakes it for live sensing)

The current answer is also written to live_state.json every step.
"""
import argparse, glob, importlib.util, json, os, re, socket, sys, threading, time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_posture as tp

BIN_NS = 250_000_000          # dashboard resolution: 4 columns per second


# ====================================================================== model engine
class Engine:
    def __init__(self, bundle, vote, threshold=0.5):
        self.threshold = threshold
        tp.FEATURE_SET = bundle.get("feature_set", "full")
        tp.NODES[:] = bundle.get("nodes", tp.NODES)
        self.pres, self.post = bundle["presence_model"], bundle["posture_model"]
        if self.pres is None or self.post is None:
            sys.exit("Model file has no presence/posture model. Retrain with the latest train_posture.py")
        self.win_ns = int(bundle["window_s"] * 1e9)
        self.stride_ns = int(bundle["stride_s"] * 1e9)
        self.freq = bundle.get("freq_mhz")
        self.freq_bad = self.freq_ok = 0
        self.pres.n_jobs = 1   # one core is much faster than spawning threads for a single window
        self.post.n_jobs = 1
        self.pcls = list(self.pres.classes_)
        self.qcls = list(self.post.classes_)
        self.buf = {n: deque() for n in tp.NODES}
        self.recent = deque(maxlen=vote)
        self.last_counts = {}
        self.last_detail = {"p_sit": None, "p_stand": None}

    def add(self, node, ts, amps, rssi, freq=None):
        if self.freq and freq is not None:
            if freq != self.freq:
                self.freq_bad += 1
                return
            self.freq_ok += 1
        if node in self.buf and amps is not None and len(amps) == tp.N_SUB:
            self.buf[node].append((ts, amps, rssi))

    def step(self, now_ns):
        if self.freq and self.freq_bad > self.freq_ok and self.freq_bad > 20:
            self.freq_bad = self.freq_ok = 0
            self.last_counts = {}
            return "channel", 0.0, "channel"
        self.freq_bad = self.freq_ok = 0
        t0 = now_ns - self.win_ns
        rec, counts, ok = {}, {}, True
        for nid in tp.NODES:
            d = self.buf[nid]
            while d and d[0][0] < now_ns - 3 * self.win_ns:
                d.popleft()
            items = [x for x in d if t0 <= x[0] < now_ns]
            counts[nid] = len(items)
            if len(items) < tp.MIN_FRAMES_PER_NODE:
                ok = False
                continue
            rec[nid] = (np.array([x[0] for x in items], dtype=np.int64),
                        np.array([x[1] for x in items], dtype=np.float32),
                        np.array([x[2] for x in items], dtype=np.float32))
        self.last_counts = counts
        if not ok:
            return None
        f = tp.window_features(rec, t0, now_ns)
        if f is None:
            return None
        X = f.reshape(1, -1)
        pp = self.pres.predict_proba(X)[0]
        p_person = float(pp[self.pcls.index("person")])
        pq = self.post.predict_proba(X)[0]
        self.last_detail = {"p_sit": float(pq[self.qcls.index("sit")]) if "sit" in self.qcls else None,
                            "p_stand": float(pq[self.qcls.index("stand")]) if "stand" in self.qcls else None}
        raw = "empty" if p_person < self.threshold else str(self.qcls[int(np.argmax(pq))])
        self.recent.append(raw)
        voted = Counter(self.recent).most_common(1)[0][0]
        return raw, p_person, voted


# ====================================================================== dashboard feed
class Feed:
    """Keeps the last ~4 minutes of CSI columns / activity / states for the web dashboard."""

    def __init__(self, mode, model_freq):
        self.lock = threading.Lock()
        self.mode, self.freq = mode, model_freq
        self.bins, self.states = deque(maxlen=960), deque(maxlen=700)
        self.cur, self.acc, self.ema = None, {}, {}
        self.seen_freq, self.last_state, self.warn = None, None, None
        self.epoch = int(time.time() * 1000)

    def add(self, node, ts, amps, rssi, freq=None):
        if node not in (1, 2, 3, 4) or amps is None or len(amps) != 128:
            return
        a = np.asarray(amps, dtype=np.float32)
        act = a > 0
        if not act.any():
            return
        norm = a / (float(a[act].mean()) + 1e-6)
        idx = int(ts // BIN_NS)
        with self.lock:
            if self.cur is None:
                self.cur = idx
            if idx > self.cur:
                self._flush()
                self.cur = idx
            acc = self.acc.get(node)
            if acc is None:
                self.acc[node] = [norm.copy(), 1, float(rssi)]
            else:
                acc[0] += norm
                acc[1] += 1
                acc[2] += float(rssi)
            if freq:
                self.seen_freq = freq

    def _flush(self):
        rec = {"i": self.cur, "n": {}}
        for n, acc in self.acc.items():
            col = acc[0] / acc[1]
            ema = self.ema.get(n)
            self.ema[n] = col.copy() if ema is None else 0.97 * ema + 0.03 * col
            m = col > 0.05
            activity = float(np.abs(col - self.ema[n])[m].mean()) if m.any() else 0.0
            rec["n"][n] = {"c": np.round(col, 2).tolist(), "a": round(activity, 4),
                           "r": round(acc[2] / acc[1], 1), "f": round(acc[1] / 0.25, 1)}
        self.acc = {}
        self.bins.append(rec)

    def add_state(self, ts_ns, raw, p_person, voted, detail):
        with self.lock:
            s = {"i": int(ts_ns // BIN_NS), "raw": raw, "p": round(p_person, 3), "v": voted,
                 "ps": None if detail.get("p_sit") is None else round(detail["p_sit"], 3),
                 "pt": None if detail.get("p_stand") is None else round(detail["p_stand"], 3)}
            self.states.append(s)
            self.last_state = s
            self.warn = "channel" if voted == "channel" else None

    def api(self, since):
        with self.lock:
            bins = [b for b in self.bins if b["i"] > since]
            states = [s for s in self.states if s["i"] > since]
            if since == 0:
                bins, states = bins[-240:], states[-130:]
            seq = max([since] + [b["i"] for b in bins] + [s["i"] for s in states])
            return {"epoch": self.epoch, "mode": self.mode, "model_freq": self.freq, "seen_freq": self.seen_freq,
                    "seq": seq, "bins": bins, "states": states, "latest": self.last_state,
                    "warn": self.warn, "server_time": time.time()}


def start_ui(feed, port, state_file):
    page = os.path.join(HERE, "dashboard.html")

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                try:
                    with open(page, "rb") as f:
                        self._send(200, f.read(), "text/html; charset=utf-8")
                except OSError:
                    self._send(404, b"dashboard.html must be in the same folder as live_predict.py", "text/plain")
            elif u.path == "/combined":
                try:
                    with open(os.path.join(HERE, "combined.html"), "rb") as f:
                        self._send(200, f.read(), "text/html; charset=utf-8")
                except OSError:
                    self._send(404, b"combined.html must be in the same folder as live_predict.py", "text/plain")
            elif u.path == "/api":
                try:
                    since = int(parse_qs(u.query).get("since", ["0"])[0])
                except ValueError:
                    since = 0
                self._send(200, json.dumps(feed.api(since)).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *a):
            pass

    try:
        srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    except OSError as e:
        print(f"Could not start dashboard on port {port}: {e}")
        return
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Dashboard:  http://localhost:{port}        (RuView + ECHORA together:  http://localhost:{port}/combined)")


# ====================================================================== helpers
def write_state(path, raw, p_person, voted):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"label": voted, "raw": raw, "p_person": round(p_person, 3), "time": time.time()}, f)
    os.replace(tmp, path)


def speak(text):
    try:
        import subprocess
        subprocess.Popen(["spd-say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def file_order(p):
    m = re.search(r"csi-(\d+)", os.path.basename(p))
    return (int(m.group(1)) if m else 0, p)


# ====================================================================== replay
def run_replay(engine, pattern, state_file, feed=None, loop=False, speed=1.0):
    files = sorted(glob.glob(pattern), key=file_order)
    if not files:
        sys.exit(f"No file matches {pattern}")
    pace = feed is not None
    counts, next_step, prev_end = Counter(), None, None
    wall0, virt0 = time.time(), None
    try:
        while True:
            for path in files:
                f0, base_f, ts = None, None, None
                with open(path) as fh:
                    for line in fh:
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if r.get("type") != "raw_csi":
                            continue
                        if f0 is None:
                            f0 = r["ts_ns"]
                            base_f = f0 if prev_end is None else prev_end + 1_000_000_000
                        ts = r["ts_ns"] - f0 + base_f
                        if virt0 is None:
                            virt0 = ts
                        if next_step is None:
                            next_step = ts + engine.win_ns
                        node, amps, rssi, fq = r.get("node_id"), r.get("amplitudes"), r.get("rssi", 0), r.get("freq_mhz")
                        engine.add(node, ts, amps, rssi, fq)
                        if feed:
                            feed.add(node, ts, amps, rssi, fq)
                        while ts >= next_step:
                            out = engine.step(next_step)
                            if out:
                                counts[out[2]] += 1
                                write_state(state_file, out[0], out[1], out[2])
                                if feed:
                                    feed.add_state(next_step, out[0], out[1], out[2], engine.last_detail)
                            next_step += engine.stride_ns
                        if pace:
                            ahead = (ts - virt0) / 1e9 / speed - (time.time() - wall0)
                            if ahead > 0.01:
                                time.sleep(min(ahead, 0.25))
                if ts is not None:
                    prev_end = ts
            if not loop:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    total = sum(counts.values())
    if total:
        print(f"Replay finished: {total} steps")
        for k, v in counts.most_common():
            print(f"  {k:6s} {v:4d}  ({v / total * 100:.0f}%)")


# ====================================================================== live
def run_live(engine, interface, port, state_file, log_path, do_speak, feed):
    spec = importlib.util.spec_from_file_location("capture", os.path.join(HERE, "record-csi-udp-raw.py"))
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003))
    sock.bind((interface, 0))
    sock.settimeout(0.1)
    try:
        import subprocess
        out = subprocess.run(["pgrep", "-f", "live_predict.py"], capture_output=True, text=True).stdout.split()
        others = [x for x in out if int(x) not in (os.getpid(), os.getppid())]
        if others:
            print(f"WARNING: another live_predict.py seems to be running (pid {', '.join(others)}).")
            print("         Stop it first:  sudo pkill -f live_predict.py   then start again.\n")
    except Exception:
        pass
    log = open(log_path, "a", buffering=1)
    log.write("time,status,raw,p_person,voted,n1,n2,n3,n4\n")
    print(f"Listening on {interface}, UDP port {port}. Log: {log_path}. Press Ctrl+C to stop.\n")
    next_step = time.time_ns() + engine.win_ns
    last_spoken, missing = None, 0
    try:
        while True:
            try:
                packet, _ = sock.recvfrom(65535)
                dec = cap.udp_payload(packet)
                if dec and dec[3] == port:
                    frame = cap.parse_csi(dec[4], dec[0])
                    if frame:
                        a = (frame["node_id"], frame["ts_ns"], frame["amplitudes"], frame["rssi"], frame["freq_mhz"])
                        engine.add(*a)
                        feed.add(*a)
            except socket.timeout:
                pass
            now = time.time_ns()
            if now >= next_step:
                next_step = now + engine.stride_ns
                out = engine.step(now)
                c = engine.last_counts
                cnt = ",".join(str(c.get(n, 0)) for n in tp.NODES)
                stamp = time.strftime("%H:%M:%S") + f".{(now // 1_000_000) % 1000:03d}"
                if out is None:
                    missing += 1
                    log.write(f"{stamp},NO_WINDOW,,,,{cnt}\n")
                    sys.stdout.write(f"\rwaiting for data from all 4 nodes (frames in last 1s: {cnt}) ...      ")
                    if do_speak and missing == 4:
                        speak("no data")
                else:
                    missing = 0
                    raw, p, voted = out
                    if raw == "channel":
                        sys.stdout.write(f"\r  !!! WRONG RADIO CHANNEL: model needs {engine.freq} MHz, router changed it !!!      ")
                    log.write(f"{stamp},OK,{raw},{p:.3f},{voted},{cnt}\n")
                    write_state(state_file, raw, p, voted)
                    feed.add_state(now, raw, p, voted, engine.last_detail)
                    sys.stdout.write(f"\r  >>> {voted.upper():6s} <<<   (raw: {raw:6s}  P(person)={p:.2f})     ")
                    if do_speak and voted != last_spoken:
                        speak(voted)
                        last_spoken = voted
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--interface", default="wlan0")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--vote", type=int, default=7, help="majority over last N steps (each step = 0.5 s)")
    ap.add_argument("--replay", default=None, help="path/glob of .jsonl recordings to replay")
    ap.add_argument("--loop", action="store_true", help="with --replay: repeat forever (dashboard demo without hardware)")
    ap.add_argument("--speed", type=float, default=1.0, help="with --replay: playback speed")
    ap.add_argument("--ui-port", type=int, default=8000, help="dashboard port (0 = no dashboard)")
    ap.add_argument("--state-file", default="live_state.json")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="say EMPTY when P(person) is below this (raise it, e.g. 0.7, if empty is never shown)")
    ap.add_argument("--log", default="live_log.csv", help="every step is appended here")
    ap.add_argument("--speak", action="store_true", help="say the label out loud when it changes")
    args = ap.parse_args()
    import joblib
    engine = Engine(joblib.load(args.model), args.vote, args.threshold)
    feed = Feed("replay" if args.replay else "live", engine.freq)
    if args.ui_port:
        start_ui(feed, args.ui_port, args.state_file)
    if args.replay:
        run_replay(engine, args.replay, args.state_file, feed if args.ui_port else None, args.loop, args.speed)
    else:
        run_live(engine, args.interface, args.port, args.state_file, args.log, args.speak, feed)


if __name__ == "__main__":
    main()
