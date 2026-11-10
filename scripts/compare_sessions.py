#!/usr/bin/env python3
"""
Compare recorded signal between groups of recordings (e.g. old vs fresh).
Shows when each group was recorded, radio channel, packet types, and per-node
signal strength / level / wobble.

Usage:
  python3 scripts/compare_sessions.py \
     old="data/sessions/session2/empty_c*" \
     holdout="data/sessions/holdout/empty_c*" \
     fresh="data/sessions/fresh/empty_f1"
"""
import glob, json, os, sys, time
from collections import Counter, defaultdict
import numpy as np

NODES = [1, 2, 3, 4]


def files_for(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        out += sorted(glob.glob(os.path.join(p, "*.jsonl"))) if os.path.isdir(p) else [p]
    return out


def summarize(name, pattern):
    files = files_for(pattern)
    if not files:
        print(f"\n### {name}: no files match {pattern}")
        return
    per = defaultdict(lambda: defaultdict(list))
    freq, ppdu, flags, lens = Counter(), Counter(), Counter(), Counter()
    t_first, t_last = None, None
    for fp in files:
        with open(fp) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") != "raw_csi":
                    continue
                a = r.get("amplitudes") or []
                lens[len(a)] += 1
                ts = r.get("ts_ns")
                if ts:
                    t_first = ts if t_first is None else min(t_first, ts)
                    t_last = ts if t_last is None else max(t_last, ts)
                if len(a) != 128 or r.get("node_id") not in NODES:
                    continue
                n = r["node_id"]
                freq[r.get("freq_mhz")] += 1
                ppdu[r.get("ppdu_type")] += 1
                flags[r.get("flags")] += 1
                arr = np.array(a, dtype=np.float32)
                act = arr[arr > 0]
                per[n]["rssi"].append(r.get("rssi", 0))
                per[n]["noise"].append(r.get("noise_floor", 0))
                per[n]["level"].append(float(act.mean()) if act.size else 0.0)
                per[n]["wobble"].append(float(act.std() / (act.mean() + 1e-6)) if act.size else 0.0)
    fmt = lambda t: time.strftime("%d %b %H:%M", time.localtime(t / 1e9)) if t else "?"
    print(f"\n### {name}: {len(files)} recordings, recorded {fmt(t_first)} to {fmt(t_last)}")
    print(f"    channel(MHz): {dict(freq)}   packet types: {dict(ppdu)}   flags: {dict(flags)}")
    print(f"    packet sizes (subcarriers): {dict(lens)}")
    print(f"    {'node':5s} {'frames':>7s} {'rssi':>6s} {'noise':>6s} {'level':>7s} {'wobble':>7s}")
    for n in NODES:
        d = per[n]
        if not d["rssi"]:
            print(f"    {n:<5d} no frames"); continue
        print(f"    {n:<5d} {len(d['rssi']):7d} {np.median(d['rssi']):6.1f} {np.median(d['noise']):6.1f} "
              f"{np.median(d['level']):7.2f} {np.median(d['wobble']):7.3f}")


def main():
    if len(sys.argv) < 2 or any("=" not in a for a in sys.argv[1:]):
        print(__doc__); sys.exit(1)
    for arg in sys.argv[1:]:
        name, pattern = arg.split("=", 1)
        summarize(name, pattern)


if __name__ == "__main__":
    main()
