#!/usr/bin/env python3
"""
ECHORA - zone + posture classifier from ESP32 CSI recordings.

Folder names decide the labels:
  empty_*                              -> zone = empty
  sit_n1, sit_n1_2, stand_n3_2, ...    -> posture = sit/stand, zone = n1/n3/...
Folders without sit/stand/empty at the start (e.g. test_01) are ignored.

Two models are trained and saved together in one file:
  1. zone model    : empty / n1 / n2 / n3 / n4   (where is the person)
  2. posture model : sit / stand                 (trained only on non-empty data)

Usage:
  python3 scripts/train_posture.py --data data/sessions/session1 \
      --exclude center test --out models/zone_posture_session1.joblib
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

LABELS = ["empty", "sit", "stand"]
NODES = [1, 2, 3, 4]
N_SUB = 128
MIN_FRAMES_PER_NODE = 5  # a window needs at least this many frames from every node


# ---------------------------------------------------------------- labels
def labels_from_folder(name):
    """Return (posture, zone), or (None, None) if the folder has no label."""
    parts = name.split("_")
    posture = parts[0] if parts[0] in LABELS else None
    if posture is None:
        return None, None
    if posture == "empty":
        return "empty", "empty"
    zone = next((p for p in parts[1:] if not p.isdigit()), None)
    return posture, zone


# ---------------------------------------------------------------- loading
NO_EVAL = False         # skip the slow leave-one-out test, just fit the final model
TRIM_START_S = 0.0      # ignore the first N seconds of every recording (sitting down / walking in)
TRIM_END_S = 0.0        # ignore the last N seconds
FREQ_FILTER = None      # e.g. 2437: keep only frames recorded on that radio channel
LAST_FREQS = {}         # channel counts of the last recording loaded


def load_recording(path):
    """Return ({node_id: (timestamps_ns, amplitudes[N,128], rssi[N])}, rejected_counts)."""
    global LAST_FREQS
    frames = defaultdict(list)
    bad = defaultdict(int)
    freqs = defaultdict(int)
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad["not_json"] += 1
                continue
            if r.get("type") != "raw_csi":
                bad["type_" + str(r.get("type"))] += 1
                continue
            nid = r.get("node_id")
            amps = r.get("amplitudes")
            fq = r.get("freq_mhz")
            freqs[fq] += 1
            if FREQ_FILTER and fq != FREQ_FILTER:
                bad["other_channel"] += 1
                continue
            if nid not in NODES:
                bad[f"node_{nid}"] += 1
                continue
            if amps is None or len(amps) != N_SUB:
                bad[f"len_{0 if amps is None else len(amps)}"] += 1
                continue
            frames[nid].append((r["ts_ns"], amps, r.get("rssi", 0)))
    LAST_FREQS = dict(freqs)
    out = {}
    for nid, lst in frames.items():
        lst.sort(key=lambda x: x[0])
        t = np.array([x[0] for x in lst], dtype=np.int64)
        a = np.array([x[1] for x in lst], dtype=np.float32)
        rs = np.array([x[2] for x in lst], dtype=np.float32)
        out[nid] = (t, a, rs)
    return out, dict(bad)


# ---------------------------------------------------------------- features
FEATURE_SET = "full"   # set from --features


def node_features_full(amps, rssi):
    """258 numbers per node: shape of the signal + how much each subcarrier moves."""
    level = amps.mean(axis=1, keepdims=True) + 1e-6   # normalise each frame by its own level
    norm = amps / level
    return np.concatenate([
        norm.mean(axis=0),            # 128: multipath shape
        norm.std(axis=0),             # 128: how much each subcarrier moves
        [rssi.mean(), rssi.std()],    # 2:   signal strength + its variation
    ])


def node_features_small(amps, rssi, motion_only):
    """A few summary numbers per node. Less detail, but harder to over-fit."""
    active = amps.mean(axis=0) > 0                     # drop always-zero guard subcarriers
    a = amps[:, active] if active.any() else amps
    level = a.mean(axis=1)
    norm = a / (level[:, None] + 1e-6)
    sc_std = norm.std(axis=0)
    jump = np.abs(np.diff(norm, axis=0)).mean() if len(norm) > 1 else 0.0
    f = [
        level.std() / (level.mean() + 1e-6),  # how much the overall level wobbles
        sc_std.mean(),                        # average subcarrier movement
        np.median(sc_std),
        np.percentile(sc_std, 90),            # strongest moving subcarriers
        jump,                                 # frame-to-frame change
        rssi.std(),
    ]
    if not motion_only:
        f += [rssi.mean(), level.mean()]      # static strength (can drift over time)
    return np.array(f, dtype=np.float64)


def node_features(amps, rssi):
    if FEATURE_SET == "full":
        return node_features_full(amps, rssi)
    return node_features_small(amps, rssi, motion_only=(FEATURE_SET == "motion"))


def window_features(rec, t0, t1):
    """Feature vector for window t0..t1, or None if any node has too few frames."""
    parts = []
    for nid in NODES:
        if nid not in rec:
            return None
        t, a, rs = rec[nid]
        i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
        if i1 - i0 < MIN_FRAMES_PER_NODE:
            return None
        parts.append(node_features(a[i0:i1], rs[i0:i1]))
    return np.concatenate(parts)


def recording_windows(rec, window_s, stride_s):
    if any(n not in rec for n in NODES):
        return []
    start = max(rec[n][0][0] for n in NODES) + int(TRIM_START_S * 1e9)
    end = min(rec[n][0][-1] for n in NODES) - int(TRIM_END_S * 1e9)
    w, s = int(window_s * 1e9), int(stride_s * 1e9)
    feats = []
    t0 = start
    while t0 + w <= end:
        f = window_features(rec, t0, t0 + w)
        if f is not None:
            feats.append(f)
        t0 += s
    return feats


# ---------------------------------------------------------------- training
def make_model():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0)


def smooth_vote(pred, k=7):
    """Majority of the last k predictions (what a live demo would show)."""
    out = []
    for i in range(len(pred)):
        chunk = list(pred[max(0, i - k + 1): i + 1])
        out.append(max(set(chunk), key=chunk.count))
    return np.array(out)


def evaluate(title, X, y, groups):
    """Leave-one-recording-out test, then train the final model on all data."""
    from sklearn.metrics import confusion_matrix
    classes = sorted(set(y))
    print(f"\n=================== {title} ===================")
    for c in classes:
        print(f"  {c:6s}: {np.sum(y == c):5d} windows from {len(set(groups[y == c]))} recordings")
    if len(classes) < 2:
        print("  Need at least 2 classes, skipping this model.")
        return None, classes, None

    if NO_EVAL:
        return make_model().fit(X, y), classes, None
    print("\n  Leave-one-recording-out test:")
    all_t, all_p, all_ps = [], [], []
    overall = None
    for g in sorted(set(groups)):
        test = groups == g
        if len(set(y[~test])) < len(classes):
            print(f"    {g:26s} skipped (no other recording of class '{y[test][0]}' to learn from)")
            continue
        m = make_model().fit(X[~test], y[~test])
        pred = m.predict(X[test])
        acc = np.mean(pred == y[test])
        guess = max(set(pred), key=list(pred).count)
        print(f"    {g:26s} true={y[test][0]:6s} accuracy={acc*100:5.1f}%  mostly predicted={guess}")
        all_t.extend(y[test])
        all_p.extend(pred)
        all_ps.extend(smooth_vote(pred))

    if all_t:
        all_t, all_p = np.array(all_t), np.array(all_p)
        overall = np.mean(all_t == all_p) * 100
        print(f"\n  Overall accuracy: {overall:.1f}%")
        sm = np.mean(all_t == np.array(all_ps)) * 100
        print(f"  Overall accuracy after voting over last 7 windows (~4 s): {sm:.1f}%")
        cm = confusion_matrix(all_t, all_p, labels=classes)
        print("\n  Confusion matrix (rows = true, columns = predicted):")
        print("          " + "".join(f"{c:>8s}" for c in classes))
        for c, row in zip(classes, cm):
            print(f"    {c:6s}" + "".join(f"{v:8d}" for v in row))

    final = make_model().fit(X, y)
    per_node = final.feature_importances_.reshape(len(NODES), -1).sum(axis=1)
    print("\n  How much this model relies on each node:")
    for n, v in zip(NODES, per_node):
        print(f"    node {n}: {v*100:5.1f}%")
    return final, classes, overall


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/sessions/session1"],
                    help="one or more session folders (all are used for training)")
    ap.add_argument("--out", default="models/zone_posture_session1.joblib")
    ap.add_argument("--window", type=float, default=1.0, help="window length in seconds")
    ap.add_argument("--stride", type=float, default=0.5, help="step between windows in seconds")
    ap.add_argument("--features", choices=["full", "compact", "motion"], default="full",
                    help="full = 1032 detailed numbers, compact = 32 summary numbers, "
                         "motion = 24 numbers that only describe movement")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip the slow self-test and only train the final model (fast)")
    ap.add_argument("--trim-start", type=float, default=4.0,
                    help="ignore the first N seconds of every recording (default 4)")
    ap.add_argument("--trim-end", type=float, default=1.0,
                    help="ignore the last N seconds of every recording (default 1)")
    ap.add_argument("--freq", type=int, default=None,
                    help="use only frames from this radio channel in MHz (2437 = channel 6, 2462 = channel 11)")
    ap.add_argument("--nodes", nargs="+", type=int, default=None,
                    help="use only these nodes, e.g. --nodes 1 2 3 (ignore node 4)")
    ap.add_argument("--skip-zone", action="store_true",
                    help="do not train a zone model (use when everything is recorded at one spot)")
    ap.add_argument("--include", nargs="*", default=None,
                    help="only use folders whose name contains one of these words")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="skip folders whose name contains one of these words, e.g. --exclude center test")
    args = ap.parse_args()
    global FEATURE_SET, FREQ_FILTER, TRIM_START_S, TRIM_END_S, NO_EVAL
    NO_EVAL = args.no_eval
    TRIM_START_S, TRIM_END_S = args.trim_start, args.trim_end
    print(f"Ignoring first {TRIM_START_S:g} s and last {TRIM_END_S:g} s of every recording")
    FEATURE_SET = args.features
    FREQ_FILTER = args.freq
    if args.nodes:
        if not set(args.nodes) <= {1, 2, 3, 4}:
            sys.exit("--nodes must be from 1 2 3 4")
        NODES[:] = sorted(set(args.nodes))
    print(f"Nodes used: {NODES}")
    print(f"Channel filter: {args.freq if args.freq else 'none (all channels mixed)'}")
    print(f"Feature set: {FEATURE_SET}")

    import joblib

    X, y_post, y_zone, groups = [], [], [], []
    print("=== Loading recordings ===")
    folders = []
    for ddir in args.data:
        tag = os.path.basename(os.path.normpath(ddir))
        for folder in sorted(glob.glob(os.path.join(ddir, "*"))):
            folders.append((tag, folder))
    for tag, folder in folders:
        name = os.path.basename(folder)
        if args.include and not any(w in name for w in args.include):
            continue
        if args.exclude and any(w in name for w in args.exclude):
            print(f"  skip  {name} (excluded)")
            continue
        posture, zone = labels_from_folder(name)
        if posture is None or zone is None:
            print(f"  skip  {name} (no label)")
            continue
        files = glob.glob(os.path.join(folder, "*.jsonl"))
        if not files:
            print(f"  skip  {name} (no .jsonl file)")
            continue
        for fpath in files:
            rec, bad = load_recording(fpath)
            missing = [n for n in NODES if n not in rec]
            if missing:
                print(f"  SKIP  {name}: nodes missing {missing}")
                continue
            feats = recording_windows(rec, args.window, args.stride)
            counts = ", ".join(f"n{n}={len(rec[n][0])}" for n in NODES)
            badtxt = " ".join(f"{k}:{v}" for k, v in sorted(bad.items())) or "0"
            fqtxt = ",".join(f"{k}:{v}" for k, v in sorted(LAST_FREQS.items(), key=lambda kv: str(kv[0])))
            print(f"  {zone:5s} {posture:5s} {tag + '/' + name:26s} windows={len(feats):4d}  "
                  f"frames: {counts}  rejected: {badtxt}  channels: {fqtxt}")
            X.extend(feats)
            y_post.extend([posture] * len(feats))
            y_zone.extend([zone] * len(feats))
            groups.extend([tag + "/" + name] * len(feats))

    if not X:
        print("No data loaded. Check --data path.")
        sys.exit(1)
    X = np.array(X)
    y_post, y_zone, groups = np.array(y_post), np.array(y_zone), np.array(groups)
    print(f"\nTotal windows: {len(X)}   features per window: {X.shape[1]}")

    if args.skip_zone:
        zone_model, zone_classes, zone_acc = None, [], None
    else:
        zone_model, zone_classes, zone_acc = evaluate("ZONE MODEL (where)", X, y_zone, groups)
    occ = y_post != "empty"
    post_model, post_classes, post_acc = evaluate(
        "POSTURE MODEL (sit / stand)", X[occ], y_post[occ], groups[occ])

    # Extra test: just "is someone there?" (empty vs any zone)
    pres = np.where(y_zone == "empty", "empty", "person")
    pres_model, pres_classes, pres_acc = evaluate("PRESENCE (empty vs person)", X, pres, groups)

    fmt = lambda v: "n/a" if v is None else f"{v:.1f}%"
    print(f"\nSUMMARY features={FEATURE_SET}  presence={fmt(pres_acc)}  "
          f"zone={fmt(zone_acc)}  posture={fmt(post_acc)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump({
        "zone_model": zone_model,
        "zone_labels": zone_classes,
        "presence_model": pres_model,
        "presence_labels": pres_classes,
        "posture_model": post_model,
        "posture_labels": post_classes,
        "nodes": NODES,
        "n_sub": N_SUB,
        "window_s": args.window,
        "stride_s": args.stride,
        "min_frames_per_node": MIN_FRAMES_PER_NODE,
        "feature_set": FEATURE_SET,
        "freq_mhz": args.freq,
        "trim_start_s": args.trim_start,
        "trim_end_s": args.trim_end,
        "feature_version": 3,
    }, args.out)
    print(f"\nSaved both models to {args.out}")


if __name__ == "__main__":
    main()
