#!/usr/bin/env python3
"""
Test a saved model on NEW recordings it never trained on.
Folder names give the true label (empty_*, sit_*, stand_*).

Usage:
  python3 scripts/predict_folder.py --model models/s2_full.joblib --data data/sessions/holdout
"""
import argparse, glob, os, sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_posture as tp


def smooth_vote(pred, k):
    out = []
    for i in range(len(pred)):
        chunk = list(pred[max(0, i - k + 1): i + 1])
        out.append(max(set(chunk), key=chunk.count))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--vote", type=int, default=7)
    ap.add_argument("--trim-start", type=float, default=4.0, help="ignore first N seconds of each recording")
    ap.add_argument("--trim-end", type=float, default=1.0, help="ignore last N seconds of each recording")
    ap.add_argument("--exclude", nargs="*", default=["test"])
    args = ap.parse_args()

    import joblib
    bundle = joblib.load(args.model)
    tp.FEATURE_SET = bundle.get("feature_set", "full")
    tp.FREQ_FILTER = bundle.get("freq_mhz")
    tp.TRIM_START_S, tp.TRIM_END_S = args.trim_start, args.trim_end
    tp.NODES[:] = bundle.get("nodes", tp.NODES)
    pres, post = bundle.get("presence_model"), bundle["posture_model"]
    if pres is None or post is None:
        print("This model file has no presence/posture model. Retrain with the latest train_posture.py")
        sys.exit(1)
    pres_pos = [l for l in bundle["presence_labels"]]
    print(f"Model: {args.model}   features: {tp.FEATURE_SET}   voting: last {args.vote} windows   ignoring first {args.trim_start:g} s / last {args.trim_end:g} s\n")

    def predict(X):
        p = pres.predict(X)
        out = np.array(["empty"] * len(X), dtype=object)
        person = p != "empty"
        if person.any():
            out[person] = post.predict(X[person])
        return list(out)

    labels = ["empty", "sit", "stand"]
    all_t, all_p, all_pv = [], [], []
    for folder in sorted(glob.glob(os.path.join(args.data, "*"))):
        name = os.path.basename(folder)
        if any(w in name for w in args.exclude):
            continue
        truth, _ = tp.labels_from_folder(name)
        if truth is None:
            continue
        for f in glob.glob(os.path.join(folder, "*.jsonl")):
            rec, _ = tp.load_recording(f)
            if any(n not in rec for n in tp.NODES):
                print(f"  {name:16s} SKIPPED (a node is missing)")
                continue
            feats = tp.recording_windows(rec, bundle["window_s"], bundle["stride_s"])
            if not feats:
                print(f"  {name:16s} SKIPPED (no windows)")
                continue
            pred = predict(np.array(feats))
            pv = smooth_vote(pred, args.vote)
            acc = np.mean([p == truth for p in pred]) * 100
            accv = np.mean([p == truth for p in pv]) * 100
            top = Counter(pred).most_common(1)[0][0]
            print(f"  {name:16s} true={truth:6s} windows={len(pred):3d} accuracy={acc:5.1f}%  "
                  f"after voting={accv:5.1f}%  mostly predicted={top}")
            all_t += [truth] * len(pred); all_p += pred; all_pv += pv

    if not all_t:
        print("No recordings found."); return
    all_t, all_p, all_pv = np.array(all_t), np.array(all_p), np.array(all_pv)
    print(f"\nOverall accuracy: {np.mean(all_t == all_p)*100:.1f}%   after voting: {np.mean(all_t == all_pv)*100:.1f}%")
    print("\nConfusion matrix after voting (rows = true, columns = predicted):")
    print("          " + "".join(f"{l:>8s}" for l in labels))
    for t in labels:
        print(f"  {t:6s}  " + "".join(f"{int(np.sum((all_t == t) & (all_pv == l))):8d}" for l in labels))


if __name__ == "__main__":
    main()
