import argparse
import json
import os
import numpy as np

def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

def cosine_change(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = (Xn[1:] * Xn[:-1]).sum(axis=1)
    change = 1.0 - sim
    change = np.concatenate([[change[0]], change], axis=0)
    return change

def moving_avg(x: np.ndarray, k: int = 5) -> np.ndarray:
    k = max(1, int(k))
    if k == 1:
        return x
    w = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, w, mode="same")

def find_segments(mask: np.ndarray):
    segs = []
    i = 0
    T = len(mask)
    while i < T:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < T and mask[j]:
            j += 1
        segs.append((i, j - 1))
        i = j
    return segs

def merge_close(segs, gap_max_steps: int):
    if not segs:
        return segs
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s - pe - 1 <= gap_max_steps:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="outputs/segments.json")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--threshold_pctl", type=float, default=75.0)
    ap.add_argument("--min_len_sec", type=float, default=2.0)
    ap.add_argument("--merge_gap_sec", type=float, default=1.0)
    args = ap.parse_args()

    d = np.load(args.npz)
    X_vis = d["X_vis"]
    X_aud = d["X_aud"]
    t0 = d["t_start"]
    t1 = d["t_end"]
    hop = float(t0[1] - t0[0]) if len(t0) > 1 else 1.0

    n = X_aud.shape[1] // 2
    aud = normalize(X_aud[:, :n].mean(axis=1))
    vis = normalize(cosine_change(X_vis))
    sal = normalize(0.6 * aud + 0.4 * vis)
    sal = moving_avg(sal, args.smooth)

    thr = np.percentile(sal, args.threshold_pctl)
    mask = sal >= thr

    segs = find_segments(mask)
    segs = merge_close(segs, gap_max_steps=int(args.merge_gap_sec / hop))

    out = []
    for s, e in segs:
        start = float(t0[s])
        end = float(t1[e])
        if end - start < args.min_len_sec:
            continue
        seg_score = float(sal[s:e+1].mean())
        out.append({
            "start": start,
            "end": end,
            "saliency": seg_score
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved segments: {args.out}")
    print(f"Count: {len(out)}")

if __name__ == "__main__":
    main()
