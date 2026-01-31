import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    def __init__(self, d_in, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)          # [B, D, T]
        y = self.net(x).squeeze(1)     # [B, T]
        return torch.sigmoid(y)


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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/pred_segments.json")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--threshold_pctl", type=float, default=75.0)
    ap.add_argument("--min_len_sec", type=float, default=2.0)
    ap.add_argument("--merge_gap_sec", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    d = np.load(args.npz)
    X = np.concatenate([d["X_vis"], d["X_aud"]], axis=1).astype(np.float32)
    t0 = d["t_start"]
    t1 = d["t_end"]
    hop = float(t0[1] - t0[0]) if len(t0) > 1 else 1.0

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = TemporalCNN(d_in=int(ckpt["d_in"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(args.device)

    with torch.no_grad():
        Xt = torch.from_numpy(X).unsqueeze(0).to(args.device)  # [1, T, D]
        y = model(Xt).squeeze(0).cpu().numpy()                  # [T]

    y_s = moving_avg(y, args.smooth)
    thr = np.percentile(y_s, args.threshold_pctl)
    mask = y_s >= thr

    segs = find_segments(mask)
    segs = merge_close(segs, gap_max_steps=int(args.merge_gap_sec / hop))

    out = []
    for s, e in segs:
        start = float(t0[s])
        end = float(t1[e])
        if end - start < args.min_len_sec:
            continue
        out.append({
            "start": start,
            "end": end,
            "saliency": float(y_s[s:e+1].mean())
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"segments": out, "saliency": y_s.tolist()}, f, indent=2)

    print(f"Saved: {args.out}")
    print(f"Segments: {len(out)}")


if __name__ == "__main__":
    main()
