import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

def cosine_change(X: np.ndarray) -> np.ndarray:
    # X: [T, D]
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="outputs/saliency.png")
    ap.add_argument("--smooth", type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.npz)
    X_vis = d["X_vis"]
    X_aud = d["X_aud"]
    t0 = d["t_start"]
    t1 = d["t_end"]
    tmid = (t0 + t1) / 2.0

    # audio proxy: mean of mel log-energy stats (first half are means)
    n = X_aud.shape[1] // 2
    aud = X_aud[:, :n].mean(axis=1)
    aud = normalize(aud)

    # visual proxy: embedding change
    vis = cosine_change(X_vis)
    vis = normalize(vis)

    # combined weak saliency
    sal = normalize(0.6 * aud + 0.4 * vis)
    sal_s = moving_avg(sal, args.smooth)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure()
    plt.plot(tmid, aud, label="audio proxy")
    plt.plot(tmid, vis, label="visual change")
    plt.plot(tmid, sal_s, label="combined (smoothed)")
    plt.xlabel("time (sec)")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot: {args.out}")

if __name__ == "__main__":
    main()
