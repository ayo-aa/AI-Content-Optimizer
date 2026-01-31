import glob
import os
import numpy as np

def normalize(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

def cosine_change(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = (Xn[1:] * Xn[:-1]).sum(axis=1)
    change = 1.0 - sim
    return np.concatenate([[change[0]], change], axis=0)

def build_examples(features_dir="data/features"):
    paths = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
    examples = []
    for p in paths:
        d = np.load(p)
        X_vis = d["X_vis"]
        X_aud = d["X_aud"]

        n = X_aud.shape[1] // 2
        aud = normalize(X_aud[:, :n].mean(axis=1))
        vis = normalize(cosine_change(X_vis))

        y = normalize(0.6 * aud + 0.4 * vis)  # weak saliency target in [0,1]

        X = np.concatenate([X_vis, X_aud], axis=1).astype(np.float32)  # [T, D]
        examples.append((X, y.astype(np.float32)))
    return examples
