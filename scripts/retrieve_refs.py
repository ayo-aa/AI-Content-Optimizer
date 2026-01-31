import argparse
import json
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


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TemporalCNN(d_in=int(ckpt["d_in"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model


def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def video_embedding(npz_path: str, model, device: str, topk: int = 10):
    d = np.load(npz_path)
    X = np.concatenate([d["X_vis"], d["X_aud"]], axis=1).astype(np.float32)  # [T, D]

    with torch.no_grad():
        y = model(torch.from_numpy(X).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

    k = min(topk, len(y))
    idx = np.argsort(-y)[:k]
    emb = X[idx].mean(axis=0).astype(np.float32)
    return l2_normalize(emb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref_embeds", default="outputs/index/ref_embeds.npy")
    ap.add_argument("--ref_ids", default="outputs/index/ref_ids.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    refE = np.load(args.ref_embeds).astype(np.float32)
    refE = refE / (np.linalg.norm(refE, axis=1, keepdims=True) + 1e-9)

    with open(args.ref_ids, "r") as f:
        ref_ids = json.load(f)

    model = load_model(args.ckpt, args.device)
    q = video_embedding(args.query_npz, model, args.device, topk=args.topk)  # [D]

    scores = refE @ q
    top = np.argsort(-scores)[: args.k]

    for rank, i in enumerate(top, start=1):
        print(f"{rank}. {ref_ids[int(i)]}  score={float(scores[int(i)]):.4f}")


if __name__ == "__main__":
    main()
