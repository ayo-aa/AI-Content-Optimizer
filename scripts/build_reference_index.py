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
        x = x.transpose(1, 2)
        y = self.net(x).squeeze(1)
        return torch.sigmoid(y)


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TemporalCNN(d_in=int(ckpt["d_in"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model


def video_embedding(npz_path: str, model, device: str, topk: int = 10):
    d = np.load(npz_path)
    X_vis = d["X_vis"]
    X_aud = d["X_aud"]
    X = np.concatenate([X_vis, X_aud], axis=1).astype(np.float32)  # [T, D]

    with torch.no_grad():
        Xt = torch.from_numpy(X).unsqueeze(0).to(device)
        y = model(Xt).squeeze(0).cpu().numpy()  # [T]

    k = min(topk, len(y))
    idx = np.argsort(-y)[:k]
    emb = X[idx].mean(axis=0)  # [D]
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--features_dir", default="data/features")
    ap.add_argument("--good_list", required=True)
    ap.add_argument("--out_dir", default="outputs/index")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.good_list, "r") as f:
        ids = [line.strip() for line in f if line.strip()]

    model = load_model(args.ckpt, args.device)

    embeds = []
    kept_ids = []
    for vid in ids:
        npz_path = os.path.join(args.features_dir, f"{vid}.npz")
        if not os.path.exists(npz_path):
            print(f"skip missing: {npz_path}")
            continue
        emb = video_embedding(npz_path, model, args.device, topk=args.topk)
        embeds.append(emb)
        kept_ids.append(vid)

    E = np.stack(embeds, axis=0)
    np.save(os.path.join(args.out_dir, "ref_embeds.npy"), E)
    with open(os.path.join(args.out_dir, "ref_ids.json"), "w") as f:
        json.dump(kept_ids, f, indent=2)

    print(f"Saved {len(kept_ids)} reference embeddings to {args.out_dir}")


if __name__ == "__main__":
    main()
