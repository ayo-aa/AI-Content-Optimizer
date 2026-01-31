import argparse
import glob
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def normalize(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def cosine_change(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = (Xn[1:] * Xn[:-1]).sum(axis=1)
    change = 1.0 - sim
    return np.concatenate([[change[0]], change], axis=0)


class WeakMomentDataset(Dataset):
    def __init__(self, npz_paths, clip_len=64):
        self.paths = npz_paths
        self.clip_len = clip_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        d = np.load(self.paths[idx])
        X_vis = d["X_vis"]
        X_aud = d["X_aud"]

        n = X_aud.shape[1] // 2
        aud = normalize(X_aud[:, :n].mean(axis=1))
        vis = normalize(cosine_change(X_vis))
        y = normalize(0.6 * aud + 0.4 * vis)  # [T]

        X = np.concatenate([X_vis, X_aud], axis=1).astype(np.float32)  # [T, D]
        T = X.shape[0]

        # sample a random clip
        if T <= self.clip_len:
            pad = self.clip_len - T
            Xp = np.pad(X, ((0, pad), (0, 0)), mode="edge")
            yp = np.pad(y, (0, pad), mode="edge")
        else:
            s = random.randint(0, T - self.clip_len)
            Xp = X[s:s + self.clip_len]
            yp = y[s:s + self.clip_len]

        return torch.from_numpy(Xp), torch.from_numpy(yp)


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
        # x: [B, T, D]
        x = x.transpose(1, 2)          # [B, D, T]
        y = self.net(x).squeeze(1)     # [B, T]
        return torch.sigmoid(y)        # keep in [0,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="data/features")
    ap.add_argument("--out_dir", default="outputs/runs/moment_v1")
    ap.add_argument("--clip_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = sorted(glob.glob(os.path.join(args.features_dir, "*.npz")))
    if len(paths) < 3:
        raise RuntimeError(f"Need at least 3 .npz feature files in {args.features_dir}")

    random.shuffle(paths)
    split = int(0.8 * len(paths))
    train_paths = paths[:split]
    val_paths = paths[split:]

    train_ds = WeakMomentDataset(train_paths, clip_len=args.clip_len)
    val_ds = WeakMomentDataset(val_paths, clip_len=args.clip_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # infer input dim
    d0 = np.load(train_paths[0])
    d_in = int(d0["X_vis"].shape[1] + d0["X_aud"].shape[1])

    model = TemporalCNN(d_in=d_in).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    os.makedirs(args.out_dir, exist_ok=True)

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for X, y in train_dl:
            X = X.to(args.device)
            y = y.to(args.device)

            pred = model(X)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for X, y in val_dl:
                X = X.to(args.device)
                y = y.to(args.device)
                pred = model(X)
                loss = loss_fn(pred, y)
                va_losses.append(loss.item())

        tr = float(np.mean(tr_losses))
        va = float(np.mean(va_losses))
        print(f"epoch {epoch:02d}  train_mse={tr:.4f}  val_mse={va:.4f}")

        if va < best_val:
            best_val = va
            ckpt = {
                "model_state": model.state_dict(),
                "d_in": d_in,
                "clip_len": args.clip_len,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

    print(f"Saved best checkpoint to {os.path.join(args.out_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
