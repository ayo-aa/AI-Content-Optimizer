import argparse
import json
import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


# -----------------------------
# Model (must match training)
# -----------------------------
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
        return torch.sigmoid(y)


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TemporalCNN(d_in=int(ckpt["d_in"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, int(ckpt["d_in"])


# -----------------------------
# Utilities
# -----------------------------
def moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    if k == 1:
        return x
    w = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, w, mode="same")


def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def find_runs(mask: np.ndarray):
    # returns list of (start_idx, end_idx) inclusive where mask is True
    runs = []
    i = 0
    T = len(mask)
    while i < T:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < T and mask[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def merge_close(runs, gap_max_steps: int):
    if not runs:
        return runs
    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if s - pe - 1 <= gap_max_steps:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def npz_paths_for_id(features_dir: str, vid: str) -> str:
    return os.path.join(features_dir, f"{vid}.npz")


# -----------------------------
# Core computations
# -----------------------------
def predict_saliency(npz_path: str, model, device: str):
    d = np.load(npz_path)
    X = np.concatenate([d["X_vis"], d["X_aud"]], axis=1).astype(np.float32)  # [T, D]
    t0 = d["t_start"].astype(np.float32)
    t1 = d["t_end"].astype(np.float32)

    with torch.no_grad():
        y = model(torch.from_numpy(X).unsqueeze(0).to(device)).squeeze(0).cpu().numpy().astype(np.float32)

    return X, y, t0, t1


def video_embedding_from_top_saliency(X: np.ndarray, y: np.ndarray, topk: int) -> np.ndarray:
    k = min(int(topk), len(y))
    idx = np.argsort(-y)[:k]
    emb = X[idx].mean(axis=0).astype(np.float32)
    return l2_normalize(emb)


def compute_structure_metrics(y_s: np.ndarray, t0: np.ndarray, t1: np.ndarray):
    T = len(y_s)
    duration = float(t1[-1]) if T > 0 else 0.0
    if T <= 1:
        return {
            "duration_sec": duration,
            "time_to_first_peak_sec": 0.0,
            "early_mean": float(y_s.mean()) if T else 0.0,
            "tail_mean": float(y_s.mean()) if T else 0.0,
            "longest_low_gap_sec": 0.0,
        }

    # define peak as being above high percentile
    high_thr = float(np.percentile(y_s, 85.0))
    low_thr = float(np.percentile(y_s, 35.0))

    # time to first peak
    peak_idxs = np.where(y_s >= high_thr)[0]
    if len(peak_idxs) == 0:
        first_peak_sec = float(t0[0])
    else:
        first_peak_sec = float(t0[int(peak_idxs[0])])

    # early density: first 20% of windows
    early_T = max(1, int(0.2 * T))
    early_mean = float(y_s[:early_T].mean())

    # tail density: last 20% of windows
    tail_T = max(1, int(0.2 * T))
    tail_mean = float(y_s[-tail_T:].mean())

    # longest low-saliency gap
    low_mask = y_s <= low_thr
    low_runs = find_runs(low_mask)
    if not low_runs:
        longest_gap = 0.0
    else:
        best = max(low_runs, key=lambda se: se[1] - se[0])
        s, e = best
        longest_gap = float(t1[e] - t0[s])

    return {
        "duration_sec": duration,
        "time_to_first_peak_sec": first_peak_sec,
        "early_mean": early_mean,
        "tail_mean": tail_mean,
        "longest_low_gap_sec": longest_gap,
        "high_thr": high_thr,
        "low_thr": low_thr,
    }


def derive_segments(y_s: np.ndarray, t0: np.ndarray, t1: np.ndarray,
                    thr_pctl: float, min_len_sec: float, merge_gap_sec: float):
    thr = float(np.percentile(y_s, thr_pctl))
    mask = y_s >= thr

    hop = float(t0[1] - t0[0]) if len(t0) > 1 else 1.0
    runs = find_runs(mask)
    runs = merge_close(runs, gap_max_steps=int(merge_gap_sec / hop))

    segs = []
    for s, e in runs:
        start = float(t0[s])
        end = float(t1[e])
        if end - start < min_len_sec:
            continue
        segs.append({
            "start": start,
            "end": end,
            "score": float(y_s[s:e + 1].mean()),
        })
    segs.sort(key=lambda x: x["start"])
    return segs


def retrieve_refs_numpy(query_emb: np.ndarray, ref_embeds_path: str, ref_ids_path: str, k: int):
    refE = np.load(ref_embeds_path).astype(np.float32)
    refE = refE / (np.linalg.norm(refE, axis=1, keepdims=True) + 1e-9)
    with open(ref_ids_path, "r") as f:
        ref_ids = json.load(f)

    scores = refE @ query_emb
    top = np.argsort(-scores)[:k]
    out = []
    for i in top:
        out.append({"id": ref_ids[int(i)], "score": float(scores[int(i)])})
    return out


def median_profile(profiles: list[dict]):
    # numeric median over same keys
    keys = [k for k in profiles[0].keys() if isinstance(profiles[0][k], (int, float))]
    med = {}
    for k in keys:
        vals = np.array([p[k] for p in profiles], dtype=np.float32)
        med[k] = float(np.median(vals))
    return med


def build_edit_recommendations(query_metrics: dict, ref_med: dict, segments: list[dict]):
    edits = []

    # 1) Trim long low-saliency gaps if query is worse than ref baseline
    if query_metrics["longest_low_gap_sec"] >= max(2.5, ref_med.get("longest_low_gap_sec", 0.0) + 1.0):
        edits.append({
            "type": "TRIM_LOW_GAP",
            "reason": "long low-saliency gap compared to reference videos",
            "confidence": 0.70
        })

    # 2) Trim tail if tail mean is low compared to refs
    if query_metrics["tail_mean"] <= ref_med.get("tail_mean", 1.0) - 0.08:
        edits.append({
            "type": "TRIM_END",
            "reason": "outro has lower saliency than reference videos",
            "confidence": 0.65
        })

    # 3) Suggest moving a strong segment earlier if first peak is late
    if query_metrics["time_to_first_peak_sec"] >= ref_med.get("time_to_first_peak_sec", 0.0) + 1.25:
        first_good = segments[0] if segments else None
        edits.append({
            "type": "SUGGEST_MOVE_EARLIER",
            "reason": "first high-saliency moment occurs later than reference videos",
            "confidence": 0.60,
            "candidate_segment": first_good
        })

    return edits


def materialize_trim_intervals_from_segments(
    t_start: np.ndarray,
    t_end: np.ndarray,
    keep_segments: list[dict],
    pad_sec: float = 0.25,
    min_trim_sec: float = 0.35
):
    """
    Build trim intervals as the complement of keep_segments.
    keep_segments: [{"start": sec, "end": sec, ...}, ...] (sorted by start).
    """
    if len(t_end) == 0:
        return []

    total_start = float(t_start[0])
    total_end = float(t_end[-1])

    if not keep_segments:
        return [{"start": total_start, "end": total_end, "kind": "trim_all_no_segments"}]

    # Merge and pad keep segments
    keeps = []
    for s in keep_segments:
        a = max(total_start, float(s["start"]) - pad_sec)
        b = min(total_end, float(s["end"]) + pad_sec)
        if b > a:
            keeps.append((a, b))
    keeps.sort()

    merged = []
    for a, b in keeps:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    keeps = [(x[0], x[1]) for x in merged]

    # Complement to get trims
    trims = []
    cur = total_start
    for a, b in keeps:
        if a - cur >= min_trim_sec:
            trims.append({"start": cur, "end": a, "kind": "between_keeps"})
        cur = max(cur, b)
    if total_end - cur >= min_trim_sec:
        trims.append({"start": cur, "end": total_end, "kind": "tail_after_keeps"})

    return trims


def load_meta(meta_path: str) -> dict:
    with open(meta_path, "r") as f:
        return json.load(f)


def non_speech_trims_from_vad(
    duration_sec: float,
    speech_segments: list,
    min_gap_sec: float = 0.45,
    pad_sec: float = 0.10,
    min_trim_sec: float = 0.20
):
    """
    Returns trim intervals that remove non-speech gaps while leaving pad_sec of context
    on each side of a gap.
    speech_segments: list of {"start": s, "end": e} sorted by time.
    """
    if duration_sec <= 0.0:
        return []

    segs = [(float(s["start"]), float(s["end"])) for s in speech_segments]
    segs = sorted(segs, key=lambda x: x[0])

    # clamp
    segs2 = []
    for s, e in segs:
        s = max(0.0, min(s, duration_sec))
        e = max(0.0, min(e, duration_sec))
        if e > s:
            segs2.append((s, e))
    segs = segs2

    if not segs:
        # no speech detected, do nothing here (montage path handles it later)
        return []

    trims = []

    # gap before first speech
    if segs[0][0] >= min_gap_sec:
        a, b = 0.0, segs[0][0]
        ts = a
        te = max(a, b - pad_sec)
        if te - ts >= min_trim_sec:
            trims.append({"start": ts, "end": te, "kind": "vad_gap_intro"})

    # gaps between speech segments
    for (s1, e1), (s2, e2) in zip(segs[:-1], segs[1:]):
        gap = s2 - e1
        if gap < min_gap_sec:
            continue
        ts = e1 + pad_sec
        te = s2 - pad_sec
        if te - ts >= min_trim_sec:
            trims.append({"start": ts, "end": te, "kind": "vad_gap_internal"})

    # gap after last speech
    last_end = segs[-1][1]
    tail_gap = duration_sec - last_end
    if tail_gap >= min_gap_sec:
        ts = last_end + pad_sec
        te = duration_sec
        if te - ts >= min_trim_sec:
            trims.append({"start": ts, "end": te, "kind": "vad_gap_tail"})

    return trims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=None)
    ap.add_argument("--query_id", required=True)
    ap.add_argument("--features_dir", default="data/features")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref_embeds", default="outputs/index/ref_embeds.npy")
    ap.add_argument("--ref_ids", default="outputs/index/ref_ids.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--seg_thr_pctl", type=float, default=75.0)
    ap.add_argument("--seg_min_len", type=float, default=2.0)
    ap.add_argument("--seg_merge_gap", type=float, default=1.0)
    ap.add_argument("--embed_topk", type=int, default=10)
    ap.add_argument("--out", default="outputs/edits.json")
    args = ap.parse_args()

    model, _ = load_model(args.ckpt, args.device)

    query_npz = npz_paths_for_id(args.features_dir, args.query_id)
    Xq, yq, t0q, t1q = predict_saliency(query_npz, model, args.device)
    yq_s = moving_avg(yq, args.smooth)

    query_emb = video_embedding_from_top_saliency(Xq, yq_s, topk=args.embed_topk)

    retrieved = retrieve_refs_numpy(query_emb, args.ref_embeds, args.ref_ids, k=args.k)

    # profiles for refs
    ref_profiles = []
    for r in retrieved:
        rid = r["id"]
        r_npz = npz_paths_for_id(args.features_dir, rid)
        Xr, yr, t0r, t1r = predict_saliency(r_npz, model, args.device)
        yr_s = moving_avg(yr, args.smooth)
        ref_profiles.append(compute_structure_metrics(yr_s, t0r, t1r))

    query_metrics = compute_structure_metrics(yq_s, t0q, t1q)
    ref_med = median_profile(ref_profiles) if ref_profiles else {}

    segments = derive_segments(
        yq_s, t0q, t1q,
        thr_pctl=args.seg_thr_pctl,
        min_len_sec=args.seg_min_len,
        merge_gap_sec=args.seg_merge_gap
    )

    edits = build_edit_recommendations(query_metrics, ref_med, segments)

    # metadata-aware trimming
    meta_path = args.meta if args.meta else os.path.join("data", "meta", f"{args.query_id}.json")
    meta = load_meta(meta_path)
    mode = meta.get("profile", {}).get("mode", "unknown")
    duration_sec = float(meta.get("profile", {}).get("duration_sec", float(t1q[-1])))

    if mode == "speech":
        trim_intervals = non_speech_trims_from_vad(
            duration_sec=duration_sec,
            speech_segments=meta.get("speech_segments", []),
            min_gap_sec=0.45,
            pad_sec=0.10
    )
    else:
        # fallback to your existing logic for now
        trim_intervals = materialize_trim_intervals_from_segments(t0q, t1q, segments)


    payload = {
        "query_id": args.query_id,
        "meta": {
            "path": meta_path,
            "mode": mode
        },
        "retrieved_refs": retrieved,
        "metrics": {
            "query": query_metrics,
            "ref_median": ref_med
        },
        "segments": segments,
        "recommendations": edits,
        "trim_intervals": trim_intervals
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved edit plan: {args.out}")
    print(f"Retrieved refs: {[r['id'] for r in retrieved]}")
    print(f"Segments: {len(segments)}  trims: {len(trim_intervals)}")


if __name__ == "__main__":
    main()
