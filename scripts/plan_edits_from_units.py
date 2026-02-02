import argparse
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def merge_intervals(intervals: List[Tuple[float, float]], min_gap: float = 0.10) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + min_gap:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e])
    return [(float(a), float(b)) for a, b in merged]


def complement_intervals(total_start: float, total_end: float, keeps: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    keeps = merge_intervals(keeps)
    trims = []
    cur = total_start
    for s, e in keeps:
        if s > cur:
            trims.append((cur, s))
        cur = max(cur, e)
    if cur < total_end:
        trims.append((cur, total_end))
    trims = [(s, e) for s, e in trims if e - s >= 0.20]
    return trims


def pick_keeps(scored_units: List[Dict[str, Any]],
               keep_threshold: float,
               min_block_units: int,
               max_drop_run: int) -> List[int]:
    """
    Keep units whose keep_score >= threshold, then smooth with simple rules:
    - if a block of keeps is very short, drop it
    - if a single low unit is between high units, keep it to preserve coherence
    """
    N = len(scored_units)
    keep_mask = np.array([u["keep_score"] >= keep_threshold for u in scored_units], dtype=bool)

    # Fill single-unit gaps between keeps
    for i in range(1, N - 1):
        if (not keep_mask[i]) and keep_mask[i - 1] and keep_mask[i + 1]:
            keep_mask[i] = True

    # Drop tiny keep islands
    i = 0
    while i < N:
        if not keep_mask[i]:
            i += 1
            continue
        j = i
        while j < N and keep_mask[j]:
            j += 1
        if (j - i) < min_block_units:
            keep_mask[i:j] = False
        i = j

    # Avoid dropping too many consecutive units (coherence)
    # If we see a long drop run, keep the best unit inside it every max_drop_run units
    i = 0
    while i < N:
        if keep_mask[i]:
            i += 1
            continue
        j = i
        while j < N and (not keep_mask[j]):
            j += 1
        drop_len = j - i
        if drop_len >= max_drop_run:
            # choose the top scoring unit inside the drop run to keep
            block = scored_units[i:j]
            best_idx = i + int(np.argmax([b["keep_score"] for b in block]))
            keep_mask[best_idx] = True
        i = j

    return [i for i in range(N) if keep_mask[i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="data/scores/<id>.json")
    ap.add_argument("--duration_sec", type=float, required=True)
    ap.add_argument("--out", required=True, help="outputs/recs/<id>_semantic_edits.json")
    ap.add_argument("--keep_threshold", type=float, default=0.50)
    ap.add_argument("--min_block_units", type=int, default=2)
    ap.add_argument("--max_drop_run", type=int, default=5)
    ap.add_argument("--pad_sec", type=float, default=0.10)
    args = ap.parse_args()

    s = load_json(args.scores)
    units = s["scored_units"]
    total_start = 0.0
    total_end = float(args.duration_sec)

    keep_idxs = pick_keeps(
        scored_units=units,
        keep_threshold=args.keep_threshold,
        min_block_units=args.min_block_units,
        max_drop_run=args.max_drop_run
    )

    keeps = []
    kept_units = []
    for i in keep_idxs:
        a = max(total_start, float(units[i]["start"]) - args.pad_sec)
        b = min(total_end, float(units[i]["end"]) + args.pad_sec)
        keeps.append((a, b))
        kept_units.append(units[i])

    keeps = merge_intervals(keeps)
    trims = complement_intervals(total_start, total_end, keeps)

    payload = {
        "query_id": os.path.splitext(os.path.basename(args.scores))[0],
        "method": "semantic_sentence_planner_v1",
        "kept_units": kept_units,
        "keep_intervals": [{"start": a, "end": b} for a, b in keeps],
        "trim_intervals": [{"start": a, "end": b, "kind": "semantic_cut"} for a, b in trims],
        "stats": {
            "num_units": len(units),
            "kept_units": len(kept_units),
            "kept_intervals": len(keeps),
            "trim_intervals": len(trims),
            "orig_duration_sec": total_end,
            "estimated_kept_sec": float(sum(b - a for a, b in keeps))
        }
    }

    save_json(payload, args.out)
    print(f"Saved edit plan: {args.out}")
    print(payload["stats"])


if __name__ == "__main__":
    main()
