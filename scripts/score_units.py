import argparse
import json
import os
import re
from typing import Dict, Any, List

import numpy as np
from sentence_transformers import SentenceTransformer


FILLER_PATTERNS = [
    r"\bi just\b",
    r"\bi wanted to\b",
    r"\bi decided\b",
    r"\bwhy not\b",
    r"\bspeaking of which\b",
    r"\bplease let me know\b",
    r"\bi'm not gonna lie\b",
    r"\bso this is just\b",
    r"\bin my opinion\b",
]

FILLER_RE = re.compile("|".join(FILLER_PATTERNS), re.IGNORECASE)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [N,D], b: [M,D]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


def is_filler(text: str) -> float:
    t = text.strip().lower()
    score = 0.0
    if len(t) <= 18:
        score += 0.25
    if FILLER_RE.search(t):
        score += 0.65
    # vague openers
    if t.startswith("so ") or t.startswith("and "):
        score += 0.15
    return float(min(1.0, score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True, help="data/units/<id>_units.json")
    ap.add_argument("--out", required=True, help="data/scores/<id>.json")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--redundancy_window", type=int, default=12)
    args = ap.parse_args()

    u = load_json(args.units)
    units: List[Dict[str, Any]] = u["units"]

    texts = [x["text"].strip() for x in units]
    starts = [float(x["start"]) for x in units]
    ends = [float(x["end"]) for x in units]

    model = SentenceTransformer(args.model)

    E = model.encode(texts, normalize_embeddings=True)
    E = np.asarray(E, dtype=np.float32)

    topic = E.mean(axis=0).astype(np.float32)              # [D]
    tn = float(np.linalg.norm(topic))
    if (not np.isfinite(tn)) or (tn < 1e-6):
        topic = E[0].copy()
        tn = float(np.linalg.norm(topic)) + 1e-9
    topic = topic / tn                                     # [D]

    relevance = E @ topic                                  # [N]

    # Redundancy: similarity to previous sentences (within a window)
    N = len(units)
    redundancy = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        j0 = max(0, i - args.redundancy_window)
        if i == 0:
            redundancy[i] = 0.0
            continue
        sims = (E[i:i+1] @ E[j0:i].T).squeeze(0)
        redundancy[i] = float(np.max(sims)) if sims.size else 0.0

    filler = np.array([is_filler(t) for t in texts], dtype=np.float32)

    # Normalize relevance to [0,1] for stable weighting
    rel = relevance.astype(np.float32)
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)

    # Keep score: prefer high relevance, penalize redundancy and filler
    keep = 0.70 * rel - 0.45 * redundancy - 0.35 * filler
    keep = keep.astype(np.float32)

    # Rescale keep score to [0,1] for interpretability
    keep01 = (keep - keep.min()) / (keep.max() - keep.min() + 1e-9)

    scored = []
    for i in range(N):
        scored.append({
            "index": i,
            "start": starts[i],
            "end": ends[i],
            "text": texts[i],
            "relevance": float(rel[i]),
            "redundancy": float(redundancy[i]),
            "filler": float(filler[i]),
            "keep_score": float(keep01[i]),
        })

    payload = {
        "video": u.get("video"),
        "source_units": os.path.basename(args.units),
        "embedding_model": args.model,
        "scored_units": scored,
    }

    save_json(payload, args.out)
    print(f"Saved scores: {args.out}")
    top = sorted(scored, key=lambda x: x["keep_score"], reverse=True)[:5]
    print("Top keep units:")
    for t in top:
        print(f"- [{t['start']:.2f}-{t['end']:.2f}] {t['keep_score']:.3f} | {t['text'][:80]}")


if __name__ == "__main__":
    main()
