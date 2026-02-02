import argparse
import json
import os
import subprocess
import tempfile
from collections import Counter

import numpy as np
import torch
from PIL import Image
import open_clip


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def extract_frame_ffmpeg(video_path: str, time_sec: float, out_path: str):
    # -ss before -i is fast and good enough for keyframes
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{time_sec:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--prompts", default="configs/vision_prompts.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    meta = load_json(args.meta)
    prompt_lib = load_json(args.prompts)

    shot_bounds = meta.get("shot_boundaries", [])
    if not shot_bounds or len(shot_bounds) < 2:
        raise RuntimeError("No shot_boundaries found in meta JSON.")

    # Build shot segments
    shots = []
    for a, b in zip(shot_bounds[:-1], shot_bounds[1:]):
        a = float(a)
        b = float(b)
        if b > a:
            shots.append({"start": a, "end": b, "mid": (a + b) / 2.0})

    device = args.device
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    # Precompute text embeddings for each category’s prompts
    categories = list(prompt_lib.keys())
    cat_text_embeds = {}
    with torch.no_grad():
        for cat in categories:
            texts = prompt_lib[cat]
            tokens = tokenizer(texts).to(device)
            txt = model.encode_text(tokens)
            txt = l2norm(txt)
            cat_text_embeds[cat] = {
                "prompts": texts,
                "embeds": txt
            }

    results = []
    label_counts = Counter()

    with tempfile.TemporaryDirectory() as td:
        for i, s in enumerate(shots):
            t = float(s["mid"])
            img_path = os.path.join(td, f"shot_{i:04d}.jpg")
            try:
                extract_frame_ffmpeg(args.video, t, img_path)
                img = Image.open(img_path).convert("RGB")
            except Exception:
                # If frame extraction fails, skip but keep a record
                results.append({
                    "shot_index": i,
                    "start": s["start"],
                    "end": s["end"],
                    "mid": s["mid"],
                    "labels": [],
                    "error": "frame_extract_failed"
                })
                continue

            image_input = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                im = model.encode_image(image_input)
                im = l2norm(im)

            # Score each category by best matching prompt
            cat_scores = []
            for cat in categories:
                txt = cat_text_embeds[cat]["embeds"]  # [P, D]
                sims = (im @ txt.T).squeeze(0)         # [P]
                best_idx = int(torch.argmax(sims).item())
                best_score = float(sims[best_idx].item())
                cat_scores.append((cat, best_score, cat_text_embeds[cat]["prompts"][best_idx]))

            cat_scores.sort(key=lambda x: x[1], reverse=True)
            top = cat_scores[: max(1, args.topk)]

            labels = []
            for cat, score, prompt in top:
                labels.append({"label": cat, "score": score, "matched_prompt": prompt})
            label_counts.update([labels[0]["label"]])

            results.append({
                "shot_index": i,
                "start": s["start"],
                "end": s["end"],
                "mid": s["mid"],
                "labels": labels
            })

    payload = {
        "video": os.path.basename(args.video),
        "mode": meta.get("profile", {}).get("mode", "unknown"),
        "model": {"name": args.model, "pretrained": args.pretrained},
        "shots": results,
        "summary": dict(label_counts)
    }

    save_json(payload, args.out)
    print(f"Saved: {args.out}")
    print("Top labels summary:", dict(label_counts))


if __name__ == "__main__":
    main()
