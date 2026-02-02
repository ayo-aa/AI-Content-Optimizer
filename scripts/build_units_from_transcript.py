import argparse
import json
import os
import re
from typing import Any, Dict, List


END_PUNCT = re.compile(r"[.!?]+$")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_sentence_units(transcript: Dict[str, Any], max_pause_sec: float = 0.9) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    cur_words = []
    cur_start = None
    cur_end = None

    def flush():
        nonlocal cur_words, cur_start, cur_end
        if not cur_words:
            return
        text = "".join([w["word"] for w in cur_words]).strip()
        if text:
            units.append({
                "start": float(cur_start),
                "end": float(cur_end),
                "text": text,
                "words": cur_words
            })
        cur_words = []
        cur_start = None
        cur_end = None

    last_word_end = None

    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            w_start = float(w["start"])
            w_end = float(w["end"])
            w_text = w["word"]

            # pause boundary
            if last_word_end is not None and (w_start - last_word_end) >= max_pause_sec:
                flush()

            if cur_start is None:
                cur_start = w_start

            cur_words.append({"start": w_start, "end": w_end, "word": w_text})
            cur_end = w_end
            last_word_end = w_end

            # punctuation boundary
            if END_PUNCT.search(w_text.strip()):
                flush()

    flush()
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_pause_sec", type=float, default=0.9)
    args = ap.parse_args()

    tr = load_json(args.transcript)
    units = build_sentence_units(tr, max_pause_sec=args.max_pause_sec)

    payload = {
        "video": tr.get("video"),
        "source_transcript": os.path.basename(args.transcript),
        "num_units": len(units),
        "units": units
    }

    save_json(payload, args.out)
    print(f"Saved units: {args.out} (n={len(units)})")


if __name__ == "__main__":
    main()
