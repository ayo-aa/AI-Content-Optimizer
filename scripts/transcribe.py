import argparse
import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List

from faster_whisper import WhisperModel


def extract_wav(video_path: str, wav_path: str, sr: int = 16000) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-acodec", "pcm_s16le",
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe_to_json(
    video_path: str,
    out_json: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    vad_filter: bool = True,
) -> None:
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "audio.wav")
        extract_wav(video_path, wav_path)

        segments, info = model.transcribe(
            wav_path,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=vad_filter,
        )

        seg_out: List[Dict[str, Any]] = []
        for seg in segments:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append({
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": w.word
                    })

            seg_out.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text,
                "words": words
            })

    payload = {
        "video": os.path.basename(video_path),
        "model": {"backend": "faster-whisper", "size": model_size, "compute_type": compute_type},
        "language": getattr(info, "language", None),
        "language_probability": float(getattr(info, "language_probability", 0.0)) if getattr(info, "language_probability", None) is not None else None,
        "segments": seg_out
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved transcript: {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large-v3"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute_type", default="int8", choices=["int8", "int8_float16", "float16", "float32"])
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--no_vad", action="store_true")
    args = ap.parse_args()

    transcribe_to_json(
        video_path=args.video,
        out_json=args.out,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=(not args.no_vad),
    )


if __name__ == "__main__":
    main()
