import wave
import argparse
import json
import os
import tempfile
import subprocess
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import webrtcvad


def extract_wav(video_path: str, wav_path: str, sr: int = 16000):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-acodec", "pcm_s16le",
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def read_wav_mono_16bit(wav_path: str):
    with wave.open(wav_path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        if nch != 1:
            raise RuntimeError(f"Expected mono wav, got {nch} channels")
        if sw != 2:
            raise RuntimeError(f"Expected 16-bit wav, got sampwidth={sw}")
        pcm = wf.readframes(nframes)
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(x), sr


def vad_speech_segments(wav: torch.Tensor, sr: int, aggressiveness: int = 2) -> List[Tuple[float, float]]:
    """
    WebRTC VAD expects 10/20/30ms frames of 16-bit PCM at 8/16/32/48k.
    Returns merged speech segments in seconds.
    """
    assert sr in (8000, 16000, 32000, 48000)
    vad = webrtcvad.Vad(aggressiveness)

    # 30ms frames are usually stable
    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)

    x = (wav * 32768.0).clamp(-32768, 32767).to(torch.int16).cpu().numpy()
    n = len(x)
    if n < frame_len:
        return []

    speech_flags = []
    times = []
    i = 0
    t = 0.0
    hop = frame_len / sr
    while i + frame_len <= n:
        frame = x[i:i + frame_len].tobytes()
        is_speech = vad.is_speech(frame, sr)
        speech_flags.append(is_speech)
        times.append(t)
        i += frame_len
        t += hop

    # Convert flags into segments
    segs = []
    in_speech = False
    start = 0.0
    for flag, t0 in zip(speech_flags, times):
        if flag and not in_speech:
            in_speech = True
            start = t0
        if not flag and in_speech:
            in_speech = False
            end = t0
            if end - start >= 0.12:
                segs.append((start, end))
    if in_speech:
        end = times[-1] + hop
        if end - start >= 0.12:
            segs.append((start, end))

    # Merge close segments (small gaps)
    merged = []
    gap = 0.25
    for s, e in segs:
        if not merged:
            merged.append([s, e])
        else:
            ps, pe = merged[-1]
            if s - pe <= gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])

    return [(float(a), float(b)) for a, b in merged]


def shot_boundaries_from_features(npz_path: str, pctl: float = 95.0) -> List[float]:
    """
    Uses large jumps in consecutive X_vis embeddings as approximate shot boundaries.
    Returns boundary times in seconds (including 0 and end).
    """
    d = np.load(npz_path)
    X = d["X_vis"].astype(np.float32)
    t0 = d["t_start"].astype(np.float32)
    t1 = d["t_end"].astype(np.float32)

    if len(X) < 3:
        return [float(t0[0]), float(t1[-1])]

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = (Xn[1:] * Xn[:-1]).sum(axis=1)
    change = 1.0 - sim  # bigger means more change

    thr = float(np.percentile(change, pctl))
    idx = np.where(change >= thr)[0] + 1  # boundary at index i

    bounds = [float(t0[0])]
    for i in idx.tolist():
        bounds.append(float(t0[i]))
    bounds.append(float(t1[-1]))

    # Deduplicate and sort
    bounds = sorted(set([round(b, 3) for b in bounds]))
    return bounds


def compute_profile(duration: float, speech_segs: List[Tuple[float, float]], shot_bounds: List[float]) -> dict:
    speech_time = sum(max(0.0, e - s) for s, e in speech_segs)
    speech_ratio = speech_time / max(duration, 1e-6)

    shot_lengths = []
    for a, b in zip(shot_bounds[:-1], shot_bounds[1:]):
        shot_lengths.append(max(0.0, b - a))
    avg_shot = float(np.mean(shot_lengths)) if shot_lengths else duration
    shot_rate = (len(shot_bounds) - 1) / max(duration, 1e-6)  # shots per second

    # Simple mode routing
    if speech_ratio >= 0.55:
        mode = "speech"
    elif speech_ratio <= 0.15 and shot_rate >= 0.12:
        mode = "montage"
    else:
        mode = "mixed"

    return {
        "duration_sec": float(duration),
        "speech_ratio": float(speech_ratio),
        "avg_shot_len_sec": float(avg_shot),
        "shot_rate_per_sec": float(shot_rate),
        "mode": mode,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--vad_aggr", type=int, default=2)
    ap.add_argument("--shot_pctl", type=float, default=95.0)
    args = ap.parse_args()

    d = np.load(args.npz)
    duration = float(d["t_end"][-1])

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "tmp.wav")
        extract_wav(args.video, wav_path, sr=args.sr)
        wav, sr = read_wav_mono_16bit(wav_path)
            # sr should already equal args.sr because ffmpeg resampled


    speech_segs = vad_speech_segments(wav, sr, aggressiveness=args.vad_aggr)
    shot_bounds = shot_boundaries_from_features(args.npz, pctl=args.shot_pctl)
    profile = compute_profile(duration, speech_segs, shot_bounds)

    payload = {
        "video": os.path.basename(args.video),
        "npz": os.path.basename(args.npz),
        "profile": profile,
        "speech_segments": [{"start": s, "end": e} for s, e in speech_segs],
        "shot_boundaries": shot_bounds
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved metadata: {args.out}")
    print(f"mode={profile['mode']} speech_ratio={profile['speech_ratio']:.2f} shots={len(shot_bounds)-1}")


if __name__ == "__main__":
    main()
