import wave
import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
from torchvision import models, transforms


@dataclass
class FeatConfig:
    window_sec: float = 2.0
    hop_sec: float = 1.0
    audio_sr: int = 16000
    n_mels: int = 64
    device: str = "cpu"


def run_ffmpeg_extract_wav(video_path: str, wav_path: str, sr: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", str(sr),
        "-vn", wav_path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", errors="ignore"))

def load_wav_stdlib(path: str):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)

    if sampwidth == 1:
        audio = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sampwidth == 2:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32)
        audio = audio / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return torch.from_numpy(audio), sr  # [N], sr


def load_audio_windows(video_path: str, cfg: FeatConfig, duration_sec: float, T: int):
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "tmp.wav")
        run_ffmpeg_extract_wav(video_path, wav_path, cfg.audio_sr)

        wav, sr = load_wav_stdlib(wav_path)  # [N]
        if sr != cfg.audio_sr:
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, cfg.audio_sr
            ).squeeze(0)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.audio_sr,
        n_mels=cfg.n_mels
    )

    target_len = int(cfg.window_sec * cfg.audio_sr)
    X_aud = []
    for t in range(T):
        start_sec = t * cfg.hop_sec
        end_sec = start_sec + cfg.window_sec
        a0 = int(start_sec * cfg.audio_sr)
        a1 = int(end_sec * cfg.audio_sr)

        chunk = wav[a0:a1]
        if chunk.numel() < target_len:
            chunk = torch.nn.functional.pad(chunk, (0, target_len - chunk.numel()))
        elif chunk.numel() > target_len:
            chunk = chunk[:target_len]

        spec = mel(chunk.unsqueeze(0))         # [1, n_mels, frames]
        spec = torch.log(spec + 1e-6).squeeze(0)  # [n_mels, frames]

        feat_mean = spec.mean(dim=1)  # [n_mels]
        feat_std = spec.std(dim=1)    # [n_mels]
        feat = torch.cat([feat_mean, feat_std], dim=0)  # [2*n_mels]
        X_aud.append(feat.cpu().numpy())

    return np.stack(X_aud, axis=0).astype(np.float32)  # [T, 2*n_mels]


def get_video_meta(cap: cv2.VideoCapture):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = float(fps) if fps and fps > 1e-6 else 30.0
    frames = int(frames) if frames and frames > 0 else 0
    duration = frames / fps if frames > 0 else None
    return fps, frames, duration


def build_visual_encoder(device: str):
    weights = models.ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=weights)
    model.heads = nn.Identity()
    model.eval().to(device)

    preprocess = weights.transforms()  # callable preset

    def tfm(frame_rgb_np):
        # frame_rgb_np is an RGB numpy array from OpenCV conversion
        img = transforms.ToPILImage()(frame_rgb_np)
        return preprocess(img)

    return model, tfm


def read_frame_at_time(cap: cv2.VideoCapture, time_sec: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_features(video_path: str, out_path: str, cfg: FeatConfig):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps, frames, duration = get_video_meta(cap)
    if duration is None:
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        end_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        duration = float(end_ms / 1000.0) if end_ms and end_ms > 0 else 0.0
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

    duration = float(duration)
    if duration < cfg.window_sec:
        raise RuntimeError(f"Video too short ({duration:.2f}s).")

    T = int(np.floor((duration - cfg.window_sec) / cfg.hop_sec) + 1)
    t_start = np.array([t * cfg.hop_sec for t in range(T)], dtype=np.float32)
    t_end = t_start + cfg.window_sec
    t_mid = (t_start + t_end) / 2.0

    device = cfg.device
    model, tfm = build_visual_encoder(device)

    X_vis = []
    for tm in tqdm(t_mid, desc="Visual"):
        frame = read_frame_at_time(cap, float(tm))
        if frame is None:
            X_vis.append(X_vis[-1] if len(X_vis) else np.zeros((768,), dtype=np.float32))
            continue

        inp = tfm(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(inp).squeeze(0).cpu().numpy().astype(np.float32)
        X_vis.append(emb)

    cap.release()

    X_vis = np.stack(X_vis, axis=0).astype(np.float32)          # [T, 768]
    X_aud = load_audio_windows(video_path, cfg, duration, T)     # [T, 2*n_mels]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        X_vis=X_vis,
        X_aud=X_aud,
        t_start=t_start,
        t_end=t_end,
        duration=np.array([duration], dtype=np.float32),
        fps=np.array([fps], dtype=np.float32),
        frame_count=np.array([frames], dtype=np.int64),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window_sec", type=float, default=2.0)
    ap.add_argument("--hop_sec", type=float, default=1.0)
    ap.add_argument("--audio_sr", type=int, default=16000)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    cfg = FeatConfig(
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        audio_sr=args.audio_sr,
        n_mels=args.n_mels,
        device=args.device,
    )

    extract_features(args.video, args.out, cfg)
    print(f"Saved features to {args.out}")


if __name__ == "__main__":
    main()
