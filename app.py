import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr


REPO_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ffprobe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        video_path
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip()
    return float(out)


def pipeline(video_path: str, keep_threshold: float, model_size: str, compute_type: str) -> tuple[str, str]:
    """
    Returns: (edited_video_path, report_json_string)
    """
    video_path = str(video_path)
    vid_id = Path(video_path).stem

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        transcripts_dir = td / "transcripts"
        units_dir = td / "units"
        scores_dir = td / "scores"
        recs_dir = td / "recs"
        edited_dir = td / "edited"

        for p in [transcripts_dir, units_dir, scores_dir, recs_dir, edited_dir]:
            p.mkdir(parents=True, exist_ok=True)

        transcript_json = transcripts_dir / f"{vid_id}.json"
        units_json = units_dir / f"{vid_id}_units.json"
        scores_json = scores_dir / f"{vid_id}.json"
        edits_json = recs_dir / f"{vid_id}_semantic_edits.json"
        edited_mp4 = edited_dir / f"{vid_id}_semantic_edited.mp4"

        # 1) ASR
        run([
            "python", str(REPO_ROOT / "scripts" / "transcribe.py"),
            "--video", video_path,
            "--out", str(transcript_json),
            "--model", model_size,
            "--compute_type", compute_type
        ])

        # 2) Build sentence units
        run([
            "python", str(REPO_ROOT / "scripts" / "build_units_from_transcript.py"),
            "--transcript", str(transcript_json),
            "--out", str(units_json)
        ])

        # 3) Score units semantically
        run([
            "python", str(REPO_ROOT / "scripts" / "score_units.py"),
            "--units", str(units_json),
            "--out", str(scores_json)
        ])

        # 4) Plan edits from units
        duration = ffprobe_duration(video_path)
        run([
            "python", str(REPO_ROOT / "scripts" / "plan_edits_from_units.py"),
            "--scores", str(scores_json),
            "--duration_sec", str(duration),
            "--out", str(edits_json),
            "--keep_threshold", str(keep_threshold)
        ])

        # 5) Apply edits
        run([
            "python", str(REPO_ROOT / "scripts" / "apply_edits.py"),
            "--video", video_path,
            "--edits_json", str(edits_json),
            "--out", str(edited_mp4),
            "--reencode"
        ])

        # Create an easy-to-read report for the UI
        report = json.load(open(edits_json, "r"))
        report["ui"] = {
            "orig_duration_sec": round(duration, 2),
            "edited_path": str(edited_mp4.name),
        }

        # Persist output outside tempdir so Gradio can serve it
        out_dir = REPO_ROOT / "outputs" / "app_runs"
        out_dir.mkdir(parents=True, exist_ok=True)

        final_video = out_dir / f"{vid_id}_edited.mp4"
        final_report = out_dir / f"{vid_id}_report.json"

        shutil.copyfile(edited_mp4, final_video)
        with open(final_report, "w") as f:
            json.dump(report, f, indent=2)

        return str(final_video), json.dumps(report, indent=2)


def ui_run(video, keep_threshold, model_size, compute_type):
    if video is None:
        raise gr.Error("Upload an mp4 first.")
    edited_path, report_str = pipeline(video, keep_threshold, model_size, compute_type)
    return edited_path, report_str


with gr.Blocks(title="AI Content Optimizer") as demo:
    gr.Markdown("# AI Content Optimizer\nUpload a video, get an edited version plus an edit report.")

    with gr.Row():
        video_in = gr.Video(label="Upload MP4", format="mp4")
        with gr.Column():
            keep_threshold = gr.Slider(0.30, 0.80, value=0.45, step=0.01, label="Keep threshold (lower keeps more)")
            model_size = gr.Dropdown(["tiny", "base", "small", "medium", "large-v3"], value="small", label="ASR model")
            compute_type = gr.Dropdown(["int8", "int8_float16", "float16", "float32"], value="int8", label="Compute type")
            run_btn = gr.Button("Optimize")

    edited_out = gr.Video(label="Edited video", format="mp4")
    report_out = gr.Code(label="Edit plan and debug report (JSON)", language="json")

    run_btn.click(
        fn=ui_run,
        inputs=[video_in, keep_threshold, model_size, compute_type],
        outputs=[edited_out, report_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
