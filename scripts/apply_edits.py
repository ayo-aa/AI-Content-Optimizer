import argparse
import json
import os
import subprocess
import tempfile


def run(cmd):
    subprocess.run(cmd, check=True)


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e])
    return [(float(a), float(b)) for a, b in merged]


def subtract_intervals(total_start, total_end, remove_intervals):
    """
    Returns keep intervals = [total_start,total_end] minus remove_intervals.
    remove_intervals must be merged and within bounds.
    """
    keep = []
    cur = total_start
    for s, e in remove_intervals:
        if e <= cur:
            continue
        if s > cur:
            keep.append((cur, min(s, total_end)))
        cur = max(cur, e)
        if cur >= total_end:
            break
    if cur < total_end:
        keep.append((cur, total_end))
    # drop tiny keeps
    keep = [(s, e) for s, e in keep if e - s >= 0.20]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--edits_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reencode", action="store_true")
    args = ap.parse_args()

    with open(args.edits_json, "r") as f:
        plan = json.load(f)

    trims = []
    for itv in plan.get("trim_intervals", []):
        s = float(itv["start"])
        e = float(itv["end"])
        if e > s:
            trims.append((s, e))

    if not trims:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        run(["cp", args.video, args.out])
        print(f"No trims. Copied original to {args.out}")
        return

    # Determine total duration from trim intervals payload if present
    # fallback: use last trim end as a weak proxy (not ideal)
    total_end = 0.0
    for itv in plan.get("trim_intervals", []):
        total_end = max(total_end, float(itv["end"]))
    total_start = 0.0

    trims = merge_intervals(trims)
    keeps = subtract_intervals(total_start, total_end, trims)

    if not keeps:
        raise RuntimeError("All content trimmed. Lower trim aggressiveness.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        seg_paths = []
        for i, (s, e) in enumerate(keeps):
            seg = os.path.join(td, f"seg_{i:03d}.mp4")
            cmd = ["ffmpeg", "-y", "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", args.video]
            if args.reencode:
                cmd += ["-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", seg]
            else:
                # stream copy is fast but can fail depending on codecs/keyframes
                cmd += ["-c", "copy", seg]
            run(cmd)
            seg_paths.append(seg)

        list_path = os.path.join(td, "list.txt")
        with open(list_path, "w") as f:
            for p in seg_paths:
                f.write(f"file '{p}'\n")

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path]
        if args.reencode:
            cmd += ["-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", args.out]
        else:
            cmd += ["-c", "copy", args.out]
        run(cmd)

    print(f"Saved edited video: {args.out}")
    print(f"Kept segments: {len(keeps)}  Trim intervals: {len(trims)}")


if __name__ == "__main__":
    main()
