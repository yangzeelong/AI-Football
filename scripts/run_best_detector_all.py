from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "Run the current best task-1 detector settings on all videos under data/."
    )
    parser.add_argument("--data-dir",
                        default="data",
                        help="Directory to scan for videos.")
    parser.add_argument(
        "--outdir",
        default="tmp/best_detector_all",
        help="Directory for observations, reports, and summary files.",
    )
    parser.add_argument(
        "--detector",
        choices=["rfdetr", "yolo"],
        default="rfdetr",
        help="Best detector backend to run.",
    )
    parser.add_argument(
        "--rfdetr-size",
        choices=["nano", "small", "base", "medium", "large"],
        default="small",
        help="Recommended RF-DETR size from detector evaluation.",
    )
    parser.add_argument(
        "--model-dir",
        default="models/rfdetr",
        help="RF-DETR weights directory.",
    )
    parser.add_argument(
        "--yolo-model",
        default="models/yolo/yolov8n.pt",
        help="YOLO model path when --detector yolo is used.",
    )
    parser.add_argument("--conf",
                        type=float,
                        default=0.25,
                        help="Detection confidence.")
    parser.add_argument("--imgsz",
                        type=int,
                        default=640,
                        help="YOLO image size.")
    parser.add_argument("--device", default="cuda:0", help="Detector device.")
    parser.add_argument("--pose-device", default=None, help="MMPose device.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit per video. Omit to process full videos.",
    )
    parser.add_argument(
        "--app-config",
        default="config/app.yaml",
        help="App filtering config YAML.",
    )
    parser.add_argument(
        "--video-labels",
        default="config/video_labels.yaml",
        help="Video label YAML used to skip ignored materials.",
    )
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help="Run videos marked with ignore: true in --video-labels.",
    )
    parser.add_argument(
        "--roi-config",
        default="config/roi.json",
        help="ROI config JSON.",
    )
    parser.add_argument(
        "--no-roi",
        action="store_true",
        help="Disable ROI filtering. ROI is enabled by default when available.",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="Only run the first N discovered videos.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos whose observation and JSON report already exist.",
    )
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Print commands only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_labels = load_video_labels(args.video_labels)
    videos = filter_ignored_videos(
        discover_videos(Path(args.data_dir)),
        video_labels,
        include_ignored=args.include_ignored,
    )
    if args.limit_videos is not None:
        videos = videos[:args.limit_videos]
    if not videos:
        raise FileNotFoundError(f"No videos found under {args.data_dir}")

    outdir = Path(args.outdir)
    if outdir.exists():
        import shutil
        shutil.rmtree(outdir)
        logger.warning(f"Removed existing file {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for video in videos:
        run = build_run(args, video, outdir)
        rows.append(execute_run(run, args.dry_run, args.skip_existing))

    if not args.dry_run:
        write_summary(outdir, rows)


def discover_videos(data_dir: Path) -> list[Path]:
    return sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def load_video_labels(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    label_path = Path(path)
    if not label_path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read video labels.") from exc
    data = yaml.safe_load(label_path.read_text(encoding="utf-8")) or {}
    videos = data.get("videos", {}) if isinstance(data, dict) else {}
    return videos if isinstance(videos, dict) else {}


def filter_ignored_videos(
    videos: list[Path],
    video_labels: dict[str, Any],
    include_ignored: bool,
) -> list[Path]:
    if include_ignored:
        return videos
    return [
        video for video in videos
        if not video_labels.get(video.name, {}).get("ignore", False)
    ]


def build_run(args: argparse.Namespace, video: Path,
              outdir: Path) -> dict[str, Any]:
    run_name = f"{safe_name(video.stem)}_{short_hash(video)}"
    run_dir = outdir / run_name
    observation_path = run_dir / "observations.jsonl"
    report_json_path = run_dir / "quality_report.json"
    report_md_path = run_dir / "quality_report.md"
    rendered_video_path = run_dir / "rendered.mp4"

    app_cmd = [
        sys.executable,
        "code/app.py",
        "--video",
        str(video),
        "--detector",
        args.detector,
        "--conf",
        str(args.conf),
        "--device",
        args.device,
        "--stride",
        str(args.stride),
        "--output-observations",
        str(observation_path),
        "--output-video",
        str(rendered_video_path),
        "--app-config",
        args.app_config,
        "--roi-config",
        args.roi_config,
    ]
    if not args.no_roi:
        app_cmd.append("--use-roi")
    if args.max_frames is not None:
        app_cmd.extend(["--max-frames", str(args.max_frames)])
    if args.pose_device:
        app_cmd.extend(["--pose-device", args.pose_device])

    if args.detector == "rfdetr":
        app_cmd.extend([
            "--rfdetr-size",
            args.rfdetr_size,
            "--model-dir",
            args.model_dir,
        ])
    else:
        app_cmd.extend(
            ["--model", args.yolo_model, "--imgsz",
             str(args.imgsz)])

    report_cmd = [
        sys.executable,
        "code/evaluate_observations.py",
        "--input",
        str(observation_path),
        "--output",
        str(report_json_path),
        "--markdown",
        str(report_md_path),
    ]

    return {
        "video": video,
        "run_dir": run_dir,
        "observation_path": observation_path,
        "rendered_video_path": rendered_video_path,
        "report_json_path": report_json_path,
        "report_md_path": report_md_path,
        "app_cmd": app_cmd,
        "report_cmd": report_cmd,
    }


def execute_run(
    run: dict[str, Any],
    dry_run: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    run["run_dir"].mkdir(parents=True, exist_ok=True)
    safe_print(f"[run] {run['run_dir'].name} -> {run['run_dir']}")
    safe_print(f"[command] app.py -> {run['observation_path']}")

    row = base_row(run)
    if dry_run:
        row["status"] = "dry_run"
        return row

    if (skip_existing and run["observation_path"].exists()
            and run["report_json_path"].exists()
            and run["rendered_video_path"].exists()):
        row["status"] = "skipped"
        row.update(load_report_metrics(run["report_json_path"]))
        return row

    started = time.perf_counter()
    try:
        # 正式任务1输出必须包含检测、单镜头track、MMPose 26 keypoints和足球轨迹。
        subprocess.run(run["app_cmd"], check=True)
        subprocess.run(run["report_cmd"], check=True)
    except subprocess.CalledProcessError as exc:
        row["status"] = "failed"
        row["returncode"] = exc.returncode
    else:
        row["status"] = "ok"
        row.update(load_report_metrics(run["report_json_path"]))
    finally:
        row["elapsed_sec"] = round(time.perf_counter() - started, 3)

    return row


def base_row(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "",
        "video": str(run["video"]),
        "frames": 0,
        "elapsed_sec": 0.0,
        "total_score": 0.0,
        "ball_present_ratio": 0.0,
        "max_missing_streak": 0,
        "avg_keypoint_confidence": 0.0,
        "max_raw_balls_per_frame": 0,
        "multi_ball_frame_ratio": 0.0,
        "false_ball_frame_ratio": 0.0,
        "untracked_person_observations": 0,
        "short_track_ratio": 0.0,
        "observations": str(run["observation_path"]),
        "rendered_video": str(run["rendered_video_path"]),
        "report": str(run["report_md_path"]),
        "returncode": "",
    }


def load_report_metrics(report_json_path: Path) -> dict[str, Any]:
    report = json.loads(report_json_path.read_text(encoding="utf-8"))
    labels = report.get("labels", {})
    return {
        "frames":
        report["frame_stats"]["total_frames"],
        "total_score":
        report["quality_score"]["total"],
        "ball_present_ratio":
        report["balls"]["ball_present_frame_ratio"],
        "max_missing_streak":
        report["balls"]["max_missing_streak"],
        "avg_keypoint_confidence":
        report["keypoints"]["avg_confidence"],
        "max_raw_balls_per_frame":
        labels.get("max_raw_balls_per_frame", 0),
        "multi_ball_frame_ratio":
        labels.get("multi_ball_frame_ratio", 0.0),
        "false_ball_frame_ratio":
        labels.get("false_ball_frame_ratio", 0.0),
        "untracked_person_observations":
        report["person_tracks"]["untracked_person_observations"],
        "short_track_ratio":
        report["person_tracks"]["short_track_ratio"],
    }


def write_summary(outdir: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = outdir / "summary.csv"
    md_path = outdir / "summary.md"
    fieldnames = [
        "status",
        "video",
        "frames",
        "elapsed_sec",
        "total_score",
        "ball_present_ratio",
        "max_missing_streak",
        "avg_keypoint_confidence",
        "max_raw_balls_per_frame",
        "multi_ball_frame_ratio",
        "false_ball_frame_ratio",
        "untracked_person_observations",
        "short_track_ratio",
        "observations",
        "rendered_video",
        "report",
        "returncode",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Best Detector Batch Summary",
        "",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| status | video | frames | total | ball ratio | raw ball max | multi ball ratio | false ball ratio | keypoint conf | untracked persons | report | rendered video |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['status']} | {Path(str(row['video'])).name} | {row['frames']} | "
            f"{row['total_score']} | {row['ball_present_ratio']} | "
            f"{row['max_raw_balls_per_frame']} | {row['multi_ball_frame_ratio']} | "
            f"{row['false_ball_frame_ratio']} | "
            f"{row['avg_keypoint_confidence']} | {row['untracked_person_observations']} | "
            f"`{row['report']}` | `{row['rendered_video']}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    safe_print(f"[done] summary: {csv_path}")
    safe_print(f"[done] markdown: {md_path}")


def safe_name(value: str) -> str:
    return "".join(
        char if char.isascii() and (char.isalnum() or char in "-_") else "_"
        for char in value).strip("-_") or "video"


def short_hash(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8",
                                         errors="replace")).hexdigest()[:8]


def safe_print(value: object) -> None:
    text = str(value).encode("gbk", errors="replace").decode("gbk")
    print(text)


if __name__ == "__main__":
    main()
