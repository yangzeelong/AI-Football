from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DETECTORS = ["yolo", "rfdetr:nano", "rfdetr:small", "rfdetr:base"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detector choices with the existing task-1 pipeline."
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Input videos to evaluate.",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=DEFAULT_DETECTORS,
        help="Detector specs, e.g. yolo rfdetr:nano rfdetr:small rfdetr:base.",
    )
    parser.add_argument(
        "--outdir",
        default="tmp/detector_eval",
        help="Directory for JSONL, reports, and summary files.",
    )
    parser.add_argument(
        "--app-config",
        default="config/app.yaml",
        help="App filtering config YAML.",
    )
    parser.add_argument(
        "--roi-config",
        default="config/roi.json",
        help="ROI config JSON.",
    )
    parser.add_argument("--use-roi", action="store_true", help="Enable ROI filtering.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Frames per run. Use 0 to process the whole video.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size.")
    parser.add_argument(
        "--ball-jump-px",
        type=float,
        default=150.0,
        help="Ball center jump threshold in pixels for summary diagnostics.",
    )
    parser.add_argument(
        "--yolo-model",
        default="models/yolo/yolov8n.pt",
        help="YOLO model path.",
    )
    parser.add_argument(
        "--rfdetr-model-dir",
        default="models/rfdetr",
        help="RF-DETR weights directory.",
    )
    parser.add_argument("--device", default="cuda:0", help="Detector device.")
    parser.add_argument(
        "--pose-model",
        choices=["rtmpose-m", "hrnet-w32", "hrnet-w48-dark"],
        default="rtmpose-m",
        help="MMPose preset used by code/app.py.",
    )
    parser.add_argument(
        "--pose-config",
        default=None,
        help="Optional MMPose config override. App default is used if omitted.",
    )
    parser.add_argument(
        "--pose-checkpoint",
        default=None,
        help="Optional MMPose checkpoint override. App default is used if omitted.",
    )
    parser.add_argument(
        "--pose-device",
        default=None,
        help="Optional MMPose device. Defaults to --device inside app.py.",
    )
    parser.add_argument(
        "--camera-id",
        default="C1",
        help="Camera ID written to each observation file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for video in [Path(path) for path in args.videos]:
        for detector_spec in args.detectors:
            run = build_run(
                args=args,
                video=video,
                detector_spec=detector_spec,
                outdir=outdir,
            )
            rows.append(
                execute_run(run, dry_run=args.dry_run, ball_jump_px=args.ball_jump_px)
            )

    if not args.dry_run:
        write_summary(outdir, rows)


def build_run(
    args: argparse.Namespace,
    video: Path,
    detector_spec: str,
    outdir: Path,
) -> dict[str, Any]:
    detector, size = parse_detector_spec(detector_spec)
    video_name = safe_name(video.stem)
    detector_name = detector if size is None else f"{detector}_{size}"
    run_dir = outdir / video_name / detector_name
    observation_path = run_dir / "observations.jsonl"
    report_json_path = run_dir / "quality_report.json"
    report_md_path = run_dir / "quality_report.md"

    app_cmd = [
        sys.executable,
        "code/app.py",
        "--video",
        str(video),
        "--detector",
        detector,
        "--conf",
        str(args.conf),
        "--device",
        args.device,
        "--stride",
        str(args.stride),
        "--pose-model",
        args.pose_model,
        "--camera-id",
        args.camera_id,
        "--output-observations",
        str(observation_path),
        "--app-config",
        args.app_config,
        "--roi-config",
        args.roi_config,
    ]
    if args.use_roi:
        app_cmd.append("--use-roi")
    if args.max_frames and args.max_frames > 0:
        app_cmd.extend(["--max-frames", str(args.max_frames)])
    if detector == "yolo":
        app_cmd.extend(["--model", args.yolo_model, "--imgsz", str(args.imgsz)])
    else:
        app_cmd.extend([
            "--rfdetr-size",
            size or "base",
            "--model-dir",
            args.rfdetr_model_dir,
        ])
    if args.pose_config:
        app_cmd.extend(["--pose-config", args.pose_config])
    if args.pose_checkpoint:
        app_cmd.extend(["--pose-checkpoint", args.pose_checkpoint])
    if args.pose_device:
        app_cmd.extend(["--pose-device", args.pose_device])

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
        "detector": detector,
        "size": size or "",
        "run_dir": run_dir,
        "observation_path": observation_path,
        "report_json_path": report_json_path,
        "report_md_path": report_md_path,
        "app_cmd": app_cmd,
        "report_cmd": report_cmd,
    }


def execute_run(
    run: dict[str, Any],
    dry_run: bool,
    ball_jump_px: float,
) -> dict[str, Any]:
    run["run_dir"].mkdir(parents=True, exist_ok=True)
    print(f"[run] {run['video']} -> {run['detector']} {run['size']}".rstrip())
    print(" ".join(str(part) for part in run["app_cmd"]))

    if dry_run:
        return base_row(run, status="dry_run")

    started = time.perf_counter()
    try:
        # 先生成任务1正式 JSONL，再用现有质量脚本生成可比较报告。
        subprocess.run(run["app_cmd"], check=True)
        subprocess.run(run["report_cmd"], check=True)
    except subprocess.CalledProcessError as exc:
        row = base_row(run, status="failed")
        row["returncode"] = exc.returncode
        row["elapsed_sec"] = round(time.perf_counter() - started, 3)
        return row

    elapsed_sec = round(time.perf_counter() - started, 3)
    report = json.loads(run["report_json_path"].read_text(encoding="utf-8"))
    ball_jumps = compute_ball_jumps(run["observation_path"], ball_jump_px)
    row = base_row(run, status="ok")
    row.update(flatten_report(report))
    row.update(ball_jumps)
    row["elapsed_sec"] = elapsed_sec
    row["fps_processed"] = round(row["frames"] / elapsed_sec, 3) if elapsed_sec else 0.0
    return row


def parse_detector_spec(spec: str) -> tuple[str, str | None]:
    if spec == "yolo":
        return "yolo", None
    if spec.startswith("rfdetr:"):
        _, size = spec.split(":", 1)
        if size not in {"nano", "small", "base", "medium", "large"}:
            raise ValueError(f"Unsupported RF-DETR size in detector spec: {spec}")
        return "rfdetr", size
    raise ValueError(f"Unsupported detector spec: {spec}")


def base_row(run: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "status": status,
        "video": str(run["video"]),
        "detector": run["detector"],
        "size": run["size"],
        "frames": 0,
        "elapsed_sec": 0.0,
        "fps_processed": 0.0,
        "total_score": 0.0,
        "keypoint_score": 0.0,
        "person_tracking_score": 0.0,
        "ball_tracking_score": 0.0,
        "stability_score": 0.0,
        "ball_present_ratio": 0.0,
        "ball_speed_p95": 0.0,
        "ball_jump_count": 0,
        "ball_jump_px_max": 0.0,
        "max_missing_streak": 0,
        "avg_keypoint_confidence": 0.0,
        "keypoint_missing_ratio": 0.0,
        "untracked_person_observations": 0,
        "short_track_ratio": 0.0,
        "report": str(run["report_md_path"]),
        "observations": str(run["observation_path"]),
        "returncode": "",
    }


def flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    score = report["quality_score"]
    frame_stats = report["frame_stats"]
    person_tracks = report["person_tracks"]
    keypoints = report["keypoints"]
    balls = report["balls"]
    return {
        "frames": frame_stats["total_frames"],
        "total_score": score["total"],
        "keypoint_score": score["keypoint"],
        "person_tracking_score": score["person_tracking"],
        "ball_tracking_score": score["ball_tracking"],
        "stability_score": score["stability"],
        "ball_present_ratio": balls["ball_present_frame_ratio"],
        "ball_speed_p95": balls["speed_px_s_p95"],
        "max_missing_streak": balls["max_missing_streak"],
        "avg_keypoint_confidence": keypoints["avg_confidence"],
        "keypoint_missing_ratio": keypoints["missing_ratio"],
        "untracked_person_observations": person_tracks["untracked_person_observations"],
        "short_track_ratio": person_tracks["short_track_ratio"],
    }


def compute_ball_jumps(
    observation_path: Path,
    jump_threshold_px: float,
) -> dict[str, Any]:
    previous: tuple[float, float] | None = None
    jump_count = 0
    max_jump = 0.0

    with observation_path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record.get("type") != "frame":
                continue
            balls = record.get("balls", [])
            if not balls:
                previous = None
                continue
            center = balls[0].get("center")
            if center is None or center[0] is None or center[1] is None:
                previous = None
                continue
            current = (float(center[0]), float(center[1]))
            if previous is not None:
                jump = ((current[0] - previous[0]) ** 2 +
                        (current[1] - previous[1]) ** 2) ** 0.5
                max_jump = max(max_jump, jump)
                if jump > jump_threshold_px:
                    jump_count += 1
            previous = current

    return {
        "ball_jump_count": jump_count,
        "ball_jump_px_max": round(max_jump, 3),
    }


def write_summary(outdir: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = outdir / "summary.csv"
    md_path = outdir / "summary.md"
    fieldnames = [
        "status",
        "video",
        "detector",
        "size",
        "frames",
        "elapsed_sec",
        "fps_processed",
        "total_score",
        "keypoint_score",
        "person_tracking_score",
        "ball_tracking_score",
        "stability_score",
        "ball_present_ratio",
        "ball_speed_p95",
        "ball_jump_count",
        "ball_jump_px_max",
        "max_missing_streak",
        "avg_keypoint_confidence",
        "keypoint_missing_ratio",
        "untracked_person_observations",
        "short_track_ratio",
        "report",
        "observations",
        "returncode",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Detector Evaluation Summary",
        "",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| status | video | detector | size | frames | fps | total | ball ratio | ball jumps | max jump px | keypoint conf | untracked persons | report |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['status']} | {Path(str(row['video'])).name} | {row['detector']} | "
            f"{row['size']} | {row['frames']} | {row['fps_processed']} | "
            f"{row['total_score']} | {row['ball_present_ratio']} | "
            f"{row['ball_jump_count']} | {row['ball_jump_px_max']} | "
            f"{row['avg_keypoint_confidence']} | {row['untracked_person_observations']} | "
            f"`{row['report']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] summary: {csv_path}")
    print(f"[done] markdown: {md_path}")


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


if __name__ == "__main__":
    main()
