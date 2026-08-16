from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep detector/app thresholds with video labels as GT.")
    parser.add_argument("--data-dir",
                        default="data",
                        help="Directory to scan for videos.")
    parser.add_argument("--outdir",
                        default="tmp/threshold_sweep",
                        help="Output directory for sweep runs.")
    parser.add_argument("--video-labels",
                        default="config/video_labels.yaml",
                        help="GT-like video label YAML.")
    parser.add_argument("--base-app-config",
                        default="config/app.yaml",
                        help="Base app YAML config.")
    parser.add_argument("--roi-config",
                        default="config/roi.json",
                        help="ROI config JSON.")
    parser.add_argument("--no-roi",
                        action="store_true",
                        help="Disable ROI filtering.")
    parser.add_argument("--include-ignored",
                        action="store_true",
                        help="Include videos marked ignore: true in labels.")
    parser.add_argument("--detector",
                        choices=["rfdetr", "yolo"],
                        default="rfdetr",
                        help="Detector backend.")
    parser.add_argument("--rfdetr-size",
                        choices=["nano", "small", "base", "medium", "large"],
                        default="small",
                        help="RF-DETR model size.")
    parser.add_argument("--model-dir",
                        default="models/rfdetr",
                        help="RF-DETR weight directory.")
    parser.add_argument("--yolo-model",
                        default="models/yolo/yolov8n.pt",
                        help="YOLO model path.")
    parser.add_argument("--imgsz",
                        type=int,
                        default=640,
                        help="YOLO image size.")
    parser.add_argument("--device", default="cuda:0", help="Detector device.")
    parser.add_argument("--pose-device", default=None, help="MMPose device.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--max-frames",
                        type=int,
                        default=None,
                        help="Optional frame limit per video.")
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Save a rendered MP4 for each sweep run.",
    )
    parser.add_argument(
        "--conf-values",
        default="0.25,0.35,0.45,0.55",
        help="Comma-separated detector confidence values.",
    )
    parser.add_argument(
        "--ball-conf-values",
        default="0.10,0.20,0.30,0.40,0.50",
        help="Comma-separated app ball min_confidence values.",
    )
    parser.add_argument("--limit-videos",
                        type=int,
                        default=None,
                        help="Only run first N discovered videos.")
    parser.add_argument("--limit-runs",
                        type=int,
                        default=None,
                        help="Only run first N threshold combinations.")
    parser.add_argument("--skip-existing",
                        action="store_true",
                        help="Reuse runs with existing reports.")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Print commands without running them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_video_labels(args.video_labels)
    videos = filter_videos(discover_videos(Path(args.data_dir)), labels,
                           args.include_ignored)
    if args.limit_videos is not None:
        videos = videos[:args.limit_videos]
    if not videos:
        raise FileNotFoundError(f"No videos found under {args.data_dir}")

    combos = build_combos(args)
    if args.limit_runs is not None:
        combos = combos[:args.limit_runs]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for combo in combos:
        for video in videos:
            run = build_run(args, video, combo, labels, outdir)
            rows.append(execute_run(run, args.dry_run, args.skip_existing))

    if not args.dry_run:
        write_summary(outdir, rows)


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
    if not isinstance(data, dict):
        raise ValueError(f"Invalid video label YAML: {label_path}")
    videos = data["videos"]
    if not isinstance(videos, dict):
        raise ValueError(f"Invalid `videos` mapping in: {label_path}")
    return videos


def discover_videos(data_dir: Path) -> list[Path]:
    return sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def filter_videos(
    videos: list[Path],
    labels: dict[str, Any],
    include_ignored: bool,
) -> list[Path]:
    if include_ignored:
        return videos
    return [
        video for video in videos
        if not labels.get(video.name, {}).get("ignore", False)
    ]


def build_combos(args: argparse.Namespace) -> list[dict[str, float]]:
    conf_values = parse_float_list(args.conf_values, "--conf-values")
    ball_conf_values = parse_float_list(args.ball_conf_values,
                                        "--ball-conf-values")
    return [
        {
            "conf": detector_conf,
            "ball_conf": ball_conf,
        }
        for detector_conf in conf_values
        for ball_conf in ball_conf_values
    ]


def parse_float_list(raw: str, name: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one number")
    return values


def build_run(
    args: argparse.Namespace,
    video: Path,
    combo: dict[str, float],
    labels: dict[str, Any],
    outdir: Path,
) -> dict[str, Any]:
    combo_name = f"det{combo['conf']:.2f}_ball{combo['ball_conf']:.2f}"
    run_name = f"{safe_name(video.stem)}_{short_hash(video)}"
    run_dir = outdir / combo_name / run_name
    app_config_path = run_dir / "app.yaml"
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
        str(combo["conf"]),
        "--device",
        args.device,
        "--stride",
        str(args.stride),
        "--output-observations",
        str(observation_path),
        "--app-config",
        str(app_config_path),
        "--roi-config",
        args.roi_config,
    ]
    if args.render_video:
        app_cmd.extend(["--output-video", str(rendered_video_path)])
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
        app_cmd.extend([
            "--model",
            args.yolo_model,
            "--imgsz",
            str(args.imgsz),
        ])

    report_cmd = [
        sys.executable,
        "code/evaluate_observations.py",
        "--input",
        str(observation_path),
        "--output",
        str(report_json_path),
        "--markdown",
        str(report_md_path),
        "--video-labels",
        args.video_labels,
    ]

    return {
        "video": video,
        "video_label": labels.get(video.name, {}),
        "combo": combo,
        "run_dir": run_dir,
        "app_config_path": app_config_path,
        "observation_path": observation_path,
        "report_json_path": report_json_path,
        "report_md_path": report_md_path,
        "rendered_video_path": rendered_video_path if args.render_video else None,
        "app_cmd": app_cmd,
        "report_cmd": report_cmd,
        "base_app_config": Path(args.base_app_config),
    }


def execute_run(
    run: dict[str, Any],
    dry_run: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    run["run_dir"].mkdir(parents=True, exist_ok=True)
    write_app_config(run["base_app_config"], run["app_config_path"],
                     float(run["combo"]["ball_conf"]))

    print(f"[run] {run['run_dir']}")
    if dry_run:
        print("[dry-run] " + " ".join(str(part) for part in run["app_cmd"]))
        return base_row(run, status="dry_run")

    if not (skip_existing and run_outputs_exist(run)):
        subprocess.run(run["app_cmd"], check=True)
        subprocess.run(run["report_cmd"], check=True)

    return row_from_report(run)


def run_outputs_exist(run: dict[str, Any]) -> bool:
    rendered_video_path = run["rendered_video_path"]
    return (
        run["report_json_path"].exists()
        and (rendered_video_path is None or rendered_video_path.exists())
    )


def write_app_config(
    base_app_config: Path,
    output_path: Path,
    ball_min_confidence: float,
) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write app configs.") from exc

    raw = yaml.safe_load(base_app_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML mapping: {base_app_config}")
    # 只覆盖球过滤阈值，其余检测/过滤配置完全沿用当前 app.yaml。
    cloned = deepcopy(raw)
    cloned["filters"]["ball"]["min_confidence"] = ball_min_confidence
    output_path.write_text(
        yaml.safe_dump(cloned, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def row_from_report(run: dict[str, Any]) -> dict[str, Any]:
    if not run["report_json_path"].exists():
        return base_row(run, status="missing_report")
    report = json.loads(run["report_json_path"].read_text(encoding="utf-8"))
    row = base_row(run, status="ok")
    score = report["quality_score"]
    labels = report["labels"]
    balls = report["balls"]
    person_tracks = report["person_tracks"]
    keypoints = report["keypoints"]
    row.update({
        "frames": report["frame_stats"]["total_frames"],
        "total_score": score["total"],
        "ball_score": score["ball_tracking"],
        "person_score": score["person_tracking"],
        "keypoint_score": score["keypoint"],
        "expected_balls": labels.get("expected_balls"),
        "ball_exact_ratio": labels.get("raw_ball_exact_count_frame_ratio"),
        "ball_in_range_ratio": labels.get("raw_ball_in_expected_range_frame_ratio"),
        "false_ball_ratio": labels.get("false_ball_frame_ratio"),
        "multi_ball_ratio": labels.get("multi_ball_frame_ratio"),
        "avg_raw_balls": labels.get("avg_raw_balls_per_frame"),
        "max_raw_balls": labels.get("max_raw_balls_per_frame"),
        "ball_present_ratio": balls["ball_present_frame_ratio"],
        "expected_persons_min": labels.get("expected_persons_min"),
        "expected_persons_max": labels.get("expected_persons_max"),
        "person_in_range_ratio": labels.get(
            "raw_person_in_expected_range_frame_ratio"),
        "short_track_ratio": person_tracks["short_track_ratio"],
        "avg_keypoint_confidence": keypoints["avg_confidence"],
    })
    return row


def base_row(run: dict[str, Any], status: str) -> dict[str, Any]:
    label = run["video_label"]
    return {
        "status": status,
        "video": str(run["video"]),
        "scenario": label.get("scenario"),
        "detector_conf": run["combo"]["conf"],
        "ball_min_confidence": run["combo"]["ball_conf"],
        "frames": None,
        "total_score": None,
        "ball_score": None,
        "person_score": None,
        "keypoint_score": None,
        "expected_balls": label.get("expected_balls"),
        "ball_exact_ratio": None,
        "ball_in_range_ratio": None,
        "false_ball_ratio": None,
        "multi_ball_ratio": None,
        "avg_raw_balls": None,
        "max_raw_balls": None,
        "ball_present_ratio": None,
        "expected_persons_min": label.get("expected_persons_min"),
        "expected_persons_max": label.get("expected_persons_max"),
        "person_in_range_ratio": None,
        "short_track_ratio": None,
        "avg_keypoint_confidence": None,
        "report": str(run["report_md_path"]),
        "observations": str(run["observation_path"]),
        "rendered_video": (
            str(run["rendered_video_path"])
            if run["rendered_video_path"] is not None else None
        ),
    }


def write_summary(outdir: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = outdir / "summary.csv"
    combo_csv_path = outdir / "combo_summary.csv"
    md_path = outdir / "summary.md"
    fieldnames = list(base_row_for_fieldnames())

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    combo_rows = aggregate_combo_rows(rows)
    with combo_csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(combo_rows[0]) if combo_rows else [])
        if combo_rows:
            writer.writeheader()
            writer.writerows(combo_rows)

    ranked = sorted(
        [row for row in rows if row["status"] == "ok"],
        key=lambda row: (
            float(row["total_score"] or 0),
            float(row["ball_score"] or 0),
            -float(row["false_ball_ratio"] or 0),
        ),
        reverse=True,
    )

    lines = [
        "# Threshold Sweep Summary",
        "",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- runs: {len(rows)}",
        "",
        "## Combo Ranking",
        "",
        "| rank | det conf | ball conf | runs | avg total | avg ball | avg exact | avg false | avg multi |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(combo_rows[:20], start=1):
        lines.append(
            f"| {index} | {row['detector_conf']} | {row['ball_min_confidence']} | "
            f"{row['ok_runs']} | {row['avg_total_score']} | {row['avg_ball_score']} | "
            f"{row['avg_ball_exact_ratio']} | {row['avg_false_ball_ratio']} | "
            f"{row['avg_multi_ball_ratio']} |")

    lines.extend([
        "",
        "## Top Runs",
        "",
        "| rank | video | scenario | det conf | ball conf | total | ball | exact | false | multi | report | rendered |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for index, row in enumerate(ranked[:20], start=1):
        lines.append(
            f"| {index} | {Path(str(row['video'])).name} | {row['scenario']} | "
            f"{row['detector_conf']} | {row['ball_min_confidence']} | "
            f"{row['total_score']} | {row['ball_score']} | "
            f"{row['ball_exact_ratio']} | {row['false_ball_ratio']} | "
            f"{row['multi_ball_ratio']} | `{row['report']}` | "
            f"`{row['rendered_video']}` |")

    lines.extend([
        "",
        "## All Runs",
        "",
        f"Full CSV: `{csv_path}`",
        f"Combo CSV: `{combo_csv_path}`",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] summary: {csv_path}")


def aggregate_combo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (float(row["detector_conf"]), float(row["ball_min_confidence"]))
        groups.setdefault(key, []).append(row)

    combo_rows = []
    for (detector_conf, ball_conf), group_rows in groups.items():
        combo_rows.append({
            "detector_conf": detector_conf,
            "ball_min_confidence": ball_conf,
            "ok_runs": len(group_rows),
            "avg_total_score": round(avg_metric(group_rows, "total_score"), 2),
            "avg_ball_score": round(avg_metric(group_rows, "ball_score"), 2),
            "avg_keypoint_score": round(avg_metric(group_rows, "keypoint_score"), 2),
            "avg_person_score": round(avg_metric(group_rows, "person_score"), 2),
            "avg_ball_exact_ratio": round(avg_metric(group_rows, "ball_exact_ratio"), 4),
            "avg_ball_in_range_ratio": round(
                avg_metric(group_rows, "ball_in_range_ratio"), 4),
            "avg_false_ball_ratio": round(avg_metric(group_rows, "false_ball_ratio"), 4),
            "avg_multi_ball_ratio": round(avg_metric(group_rows, "multi_ball_ratio"), 4),
            "avg_raw_balls": round(avg_metric(group_rows, "avg_raw_balls"), 3),
            "max_raw_balls": max(
                int(row["max_raw_balls"] or 0) for row in group_rows),
        })
    return sorted(
        combo_rows,
        key=lambda row: (
            float(row["avg_total_score"]),
            float(row["avg_ball_score"]),
            -float(row["avg_false_ball_ratio"]),
            -float(row["avg_multi_ball_ratio"]),
        ),
        reverse=True,
    )


def avg_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        float(row[key]) for row in rows
        if row.get(key) is not None
    ]
    return sum(values) / len(values) if values else 0.0


def base_row_for_fieldnames() -> dict[str, Any]:
    return {
        "status": None,
        "video": None,
        "scenario": None,
        "detector_conf": None,
        "ball_min_confidence": None,
        "frames": None,
        "total_score": None,
        "ball_score": None,
        "person_score": None,
        "keypoint_score": None,
        "expected_balls": None,
        "ball_exact_ratio": None,
        "ball_in_range_ratio": None,
        "false_ball_ratio": None,
        "multi_ball_ratio": None,
        "avg_raw_balls": None,
        "max_raw_balls": None,
        "ball_present_ratio": None,
        "expected_persons_min": None,
        "expected_persons_max": None,
        "person_in_range_ratio": None,
        "short_track_ratio": None,
        "avg_keypoint_confidence": None,
        "report": None,
        "observations": None,
        "rendered_video": None,
    }


def safe_name(value: str) -> str:
    normalized = []
    for char in value.lower():
        if char.isascii() and (char.isalnum() or char in "-_"):
            normalized.append(char)
        elif char in " .":
            normalized.append("_")
    return "".join(normalized).strip("_-") or "video"


def short_hash(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]


if __name__ == "__main__":
    main()
