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

from app_config_utils import merge_yaml


DEFAULT_APP_CONFIG = "config/app.yaml"
DEFAULT_ROI_CONFIG = "config/roi.json"
DEFAULT_VIDEO_LABELS = "config/video_labels.yaml"
DEFAULT_OUTPUT_ROOT = "tmp/det_resolution_compare"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RF-DETR detector input resolutions.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--config",
        default=DEFAULT_APP_CONFIG,
        help="Base app YAML config.",
    )
    parser.add_argument(
        "--roi-config",
        default=DEFAULT_ROI_CONFIG,
        help="Path to ROI config JSON.",
    )
    parser.add_argument(
        "--video-labels",
        default=DEFAULT_VIDEO_LABELS,
        help="Video labels YAML used by the evaluator.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory used to store all comparison runs.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Detector device, e.g. cpu, 0, cuda:0.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[640, 960],
        help="Detector input resolutions to compare.",
    )
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Process every Nth frame.",
    )
    sampling_group.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Desired processing fps; stride will be inferred from source fps when set.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N processed frames.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendered video output and write observations JSONL only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = Path(args.video)
    base_app_config = args.config
    roi_config = args.roi_config
    video_labels = args.video_labels
    output_root = Path(args.output_root)
    run_root = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for resolution in args.resolutions:
        run = build_run(
            video=video,
            run_root=run_root,
            base_app_config=base_app_config,
            roi_config=roi_config,
            video_labels=video_labels,
            device=args.device,
            resolution=resolution,
            stride=args.stride,
            target_fps=args.target_fps,
            max_frames=args.max_frames,
            no_render=args.no_render,
        )
        rows.append(execute_run(run, args.dry_run))

    write_summary(run_root, rows)


def build_run(
    video: Path,
    run_root: Path,
    base_app_config: str,
    roi_config: str,
    video_labels: str,
    device: str,
    resolution: int,
    stride: int | None,
    target_fps: float | None,
    max_frames: int | None,
    no_render: bool,
) -> dict[str, Any]:
    name = f"res_{resolution}"
    run_dir = run_root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    observation_path = run_dir / "observations.jsonl"
    report_json_path = run_dir / "quality_report.json"
    report_md_path = run_dir / "quality_report.md"
    rendered_video_path = run_dir / "rendered.mp4"
    app_config_path = run_dir / "app.yaml"
    command_path = run_dir / "command.txt"

    overrides = {"models": {"detector": {"input_resolution": resolution}}}
    merge_yaml(base_app_config, app_config_path, overrides)

    app_cmd = build_app_cmd(
        video=video,
        app_config=app_config_path,
        roi_config=roi_config,
        device=device,
        stride=stride,
        target_fps=target_fps,
        max_frames=max_frames,
        no_render=no_render,
    )
    report_cmd = build_report_cmd(
        observation_path=observation_path,
        report_json_path=report_json_path,
        report_md_path=report_md_path,
        video_labels=video_labels,
    )
    command_path.write_text(
        "\n".join([
            "# app command",
            " ".join(map(_quote, app_cmd)),
            "",
            "# report command",
            " ".join(map(_quote, report_cmd)),
            "",
        ]),
        encoding="utf-8",
    )

    return {
        "name": name,
        "resolution": resolution,
        "video": video,
        "run_dir": run_dir,
        "app_config_path": app_config_path,
        "observation_path": observation_path,
        "rendered_video_path": rendered_video_path,
        "report_json_path": report_json_path,
        "report_md_path": report_md_path,
        "command_path": command_path,
        "app_cmd": app_cmd,
        "report_cmd": report_cmd,
        "stride": stride,
        "target_fps": target_fps,
        "max_frames": max_frames,
    }


def build_app_cmd(
    video: Path,
    app_config: Path,
    roi_config: str,
    device: str,
    stride: int | None,
    target_fps: float | None,
    max_frames: int | None,
    no_render: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "code/app.py",
        "--video",
        str(video),
        "--output-dir",
        str(app_config.parent),
        "--config",
        str(app_config),
        "--device",
        str(device),
        "--use-roi",
    ]
    if no_render:
        cmd.append("--no-render")
    if target_fps is not None:
        cmd.extend(["--target-fps", str(target_fps)])
    else:
        cmd.extend(["--stride", str(stride if stride is not None else 1)])
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    return cmd


def build_report_cmd(
    observation_path: Path,
    report_json_path: Path,
    report_md_path: Path,
    video_labels: str,
) -> list[str]:
    return [
        sys.executable,
        "code/evaluate_observations.py",
        "--input",
        str(observation_path),
        "--output",
        str(report_json_path),
        "--markdown",
        str(report_md_path),
        "--video-labels",
        str(video_labels),
    ]


def execute_run(run: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    row = {
        "name": run["name"],
        "resolution": run["resolution"],
        "video": str(run["video"]),
        "status": "dry_run" if dry_run else "",
        "elapsed_sec": 0.0,
        "frames": "",
        "quality_score": "",
        "ball_present_ratio": "",
        "max_missing_streak": "",
        "false_ball_frame_ratio": "",
        "observations": str(run["observation_path"]),
        "rendered_video": str(run["rendered_video_path"]),
        "report": str(run["report_md_path"]),
        "command": str(run["command_path"]),
        "returncode": "",
    }
    if dry_run:
        print_run(run)
        return row

    print_run(run)
    started = time.perf_counter()
    try:
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


def load_report_metrics(report_json_path: Path) -> dict[str, Any]:
    report = json.loads(report_json_path.read_text(encoding="utf-8"))
    frame_stats = report.get("frame_stats", {})
    quality_score = report.get("quality_score", {})
    balls = report.get("balls", {})
    labels = report.get("labels", {})
    return {
        "frames": frame_stats.get("total_frames", ""),
        "quality_score": quality_score.get("total", ""),
        "ball_present_ratio": balls.get("ball_present_frame_ratio", ""),
        "max_missing_streak": balls.get("max_missing_streak", ""),
        "false_ball_frame_ratio": labels.get("false_ball_frame_ratio", ""),
    }


def write_summary(run_root: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = run_root / "summary.csv"
    md_path = run_root / "summary.md"
    fieldnames = [
        "name",
        "resolution",
        "status",
        "video",
        "frames",
        "quality_score",
        "ball_present_ratio",
        "max_missing_streak",
        "false_ball_frame_ratio",
        "elapsed_sec",
        "observations",
        "rendered_video",
        "report",
        "command",
        "returncode",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    lines = [
        "# Detector Resolution Compare",
        "",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| name | status | resolution | frames | score | ball present | max missing | false ball | elapsed | report | render |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['status']} | {row['resolution']} | {row.get('frames', '')} | "
            f"{row.get('quality_score', '')} | {row.get('ball_present_ratio', '')} | "
            f"{row.get('max_missing_streak', '')} | {row.get('false_ball_frame_ratio', '')} | "
            f"{row.get('elapsed_sec', '')} | `{row['report']}` | `{row['rendered_video']}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] summary: {csv_path}")
    print(f"[done] markdown: {md_path}")


def print_run(run: dict[str, Any]) -> None:
    print(f"[run] {run['name']} -> {run['run_dir']}")
    print(f"[app] {' '.join(map(_quote, run['app_cmd']))}")
    print(f"[report] {' '.join(map(_quote, run['report_cmd']))}")


def _quote(value: object) -> str:
    text = str(value)
    if not text or any(char.isspace() for char in text):
        return f'"{text}"'
    return text


if __name__ == "__main__":
    main()
