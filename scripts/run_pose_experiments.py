from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run configurable pose experiments from a YAML file.")
    parser.add_argument(
        "--config",
        default="config/pose_experiments.yaml",
        help="Experiment YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_root = Path(config["output_root"])
    prefix = "dry_run_" if args.dry_run else ""
    run_root = output_root / f"{prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)

    base_app_config = config["base_app_config"]
    defaults = config.get("defaults", {})
    video = Path(config["video"])
    roi_config = config["roi_config"]
    video_labels = config.get("video_labels", "config/video_labels.yaml")
    experiments = config.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments configured.")

    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        run = build_run(
            video=video,
            run_root=run_root,
            defaults=defaults,
            base_app_config=base_app_config,
            roi_config=roi_config,
            video_labels=video_labels,
            experiment=experiment,
        )
        rows.append(execute_run(run, args.dry_run))

    write_summary(run_root, rows)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read experiment config.") from exc
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Experiment config must be a mapping: {config_path}")
    return data


def build_run(
    video: Path,
    run_root: Path,
    defaults: dict[str, Any],
    base_app_config: str,
    roi_config: str,
    video_labels: str,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    name = str(experiment["name"])
    run_dir = run_root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    observation_path = run_dir / "observations.jsonl"
    report_json_path = run_dir / "quality_report.json"
    report_md_path = run_dir / "quality_report.md"
    output_video = bool(experiment.get("output_video",
                                       defaults.get("output_video", True)))
    rendered_video_path = run_dir / "rendered.mp4" if output_video else None
    app_config_path = run_dir / "app.yaml"
    command_path = run_dir / "command.txt"

    merged_app_config = merge_app_config(
        base_app_config,
        experiment.get("app_overrides", {}),
        app_config_path,
    )
    app_cmd = build_app_cmd(
        video=video,
        observation_path=observation_path,
        rendered_video_path=rendered_video_path,
        app_config=app_config_path,
        roi_config=roi_config,
        defaults=defaults,
        experiment=experiment,
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
        "video": video,
        "run_dir": run_dir,
        "app_config_path": app_config_path,
        "merged_app_config": merged_app_config,
        "observation_path": observation_path,
        "rendered_video_path": rendered_video_path,
        "report_json_path": report_json_path,
        "report_md_path": report_md_path,
        "command_path": command_path,
        "app_cmd": app_cmd,
        "report_cmd": report_cmd,
        "pose_model": experiment["pose_model"],
        "stride": experiment["stride"],
        "target_fps": experiment.get("target_fps"),
        "max_frames": experiment.get("max_frames"),
    }


def merge_app_config(base_app_config: str, overrides: dict[str, Any],
                     output_path: Path) -> dict[str, Any]:
    base = load_config(base_app_config)
    merged = deepcopy(base)
    deep_update(merged, overrides)
    write_yaml(merged, output_path)
    return merged


def build_app_cmd(
    video: Path,
    observation_path: Path,
    rendered_video_path: Path,
    app_config: Path,
    roi_config: str,
    defaults: dict[str, Any],
    experiment: dict[str, Any],
) -> list[str]:
    cmd = [
        sys.executable,
        "code/app.py",
        "--video",
        str(video),
        "--detector",
        str(experiment.get("detector", defaults.get("detector", "rfdetr"))),
        "--conf",
        str(experiment.get("conf", defaults.get("conf", 0.25))),
        "--device",
        str(experiment.get("device", defaults.get("device", "cuda:0"))),
        "--stride",
        str(experiment["stride"]),
        "--pose-model",
        str(experiment["pose_model"]),
        "--output-observations",
        str(observation_path),
        "--app-config",
        str(app_config),
        "--roi-config",
        str(roi_config),
    ]
    if experiment.get("target_fps") is not None:
        cmd.extend(["--target-fps", str(experiment["target_fps"])])
    if rendered_video_path is not None:
        cmd.extend(["--output-video", str(rendered_video_path)])
    if experiment.get("use_roi", defaults.get("use_roi", True)):
        cmd.append("--use-roi")
    if experiment.get("max_frames") is not None:
        cmd.extend(["--max-frames", str(experiment["max_frames"])])
    pose_device = experiment.get("pose_device", defaults.get("pose_device"))
    if pose_device:
        cmd.extend(["--pose-device", str(pose_device)])

    detector = str(experiment.get("detector", defaults.get("detector", "rfdetr")))
    if detector == "rfdetr":
        cmd.extend([
            "--rfdetr-size",
            str(experiment.get("rfdetr_size", defaults.get("rfdetr_size", "small"))),
            "--model-dir",
            str(experiment.get("model_dir", defaults.get("model_dir", "models/rfdetr"))),
        ])
    else:
        cmd.extend([
            "--model",
            str(experiment.get("yolo_model", defaults.get("yolo_model", "models/yolo/yolov8n.pt"))),
            "--imgsz",
            str(experiment.get("imgsz", defaults.get("imgsz", 640))),
        ])
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
        "video": str(run["video"]),
        "pose_model": run["pose_model"],
        "stride": run["stride"],
        "target_fps": run.get("target_fps", ""),
        "max_frames": run["max_frames"],
        "status": "dry_run" if dry_run else "",
        "elapsed_sec": 0.0,
        "observations": str(run["observation_path"]),
        "rendered_video": str(run["rendered_video_path"])
        if run["rendered_video_path"] is not None else "",
        "report": str(run["report_md_path"]),
        "command": str(run["command_path"]),
        "quality_score": 0.0,
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
    return {
        "frames": report["frame_stats"]["total_frames"],
        "quality_score": report["quality_score"]["total"],
        "avg_keypoint_confidence": report["keypoints"]["avg_confidence"],
        "ball_present_ratio": report["balls"]["ball_present_frame_ratio"],
        "short_track_ratio": report["person_tracks"]["short_track_ratio"],
    }


def write_summary(run_root: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = run_root / "summary.csv"
    md_path = run_root / "summary.md"
    fieldnames = [
        "name",
        "status",
        "video",
        "pose_model",
        "stride",
        "target_fps",
        "max_frames",
        "frames",
        "quality_score",
        "avg_keypoint_confidence",
        "ball_present_ratio",
        "short_track_ratio",
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
        "# Pose Experiment Summary",
        "",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| name | status | pose | stride | max_frames | frames | score | report | render |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['status']} | {row['pose_model']} | {row['stride']} | "
            f"{row['max_frames']} | {row.get('frames', '')} | {row.get('quality_score', '')} | "
            f"`{row['report']}` | `{row['rendered_video']}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] summary: {csv_path}")
    print(f"[done] markdown: {md_path}")


def deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def write_yaml(data: dict[str, Any], path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write experiment config.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


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
