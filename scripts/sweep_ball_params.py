from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path


DEFAULT_COMBOS = [
    # 先从保守到激进跑几组，方便看球检测的召回率变化。
    {"name": "c025_i640_b010", "conf": 0.25, "imgsz": 640, "ball_conf": 0.10},
    {"name": "c015_i960_b010", "conf": 0.15, "imgsz": 960, "ball_conf": 0.10},
    {"name": "c010_i1280_b010", "conf": 0.10, "imgsz": 1280, "ball_conf": 0.10},
    {"name": "c010_i1280_b008", "conf": 0.10, "imgsz": 1280, "ball_conf": 0.08},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep detection parameters and collect observation quality reports."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--base-app-config",
        default="config/app.yaml",
        help="Base app YAML config used as a template.",
    )
    parser.add_argument(
        "--roi-config",
        default="config/roi.json",
        help="ROI config path.",
    )
    parser.add_argument(
        "--use-roi",
        action="store_true",
        help="Enable ROI filtering during each sweep run.",
    )
    parser.add_argument(
        "--outdir",
        default="tmp/ball_sweep",
        help="Output directory for all sweep runs.",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name.")
    parser.add_argument("--device", default="cuda:0", help="YOLO device.")
    parser.add_argument("--tracker", default="botsort.yaml", help="Tracker config.")
    parser.add_argument("--pose-config", default=None, help="Optional MMPose config path.")
    parser.add_argument("--pose-checkpoint", default=None, help="Optional MMPose checkpoint path.")
    parser.add_argument("--pose-device", default=None, help="Optional MMPose device.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick smoke runs.",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help="Only run the first N parameter combinations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    base_config_path = Path(args.base_app_config)
    roi_config_path = Path(args.roi_config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    combos = DEFAULT_COMBOS[: args.limit_runs] if args.limit_runs else DEFAULT_COMBOS
    rows: list[dict[str, object]] = []

    for combo in combos:
        run_dir = outdir / combo["name"]
        run_dir.mkdir(parents=True, exist_ok=True)

        app_config_path = run_dir / "app.yaml"
        observation_path = run_dir / "observations.jsonl"
        report_json_path = run_dir / "quality_report.json"
        report_md_path = run_dir / "quality_report.md"

        build_app_config(
            base_config_path=base_config_path,
            output_path=app_config_path,
            ball_min_confidence=float(combo["ball_conf"]),
        )

        run_app(
            video_path=video_path,
            app_config_path=app_config_path,
            roi_config_path=roi_config_path,
            observation_path=observation_path,
            model=args.model,
            device=args.device,
            tracker=args.tracker,
            pose_config=args.pose_config,
            pose_checkpoint=args.pose_checkpoint,
            pose_device=args.pose_device,
            conf=float(combo["conf"]),
            imgsz=int(combo["imgsz"]),
            ball_min_confidence=float(combo["ball_conf"]),
            max_frames=args.max_frames,
            use_roi=args.use_roi,
        )

        run_report(
            observation_path=observation_path,
            report_json_path=report_json_path,
            report_md_path=report_md_path,
        )

        report = json.loads(report_json_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": combo["name"],
                "conf": combo["conf"],
                "imgsz": combo["imgsz"],
                "ball_min_conf": combo["ball_conf"],
                "total_score": report["quality_score"]["total"],
                "ball_present_ratio": report["balls"]["ball_present_frame_ratio"],
                "max_missing_streak": report["balls"]["max_missing_streak"],
                "short_track_ratio": report["person_tracks"]["short_track_ratio"],
                "avg_keypoint_confidence": report["keypoints"]["avg_confidence"],
                "output_dir": str(run_dir),
            }
        )

    write_summary(outdir, rows)


def build_app_config(
    base_config_path: Path,
    output_path: Path,
    ball_min_confidence: float,
) -> None:
    yaml = load_yaml()
    # 直接复制基础配置并只改球相关阈值，避免把其他字段意外改掉。
    data = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML mapping: {base_config_path}")
    cloned = deepcopy(data)
    cloned["filters"]["ball"]["min_confidence"] = ball_min_confidence
    output_path.write_text(
        yaml.safe_dump(cloned, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - 运行时依赖缺失时直接报错
        raise RuntimeError(
            "PyYAML is required. Install it with `pip install PyYAML`."
        ) from exc
    return yaml


def run_app(
    video_path: Path,
    app_config_path: Path,
    roi_config_path: Path,
    observation_path: Path,
    model: str,
    device: str,
    tracker: str,
    pose_config: str | None,
    pose_checkpoint: str | None,
    pose_device: str | None,
    conf: float,
    imgsz: int,
    ball_min_confidence: float,
    max_frames: int | None,
    use_roi: bool,
) -> None:
    cmd = [
        sys.executable,
        "code/app.py",
        "--video",
        str(video_path),
        "--model",
        model,
        "--device",
        device,
        "--tracker",
        tracker,
        "--conf",
        str(conf),
        "--imgsz",
        str(imgsz),
        "--output-observations",
        str(observation_path),
        "--app-config",
        str(app_config_path),
        "--roi-config",
        str(roi_config_path),
    ]
    if use_roi:
        cmd.append("--use-roi")
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    if pose_config and pose_checkpoint:
        cmd.extend(["--pose-config", pose_config, "--pose-checkpoint", pose_checkpoint])
        if pose_device:
            cmd.extend(["--pose-device", pose_device])

    print(
        f"[run] {observation_path.parent.name}: conf={conf} imgsz={imgsz} "
        f"ball_min_conf={ball_min_confidence} use_roi={use_roi}"
    )
    subprocess.run(cmd, check=True)


def run_report(
    observation_path: Path,
    report_json_path: Path,
    report_md_path: Path,
) -> None:
    cmd = [
        sys.executable,
        "code/evaluate_observations.py",
        "--input",
        str(observation_path),
        "--output",
        str(report_json_path),
        "--markdown",
        str(report_md_path),
    ]
    subprocess.run(cmd, check=True)


def write_summary(outdir: Path, rows: list[dict[str, object]]) -> None:
    csv_path = outdir / "summary.csv"
    md_path = outdir / "summary.md"

    fieldnames = [
        "name",
        "conf",
        "imgsz",
        "ball_min_conf",
        "total_score",
        "ball_present_ratio",
        "max_missing_streak",
        "short_track_ratio",
        "avg_keypoint_confidence",
        "output_dir",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Sweep Summary",
        "",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| name | conf | imgsz | ball min conf | total score | ball present ratio | max missing streak | short track ratio | avg keypoint conf |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['conf']} | {row['imgsz']} | {row['ball_min_conf']} | {row['total_score']} | {row['ball_present_ratio']} | {row['max_missing_streak']} | {row['short_track_ratio']} | {row['avg_keypoint_confidence']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] summary saved to {csv_path}")


if __name__ == "__main__":
    main()
