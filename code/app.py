from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

from football_vision import (
    RfdetrDetector,
    RoiAnnotator,
    RoiManager,
    YoloDetector,
)
from app_config import AppConfig
from app_pipeline import run_video_app
from pose_estimation import MMPoseTopDownEstimator


DEFAULT_ROI_CONFIG = "config/roi.json"
DEFAULT_APP_CONFIG = "config/app.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI football video analysis prototype.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store rendered video, observations, and run metadata.",
    )
    parser.add_argument("--roi-config",
                        default=DEFAULT_ROI_CONFIG,
                        help="Path to ROI config JSON.")
    parser.add_argument("--config",
                        default=DEFAULT_APP_CONFIG,
                        help="Path to app YAML config.")

    parser.add_argument("--draw-roi",
                        action="store_true",
                        help="Open an interactive ROI annotation window.")
    parser.add_argument("--roi-frame",
                        type=int,
                        default=0,
                        help="Frame index used for ROI annotation.")
    parser.add_argument("--use-roi",
                        action="store_true",
                        help="Filter detections by saved ROI.")

    parser.add_argument("--show",
                        action="store_true",
                        help="Show rendered detections with OpenCV.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendered video output and write observations JSONL only.",
    )
    parser.add_argument("--camera-id",
                        default="C1",
                        help="Camera ID stored in observation output.")
    parser.add_argument("--display-width",
                        type=int,
                        default=1920,
                        help="OpenCV preview window width.")
    parser.add_argument("--display-height",
                        type=int,
                        default=1080,
                        help="OpenCV preview window height.")
    parser.add_argument("--device", default="cuda:0", help="Detector device, e.g. cpu, 0, cuda:0.")
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument("--stride",
                                type=int,
                                default=None,
                                help="Process every Nth frame.")
    sampling_group.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Desired processing fps; stride will be inferred from source fps when set.",
    )
    parser.add_argument("--max-frames",
                        type=int,
                        default=None,
                        help="Stop after N processed frames.")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Overlay per-frame debug details on the rendered video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    roi_manager = RoiManager(args.roi_config)
    app_config = AppConfig.from_yaml(args.config)

    if args.draw_roi:
        annotator = RoiAnnotator(roi_manager)
        config = annotator.annotate(video_path, frame_index=args.roi_frame)
        logger.info("saved ROI: video={} points={} path={}", config.video_key,
                    len(config.points), args.roi_config)
        return

    if args.output_dir is None:
        raise ValueError("--output-dir is required when running video analysis.")

    stride, target_fps = _normalize_sampling_args(args.stride, args.target_fps)
    output_dir, output_observations, output_video = _resolve_output_paths(args)
    _save_run_metadata(output_dir, args, app_config, output_observations,
                       output_video, stride, target_fps)

    detector = build_detector(app_config, args.device)
    pose_config, pose_checkpoint = resolve_pose_paths(app_config)
    pose_estimator = MMPoseTopDownEstimator(
        config_path=pose_config,
        checkpoint_path=pose_checkpoint,
        device=args.device,
        flip_test=app_config.pose.flip_test,
    )
    run_video_app(
        video_path=video_path,
        detector=detector,
        roi_manager=roi_manager,
        output_observations=str(output_observations),
        camera_id=args.camera_id,
        use_roi=args.use_roi,
        show=args.show,
        stride=stride,
        target_fps=target_fps,
        max_frames=args.max_frames,
        display_width=args.display_width,
        display_height=args.display_height,
        output_video=str(output_video) if output_video is not None else None,
        tracker=app_config.runtime.tracker,
        pose_estimator=pose_estimator,
        app_config=app_config,
        debug=args.debug,
    )


def build_detector(app_config: AppConfig, device: str):
    detector_cfg = app_config.detector
    confidence = detector_cfg.confidence
    class_names = sorted(detector_cfg.class_names)
    backend = detector_cfg.backend.lower()
    logger.info(
        "loading detector: backend={} resolution={} conf={} device={}",
        backend,
        detector_cfg.input_resolution,
        confidence,
        device,
    )
    common = {
        "confidence": confidence,
        "class_names": class_names,
        "resolution": detector_cfg.input_resolution,
        "preserve_aspect_ratio": detector_cfg.preserve_aspect_ratio,
        "device": device,
        "half": detector_cfg.half,
    }
    if backend == "rfdetr":
        return RfdetrDetector(
            **common,
            size=detector_cfg.rfdetr_size,
            model_dir=detector_cfg.rfdetr_model_dir,
        )
    if backend in {"yolo", "yolov8"}:
        return YoloDetector(
            **common,
            model_path=detector_cfg.yolo_model_path,
        )
    raise ValueError(
        f"Unsupported detector backend: {detector_cfg.backend}. "
        "Expected rfdetr or yolov8."
    )


def resolve_pose_paths(app_config: AppConfig) -> tuple[str, str]:
    pose_cfg = app_config.pose
    preset = pose_cfg.preset
    pose_config = pose_cfg.config_path
    pose_checkpoint = pose_cfg.checkpoint_path
    logger.info("loading pose model: preset={} config={} checkpoint={}",
                preset, pose_config, pose_checkpoint)
    return pose_config, pose_checkpoint


def _normalize_sampling_args(stride: int | None,
                             target_fps: float | None) -> tuple[int, float | None]:
    if stride is None and target_fps is None:
        return 1, None
    return (stride if stride is not None else 1), target_fps


def _resolve_output_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path | None]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = None if args.no_render else output_dir / "rendered.mp4"
    return output_dir, output_dir / "observations.jsonl", output_video


def _save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    app_config: AppConfig,
    output_observations: Path,
    output_video: Path | None,
    stride: int,
    target_fps: float | None,
) -> None:
    config_copy_path = output_dir / "app_config.yaml"
    run_args_path = output_dir / "run_args.json"
    config_source = Path(args.config)
    config_copy_path.write_text(config_source.read_text(encoding="utf-8"),
                                encoding="utf-8")
    payload = {
        "cli": _jsonify(vars(args)),
        "resolved": {
            "config": str(config_source),
            "output_dir": str(output_dir),
            "output_observations": str(output_observations),
            "output_video": str(output_video) if output_video is not None else None,
            "stride": stride,
            "target_fps": target_fps,
            "detector": app_config.detector.backend,
            "detector_model": (
                app_config.detector.yolo_model_path
                if app_config.detector.backend in {"yolo", "yolov8"}
                else app_config.detector.rfdetr_size
            ),
            "detector_input_resolution": app_config.detector.input_resolution,
            "pose_preset": app_config.pose.preset,
            "tracker": app_config.runtime.tracker,
        },
    }
    run_args_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonify(item) for item in value)
    return value


if __name__ == "__main__":
    main()
