from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from football_vision import (
    RfdetrDetector,
    RoiAnnotator,
    RoiManager,
    YoloDetector,
    run_detection_preview,
)
from app_config import AppConfig
from app_pipeline import run_single_view_app
from pose_estimation import MMPoseTopDownEstimator


DEFAULT_ROI_CONFIG = "config/roi.json"
DEFAULT_APP_CONFIG = "config/app.yaml"
DEFAULT_CLASSES = ["person", "sports ball"]
DEFAULT_POSE_CONFIG = (
    "models/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/"
    "rtmpose-m_8xb64-270e_coco-wholebody-256x192.py"
)
DEFAULT_POSE_CHECKPOINT = (
    "models/mmpose/rtmpose-wholebody/"
    "rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth"
)
POSE_PRESETS = {
    "rtmpose-m": (
        DEFAULT_POSE_CONFIG,
        DEFAULT_POSE_CHECKPOINT,
    ),
    "hrnet-w32": (
        "models/mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/"
        "coco-wholebody/td-hm_hrnet-w32_8xb64-210e_coco-wholebody-256x192.py",
        "models/mmpose/hrnet/"
        "hrnet_w32_coco_wholebody_256x192-853765cd_20200918.pth",
    ),
    "hrnet-w48-dark": (
        "models/mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/"
        "coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py",
        "models/mmpose/hrnet/"
        "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI football video analysis prototype.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--roi-config",
                        default=DEFAULT_ROI_CONFIG,
                        help="Path to ROI config JSON.")
    parser.add_argument(
        "--app-config",
        default=DEFAULT_APP_CONFIG,
        help="Path to app filtering config YAML.",
    )

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
    parser.add_argument("--output-video",
                        default=None,
                        help="Save rendered detection result to a video file.")
    parser.add_argument("--output-observations",
                        default=None,
                        help="Save single-view 2D observations to JSONL.")
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
    parser.add_argument("--detector",
                        choices=["rfdetr", "yolo"],
                        default="rfdetr",
                        help="Detection backend.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path or model name.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--rfdetr-size",
                        choices=["nano", "small", "base", "medium", "large"],
                        default="base",
                        help="RF-DETR model size.")
    parser.add_argument("--model-dir",
                        default="models/rfdetr",
                        help="Directory for RF-DETR model weights.")
    parser.add_argument("--device", default="cuda:0", help="Detector device, e.g. cpu, 0, cuda:0.")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=DEFAULT_CLASSES,
        help="COCO class names to keep.",
    )

    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--target-fps",
                        type=float,
                        default=None,
                        help="Desired processing fps; stride will be inferred from source fps when set.")
    parser.add_argument("--max-frames",
                        type=int,
                        default=None,
                        help="Stop after N processed frames.")
    parser.add_argument("--perf-log",
                        action="store_true",
                        help="Log per-frame performance metrics.")
    parser.add_argument("--perf-every",
                        type=int,
                        default=30,
                        help="Log performance every N processed frames.")
    parser.add_argument("--tracker",
                        default="botsort.yaml",
                        help="Ultralytics tracker config, e.g. botsort.yaml.")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Overlay per-frame debug details on the rendered video.")
    parser.add_argument("--pose-model",
                        choices=sorted(POSE_PRESETS),
                        default="hrnet-w48-dark",
                        help="Named MMPose preset. Explicit --pose-config or "
                        "--pose-checkpoint values override this preset.")
    parser.add_argument("--pose-config",
                        default=None,
                        help="MMPose config path override.")
    parser.add_argument("--pose-checkpoint",
                        default=None,
                        help="MMPose checkpoint path override.")
    parser.add_argument("--pose-device",
                        default=None,
                        help="MMPose device. Defaults to --device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    roi_manager = RoiManager(args.roi_config)

    if args.draw_roi:
        annotator = RoiAnnotator(roi_manager)
        config = annotator.annotate(video_path, frame_index=args.roi_frame)
        logger.info("saved ROI: video={} points={} path={}", config.video_key,
                    len(config.points), args.roi_config)
        return

    detector = build_detector(args)
    if args.output_observations:
        app_config = AppConfig.from_yaml(args.app_config)
        pose_config, pose_checkpoint = resolve_pose_paths(args)
        pose_estimator = MMPoseTopDownEstimator(
            config_path=pose_config,
            checkpoint_path=pose_checkpoint,
            device=args.pose_device or args.device,
        )
        run_single_view_app(
            video_path=video_path,
            detector=detector,
            roi_manager=roi_manager,
            output_observations=args.output_observations,
            camera_id=args.camera_id,
            use_roi=args.use_roi,
            show=args.show,
            stride=args.stride,
            target_fps=args.target_fps,
            max_frames=args.max_frames,
            display_width=args.display_width,
            display_height=args.display_height,
            output_video=args.output_video,
            tracker=args.tracker,
            pose_estimator=pose_estimator,
            app_config=app_config,
            debug=args.debug,
        )
        return

    run_detection_preview(
        video_path=video_path,
        detector=detector,
        roi_manager=roi_manager,
        use_roi=args.use_roi,
        show=args.show,
        stride=args.stride,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        display_width=args.display_width,
        display_height=args.display_height,
        perf_log=args.perf_log,
        perf_every=args.perf_every,
        output_video=args.output_video,
        debug=args.debug,
    )


def build_detector(args: argparse.Namespace):
    logger.info("loading detector: backend={} conf={} device={}",
                args.detector, args.conf, args.device or "auto")
    if args.detector == "rfdetr":
        logger.info("RF-DETR size={}", args.rfdetr_size)
        return RfdetrDetector(
            confidence=args.conf,
            class_names=args.classes,
            size=args.rfdetr_size,
            device=args.device,
            model_dir=args.model_dir,
        )
    logger.info("YOLO model={} imgsz={}", args.model, args.imgsz)
    return YoloDetector(
        model_path=args.model,
        confidence=args.conf,
        image_size=args.imgsz,
        class_names=args.classes,
        device=args.device,
    )


def resolve_pose_paths(args: argparse.Namespace) -> tuple[str, str]:
    preset_config, preset_checkpoint = POSE_PRESETS[args.pose_model]
    pose_config = args.pose_config or preset_config
    pose_checkpoint = args.pose_checkpoint or preset_checkpoint
    logger.info("loading pose model: preset={} config={} checkpoint={}",
                args.pose_model, pose_config, pose_checkpoint)
    return pose_config, pose_checkpoint


if __name__ == "__main__":
    main()
