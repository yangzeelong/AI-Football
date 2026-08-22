from __future__ import annotations

from pathlib import Path

import cv2
from loguru import logger
from tqdm import tqdm

from football_vision import (
    Color,
    ResultVideoWriter,
    RoiManager,
    VideoReader,
    VideoShow,
    YoloDetector,
    render_detection_frame,
)
from observations import (
    JsonlObservationWriter,
    PersonObservation2D,
    VideoObservationMetadata,
    empty_project_26,
    frame_observation,
)
from app_config import AppConfig
from keypoint_smoothing import KeypointTemporalSmoother
from pose_estimation import MMPoseTopDownEstimator, person_detections
from tracking import FootballTracker


SKELETON_26: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_big_toe"),
    ("left_ankle", "left_small_toe"),
    ("left_ankle", "left_heel"),
    ("right_ankle", "right_big_toe"),
    ("right_ankle", "right_small_toe"),
    ("right_ankle", "right_heel"),
    ("neck", "thorax"),
    ("thorax", "pelvis"),
)


def run_single_view_app(
    video_path: str | Path,
    detector: YoloDetector,
    roi_manager: RoiManager,
    output_observations: str | Path,
    camera_id: str = "C1",
    use_roi: bool = False,
    show: bool = False,
    stride: int = 1,
    target_fps: float | None = None,
    max_frames: int | None = None,
    display_width: int = 1920,
    display_height: int = 1080,
    output_video: str | Path | None = None,
    tracker: str = "botsort.yaml",
    pose_estimator: MMPoseTopDownEstimator | None = None,
    app_config: AppConfig | None = None,
) -> None:
    football_tracker = FootballTracker()
    app_config = app_config or AppConfig.default()

    with VideoReader(video_path, stride=stride, target_fps=target_fps) as reader:
        # JSONL 自带视频信息，回放和质量评估脚本无需再额外传 video 参数。
        metadata = VideoObservationMetadata(
            video_path=str(video_path),
            fps=reader.fps,
            width=reader.width,
            height=reader.height,
            frame_count=reader.frame_count,
            stride=reader.stride,
            camera_id=camera_id,
        )
        observation_writer = JsonlObservationWriter(output_observations,
                                                    metadata)
        output_fps = reader.fps / reader.stride if reader.fps > 0 else 30.0
        keypoint_smoother = KeypointTemporalSmoother(
            app_config.keypoint_smoothing,
            sample_fps=output_fps,
        )
        logger.info(
            "single-view app loaded: path={} camera={} size={}x{} fps={:.2f} frames={} stride={}",
            video_path,
            camera_id,
            reader.width,
            reader.height,
            reader.fps,
            reader.frame_count,
            reader.stride,
        )
        roi = roi_manager.get(video_path) if use_roi else None
        viewer = VideoShow(fps=reader.fps,
                           display_width=display_width,
                           display_height=display_height) if show else None
        writer = ResultVideoWriter(
            output_video,
            fps=output_fps,
            frame_size=(reader.width, reader.height),
        ) if output_video else None

        processed = 0
        total_frames = _processed_frame_total(reader.frame_count,
                                              reader.stride, max_frames)
        try:
            # tqdm 根据视频总帧数和 stride 估算处理进度，长视频运行时更容易判断剩余时间。
            for frame in tqdm(reader.frames(max_frames=max_frames),
                              total=total_frames,
                              desc="processing",
                              unit="frame"):
                detections = detector.track(frame, tracker=tracker)
                detections = roi_manager.filter_detections(detections, roi)
                # ROI 之后再做 YAML 阈值过滤，保证输出和可视化使用同一批观测。
                detections = app_config.filter_detections(detections)
                raw_detection_counts = _raw_detection_counts(detections)

                person_boxes = person_detections(detections)
                persons = _person_boxes_without_pose(person_boxes)
                if pose_estimator:
                    persons = pose_estimator.estimate(frame, person_boxes)
                    persons = keypoint_smoother.update(persons)
                balls = football_tracker.update(frame, detections)

                observation = frame_observation(
                    frame=frame,
                    camera_id=camera_id,
                    persons=persons,
                    balls=balls,
                    raw_detection_counts=raw_detection_counts,
                )
                observation_writer.write(observation)

                rendered = None
                should_continue = True
                if viewer or writer:
                    rendered = render_detection_frame(frame, detections, roi,
                                                      reader.fps)
                    _draw_person_keypoints(rendered, persons)
                    _draw_tracked_balls(rendered, balls)

                if viewer and rendered is not None:
                    should_continue = viewer.show(rendered)
                if writer and rendered is not None:
                    writer.write(rendered)

                processed += 1
                if not should_continue:
                    break
        finally:
            observation_writer.close()
            if viewer:
                viewer.close()
            if writer:
                writer.close()

        logger.info("wrote observations: path={}", output_observations)
        logger.info("processed {} frames from {}", processed, video_path)


def _processed_frame_total(frame_count: int, stride: int,
                           max_frames: int | None) -> int | None:
    if frame_count <= 0:
        return max_frames
    total = (frame_count + stride - 1) // stride
    if max_frames is not None:
        total = min(total, max_frames)
    return total


def _person_boxes_without_pose(person_boxes) -> list[PersonObservation2D]:
    return [
        PersonObservation2D(
            track_id=detection.track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            keypoints=empty_project_26(),
        ) for detection in person_boxes
    ]


def _raw_detection_counts(detections) -> dict[str, int]:
    person_count = sum(1 for detection in detections if detection.label == "person")
    ball_count = sum(
        1 for detection in detections
        if detection.label in {"sports ball", "ball", "football", "soccer ball"}
    )
    return {
        "total": len(detections),
        "person": person_count,
        "ball": ball_count,
    }


def _draw_person_keypoints(image, persons) -> None:
    for person in persons:
        points = {keypoint.name: keypoint for keypoint in person.keypoints}
        for start, end in SKELETON_26:
            start_point = points.get(start)
            end_point = points.get(end)
            if not start_point or not end_point:
                continue
            if start_point.x is None or end_point.x is None:
                continue
            cv2.line(
                image,
                (int(start_point.x), int(start_point.y)),
                (int(end_point.x), int(end_point.y)),
                Color.CYAN.value,
                2,
            )
        for keypoint in person.keypoints:
            if keypoint.x is None or keypoint.y is None:
                continue
            cv2.circle(image, (int(keypoint.x), int(keypoint.y)), 3,
                       Color.YELLOW.value, -1)


def _draw_tracked_balls(image, balls) -> None:
    for ball in balls:
        center = (int(ball.center[0]), int(ball.center[1]))
        color = Color.ORANGE.value if ball.state == "observed" else Color.BLUE.value
        cv2.circle(image, center, 8, color, -1)
        cv2.putText(
            image,
            f"ball id={ball.track_id} {ball.state}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
