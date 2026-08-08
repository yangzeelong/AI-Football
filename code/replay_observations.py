from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from loguru import logger

from football_vision import Color, ResultVideoWriter, VideoReader, VideoShow
from app_config import AppConfig
from app_pipeline import SKELETON_26


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay observation JSONL on top of the source video.")
    parser.add_argument("--observations",
                        required=True,
                        help="Observation JSONL path.")
    parser.add_argument(
        "--app-config",
        default="config/app.yaml",
        help="YAML config for person/ball box filtering.",
    )
    parser.add_argument("--show",
                        action="store_true",
                        help="Show replay window.")
    parser.add_argument("--output-video",
                        default=None,
                        help="Write replay video.")
    parser.add_argument("--max-frames",
                        type=int,
                        default=None,
                        help="Stop after N rendered frames.")
    parser.add_argument("--display-width",
                        type=int,
                        default=1920,
                        help="Preview window width.")
    parser.add_argument("--display-height",
                        type=int,
                        default=1080,
                        help="Preview window height.")
    parser.add_argument("--hide-keypoints",
                        action="store_true",
                        help="Do not draw keypoints.")
    parser.add_argument("--hide-boxes",
                        action="store_true",
                        help="Do not draw boxes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show and not args.output_video:
        raise ValueError("Use --show, --output-video, or both.")

    # 回放只依赖 JSONL：视频路径、stride、相机信息都从第一行 metadata 读取。
    metadata, observations = load_observations(args.observations)
    video = metadata["video"]
    app_config = AppConfig.from_yaml(args.app_config)
    replay_observations(
        video_path=video["video_path"],
        observations=observations,
        app_config=app_config,
        show=args.show,
        output_video=args.output_video,
        stride=int(video["stride"]),
        max_frames=args.max_frames,
        display_width=args.display_width,
        display_height=args.display_height,
        draw_keypoints=not args.hide_keypoints,
        draw_boxes=not args.hide_boxes,
    )


def load_observations(
    path: str | Path, ) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    observations: dict[int, dict[str, Any]] = {}
    metadata: dict[str, Any] | None = None
    record_index = 0
    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record_index += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_no}: {path}") from exc
            if record_index == 1:
                # 不兼容旧格式；第一条非空记录必须是 metadata。
                _validate_metadata(record, path)
                metadata = record
                continue
            _validate_frame_record(record, line_no, path)
            observations[int(record["frame_index"])] = record
    if metadata is None:
        raise ValueError(f"Observation JSONL is empty: {path}")
    return metadata, observations


def replay_observations(
    video_path: str | Path,
    observations: dict[int, dict[str, Any]],
    app_config: AppConfig,
    show: bool,
    output_video: str | Path | None,
    stride: int,
    max_frames: int | None,
    display_width: int,
    display_height: int,
    draw_keypoints: bool,
    draw_boxes: bool,
) -> None:
    with VideoReader(video_path, stride=stride) as reader:
        logger.info(
            f"Replaying {reader.frame_count} frames from {video_path}, fps={reader.fps}"
        )
        viewer = VideoShow(fps=reader.fps,
                           display_width=display_width,
                           display_height=display_height) if show else None
        output_fps = reader.fps / reader.stride if reader.fps > 0 else 30.0
        writer = ResultVideoWriter(
            output_video,
            fps=output_fps,
            frame_size=(reader.width, reader.height),
        ) if output_video else None

        rendered_count = 0
        try:
            for frame in reader.frames(max_frames=max_frames):
                observation = observations.get(frame.index)
                canvas = frame.image.copy()
                draw_overlay(canvas,
                             frame.index,
                             reader.fps,
                             frame.timestamp_sec,
                             observation,
                             app_config,
                             draw_keypoints=draw_keypoints,
                             draw_boxes=draw_boxes)

                should_continue = True
                if viewer:
                    should_continue = viewer.show(canvas)
                if writer:
                    writer.write(canvas)

                rendered_count += 1
                if not should_continue:
                    break
        finally:
            if viewer:
                viewer.close()
            if writer:
                writer.close()


def draw_overlay(
    image,
    frame_index: int,
    fps: float,
    timestamp_sec: float,
    observation: dict[str, Any] | None,
    app_config: AppConfig,
    draw_keypoints: bool,
    draw_boxes: bool,
) -> None:
    persons = observation.get("persons", []) if observation else []
    balls = observation.get("balls", []) if observation else []
    # 回放阶段再次应用同一份 YAML 过滤规则，便于调参后直接对比显示效果。
    persons = [
        person for person in persons
        if app_config.passes_box("person", float(person.get(
            "confidence", 0.0)), tuple(person.get("bbox", (0, 0, 0, 0))))
    ]
    balls = [
        ball for ball in balls if app_config.passes_box(
            "sports ball", float(ball.get("confidence", 0.0)),
            tuple(ball.get("bbox", (0, 0, 0, 0))))
    ]

    if draw_boxes:
        for person in persons:
            _draw_person_box(image, person)
        for ball in balls:
            _draw_ball(image, ball)

    if draw_keypoints:
        for person in persons:
            _draw_keypoints(image, person.get("keypoints", []))

    cv2.putText(
        image,
        f"frame={frame_index} time={timestamp_sec:.2f}s persons={len(persons)} balls={len(balls)} fps={fps}",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        Color.GREEN.value,
        2,
    )


def _draw_person_box(image, person: dict[str, Any]) -> None:
    x1, y1, x2, y2 = [int(value) for value in person.get("bbox", (0, 0, 0, 0))]
    cv2.rectangle(image, (x1, y1), (x2, y2), Color.GREEN.value, 2)
    label = f"person id={person.get('track_id')} {float(person.get('confidence', 0.0)):.2f}"
    cv2.putText(image, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, Color.GREEN.value, 2)


def _draw_ball(image, ball: dict[str, Any]) -> None:
    x1, y1, x2, y2 = [int(value) for value in ball.get("bbox", (0, 0, 0, 0))]
    center = ball.get("center", [(x1 + x2) / 2, (y1 + y2) / 2])
    color = Color.ORANGE.value if ball.get(
        "state") == "observed" else Color.BLUE.value
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.circle(image, (int(center[0]), int(center[1])), 6, color, -1)
    label = f"ball id={ball.get('track_id')} {ball.get('state', 'observed')}"
    cv2.putText(image, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)


def _draw_keypoints(image, keypoints: list[dict[str, Any]]) -> None:
    points = {keypoint.get("name"): keypoint for keypoint in keypoints}
    for start, end in SKELETON_26:
        start_point = points.get(start)
        end_point = points.get(end)
        if not start_point or not end_point:
            continue
        if start_point.get("x") is None or end_point.get("x") is None:
            continue
        cv2.line(
            image,
            (int(start_point["x"]), int(start_point["y"])),
            (int(end_point["x"]), int(end_point["y"])),
            Color.CYAN.value,
            2,
        )

    for keypoint in keypoints:
        if keypoint.get("x") is None or keypoint.get("y") is None:
            continue
        cv2.circle(image, (int(keypoint["x"]), int(keypoint["y"])), 3,
                   Color.YELLOW.value, -1)


def _validate_metadata(record: dict[str, Any], path: str | Path) -> None:
    if record["type"] != "metadata":
        raise ValueError(f"First JSONL record must be metadata: {path}")
    video = record["video"]
    required = ("video_path", "fps", "width", "height", "frame_count",
                "stride", "camera_id")
    for key in required:
        _ = video[key]


def _validate_frame_record(record: dict[str, Any], line_no: int,
                           path: str | Path) -> None:
    if record["type"] != "frame":
        raise ValueError(f"Expected frame record at line {line_no}: {path}")
    _ = record["frame_index"]
    _ = record["timestamp_sec"]
    _ = record["camera_id"]
    _ = record["persons"]
    _ = record["balls"]


if __name__ == "__main__":
    main()
