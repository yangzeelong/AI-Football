#!/usr/bin/env python3
"""Render AI-Football JSONL observations back onto the source video.

This script is intentionally model-free. It reads the source video and the
per-frame observations produced by the SDK, then writes a debug/report video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2


SKELETON_EDGES = (
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
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

PERSON_COLOR = (70, 220, 80)       # BGR
BALL_COLOR = (40, 150, 255)        # BGR
CONE_COLOR = (60, 60, 235)         # BGR 红：锥（三分类版才有，跟踪链路不带它）
SKELETON_COLOR = (255, 200, 40)    # BGR
KEYPOINT_COLOR = (40, 220, 255)    # BGR
PANEL_COLOR = (25, 25, 25)
TEXT_COLOR = (245, 245, 245)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render SDK/Python observation JSONL to an MP4 video."
    )
    parser.add_argument(
        "--observations",
        required=True,
        help="Input observations.jsonl.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output rendered MP4 path.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Override source video path from JSONL metadata.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Render at most N source frames.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS; defaults to source FPS.",
    )
    parser.add_argument(
        "--hide-boxes",
        action="store_true",
        help="Do not draw person or ball boxes.",
    )
    parser.add_argument(
        "--hide-keypoints",
        action="store_true",
        help="Do not draw person keypoints and skeletons.",
    )
    parser.add_argument(
        "--hide-panel",
        action="store_true",
        help="Do not draw the frame information panel.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Skip to this source frame before rendering (default 0).",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Second observations.jsonl to render side by side "
             "(left = --observations, right = this file). Source video is decoded once.",
    )
    parser.add_argument("--label-a", default="A", help="Label drawn on the left half.")
    parser.add_argument("--label-b", default="B", help="Label drawn on the right half.")
    parser.add_argument(
        "--crf", type=int, default=23,
        help="H.264 quality when PyAV/libx264 is used (lower = better, default 23).")
    parser.add_argument(
        "--no-pyav", action="store_true",
        help="Force the OpenCV writer (mp4v) instead of PyAV/libx264.")
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Also show detector_counts (person/ball/cone) in the panel when the JSONL has it.",
    )
    return parser.parse_args()


def load_observations(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    metadata: dict[str, Any] | None = None
    observations: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc

            record_type = record.get("type")
            if metadata is None:
                if record_type != "metadata":
                    raise ValueError("The first JSONL record must be metadata")
                metadata = record
                continue
            if record_type != "frame":
                raise ValueError(
                    f"Expected a frame record at {path}:{line_number}, "
                    f"got {record_type!r}"
                )
            if "frame_index" not in record:
                raise ValueError(f"Missing frame_index at {path}:{line_number}")
            observations[int(record["frame_index"])] = record

    if metadata is None:
        raise ValueError(f"Observation JSONL is empty: {path}")
    if not isinstance(metadata.get("video"), dict):
        raise ValueError(f"Metadata has no video section: {path}")
    return metadata, observations


def resolve_video_path(
    metadata: dict[str, Any], observations_path: Path, override: str | None
) -> Path:
    raw_path = override or str(metadata["video"].get("video_path", ""))
    if not raw_path:
        raise ValueError("No source video path in metadata; use --video")

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate

    candidates = [
        candidate,
        observations_path.parent / candidate,
        Path.cwd() / candidate,
        Path(__file__).resolve().parents[1] / candidate,
    ]
    for item in candidates:
        if item.exists():
            return item.resolve()
    checked = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"Source video not found; checked: {checked}")


class PyAvWriter:
    """用 PyAV/libx264 写 MP4：同画质比 OpenCV 的 mp4v 小一个量级，且不限制 fps。

    本机两个解释器的 OpenCV 都没有 H.264 编码器（只有 mp4v），而 ai-football 环境里
    的 PyAV 自带 libx264/libx265 —— 所以默认优先走这里，失败再退回 OpenCV。
    """

    def __init__(self, path: Path, fps: float, width: int, height: int, crf: int = 23):
        import av

        path.parent.mkdir(parents=True, exist_ok=True)
        self._out = av.open(str(path), mode="w", options={"movflags": "+faststart"})
        self._stream = self._out.add_stream("libx264", rate=max(1, int(round(fps))))
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": str(crf), "preset": "veryfast"}
        self._path = path
        self._frames = 0
        self._crf = crf

    def write(self, frame: Any) -> None:
        import av

        vf = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in self._stream.encode(vf):
            self._out.mux(packet)
        self._frames += 1

    def release(self) -> None:
        try:
            for packet in self._stream.encode():
                self._out.mux(packet)
        finally:
            self._out.close()
            print(f"[render] pyav/libx264 crf={self._crf} frames={self._frames} "
                  f"output={self._path}")


def create_writer(path: Path, fps: float, width: int, height: int,
                  crf: int = 23, use_pyav: bool = True) -> Any:
    """优先 PyAV/libx264，失败退回 OpenCV（mp4v）。"""
    if use_pyav:
        try:
            writer = PyAvWriter(path, fps, width, height, crf)
            print(f"[render] codec=libx264(crf={crf}) output={path}")
            return writer
        except Exception as exc:                  # noqa: BLE001 - 任何失败都退回 OpenCV
            print(f"[render] PyAV 不可用（{type(exc).__name__}: {exc}）→ 退回 OpenCV mp4v",
                  flush=True)
    return _create_writer_opencv(path, fps, width, height)


def _create_writer_opencv(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if writer.isOpened():
            print(f"[render] codec={codec} output={path}")
            return writer
        writer.release()
    raise RuntimeError(f"Unable to open an MP4 writer for {path}")


def draw_text(
    image: Any,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = TEXT_COLOR,
    scale: float = 0.58,
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_person(image: Any, person: dict[str, Any], draw_boxes: bool, draw_keypoints: bool) -> None:
    bbox = person.get("bbox", (0, 0, 0, 0))
    if len(bbox) < 4:
        return
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox[:4]]
    if draw_boxes:
        cv2.rectangle(image, (x0, y0), (x1, y1), PERSON_COLOR, 2, cv2.LINE_AA)
        track_id = person.get("track_id", "?")
        confidence = float(person.get("confidence", 0.0))
        draw_text(image, f"person id={track_id} {confidence:.2f}",
                  (x0, max(24, y0 - 8)), PERSON_COLOR)
    if draw_keypoints:
        draw_keypoints_for_person(image, person.get("keypoints", []))


def draw_keypoints_for_person(image: Any, keypoints: list[dict[str, Any]]) -> None:
    points: dict[str, tuple[int, int]] = {}
    for keypoint in keypoints:
        name = keypoint.get("name")
        x = keypoint.get("x")
        y = keypoint.get("y")
        if not name or x is None or y is None:
            continue
        if keypoint.get("state") == "missing":
            continue
        points[str(name)] = (int(round(float(x))), int(round(float(y))))

    for start, end in SKELETON_EDGES:
        if start in points and end in points:
            cv2.line(image, points[start], points[end], SKELETON_COLOR, 2, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(image, point, 3, KEYPOINT_COLOR, -1, cv2.LINE_AA)


def draw_ball(image: Any, ball: dict[str, Any], draw_boxes: bool) -> None:
    bbox = ball.get("bbox", (0, 0, 0, 0))
    if len(bbox) < 4:
        return
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox[:4]]
    center = ball.get("center")
    if not center or len(center) < 2:
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
    if draw_boxes:
        cv2.rectangle(image, (x0, y0), (x1, y1), BALL_COLOR, 2, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), 6, BALL_COLOR, -1, cv2.LINE_AA)
        state = ball.get("state", "observed")
        draw_text(image, f"ball id={ball.get('track_id', '?')} {state}",
                  (x0, max(24, y0 - 8)), BALL_COLOR)


def draw_cone(image: Any, cone: dict[str, Any], draw_boxes: bool) -> None:
    """画锥框。只有三分类微调模型会输出 `cones`（COCO 版是空数组）。"""
    bbox = cone.get("bbox", (0, 0, 0, 0))
    if len(bbox) < 4:
        return
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox[:4]]
    if not draw_boxes:
        return
    cv2.rectangle(image, (x0, y0), (x1, y1), CONE_COLOR, 2, cv2.LINE_AA)
    score = float(cone.get("confidence", 0.0))
    draw_text(image, f"cone {score:.2f}", (x0, max(24, y0 - 8)), CONE_COLOR, scale=0.5)


def draw_panel(
    image: Any,
    frame_index: int,
    timestamp_sec: float,
    fps: float,
    persons: list[dict[str, Any]],
    balls: list[dict[str, Any]],
    cones: list[dict[str, Any]] | None = None,
    counts: dict[str, Any] | None = None,
) -> None:
    cones = cones or []
    lines = [
        f"AI-Football replay  frame={frame_index}  time={timestamp_sec:.2f}s",
        f"persons={len(persons)}  balls={len(balls)}  cones={len(cones)}  source_fps={fps:.2f}",
    ]
    if counts:
        # detector_counts 是「检测器过滤前」的逐帧计数，含 cones（跟踪阶段没有锥通道，
        # 所以锥只在这个字段里能看见；框本身没进 observations）。
        lines.append(
            "detector: total=%s person=%s ball=%s cone=%s"
            % (counts.get("total", "—"), counts.get("person", "—"),
               counts.get("ball", "—"), counts.get("cone", "—"))
        )
    padding = 12
    line_height = 25
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0]
                for line in lines) + padding * 2
    height = line_height * len(lines) + padding
    overlay = image.copy()
    cv2.rectangle(overlay, (12, 12), (12 + width, 12 + height), PANEL_COLOR, -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)
    for index, line in enumerate(lines):
        draw_text(image, line, (12 + padding, 12 + 20 + index * line_height),
                  TEXT_COLOR, scale=0.62)


def render(
    observations_path: Path,
    output_path: Path,
    video_override: str | None,
    max_frames: int | None,
    fps_override: float | None,
    draw_boxes: bool,
    draw_keypoints: bool,
    draw_info_panel: bool,
    start_frame: int = 0,
    compare_path: Path | None = None,
    label_a: str = "A",
    label_b: str = "B",
    show_counts: bool = False,
    crf: int = 23,
    use_pyav: bool = True,
) -> int:
    metadata, observations = load_observations(observations_path)
    compare_observations: dict[int, dict[str, Any]] = {}
    if compare_path is not None:
        _, compare_observations = load_observations(compare_path)
    video_path = resolve_video_path(metadata, observations_path, video_override)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or float(metadata["video"].get("fps", 30.0))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_width <= 0 or source_height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid source video dimensions: {video_path}")
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    side_by_side = compare_path is not None
    output_fps = fps_override or source_fps or 30.0
    writer = create_writer(output_path, output_fps,
                           source_width * (2 if side_by_side else 1), source_height,
                           crf, use_pyav)
    last_observed_frame = max(observations) if observations else -1
    rendered = 0
    try:
        while max_frames is None or rendered < max_frames:
            frame_index = start_frame + rendered
            if frame_index > last_observed_frame:
                break
            ok, frame = capture.read()
            if not ok:
                break

            left = frame.copy() if side_by_side else frame
            halves: list[tuple[Any, dict[int, dict[str, Any]], str]] = [(left, observations, label_a)]
            if side_by_side:
                right = frame.copy()
                halves.append((right, compare_observations, label_b))

            for target, table, label in halves:
                observation = table.get(frame_index)
                persons = observation.get("persons", []) if observation else []
                balls = observation.get("balls", []) if observation else []
                timestamp_sec = (
                    float(observation.get("timestamp_sec", frame_index / source_fps))
                    if observation
                    else frame_index / source_fps
                )
                for person in persons:
                    draw_person(target, person, draw_boxes, draw_keypoints)
                for ball in balls:
                    draw_ball(target, ball, draw_boxes)
                for cone in (observation or {}).get("cones", []):
                    draw_cone(target, cone, draw_boxes)
                if draw_info_panel:
                    draw_panel(target, frame_index, timestamp_sec, source_fps, persons, balls,
                               (observation or {}).get("cones", []),
                               (observation or {}).get("detector_counts") if show_counts else None)
                draw_text(target, label, (16, source_height - 20), (60, 240, 255), scale=0.9, thickness=2)

            if side_by_side:
                canvas = cv2.hconcat([left, right])
                cv2.line(canvas, (source_width, 0), (source_width, source_height), (70, 70, 70), 2)
                writer.write(canvas)
            else:
                writer.write(left)
            rendered += 1
            if rendered % 300 == 0:
                print(f"[render] frames={rendered}", flush=True)
    finally:
        capture.release()
        writer.release()

    print(f"[done] video={video_path}")
    print(f"[done] frames={rendered} fps={output_fps:.3f} output={output_path}")
    return rendered


def main() -> int:
    args = parse_args()
    try:
        render(
            observations_path=Path(args.observations).expanduser().resolve(),
            output_path=Path(args.output).expanduser().resolve(),
            video_override=args.video,
            max_frames=args.max_frames,
            fps_override=args.fps,
            draw_boxes=not args.hide_boxes,
            draw_keypoints=not args.hide_keypoints,
            draw_info_panel=not args.hide_panel,
            start_frame=max(0, args.start_frame),
            compare_path=(Path(args.compare).expanduser().resolve()
                          if args.compare else None),
            label_a=args.label_a,
            label_b=args.label_b,
            show_counts=args.show_counts,
            crf=args.crf,
            use_pyav=not args.no_pyav,
        )
    except (OSError, ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
