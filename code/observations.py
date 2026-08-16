from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from football_vision import BBox, Detection, VideoFrame


PROJECT_26_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "neck",
    "pelvis",
    "thorax",
)

WHOLEBODY_TO_PROJECT_26: dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
    "left_big_toe": 17,
    "left_small_toe": 18,
    "left_heel": 19,
    "right_big_toe": 20,
    "right_small_toe": 21,
    "right_heel": 22,
}


@dataclass(frozen=True)
class Keypoint2D:
    name: str
    x: float | None
    y: float | None
    confidence: float
    state: str = "observed"


@dataclass(frozen=True)
class PersonObservation2D:
    track_id: int | None
    bbox: BBox
    confidence: float
    keypoints: list[Keypoint2D]
    state: str = "observed"


@dataclass(frozen=True)
class BallObservation2D:
    track_id: int | None
    bbox: BBox
    center: tuple[float, float]
    confidence: float
    state: str = "observed"


@dataclass(frozen=True)
class FrameObservation2D:
    frame_index: int
    timestamp_sec: float
    camera_id: str
    persons: list[PersonObservation2D]
    balls: list[BallObservation2D]
    raw_detection_counts: dict[str, int]


@dataclass(frozen=True)
class VideoObservationMetadata:
    video_path: str
    fps: float
    width: int
    height: int
    frame_count: int
    stride: int
    camera_id: str


class JsonlObservationWriter:

    def __init__(self, output_path: str | Path,
                 metadata: VideoObservationMetadata) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", encoding="utf-8")
        # 第一行写视频级 metadata，后续回放和评估脚本只依赖 JSONL 即可定位原视频。
        self._write_record({
            "type": "metadata",
            "schema_version": 1,
            "video": asdict(metadata),
        })

    def write(self, observation: FrameObservation2D) -> None:
        # 每一帧都显式标记 type，避免 metadata 和 frame 记录被误读。
        data = {"type": "frame"}
        data.update(asdict(observation))
        self._write_record(data)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "JsonlObservationWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _write_record(self, record: dict) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False))
        self._file.write("\n")


def empty_project_26() -> list[Keypoint2D]:
    return [
        Keypoint2D(name=name, x=None, y=None, confidence=0.0, state="missing")
        for name in PROJECT_26_KEYPOINTS
    ]


def ball_observations_from_detections(
    detections: Iterable[Detection],
    ball_labels: set[str] | None = None,
) -> list[BallObservation2D]:
    labels = ball_labels or {"sports ball", "ball"}
    return [
        BallObservation2D(
            track_id=detection.track_id,
            bbox=detection.bbox,
            center=detection.center,
            confidence=detection.confidence,
        ) for detection in detections if detection.label in labels
    ]


def frame_observation(
    frame: VideoFrame,
    camera_id: str,
    persons: list[PersonObservation2D] | None = None,
    balls: list[BallObservation2D] | None = None,
    raw_detection_counts: dict[str, int] | None = None,
) -> FrameObservation2D:
    return FrameObservation2D(
        frame_index=frame.index,
        timestamp_sec=frame.timestamp_sec,
        camera_id=camera_id,
        persons=persons or [],
        balls=balls or [],
        raw_detection_counts=raw_detection_counts or {},
    )
