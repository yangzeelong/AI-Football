"""Read `observations.jsonl`, the one interface the C++ side writes.

A port of `ObservationJsonlReader.cpp`: one metadata line, then one line per
frame with the persons, their keypoints and the ball candidates. Only the fields
the touch logic needs are kept, and every number the C++ declares as `float` is
narrowed to `numpy.float32` here so the single precision arithmetic downstream
rounds exactly like the original.

The reader streams: a whole recording is 16037 lines and ~90 MB, and there is no
reason to hold all of it as Python objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

F32 = np.float32

#: `TouchFrameInput::Person::feet` follows kFootKeypointIndices, which is one
#: contiguous run of the pose model's keypoints.
FOOT_KEYPOINT_COUNT = 8
FOOT_KEYPOINT_FIRST_INDEX = 15


@dataclass
class Ball:
    """One ball candidate on one frame."""

    x: float = F32(0.0)
    y: float = F32(0.0)
    confidence: float = F32(0.0)
    track_id: int = -1
    #: False when the ball was extrapolated, which is not touch evidence.
    observed: bool = False


@dataclass
class FootPoint:
    """One foot keypoint; `valid` false means missing or interpolated."""

    x: float = F32(0.0)
    y: float = F32(0.0)
    confidence: float = F32(0.0)
    valid: bool = False


@dataclass
class Person:
    track_id: int = -1
    height_px: float = F32(0.0)
    feet: list[FootPoint] = field(
        default_factory=lambda: [FootPoint() for _ in range(FOOT_KEYPOINT_COUNT)]
    )


@dataclass
class FrameInput:
    """One frame of contact evidence, before any threshold is applied."""

    frame_index: int = 0
    timestamp_sec: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    balls: list[Ball] = field(default_factory=list)
    persons: list[Person] = field(default_factory=list)


@dataclass
class VideoMetadata:
    video_path: str = ""
    camera_id: str = ""
    fps: float = 0.0
    width: int = 0
    height: int = 0
    stride: int = 1


def _number_at(array: object, index: int) -> float:
    """`NumberAt()`: a missing or non numeric element reads as 0."""
    if not isinstance(array, list) or len(array) <= index:
        return F32(0.0)
    value = array[index]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return F32(0.0)
    return F32(value)


def _is_observed(state: object) -> bool:
    return isinstance(state, str) and state == "observed"


def parse_metadata(line: str) -> VideoMetadata | None:
    """The `{"type":"metadata", ...}` line, or None when it does not parse."""
    try:
        root = json.loads(line)
    except ValueError:
        return None
    video = root.get("video")
    if not isinstance(video, dict):
        return None
    return VideoMetadata(
        video_path=str(video.get("video_path", "")),
        camera_id=str(video.get("camera_id", "")),
        fps=float(video.get("fps", 0.0)),
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        stride=int(video.get("stride", 1)),
    )


def parse_frame(line: str, metadata: VideoMetadata) -> FrameInput | None:
    """`ParseFrameLine()`: the fields the touch logic reads, nothing else."""
    try:
        root = json.loads(line)
    except ValueError:
        return None

    frame = FrameInput(
        frame_index=int(root.get("frame_index", 0)),
        timestamp_sec=float(root.get("timestamp_sec", 0.0)),
        frame_width=metadata.width,
        frame_height=metadata.height,
    )

    persons = root.get("persons")
    if isinstance(persons, list):
        for person in persons:
            if not isinstance(person, dict):
                continue
            parsed = Person(track_id=int(person.get("track_id", -1)))
            box = person.get("bbox")
            if isinstance(box, list) and len(box) >= 4:
                height = _number_at(box, 3) - _number_at(box, 1)
                parsed.height_px = height if height > 0.0 else F32(0.0)

            keypoints = person.get("keypoints")
            if isinstance(keypoints, list):
                for index, keypoint in enumerate(keypoints):
                    slot = index - FOOT_KEYPOINT_FIRST_INDEX
                    if slot < 0 or slot >= FOOT_KEYPOINT_COUNT:
                        continue
                    if not isinstance(keypoint, dict):
                        continue
                    foot = parsed.feet[slot]
                    foot.confidence = F32(keypoint.get("confidence", 0.0))
                    # Interpolated and missing keypoints are not evidence of a
                    # touch: only a real observation may move the ball to a foot.
                    foot.valid = (
                        _is_observed(keypoint.get("state"))
                        and isinstance(keypoint.get("x"), (int, float))
                        and not isinstance(keypoint.get("x"), bool)
                        and isinstance(keypoint.get("y"), (int, float))
                        and not isinstance(keypoint.get("y"), bool)
                    )
                    if not foot.valid:
                        continue
                    foot.x = F32(keypoint["x"])
                    foot.y = F32(keypoint["y"])
            frame.persons.append(parsed)

    balls = root.get("balls")
    if isinstance(balls, list):
        for ball in balls:
            if not isinstance(ball, dict):
                continue
            parsed_ball = Ball(
                track_id=int(ball.get("track_id", -1)),
                confidence=F32(ball.get("confidence", 0.0)),
                observed=_is_observed(ball.get("state")),
            )
            center = ball.get("center")
            if isinstance(center, list) and len(center) >= 2:
                parsed_ball.x = _number_at(center, 0)
                parsed_ball.y = _number_at(center, 1)
            else:
                box = ball.get("bbox")
                parsed_ball.x = F32(
                    F32(0.5) * (_number_at(box, 0) + _number_at(box, 2))
                )
                parsed_ball.y = F32(
                    F32(0.5) * (_number_at(box, 1) + _number_at(box, 3))
                )
            frame.balls.append(parsed_ball)

    return frame


def iter_frames(path: str) -> Iterator[FrameInput]:
    """Yield every frame line of the file, the metadata line absorbed."""
    metadata = VideoMetadata()
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.rstrip("\n").rstrip("\r")
            if not stripped:
                continue
            # Same cheap test as the C++ reader: the metadata line is a one off.
            if '"type":"metadata"' in stripped or '"type": "metadata"' in stripped:
                parsed = parse_metadata(stripped)
                if parsed is not None:
                    metadata = parsed
                continue
            frame = parse_frame(stripped, metadata)
            if frame is not None:
                yield frame


def read_metadata(path: str) -> VideoMetadata:
    """The metadata line alone, without walking the frames."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if '"type":"metadata"' in stripped or '"type": "metadata"' in stripped:
                parsed = parse_metadata(stripped)
                if parsed is not None:
                    return parsed
            else:
                # The metadata line is always first; do not read the whole file.
                return VideoMetadata()
    return VideoMetadata()
