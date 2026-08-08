from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from football_vision import BBox, Detection, VideoFrame
from observations import BallObservation2D


@dataclass
class BallTrack:
    track_id: int
    center: tuple[float, float]
    velocity_px_s: tuple[float, float]
    bbox_size: tuple[float, float]
    confidence: float
    timestamp_sec: float
    missed_frames: int = 0


class FootballTracker:

    def __init__(
        self,
        ball_labels: set[str] | None = None,
        max_missed_frames: int = 12,
        max_association_distance_px: float = 180.0,
        confidence_decay: float = 0.75,
    ) -> None:
        self.ball_labels = ball_labels or {"sports ball", "ball"}
        self.max_missed_frames = max(0, max_missed_frames)
        self.max_association_distance_px = max_association_distance_px
        self.confidence_decay = confidence_decay
        self._track: BallTrack | None = None
        self._next_track_id = 1

    def update(self, frame: VideoFrame,
               detections: Sequence[Detection]) -> list[BallObservation2D]:
        ball_detections = [
            detection for detection in detections
            if detection.label in self.ball_labels
        ]
        detection = self._select_detection(frame.timestamp_sec, ball_detections)

        if detection is None:
            return self._predict(frame.timestamp_sec)

        self._update_track(frame.timestamp_sec, detection)
        if self._track is None:
            return []
        return [
            BallObservation2D(
                track_id=self._track.track_id,
                bbox=detection.bbox,
                center=self._track.center,
                confidence=detection.confidence,
                state="observed",
            )
        ]

    def _select_detection(
        self,
        timestamp_sec: float,
        detections: Sequence[Detection],
    ) -> Detection | None:
        if not detections:
            return None
        if self._track is None:
            return max(detections, key=lambda detection: detection.confidence)

        predicted_center = self._predicted_center(timestamp_sec)
        candidates = [
            detection for detection in detections
            if _distance(predicted_center, detection.center) <=
            self.max_association_distance_px
        ]
        if not candidates:
            return max(detections, key=lambda detection: detection.confidence)
        return min(
            candidates,
            key=lambda detection: _distance(predicted_center, detection.center),
        )

    def _update_track(self, timestamp_sec: float, detection: Detection) -> None:
        center = detection.center
        width = detection.bbox[2] - detection.bbox[0]
        height = detection.bbox[3] - detection.bbox[1]
        if self._track is None:
            self._track = BallTrack(
                track_id=detection.track_id or self._next_track_id,
                center=center,
                velocity_px_s=(0.0, 0.0),
                bbox_size=(width, height),
                confidence=detection.confidence,
                timestamp_sec=timestamp_sec,
            )
            self._next_track_id = max(self._next_track_id,
                                      self._track.track_id + 1)
            return

        dt = timestamp_sec - self._track.timestamp_sec
        velocity = self._track.velocity_px_s
        if dt > 0:
            velocity = (
                (center[0] - self._track.center[0]) / dt,
                (center[1] - self._track.center[1]) / dt,
            )

        self._track.center = center
        self._track.velocity_px_s = velocity
        self._track.bbox_size = (width, height)
        self._track.confidence = detection.confidence
        self._track.timestamp_sec = timestamp_sec
        self._track.missed_frames = 0

    def _predict(self, timestamp_sec: float) -> list[BallObservation2D]:
        if self._track is None:
            return []

        self._track.missed_frames += 1
        if self._track.missed_frames > self.max_missed_frames:
            self._track = None
            return []

        center = self._predicted_center(timestamp_sec)
        self._track.center = center
        self._track.timestamp_sec = timestamp_sec
        self._track.confidence *= self.confidence_decay
        return [
            BallObservation2D(
                track_id=self._track.track_id,
                bbox=_bbox_from_center(center, self._track.bbox_size),
                center=center,
                confidence=self._track.confidence,
                state="predicted",
            )
        ]

    def _predicted_center(self, timestamp_sec: float) -> tuple[float, float]:
        if self._track is None:
            return (0.0, 0.0)
        dt = max(0.0, timestamp_sec - self._track.timestamp_sec)
        return (
            self._track.center[0] + self._track.velocity_px_s[0] * dt,
            self._track.center[1] + self._track.velocity_px_s[1] * dt,
        )


def _bbox_from_center(center: tuple[float, float],
                      size: tuple[float, float]) -> BBox:
    width, height = size
    return (
        center[0] - width / 2.0,
        center[1] - height / 2.0,
        center[0] + width / 2.0,
        center[1] + height / 2.0,
    )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy)**0.5
