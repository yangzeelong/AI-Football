from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from football_vision import BBox, Detection, VideoFrame
from observations import BallObservation2D, PersonObservation2D


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
               detections: Sequence[Detection],
               persons: Sequence[PersonObservation2D] | None = None) -> list[BallObservation2D]:
        ball_detections = [
            detection for detection in detections
            if detection.label in self.ball_labels
        ]
        detection = self._select_detection(frame, ball_detections, persons)

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
        frame: VideoFrame,
        detections: Sequence[Detection],
        persons: Sequence[PersonObservation2D] | None = None,
    ) -> Detection | None:
        if not detections:
            return None
        image_height, image_width = frame.image.shape[:2]
        if self._track is None:
            scored = [
                (self._score_detection(detection, None, persons, image_width,
                                       image_height),
                 detection) for detection in detections
            ]
            score, detection = max(scored, key=lambda item: item[0])
            return detection if score >= 0.18 else None

        predicted_center = self._predicted_center(frame.timestamp_sec)
        scored = [
            (self._score_detection(detection, predicted_center, persons,
                                   image_width, image_height), detection)
            for detection in detections
        ]
        score, detection = max(scored, key=lambda item: item[0])
        return detection if score >= 0.15 else None

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

    def _score_detection(
        self,
        detection: Detection,
        predicted_center: tuple[float, float] | None,
        persons: Sequence[PersonObservation2D] | None,
        image_width: int,
        image_height: int,
    ) -> float:
        confidence_score = float(detection.confidence)
        support_score = ball_support_score(detection.center, persons, image_width,
                                           image_height)
        association_score = 0.0
        if predicted_center is not None:
            distance = _distance(predicted_center, detection.center)
            radius = self.max_association_distance_px
            if self._track is not None:
                radius += min(120.0, self._track.missed_frames * 20.0)
            if radius > 0:
                association_score = max(0.0, 1.0 - distance / radius)

        score = confidence_score
        score += 0.45 * support_score
        score += 0.35 * association_score
        score -= _background_penalty(detection.center, image_height,
                                     support_score, association_score)
        return score


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


def ball_support_score(center: tuple[float, float],
                       persons: Sequence[PersonObservation2D] | None,
                       image_width: int,
                       image_height: int) -> float:
    if not persons:
        return 0.0

    best = 0.0
    support_radius = max(36.0, min(image_width, image_height) * 0.08)
    for person in persons:
        foot_points = _person_foot_points(person)
        if foot_points:
            distance = min(_distance(center, point) for point in foot_points)
            if distance <= support_radius:
                best = max(best, max(0.0, 1.0 - distance / support_radius))
                continue

        x1, y1, x2, y2 = person.bbox
        width = x2 - x1
        height = y2 - y1
        x_margin = max(12.0, width * 0.08)
        upper_y = y2 - max(18.0, height * 0.35)
        lower_y = y2 + max(12.0, height * 0.1)
        if x1 - x_margin <= center[0] <= x2 + x_margin and upper_y <= center[
                1] <= lower_y:
            best = max(best, 0.7)

    return best


def _person_foot_points(person: PersonObservation2D) -> list[tuple[float, float]]:
    foot_keypoints = {
        "left_ankle",
        "right_ankle",
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
    }
    points: list[tuple[float, float]] = []
    for keypoint in person.keypoints:
        if keypoint.name not in foot_keypoints:
            continue
        if keypoint.x is None or keypoint.y is None:
            continue
        if keypoint.confidence < 0.2:
            continue
        points.append((float(keypoint.x), float(keypoint.y)))
    return points


def _background_penalty(center: tuple[float, float], image_height: int,
                        support_score: float,
                        association_score: float) -> float:
    if support_score > 0.0 or association_score > 0.2:
        return 0.0
    y_ratio = center[1] / max(1.0, float(image_height))
    if y_ratio < 0.25:
        return 0.55
    if y_ratio < 0.4:
        return 0.25
    return 0.0
