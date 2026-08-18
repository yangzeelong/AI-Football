from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot, pi

from app_config import KeypointSmoothingConfig
from football_vision import BBox
from observations import Keypoint2D, PersonObservation2D


VIRTUAL_KEYPOINTS = {"neck", "pelvis", "thorax"}


@dataclass
class _TrackHistory:
    bbox_centers: deque[tuple[float, float]] = field(default_factory=deque)
    keypoints: dict[str, Keypoint2D] = field(default_factory=dict)
    filters: dict[str, "_KeypointOneEuroFilter"] = field(default_factory=dict)


@dataclass
class _OneEuroAxisState:
    value: float | None = None
    derivative: float = 0.0


class KeypointTemporalSmoother:

    def __init__(
        self,
        config: KeypointSmoothingConfig,
        sample_fps: float = 30.0,
    ) -> None:
        self.config = config
        self._sample_fps = sample_fps if sample_fps > 0 else 30.0
        self._tracks: dict[int, _TrackHistory] = {}

    def update(
        self,
        persons: list[PersonObservation2D],
    ) -> list[PersonObservation2D]:
        if not self.config.enabled:
            return persons

        smoothed = []
        for person in persons:
            if person.track_id is None:
                smoothed.append(person)
                continue
            history = self._tracks.setdefault(person.track_id, _TrackHistory())
            is_static = self._update_static_state(history, person.bbox)
            keypoints = self._smooth_keypoints(
                person.keypoints,
                history,
                is_static,
            )
            smoothed.append(
                PersonObservation2D(
                    track_id=person.track_id,
                    bbox=person.bbox,
                    confidence=person.confidence,
                    keypoints=keypoints,
                    state=person.state,
                ))
        return smoothed

    def _update_static_state(
        self,
        history: _TrackHistory,
        bbox: BBox,
    ) -> bool:
        history.bbox_centers.append(_bbox_center(bbox))
        while len(history.bbox_centers) > self.config.static_window:
            history.bbox_centers.popleft()
        if len(history.bbox_centers) < self.config.static_window:
            return False

        centers = list(history.bbox_centers)
        max_step = max(
            hypot(curr[0] - prev[0], curr[1] - prev[1])
            for prev, curr in zip(centers, centers[1:]))
        return max_step <= self.config.static_motion_px

    def _smooth_keypoints(
        self,
        keypoints: list[Keypoint2D],
        history: _TrackHistory,
        is_static: bool,
    ) -> list[Keypoint2D]:
        smoothed_by_name: dict[str, Keypoint2D] = {}
        for keypoint in keypoints:
            if keypoint.name in VIRTUAL_KEYPOINTS:
                continue
            previous = history.keypoints.get(keypoint.name)
            smoothed = self._smooth_one(keypoint, previous, history,
                                        is_static)
            smoothed_by_name[keypoint.name] = smoothed

        _set_virtual_midpoint(smoothed_by_name, "neck", "left_shoulder",
                              "right_shoulder")
        _set_virtual_midpoint(smoothed_by_name, "pelvis", "left_hip",
                              "right_hip")
        _set_virtual_midpoint(smoothed_by_name, "thorax", "neck", "pelvis")

        ordered = [
            smoothed_by_name.get(keypoint.name, keypoint)
            for keypoint in keypoints
        ]
        history.keypoints = {keypoint.name: keypoint for keypoint in ordered}
        return ordered

    def _smooth_one(
        self,
        keypoint: Keypoint2D,
        previous: Keypoint2D | None,
        history: _TrackHistory,
        is_static: bool,
    ) -> Keypoint2D:
        if previous is None:
            return keypoint
        if keypoint.x is None or keypoint.y is None:
            return _copy_position(previous, keypoint, "predicted")
        if previous.x is None or previous.y is None:
            return keypoint

        distance = hypot(keypoint.x - previous.x, keypoint.y - previous.y)
        if self.config.method == "one_euro":
            return self._smooth_one_euro(keypoint, previous, history, distance,
                                         is_static)
        return self._smooth_ema(keypoint, previous, distance, is_static)

    def _smooth_ema(
        self,
        keypoint: Keypoint2D,
        previous: Keypoint2D,
        distance: float,
        is_static: bool,
    ) -> Keypoint2D:
        if keypoint.confidence < self.config.low_confidence:
            return _blend_position(previous, keypoint,
                                   self.config.low_confidence_alpha,
                                   "low_confidence_smoothed")
        if is_static and distance <= self.config.deadband_px:
            # 静止小位移也要低速跟随当前观测，避免画面看起来停在上一帧。
            return _blend_position(previous, keypoint, self.config.static_alpha,
                                   "smoothed")
        if self._is_static_jitter(keypoint, previous, distance, is_static):
            return _blend_position(previous, keypoint, self.config.jitter_alpha,
                                   "jitter_suppressed")

        alpha = self.config.static_alpha if is_static else self.config.moving_alpha
        return Keypoint2D(
            name=keypoint.name,
            x=previous.x * (1.0 - alpha) + keypoint.x * alpha,
            y=previous.y * (1.0 - alpha) + keypoint.y * alpha,
            confidence=keypoint.confidence,
            state="smoothed",
        )

    def _smooth_one_euro(
        self,
        keypoint: Keypoint2D,
        previous: Keypoint2D,
        history: _TrackHistory,
        distance: float,
        is_static: bool,
    ) -> Keypoint2D:
        if (self.config.sensitive_keypoints
                and keypoint.name not in self.config.sensitive_keypoints):
            return keypoint

        current = keypoint
        state = "one_euro_smoothed"
        if keypoint.confidence < self.config.low_confidence:
            current = _blend_position(previous, keypoint,
                                      self.config.low_confidence_alpha,
                                      "low_confidence_smoothed")
            state = current.state
        elif is_static and distance > self.config.max_static_jump_px:
            current = _blend_position(previous, keypoint, self.config.jitter_alpha,
                                      "jitter_suppressed")
            state = current.state

        keypoint_filter = history.filters.get(keypoint.name)
        if keypoint_filter is None:
            keypoint_filter = _KeypointOneEuroFilter(self.config,
                                                     self._sample_fps)
            keypoint_filter.reset(float(previous.x), float(previous.y))
            history.filters[keypoint.name] = keypoint_filter
        x, y = keypoint_filter.update(float(current.x), float(current.y))
        return Keypoint2D(
            name=keypoint.name,
            x=x,
            y=y,
            confidence=keypoint.confidence,
            state=state,
        )

    def _is_static_jitter(
        self,
        keypoint: Keypoint2D,
        previous: Keypoint2D,
        distance: float,
        is_static: bool,
    ) -> bool:
        if not is_static:
            return False
        if keypoint.name not in self.config.sensitive_keypoints:
            return False
        if distance <= self.config.max_static_jump_px:
            return False
        return keypoint.confidence <= previous.confidence + 0.2


class _KeypointOneEuroFilter:

    def __init__(self, config: KeypointSmoothingConfig,
                 sample_fps: float) -> None:
        self.config = config
        self._dt = 1.0 / sample_fps
        self._x = _OneEuroAxisState()
        self._y = _OneEuroAxisState()

    def update(self, x: float, y: float) -> tuple[float, float]:
        return (
            self._filter_axis(self._x, x),
            self._filter_axis(self._y, y),
        )

    def reset(self, x: float, y: float) -> None:
        self._x.value = x
        self._x.derivative = 0.0
        self._y.value = y
        self._y.derivative = 0.0

    def _filter_axis(self, state: _OneEuroAxisState, value: float) -> float:
        if state.value is None:
            state.value = value
            return value

        derivative = (value - state.value) / self._dt
        derivative_alpha = _smoothing_alpha(self.config.d_cutoff, self._dt)
        state.derivative = _exponential_smooth(
            derivative,
            state.derivative,
            derivative_alpha,
        )

        cutoff = self.config.min_cutoff + self.config.beta * abs(
            state.derivative)
        value_alpha = _smoothing_alpha(cutoff, self._dt)
        state.value = _exponential_smooth(value, state.value, value_alpha)
        return state.value


def _copy_position(
    previous: Keypoint2D,
    current: Keypoint2D,
    state: str,
) -> Keypoint2D:
    return Keypoint2D(
        name=current.name,
        x=previous.x,
        y=previous.y,
        confidence=current.confidence,
        state=state,
    )


def _blend_position(
    previous: Keypoint2D,
    current: Keypoint2D,
    alpha: float,
    state: str,
) -> Keypoint2D:
    return Keypoint2D(
        name=current.name,
        x=previous.x * (1.0 - alpha) + current.x * alpha,
        y=previous.y * (1.0 - alpha) + current.y * alpha,
        confidence=current.confidence,
        state=state,
    )


def _smoothing_alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _exponential_smooth(value: float, previous: float, alpha: float) -> float:
    return alpha * value + (1.0 - alpha) * previous


def _set_virtual_midpoint(
    keypoints: dict[str, Keypoint2D],
    target: str,
    left: str,
    right: str,
) -> None:
    left_point = keypoints.get(left)
    right_point = keypoints.get(right)
    if left_point is None or right_point is None:
        return
    if left_point.x is None or left_point.y is None:
        return
    if right_point.x is None or right_point.y is None:
        return
    keypoints[target] = Keypoint2D(
        name=target,
        x=(left_point.x + right_point.x) / 2.0,
        y=(left_point.y + right_point.y) / 2.0,
        confidence=min(left_point.confidence, right_point.confidence),
        state="virtual",
    )


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
