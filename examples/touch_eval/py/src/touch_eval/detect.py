"""The touch detector, ported from `TouchDetector.cpp` line by line.

The pipeline per frame is: build the evidence (one candidate per ball, person
and side, reduced to the closest keypoint of that side), pick the reported pair
(the followed ball when it has a candidate), turn that distance into a contact
level with hysteresis, and let an aggregator turn runs of contact frames into
events. A refractory rule then drops events the detector has just reported.

Everything the C++ does in `float` is done in `numpy.float32` here, in the same
order, because the reported distances, thresholds and therefore the contact
decisions depend on single precision rounding.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

from .observations import FrameInput

F32 = np.float32

#: Slot order of `Person::feet`: left ankle, right ankle, left big toe, left
#: small toe, left heel, right big toe, right small toe, right heel.
_LEFT_FOOT_SLOTS = (0, 2, 3, 4)

_INFINITY = float("inf")


def _clamp_f32(value: float, low: float, high: float) -> float:
    if value < low:
        return F32(low)
    if value > high:
        return F32(high)
    return F32(value)


def is_left_foot_slot(slot: int) -> bool:
    """`IsLeftFootSlot()`: the left keypoints of the pose model."""
    return slot in _LEFT_FOOT_SLOTS


@dataclass
class ContactCandidate:
    """One (ball, person, side) pair reduced to its closest foot keypoint."""

    distance_px: float = F32(0.0)
    person_height_px: float = F32(0.0)
    person_track_id: int = -1
    left_foot: bool = True
    foot_slot: int = -1
    ball_x: float = F32(0.0)
    ball_y: float = F32(0.0)
    ball_track_id: int = -1
    ball_confidence: float = F32(0.0)
    ball_tracked: bool = False


class ContactLevel:
    """`ContactLevel`: no evidence, inside the hysteresis band, or contact."""

    NONE = 0
    BAND = 1
    CONTACT = 2


@dataclass
class FrameContact:
    """Per frame evidence after the thresholds are applied."""

    frame_index: int = 0
    timestamp_sec: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    person_count: int = 0
    observed_ball_count: int = 0
    valid_foot_point_count: int = 0
    persons_with_feet: int = 0
    has_ball: bool = False
    level: int = ContactLevel.NONE
    candidate: ContactCandidate = field(default_factory=ContactCandidate)
    threshold_px: float = F32(0.0)

    def has_contact(self) -> bool:
        return self.level == ContactLevel.CONTACT


@dataclass
class FrameEvidence:
    """Everything one frame contributes apart from the thresholds."""

    frame_index: int = 0
    timestamp_sec: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    person_count: int = 0
    ball_count: int = 0
    observed_ball_count: int = 0
    valid_foot_point_count: int = 0
    persons_with_feet: int = 0
    candidates: list[ContactCandidate] = field(default_factory=list)


@dataclass
class TouchEvent:
    """One detected touch."""

    frame_index: int = 0
    timestamp_sec: float = 0.0
    first_frame_index: int = 0
    first_timestamp_sec: float = 0.0
    person_track_id: int = -1
    left_foot: bool = True
    distance_px: float = F32(0.0)
    threshold_px: float = F32(0.0)
    ball_confidence: float = F32(0.0)
    ball_tracked: bool = False
    ball_track_id: int = -1
    contact_frames: int = 0
    speed_before_px_per_sec: float = F32(0.0)
    speed_after_px_per_sec: float = F32(0.0)


def contact_threshold_px(
    params, frame_width: int, frame_height: int, person_height_px: float
) -> float:
    """`ContactThresholdPx()`: the enter threshold for one person, one frame."""
    low = F32(max(0.0, params.minContactPx))
    high = F32(max(low, params.maxContactPx))

    # Frame level term: a fraction of the shorter side of the source frame.
    shorter_side = F32(min(frame_width if frame_width > 0 else 0,
                           frame_height if frame_height > 0 else 0))
    frame_term = _clamp_f32(F32(shorter_side * params.contactDistanceRatio), low, high)

    # Person level term: a distant player's feet are small, so the ball must be
    # closer to them. Never wider than the frame term.
    person_term = _clamp_f32(F32(person_height_px * params.personHeightRatio), low, high)
    return F32(min(frame_term, person_term))


def collect_contact_candidates(
    frame: FrameInput, params, max_candidates: int
) -> list[ContactCandidate]:
    """`CollectContactCandidates()`: one candidate per ball, person and side."""
    candidates: list[ContactCandidate] = []
    if max_candidates == 0:
        return candidates

    for ball in frame.balls:
        if not (ball.observed and ball.confidence >= params.ballMinConfidence):
            continue
        for person in frame.persons:
            for side in (0, 1):
                left_side = side == 0
                best = ContactCandidate()
                found = False
                for slot in range(len(person.feet)):
                    if is_left_foot_slot(slot) != left_side:
                        continue
                    foot = person.feet[slot]
                    if not foot.valid:
                        continue
                    if foot.confidence < params.keypointConfidence:
                        continue
                    distance = F32(
                        np.hypot(
                            F32(ball.x - foot.x),
                            F32(ball.y - foot.y),
                        )
                    )
                    if found and distance >= best.distance_px:
                        continue
                    best.distance_px = distance
                    best.person_height_px = person.height_px
                    best.person_track_id = person.track_id
                    best.left_foot = left_side
                    best.foot_slot = slot
                    best.ball_x = ball.x
                    best.ball_y = ball.y
                    best.ball_track_id = ball.track_id
                    best.ball_confidence = ball.confidence
                    best.ball_tracked = ball.track_id > 0
                    found = True
                if found:
                    candidates.append(best)

    candidates.sort(key=lambda candidate: candidate.distance_px)
    return candidates[:max_candidates]


def build_frame_evidence(frame: FrameInput, params) -> FrameEvidence:
    """`BuildFrameEvidence()`: the counts the reports quote, plus the candidates."""
    evidence = FrameEvidence(
        frame_index=frame.frame_index,
        timestamp_sec=frame.timestamp_sec,
        frame_width=frame.frame_width,
        frame_height=frame.frame_height,
        person_count=len(frame.persons),
        ball_count=len(frame.balls),
    )
    for ball in frame.balls:
        if ball.observed and ball.confidence >= params.ballMinConfidence:
            evidence.observed_ball_count += 1
    for person in frame.persons:
        valid_points = sum(1 for foot in person.feet if foot.valid)
        evidence.valid_foot_point_count += valid_points
        if valid_points > 0:
            evidence.persons_with_feet += 1
    evidence.candidates = collect_contact_candidates(
        frame, params, max(1, params.candidateCount) if params.candidateCount > 0 else 1
    )
    return evidence


def select_contact(evidence: FrameEvidence, params) -> FrameContact:
    """`SelectContact()`: apply the thresholds of `params` to one frame."""
    out = FrameContact(
        frame_index=evidence.frame_index,
        timestamp_sec=evidence.timestamp_sec,
        frame_width=evidence.frame_width,
        frame_height=evidence.frame_height,
        person_count=evidence.person_count,
        observed_ball_count=evidence.observed_ball_count,
        valid_foot_point_count=evidence.valid_foot_point_count,
        persons_with_feet=evidence.persons_with_feet,
    )
    if not evidence.candidates:
        return out

    # The candidates are sorted by distance, so index 0 is the closest pair. A
    # tracked ball wins whenever one exists: the pipeline follows that ball, so
    # a touch of any other candidate must not masquerade as one of them.
    chosen = 0
    if params.preferTrackedBall:
        for index, candidate in enumerate(evidence.candidates):
            if candidate.ball_tracked:
                chosen = index
                break

    out.candidate = evidence.candidates[chosen]
    out.has_ball = True
    out.threshold_px = contact_threshold_px(
        params, evidence.frame_width, evidence.frame_height, out.candidate.person_height_px
    )
    if out.candidate.distance_px <= out.threshold_px:
        out.level = ContactLevel.CONTACT
    elif out.candidate.distance_px <= F32(out.threshold_px * params.releaseRatio):
        out.level = ContactLevel.BAND
    else:
        out.level = ContactLevel.NONE
    return out


@dataclass
class _ClusterFrame:
    frame: FrameContact
    in_contact: bool


@dataclass
class _BallSample:
    timestamp_sec: float
    x: float
    y: float


class TouchEventAggregator:
    """`TouchEventAggregator`: runs of contact frames become events."""

    def __init__(self, params) -> None:
        self.m_params = params
        self.reset()

    def reset(self) -> None:
        self.m_rejected_by_speed_gate = 0
        self.m_rejected_by_ball_gate = 0
        self.m_cluster: list[_ClusterFrame] = []
        self.m_active = False
        self.m_gap_frames = 0
        self.m_ball_history: deque[_BallSample] = deque()

    # --- ball speed diagnostics ----------------------------------------------
    def _remember_ball(self, frame: FrameContact) -> None:
        if not frame.has_ball:
            return
        self.m_ball_history.append(
            _BallSample(
                timestamp_sec=frame.timestamp_sec,
                x=frame.candidate.ball_x,
                y=frame.candidate.ball_y,
            )
        )
        keep_sec = max(0.05, float(self.m_params.speedWindowSec)) * 4.0
        newest = self.m_ball_history[-1].timestamp_sec
        while (
            len(self.m_ball_history) > 1
            and newest - self.m_ball_history[0].timestamp_sec > keep_sec
        ):
            self.m_ball_history.popleft()

    def _ball_speed(self, from_sec: float, to_sec: float) -> float:
        if len(self.m_ball_history) < 2:
            return F32(0.0)
        if to_sec <= from_sec:
            return F32(0.0)

        first: _BallSample | None = None
        last: _BallSample | None = None
        for sample in self.m_ball_history:
            if sample.timestamp_sec < from_sec or sample.timestamp_sec > to_sec:
                continue
            if first is None:
                first = sample
            last = sample
        if first is None or last is None or first is last:
            return F32(0.0)

        dt = last.timestamp_sec - first.timestamp_sec
        if dt <= 1e-6:
            return F32(0.0)
        distance = math.hypot(
            float(last.x) - float(first.x), float(last.y) - float(first.y)
        )
        return F32(distance / dt)

    # --- state machine --------------------------------------------------------
    def push(self, frame: FrameContact) -> list[TouchEvent]:
        self._remember_ball(frame)

        if frame.level != ContactLevel.NONE and not self.m_active:
            self.m_active = True
            self.m_gap_frames = 0
            self.m_cluster = []
        if not self.m_active:
            return []

        in_contact = frame.level == ContactLevel.CONTACT
        if frame.level == ContactLevel.NONE:
            # No admissible evidence: tolerate a few frames, then close. The
            # hysteresis band does not consume this budget, so a ball sitting
            # next to the foot keeps the cluster alive while a lost ball ends it.
            if self.m_gap_frames >= self.m_params.maxGapFrames:
                return self.close_cluster()
            self.m_gap_frames += 1
        else:
            self.m_gap_frames = 0

        self.m_cluster.append(_ClusterFrame(frame=frame, in_contact=in_contact))

        if len(self.m_cluster) >= max(1, self.m_params.maxClusterFrames):
            return self.close_cluster()
        return []

    def flush(self) -> list[TouchEvent]:
        return self.close_cluster()

    # --- cluster to events ----------------------------------------------------
    @staticmethod
    def _peak_index(cluster: Sequence[_ClusterFrame], start: int, end: int) -> int:
        """`peakIndex`: the farthest excursion between two minima."""
        best = start + 1
        best_distance = F32(-1.0)
        for index in range(start + 1, end):
            entry = cluster[index].frame
            distance = entry.candidate.distance_px if entry.has_ball else _INFINITY
            if distance > best_distance:
                best_distance = distance
                best = index
        return best

    def close_cluster(self) -> list[TouchEvent]:
        events: list[TouchEvent] = []
        if not self.m_active:
            return events

        cluster = self.m_cluster
        self.m_cluster = []
        self.m_active = False
        self.m_gap_frames = 0

        contact_indices = [i for i, entry in enumerate(cluster) if entry.in_contact]
        if len(contact_indices) < self.m_params.minContactFrames:
            return events

        # Local minima of the distance over the contact frames of the cluster.
        minima: list[int] = []
        count = len(contact_indices)
        for k in range(count):
            distance = cluster[contact_indices[k]].frame.candidate.distance_px
            below_left = k == 0 or distance <= cluster[
                contact_indices[k - 1]
            ].frame.candidate.distance_px
            below_right = k + 1 == count or distance <= cluster[
                contact_indices[k + 1]
            ].frame.candidate.distance_px
            if below_left and below_right:
                minima.append(contact_indices[k])
        if not minima:
            minima.append(contact_indices[0])

        # Merge minima that are too close to be two separate touches, and refuse
        # to split when the distance never rises well above the exit threshold.
        kept: list[int] = []
        for candidate in minima:
            if not kept:
                kept.append(candidate)
                continue
            previous = kept[-1]
            too_close = (candidate - previous) < max(1, self.m_params.minSplitFrames)
            rise_frames = 0
            if not too_close:
                release = F32(
                    min(
                        cluster[previous].frame.threshold_px,
                        cluster[candidate].frame.threshold_px,
                    )
                    * self.m_params.releaseRatio
                )
                for index in range(previous + 1, candidate):
                    between = cluster[index].frame
                    # A frame without an admissible ball counts as a rise: the
                    # ball left the neighbourhood of the foot entirely.
                    if not between.has_ball or between.candidate.distance_px > release:
                        rise_frames += 1
            separated = rise_frames >= max(1, self.m_params.minRiseFrames)
            if too_close or not separated:
                if (
                    cluster[candidate].frame.candidate.distance_px
                    < cluster[previous].frame.candidate.distance_px
                ):
                    kept[-1] = candidate
                continue
            kept.append(candidate)

        for index, minimum in enumerate(kept):
            # The event spans the encounter that contains its minimum; two
            # events are separated at the farthest excursion between them, so
            # the contact frames of one touch never leak into the next one.
            scan_from = 0 if index == 0 else self._peak_index(cluster, kept[index - 1], minimum)
            scan_to = (
                self._peak_index(cluster, minimum, kept[index + 1])
                if index + 1 < len(kept)
                else len(cluster)
            )

            first = minimum
            last = minimum
            contact_frames = 0
            have_first = False
            for index_in_cluster in range(scan_from, scan_to):
                if not cluster[index_in_cluster].in_contact:
                    continue
                if not have_first:
                    first = index_in_cluster
                    have_first = True
                last = index_in_cluster
                contact_frames += 1
            if contact_frames == 0:
                first = minimum
                last = minimum
            # A split segment still has to look like a touch on its own.
            if contact_frames < max(1, self.m_params.minContactFrames):
                continue

            reported = minimum
            if self.m_params.eventTimeMode == 1:
                reported = first
            elif self.m_params.eventTimeMode == 2:
                reported = last
            elif self.m_params.eventTimeMode == 3:
                reported = (first + last) // 2

            frame = cluster[reported].frame
            event = TouchEvent(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                first_frame_index=cluster[first].frame.frame_index,
                first_timestamp_sec=cluster[first].frame.timestamp_sec,
                person_track_id=frame.candidate.person_track_id,
                left_foot=frame.candidate.left_foot,
                distance_px=frame.candidate.distance_px,
                threshold_px=frame.threshold_px,
                ball_confidence=frame.candidate.ball_confidence,
                ball_tracked=frame.candidate.ball_tracked,
                ball_track_id=frame.candidate.ball_track_id,
                contact_frames=contact_frames,
                speed_before_px_per_sec=self._ball_speed(
                    frame.timestamp_sec - float(self.m_params.speedWindowSec),
                    frame.timestamp_sec,
                ),
                speed_after_px_per_sec=self._ball_speed(
                    frame.timestamp_sec,
                    frame.timestamp_sec + float(self.m_params.speedWindowSec),
                ),
            )
            if self.m_params.requireTrackedBall and not event.ball_tracked:
                self.m_rejected_by_ball_gate += 1
                continue
            if not self._passes_speed_gate(event):
                self.m_rejected_by_speed_gate += 1
                continue
            events.append(event)
        return events

    def _passes_speed_gate(self, event: TouchEvent) -> bool:
        check_change = self.m_params.minSpeedChangePxPerSec > 0.0
        check_ratio = self.m_params.minSpeedRatio > 1.0
        if not check_change and not check_ratio:
            return True

        before = event.speed_before_px_per_sec
        after = event.speed_after_px_per_sec
        if check_change and abs(F32(after - before)) < self.m_params.minSpeedChangePxPerSec:
            return False
        if check_ratio and before > 0.0 and after > 0.0:
            # Both speeds are known, so the ratio says something.
            high = max(before, after)
            low = min(before, after)
            if F32(high / low) < self.m_params.minSpeedRatio:
                return False
        return True


class TouchDetector:
    """`TouchDetector`: the aggregator plus the refractory rule."""

    def __init__(self, params) -> None:
        self.m_params = params
        self.m_aggregator = TouchEventAggregator(params)
        self.reset()

    def reset(self) -> None:
        self.m_aggregator.reset()
        self.m_recent_events: list[TouchEvent] = []
        self.m_last_contact = FrameContact()
        self.m_suppressed_by_refractory = 0
        self.m_suppressed_by_person_refractory = 0

    def last_contact(self) -> FrameContact:
        return self.m_last_contact

    @property
    def suppressed_by_refractory(self) -> int:
        return self.m_suppressed_by_refractory

    @property
    def suppressed_by_person_refractory(self) -> int:
        return self.m_suppressed_by_person_refractory

    def apply_refractory(self, events: Sequence[TouchEvent]) -> list[TouchEvent]:
        foot_window = self.m_params.refractoryFrames
        person_window = self.m_params.personRefractoryFrames
        if (foot_window <= 0 and person_window <= 0) or not events:
            self.m_recent_events.extend(events)
            return list(events)
        # The history has to reach as far back as the wider of the two rules.
        history = max(foot_window, person_window)

        kept: list[TouchEvent] = []
        for event in events:
            same_foot = False
            same_person = False
            for recent in self.m_recent_events:
                if recent.person_track_id != event.person_track_id:
                    continue
                gap = event.frame_index - recent.frame_index
                if gap < 0:
                    continue
                if (
                    foot_window > 0
                    and recent.left_foot == event.left_foot
                    and gap < foot_window
                ):
                    same_foot = True
                    break
                if person_window > 0 and gap < person_window:
                    same_person = True
            if same_foot:
                self.m_suppressed_by_refractory += 1
                continue
            if same_person:
                self.m_suppressed_by_person_refractory += 1
                continue
            self.m_recent_events.append(event)
            kept.append(event)

        # Only events inside the refractory window can matter.
        if kept:
            newest = kept[-1].frame_index
        elif self.m_recent_events:
            newest = self.m_recent_events[-1].frame_index
        else:
            newest = events[-1].frame_index
        while (
            self.m_recent_events
            and newest - self.m_recent_events[0].frame_index >= history
        ):
            self.m_recent_events.pop(0)
        return kept

    def push_evidence(self, evidence: FrameEvidence) -> list[TouchEvent]:
        self.m_last_contact = select_contact(evidence, self.m_params)
        return self.apply_refractory(self.m_aggregator.push(self.m_last_contact))

    def push(self, frame: FrameInput) -> list[TouchEvent]:
        return self.push_evidence(build_frame_evidence(frame, self.m_params))

    def flush(self) -> list[TouchEvent]:
        return self.apply_refractory(self.m_aggregator.flush())


@dataclass
class ReplayResult:
    events: list[TouchEvent] = field(default_factory=list)
    contacts: list[FrameContact] = field(default_factory=list)
    has_contacts: bool = False
    suppressed_by_refractory: int = 0
    suppressed_by_person_refractory: int = 0
    rejected_by_speed_gate: int = 0
    rejected_by_ball_gate: int = 0


def replay(frames: Iterable, params, keep_contacts: bool = False) -> ReplayResult:
    """`Replay()`: the same detector the online app runs, over cached frames.

    `frames` may hold `FrameEvidence` (what the C++ cache stores) or raw
    `FrameInput` lines, which are converted on the fly.
    """
    out = ReplayResult(has_contacts=keep_contacts)
    detector = TouchDetector(params)
    for item in frames:
        evidence = item if isinstance(item, FrameEvidence) else build_frame_evidence(item, params)
        out.events.extend(detector.push_evidence(evidence))
        if keep_contacts:
            out.contacts.append(detector.last_contact())
    out.events.extend(detector.flush())
    out.suppressed_by_refractory = detector.suppressed_by_refractory
    out.suppressed_by_person_refractory = detector.suppressed_by_person_refractory
    out.rejected_by_speed_gate = detector.m_aggregator.m_rejected_by_speed_gate
    out.rejected_by_ball_gate = detector.m_aggregator.m_rejected_by_ball_gate
    return out
