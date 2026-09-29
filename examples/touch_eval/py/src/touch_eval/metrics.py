"""Scoring, ported from `TouchMetrics.cpp`.

The measurement has two halves and they answer different questions:

* the **count aware** score treats every annotation as a window with as many
  slots as it claims touches (`(2次)` is two slots) and asks the detector to fill
  them. It is the number that says whether the detector found the touches that
  happened, and the one the report leads with;
* the **auxiliary** score is a plain one-to-one match inside a tolerance, kept
  because it is comparable with older runs.

Both rest on the same optimal assignment, so a slot never goes to whichever
event the sweep happened to see first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from . import detect as detect_module
from .detect import FrameContact, FrameEvidence, TouchEvent, replay
from .gt import GroundTruth, GroundTruthEvent
from .observations import VideoMetadata
from .params import DetectorParams

INFINITY = float("inf")


# ---------------------------------------------------------------------------
# windows and anchors
# ---------------------------------------------------------------------------


class Anchor:
    """`PointWindowAnchor`: which second a handwritten whole second names."""

    CENTRE = 0
    START = 1
    END = 2


def anchor_name(anchor: int) -> str:
    if anchor == Anchor.START:
        return "start"
    if anchor == Anchor.CENTRE:
        return "centre"
    return "end"


def anchor_rule(anchor: int) -> str:
    if anchor == Anchor.START:
        return "a row naming second S owns S..S+w"
    if anchor == Anchor.CENTRE:
        return "a row naming second S owns S-w/2..S+w/2"
    return "a row naming second S owns S-w..S"


def alternative_anchor(anchor: int) -> int:
    """The reading the report prints beside the one the caller asked for."""
    return Anchor.START if anchor == Anchor.CENTRE else Anchor.CENTRE


@dataclass
class GtWindow:
    from_sec: float = 0.0
    to_sec: float = 0.0


def window_of(
    event: GroundTruthEvent, point_window_sec: float, pad_sec: float, anchor: int
) -> GtWindow:
    """`WindowOf()`: the span of time one annotation stands for."""
    if event.is_interval():
        return GtWindow(event.start_sec - pad_sec, event.end_sec + pad_sec)
    if anchor == Anchor.START:
        return GtWindow(
            event.start_sec - pad_sec, event.start_sec + point_window_sec + pad_sec
        )
    if anchor == Anchor.END:
        return GtWindow(
            event.start_sec - point_window_sec - pad_sec, event.start_sec + pad_sec
        )
    half = 0.5 * point_window_sec
    return GtWindow(event.start_sec - half - pad_sec, event.start_sec + half + pad_sec)


def within_window(window: GtWindow, time_sec: float) -> bool:
    return window.from_sec <= time_sec < window.to_sec


def signed_dt(event: GroundTruthEvent, time_sec: float) -> float:
    """`SignedDt()`: negative before the annotation, positive after it."""
    if time_sec < event.start_sec:
        return time_sec - event.start_sec
    if time_sec > event.end_sec:
        return time_sec - event.end_sec
    return 0.0


def signed_offset(event: GroundTruthEvent, time_sec: float) -> float:
    """`SignedOffset()`: the same quantity, used by the count aware path."""
    if event.is_interval():
        return signed_dt(event, time_sec)
    return time_sec - event.start_sec


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------


@dataclass
class EvaluationOptions:
    """`EvaluationOptions`, with the defaults the C++ currently ships."""

    params: DetectorParams = field(default_factory=DetectorParams)
    tolerances: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0])
    #: The sweep starts at zero slack: an event counts for a row only when it
    #: falls inside the span that row writes (`[S-1, S)` for a whole second, the
    #: written span for an interval). The wider points stay as a diagnostic -
    #: they answer "how much slack would it take", not "did we find the touch".
    tolerance_frames: list[int] = field(default_factory=lambda: [0, 10, 30, 60, 120])
    #: End: a row that says "00:07S (2次)" claims the second that ends there.
    point_window_anchor: int = Anchor.END
    #: Zero, like the first sweep point: the reported one-to-one number is the
    #: strict one, so it reads the same way as the count aware metric. The
    #: parameter is kept - a caller that wants the +-0.5 s reading passes it.
    primary_tolerance_frames: int = 0
    primary_tolerance_sec: float = 0.0
    tolerances_in_seconds: bool = False
    sweep_ratios: list[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.015, 0.02, 0.03, 0.045, 0.06, 0.08, 0.12, 0.2]
    )
    window_sec: float = 0.5
    point_window_sec: float = 0.0
    window_pad_sec: float = 0.0
    adjacency_sec: float = 1.5
    duplicate_sec: float = 0.5


def effective_point_window(ground_truth: GroundTruth, options: EvaluationOptions) -> float:
    """`EffectivePointWindow()`: the CLI wins, the source decides otherwise."""
    if options.point_window_sec > 0.0:
        return options.point_window_sec
    return ground_truth.default_point_window_sec


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------


def max_cardinality_match(preference: Sequence[Sequence[int]]) -> tuple[int, list[int]]:
    """`MaxCardinalityMatch()`: Kuhn's augmenting paths, iteratively.

    Returns the matched count and, per left node, its right node (-1 when
    unmatched). The preference lists are nearest first, so among the assignments
    of maximum size this keeps the closer pairs.
    """
    left_count = len(preference)
    match_of_event = [-1] * left_count
    right_count = 0
    for candidates in preference:
        for right in candidates:
            if right + 1 > right_count:
                right_count = right + 1
    match_of_right = [-1] * right_count

    def augment(start: int) -> bool:
        visited = [False] * right_count
        stack_left = [start]
        stack_cursor = [0]
        while stack_left:
            left = stack_left[-1]
            if stack_cursor[-1] >= len(preference[left]):
                stack_left.pop()
                stack_cursor.pop()
                continue
            right = preference[left][stack_cursor[-1]]
            stack_cursor[-1] += 1
            if visited[right]:
                continue
            visited[right] = True
            if match_of_right[right] == -1:
                free_right = right
                for index in range(len(stack_left) - 1, -1, -1):
                    event = stack_left[index]
                    previous = match_of_event[event]
                    match_of_event[event] = free_right
                    match_of_right[free_right] = event
                    free_right = previous
                    if free_right == -1:
                        break
                return True
            stack_left.append(match_of_right[right])
            stack_cursor.append(0)
        return False

    matched = 0
    for index in range(left_count):
        if augment(index):
            matched += 1
    return matched, match_of_event


@dataclass
class MatchOutcome:
    true_positives: int = 0
    event_gt: list[int] = field(default_factory=list)
    gt_event: list[int] = field(default_factory=list)
    signed_dt_sec: list[float] = field(default_factory=list)


def match_events(
    ground_truth: GroundTruth, events: Sequence[TouchEvent], tolerance_sec: float
) -> MatchOutcome:
    """`MatchEvents()`: one-to-one match inside a tolerance."""
    outcome = MatchOutcome(
        event_gt=[-1] * len(events), gt_event=[-1] * len(ground_truth.events)
    )
    per_event: list[list[tuple[float, float, int]]] = [[] for _ in events]
    for g, annotation in enumerate(ground_truth.events):
        for e, event in enumerate(events):
            offset = signed_dt(annotation, event.timestamp_sec)
            if abs(offset) > tolerance_sec:
                continue
            per_event[e].append((abs(offset), offset, g))
    for candidates in per_event:
        candidates.sort(key=lambda item: (item[0], item[2]))

    preference = [[item[2] for item in candidates] for candidates in per_event]
    matched, chosen_gt = max_cardinality_match(preference)
    outcome.true_positives = matched

    chosen: list[tuple[int, float]] = []
    for e, right in enumerate(chosen_gt):
        if right < 0:
            continue
        outcome.event_gt[e] = right
        outcome.gt_event[right] = e
        for candidate in per_event[e]:
            if candidate[2] == right:
                chosen.append((e, candidate[1]))
                break
    chosen.sort(key=lambda item: item[0])
    outcome.signed_dt_sec = [value for _, value in chosen]
    return outcome


@dataclass
class InstanceMatch:
    true_positives: int = 0
    event_gt: list[int] = field(default_factory=list)
    gt_assigned: list[int] = field(default_factory=list)
    signed_dt_sec: list[float] = field(default_factory=list)


def match_instances(
    ground_truth: GroundTruth,
    events: Sequence[TouchEvent],
    point_window_sec: float,
    pad_sec: float,
    anchor: int,
) -> InstanceMatch:
    """`MatchInstances()`: every annotation expands into `count` slots."""
    match = InstanceMatch(
        event_gt=[-1] * len(events), gt_assigned=[0] * len(ground_truth.events)
    )
    slot_gt: list[int] = []
    per_event: list[list[tuple[float, int, int]]] = [[] for _ in events]
    for g, annotation in enumerate(ground_truth.events):
        capacity = max(1, annotation.count)
        first_slot = len(slot_gt)
        slot_gt.extend([g] * capacity)
        window = window_of(annotation, point_window_sec, pad_sec, anchor)
        for e, event in enumerate(events):
            if not within_window(window, event.timestamp_sec):
                continue
            offset = abs(signed_offset(annotation, event.timestamp_sec))
            for k in range(capacity):
                per_event[e].append((offset, first_slot + k, event.frame_index))
    for candidates in per_event:
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    preference = [[item[1] for item in candidates] for candidates in per_event]
    matched, chosen_slot = max_cardinality_match(preference)
    match.true_positives = matched

    for e, slot in enumerate(chosen_slot):
        if slot < 0:
            continue
        gt_index = slot_gt[slot]
        match.event_gt[e] = gt_index
        match.gt_assigned[gt_index] += 1
    for e, slot in enumerate(chosen_slot):
        if slot < 0:
            continue
        match.signed_dt_sec.append(
            signed_offset(ground_truth.events[match.event_gt[e]], events[e].timestamp_sec)
        )
    return match


# ---------------------------------------------------------------------------
# percentiles and scores
# ---------------------------------------------------------------------------


@dataclass
class Percentiles:
    samples: int = 0
    mean: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0


def compute_percentiles(values: Iterable[float]) -> Percentiles:
    """`ComputePercentiles()`: rank = fraction * (n-1), rounded to an index."""
    ordered = sorted(values)
    result = Percentiles()
    if not ordered:
        return result
    result.samples = len(ordered)
    result.min = ordered[0]
    result.max = ordered[-1]
    result.mean = math.fsum(ordered) / len(ordered)

    def at(fraction: float) -> float:
        rank = fraction * (len(ordered) - 1)
        index = int(rank + 0.5)
        if index >= len(ordered):
            index = len(ordered) - 1
        return ordered[index]

    result.p50 = at(0.50)
    result.p90 = at(0.90)
    result.p95 = at(0.95)
    result.p99 = at(0.99)
    return result


@dataclass
class InstanceScore:
    point_window_sec: float = 0.0
    window_pad_sec: float = 0.0
    capacity: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    windows_with_event: int = 0
    windows_exact: int = 0
    suppressed_by_refractory: int = 0
    suppressed_by_person_refractory: int = 0
    rejected_by_speed_gate: int = 0
    rejected_by_ball_gate: int = 0
    signed_dt_sec: Percentiles = field(default_factory=Percentiles)
    abs_dt_sec: Percentiles = field(default_factory=Percentiles)


def _fill_rates(score) -> None:
    if score.true_positives + score.false_positives > 0:
        score.precision = score.true_positives / (
            score.true_positives + score.false_positives
        )
    if score.true_positives + score.false_negatives > 0:
        score.recall = score.true_positives / (
            score.true_positives + score.false_negatives
        )
    if score.precision + score.recall > 0.0:
        score.f1 = (
            2.0 * score.precision * score.recall / (score.precision + score.recall)
        )


def score_instances(
    ground_truth: GroundTruth, events: Sequence[TouchEvent], options: EvaluationOptions
) -> tuple[InstanceScore, InstanceMatch]:
    """`ScoreInstances()`: the count aware score, plus the match behind it."""
    score = InstanceScore(
        point_window_sec=effective_point_window(ground_truth, options),
        window_pad_sec=options.window_pad_sec,
        capacity=ground_truth.touch_count(),
    )
    match = match_instances(
        ground_truth,
        events,
        score.point_window_sec,
        score.window_pad_sec,
        options.point_window_anchor,
    )
    score.true_positives = match.true_positives
    score.false_positives = len(events) - match.true_positives
    score.false_negatives = score.capacity - match.true_positives
    _fill_rates(score)
    for g, assigned in enumerate(match.gt_assigned):
        if assigned > 0:
            score.windows_with_event += 1
        if assigned == max(1, ground_truth.events[g].count):
            score.windows_exact += 1
    score.signed_dt_sec = compute_percentiles(match.signed_dt_sec)
    score.abs_dt_sec = compute_percentiles(abs(value) for value in match.signed_dt_sec)
    return score, match


@dataclass
class ToleranceScore:
    tolerance_sec: float = 0.0
    tolerance_frames: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    signed_dt_sec: Percentiles = field(default_factory=Percentiles)
    abs_dt_sec: Percentiles = field(default_factory=Percentiles)


def score_tolerance(
    ground_truth: GroundTruth,
    events: Sequence[TouchEvent],
    tolerance_sec: float,
    fps: float,
) -> ToleranceScore:
    """`ScoreTolerance()`: the auxiliary one-to-one score."""
    score = ToleranceScore(tolerance_sec=tolerance_sec)
    if fps > 0.0:
        score.tolerance_frames = int(round(tolerance_sec * fps))
    match = match_events(ground_truth, events, tolerance_sec)
    score.true_positives = match.true_positives
    score.false_positives = len(events) - match.true_positives
    # The auxiliary score counts rows, not slots, so it stays comparable with
    # the numbers published before the count aware metric existed.
    score.false_negatives = len(ground_truth.events) - match.true_positives
    _fill_rates(score)
    signed = []
    for e, gt_index in enumerate(match.event_gt):
        if gt_index < 0:
            continue
        signed.append(signed_dt(ground_truth.events[gt_index], events[e].timestamp_sec))
    score.signed_dt_sec = compute_percentiles(signed)
    score.abs_dt_sec = compute_percentiles(abs(value) for value in signed)
    return score


@dataclass
class CountBucket:
    gt_index: int = 0
    start_sec: float = 0.0
    end_sec: float = 0.0
    span_from_sec: float = 0.0
    span_to_sec: float = 0.0
    gt_count: int = 0
    predicted_count: int = 0
    exact: bool = False


@dataclass
class CountScore:
    records: int = 0
    exact_records: int = 0
    over_records: int = 0
    under_records: int = 0
    gt_touches: int = 0
    predicted_in_windows: int = 0
    mean_abs_count_error: float = 0.0
    buckets: list[CountBucket] = field(default_factory=list)


def score_counts(
    ground_truth: GroundTruth,
    events: Sequence[TouchEvent],
    point_window_sec: float,
    pad_sec: float,
    anchor: int,
) -> CountScore:
    """`ScoreCounts()`: per annotation, how many touches were detected in it."""
    score = CountScore(
        records=len(ground_truth.events), gt_touches=ground_truth.touch_count()
    )
    for annotation in ground_truth.events:
        bucket = CountBucket(
            gt_index=annotation.index,
            start_sec=annotation.start_sec,
            end_sec=annotation.end_sec,
            gt_count=annotation.count,
        )
        window = window_of(annotation, point_window_sec, pad_sec, anchor)
        bucket.predicted_count = sum(
            1 for event in events if within_window(window, event.timestamp_sec)
        )
        bucket.span_from_sec = window.from_sec
        bucket.span_to_sec = window.to_sec
        bucket.exact = bucket.predicted_count == bucket.gt_count
        if bucket.exact:
            score.exact_records += 1
        elif bucket.predicted_count > bucket.gt_count:
            score.over_records += 1
        else:
            score.under_records += 1
        score.predicted_in_windows += bucket.predicted_count
        score.mean_abs_count_error += abs(bucket.predicted_count - bucket.gt_count)
        score.buckets.append(bucket)
    if score.buckets:
        score.mean_abs_count_error /= len(score.buckets)
    return score


class Verdict:
    """`EventVerdict`: why an event that matched no annotation was kept out."""

    ASSIGNED = "assigned"
    DUPLICATE = "duplicate"
    INSIDE_NO_TOUCH = "inside_no_touch"
    IN_WINDOW_NO_SLOT = "in_window_no_slot"
    ADJACENT = "adjacent"
    UNANNOTATED = "unannotated"


@dataclass
class FalsePositiveBreakdown:
    duplicates: int = 0
    inside_no_touch: int = 0
    in_window_no_slot: int = 0
    adjacent: int = 0
    unannotated: int = 0
    event_verdict: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


def describe_event(event: TouchEvent) -> str:
    """`DescribeEvent()`: the one line the false positive examples quote."""
    return (
        f"{event.timestamp_sec:.2f} s frame {event.frame_index} "
        f"p{event.person_track_id} {'left' if event.left_foot else 'right'} "
        f"{event.distance_px:.1f} px"
    )


def classify_false_positives(
    ground_truth: GroundTruth,
    events: Sequence[TouchEvent],
    event_gt: Sequence[int],
    options: EvaluationOptions,
) -> FalsePositiveBreakdown:
    """`ClassifyFalsePositives()`: a mutually exclusive reason per event."""
    breakdown = FalsePositiveBreakdown(
        event_verdict=[Verdict.ASSIGNED] * len(events)
    )
    point_window_sec = effective_point_window(ground_truth, options)
    windows = [
        window_of(annotation, point_window_sec, options.window_pad_sec,
                  options.point_window_anchor)
        for annotation in ground_truth.events
    ]

    for e, event in enumerate(events):
        if event_gt[e] >= 0:
            continue
        inside_no_touch = False
        for annotation in ground_truth.no_touch:
            window = window_of(annotation, point_window_sec, options.window_pad_sec,
                               options.point_window_anchor)
            if within_window(window, event.timestamp_sec):
                inside_no_touch = True
                break

        duplicate = False
        nearest_assigned_sec = 0.0
        for other, other_gt in enumerate(event_gt):
            if duplicate:
                break
            if other_gt < 0:
                continue
            if events[other].person_track_id != event.person_track_id:
                continue
            if events[other].left_foot != event.left_foot:
                continue
            distance = abs(events[other].timestamp_sec - event.timestamp_sec)
            if distance <= options.duplicate_sec:
                duplicate = True
                nearest_assigned_sec = distance

        nearest_window_sec = INFINITY
        for window in windows:
            if within_window(window, event.timestamp_sec):
                distance = 0.0
            else:
                before = window.from_sec - event.timestamp_sec
                after = event.timestamp_sec - window.to_sec
                distance = before if before > 0.0 else after
            nearest_window_sec = min(nearest_window_sec, distance)

        example = describe_event(event)
        if inside_no_touch:
            breakdown.inside_no_touch += 1
            breakdown.event_verdict[e] = Verdict.INSIDE_NO_TOUCH
            example += " - inside a no_touch span"
        elif duplicate:
            breakdown.duplicates += 1
            breakdown.event_verdict[e] = Verdict.DUPLICATE
            example += (
                f" - same person and foot as an assigned event "
                f"{nearest_assigned_sec:.2f} s away"
            )
        elif nearest_window_sec == 0.0:
            breakdown.in_window_no_slot += 1
            breakdown.event_verdict[e] = Verdict.IN_WINDOW_NO_SLOT
            example += " - inside an annotation window, but every slot was taken"
        elif nearest_window_sec <= options.adjacency_sec:
            breakdown.adjacent += 1
            breakdown.event_verdict[e] = Verdict.ADJACENT
            example += f" - {nearest_window_sec:.2f} s outside the nearest window"
        else:
            breakdown.unannotated += 1
            breakdown.event_verdict[e] = Verdict.UNANNOTATED
            example += f" - no annotation within {options.adjacency_sec} s"
        if len(breakdown.examples) < 24:
            breakdown.examples.append(example)
    return breakdown


# ---------------------------------------------------------------------------
# frame diagnostics
# ---------------------------------------------------------------------------


@dataclass
class WindowStats:
    has_detected_ball: bool = False
    has_usable_ball: bool = False
    has_usable_feet: bool = False
    nearest_distance_px: float = INFINITY
    nearest_usable_distance_px: float = INFINITY
    nearest_person_height_px: float = 0.0
    nearest_frame_width: int = 0
    nearest_frame_height: int = 0
    nearest_timestamp_sec: float = 0.0
    max_ball_confidence: float = 0.0
    frames: int = 0


def scan_window(
    frames: Sequence[FrameEvidence], timestamps: Sequence[float],
    from_sec: float, to_sec: float
) -> WindowStats:
    """`ScanWindow()`: what the frames of one window contained."""
    stats = WindowStats()
    begin = _lower_bound(timestamps, from_sec)
    for index in range(begin, len(frames)):
        frame = frames[index]
        if frame.timestamp_sec > to_sec:
            break
        stats.frames += 1
        if frame.ball_count > 0:
            stats.has_detected_ball = True
        if frame.observed_ball_count > 0:
            stats.has_usable_ball = True
        if frame.valid_foot_point_count > 0:
            stats.has_usable_feet = True
        if not frame.candidates:
            continue
        nearest = frame.candidates[0]
        if nearest.distance_px < stats.nearest_distance_px:
            stats.nearest_distance_px = nearest.distance_px
            stats.nearest_person_height_px = nearest.person_height_px
            stats.nearest_frame_width = frame.frame_width
            stats.nearest_frame_height = frame.frame_height
            stats.nearest_timestamp_sec = frame.timestamp_sec
        if frame.observed_ball_count > 0 and (
            nearest.distance_px < stats.nearest_usable_distance_px
        ):
            stats.nearest_usable_distance_px = nearest.distance_px
        stats.max_ball_confidence = max(
            stats.max_ball_confidence, float(nearest.ball_confidence)
        )
    return stats


def _lower_bound(values: Sequence[float], target: float) -> int:
    low = 0
    high = len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low


@dataclass
class FailureGroup:
    reason: str = ""
    count: int = 0
    examples: list[str] = field(default_factory=list)


def add_failure(groups: list[FailureGroup], reason: str, example: str) -> None:
    """`AddFailure()`: group by reason, keep the first eight examples."""
    for group in groups:
        if group.reason != reason:
            continue
        group.count += 1
        if len(group.examples) < 8:
            group.examples.append(example)
        return
    groups.append(FailureGroup(reason=reason, count=1, examples=[example]))


@dataclass
class Diagnostics:
    records: int = 0
    with_detected_ball: int = 0
    with_usable_ball: int = 0
    with_usable_feet: int = 0
    nearest_distance_px: Percentiles = field(default_factory=Percentiles)
    nearest_usable_distance_px: Percentiles = field(default_factory=Percentiles)
    background_nearest_usable_distance_px: Percentiles = field(
        default_factory=Percentiles
    )
    person_height_px: Percentiles = field(default_factory=Percentiles)


@dataclass
class SweepPoint:
    contact_distance_ratio: float = 0.0
    threshold_px_ref: float = 0.0
    tolerance_sec: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    instance_precision: float = 0.0
    instance_recall: float = 0.0
    instance_f1: float = 0.0


@dataclass
class LowQualityScore:
    rows: int = 0
    touches: int = 0
    excluded_events: int = 0
    instance: InstanceScore = field(default_factory=InstanceScore)
    tolerances: list[ToleranceScore] = field(default_factory=list)
    primary: ToleranceScore = field(default_factory=ToleranceScore)
    counts: CountScore = field(default_factory=CountScore)
    windows_with_usable_ball: int = 0
    windows_with_usable_feet: int = 0
    person_height_px: Percentiles = field(default_factory=Percentiles)
    nearest_usable_distance_px: Percentiles = field(default_factory=Percentiles)


@dataclass
class EvaluationResult:
    frames: int = 0
    duration_sec: float = 0.0
    gt_events: int = 0
    gt_touches: int = 0
    predicted_events: int = 0
    covered_from_sec: float = 0.0
    covered_to_sec: float = 0.0
    gt_events_covered: int = 0
    partial_run: bool = False
    fps: float = 0.0
    tolerances_in_frames: bool = False
    instance: InstanceScore = field(default_factory=InstanceScore)
    anchor_alternative: int = Anchor.CENTRE
    anchor_alternative_instance: InstanceScore = field(default_factory=InstanceScore)
    anchor_alternative_counts: CountScore = field(default_factory=CountScore)
    has_anchor_alternative: bool = False
    low_quality: LowQualityScore = field(default_factory=LowQualityScore)
    false_positives: FalsePositiveBreakdown = field(
        default_factory=FalsePositiveBreakdown
    )
    tolerances: list[ToleranceScore] = field(default_factory=list)
    primary: ToleranceScore = field(default_factory=ToleranceScore)
    counts: CountScore = field(default_factory=CountScore)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    failures: list[FailureGroup] = field(default_factory=list)
    sweep: list[SweepPoint] = field(default_factory=list)
    events: list[TouchEvent] = field(default_factory=list)
    event_gt_index: list[int] = field(default_factory=list)
    contacts: list[FrameContact] = field(default_factory=list)


def resolve_frame_rate(
    frames: Sequence[FrameEvidence], ground_truth: GroundTruth
) -> float:
    """`ResolveFrameRate()`: the annotation's fps, else the frame spacing."""
    if ground_truth.fps > 0.0:
        return ground_truth.fps
    for index in range(1, len(frames)):
        step = frames[index].timestamp_sec - frames[index - 1].timestamp_sec
        if step > 1e-9:
            return 1.0 / step
    return 0.0


def resolve_tolerances(
    options: EvaluationOptions,
    ground_truth: GroundTruth,
    frames: Sequence[FrameEvidence],
) -> tuple[EvaluationOptions, bool, float]:
    """`ResolveTolerances()`: frame tolerances become seconds when fps is known."""
    fps = resolve_frame_rate(frames, ground_truth)
    used_frames = False
    if options.tolerances_in_seconds or fps <= 0.0 or not options.tolerance_frames:
        return options, used_frames, fps
    effective = EvaluationOptions(**vars(options))
    effective.tolerances = [count / fps for count in options.tolerance_frames]
    if options.primary_tolerance_frames > 0:
        effective.primary_tolerance_sec = options.primary_tolerance_frames / fps
    used_frames = True
    return effective, used_frames, fps


def without_low_quality(ground_truth: GroundTruth) -> GroundTruth:
    """`WithoutLowQuality()`: the same annotations, flagged rows removed."""
    clean = GroundTruth(
        source=ground_truth.source,
        source_kind=ground_truth.source_kind,
        fps=ground_truth.fps,
        default_point_window_sec=ground_truth.default_point_window_sec,
        warnings=list(ground_truth.warnings),
    )
    clean.events = [e for e in ground_truth.events if not e.low_quality]
    clean.no_touch = [e for e in ground_truth.no_touch if not e.low_quality]
    return clean


def drop_low_quality_events(
    ground_truth: GroundTruth, events: Sequence[TouchEvent], options: EvaluationOptions
) -> tuple[list[TouchEvent], int]:
    """`DropLowQualityEvents()`: predictions only a flagged row explains."""
    point_window_sec = effective_point_window(ground_truth, options)
    windows = [
        window_of(row, point_window_sec, options.window_pad_sec,
                  options.point_window_anchor)
        for row in list(ground_truth.events) + list(ground_truth.no_touch)
        if row.low_quality
    ]
    kept: list[TouchEvent] = []
    excluded = 0
    for event in events:
        if any(within_window(window, event.timestamp_sec) for window in windows):
            excluded += 1
            continue
        kept.append(event)
    return kept, excluded


# ---------------------------------------------------------------------------
# the whole evaluation
# ---------------------------------------------------------------------------


def evaluate_frames(
    frames: Sequence[FrameEvidence],
    ground_truth: GroundTruth,
    options: EvaluationOptions,
    video: VideoMetadata | None = None,
) -> EvaluationResult:
    """`EvaluateFrames()`: replay once, score in every reading of the rules."""
    result = EvaluationResult(
        frames=len(frames),
        gt_events=len(ground_truth.events),
        gt_touches=ground_truth.touch_count(),
    )
    effective, used_frames, fps = resolve_tolerances(options, ground_truth, frames)
    result.fps = fps
    result.tolerances_in_frames = used_frames

    if frames:
        result.duration_sec = frames[-1].timestamp_sec - frames[0].timestamp_sec
        result.covered_from_sec = frames[0].timestamp_sec
        result.covered_to_sec = frames[-1].timestamp_sec

    for annotation in ground_truth.events:
        if (
            annotation.end_sec + effective.primary_tolerance_sec
            < result.covered_from_sec
            or annotation.start_sec - effective.primary_tolerance_sec
            > result.covered_to_sec
        ):
            continue
        result.gt_events_covered += 1
    result.partial_run = result.gt_events_covered < result.gt_events

    played = replay(frames, effective.params, keep_contacts=True)
    result.events = played.events
    result.contacts = played.contacts
    result.predicted_events = len(played.events)

    result.instance, instance_match = score_instances(
        ground_truth, played.events, effective
    )
    result.instance.suppressed_by_refractory = played.suppressed_by_refractory
    result.instance.suppressed_by_person_refractory = (
        played.suppressed_by_person_refractory
    )
    result.instance.rejected_by_speed_gate = played.rejected_by_speed_gate
    result.instance.rejected_by_ball_gate = played.rejected_by_ball_gate
    result.event_gt_index = instance_match.event_gt

    result.false_positives = classify_false_positives(
        ground_truth, played.events, result.event_gt_index, effective
    )

    for tolerance in effective.tolerances:
        result.tolerances.append(
            score_tolerance(ground_truth, played.events, tolerance, result.fps)
        )
    result.primary = score_tolerance(
        ground_truth, played.events, effective.primary_tolerance_sec, result.fps
    )

    result.counts = score_counts(
        ground_truth,
        played.events,
        result.instance.point_window_sec,
        effective.window_pad_sec,
        effective.point_window_anchor,
    )

    # The same events read with the other anchor, so the choice is visible.
    flipped = EvaluationOptions(**vars(effective))
    flipped.point_window_anchor = alternative_anchor(effective.point_window_anchor)
    result.anchor_alternative = flipped.point_window_anchor
    result.anchor_alternative_instance, _ = score_instances(
        ground_truth, played.events, flipped
    )
    result.anchor_alternative_instance.suppressed_by_refractory = (
        played.suppressed_by_refractory
    )
    result.anchor_alternative_instance.rejected_by_speed_gate = (
        played.rejected_by_speed_gate
    )
    result.anchor_alternative_instance.rejected_by_ball_gate = (
        played.rejected_by_ball_gate
    )
    result.anchor_alternative_counts = score_counts(
        ground_truth,
        played.events,
        result.anchor_alternative_instance.point_window_sec,
        flipped.window_pad_sec,
        flipped.point_window_anchor,
    )
    result.has_anchor_alternative = bool(ground_truth.events)

    # --- frame diagnostics ---------------------------------------------------
    timestamps = [frame.timestamp_sec for frame in frames]
    all_nearest: list[float] = []
    all_nearest_usable: list[float] = []
    for frame in frames:
        if not frame.candidates:
            continue
        all_nearest.append(frame.candidates[0].distance_px)
        if frame.observed_ball_count > 0:
            all_nearest_usable.append(frame.candidates[0].distance_px)
    result.diagnostics.background_nearest_usable_distance_px = compute_percentiles(
        all_nearest_usable
    )

    window_nearest: list[float] = []
    window_nearest_usable: list[float] = []
    window_heights: list[float] = []
    for g, annotation in enumerate(ground_truth.events):
        stats = scan_window(
            frames,
            timestamps,
            annotation.start_sec - effective.window_sec,
            annotation.end_sec + effective.window_sec,
        )
        result.diagnostics.records += 1
        if stats.has_detected_ball:
            result.diagnostics.with_detected_ball += 1
        if stats.has_usable_ball:
            result.diagnostics.with_usable_ball += 1
        if stats.has_usable_feet:
            result.diagnostics.with_usable_feet += 1
        if stats.nearest_distance_px < INFINITY:
            window_nearest.append(stats.nearest_distance_px)
            window_heights.append(stats.nearest_person_height_px)
        if stats.nearest_usable_distance_px < INFINITY:
            window_nearest_usable.append(stats.nearest_usable_distance_px)

        if instance_match.gt_assigned[g] > 0:
            continue
        example = (
            f"gt#{annotation.index} @ {_format_seconds(annotation.start_sec)}"
        )
        if annotation.is_interval():
            example += f"-{_format_seconds(annotation.end_sec)}"
        example += (
            f" (window {_format_seconds(annotation.start_sec - effective.window_sec)}"
            f"-{_format_seconds(annotation.end_sec + effective.window_sec)})"
        )
        if not stats.has_detected_ball:
            reason = "no ball detected in the window"
        elif not stats.has_usable_ball:
            reason = "ball detected but never admissible (extrapolated or low confidence)"
        elif not stats.has_usable_feet:
            reason = "no usable foot keypoints (missing or interpolated)"
            example += f", nearest {stats.nearest_distance_px:.2f} px"
        else:
            threshold = detect_module.contact_threshold_px(
                effective.params,
                stats.nearest_frame_width,
                stats.nearest_frame_height,
                detect_module.F32(stats.nearest_person_height_px),
            )
            if stats.nearest_usable_distance_px <= threshold:
                reason = (
                    "contact frames present but no accepted event (too short or merged)"
                )
                example += (
                    f", nearest {stats.nearest_usable_distance_px:.2f} px "
                    f"<= tau {threshold:.2f} px"
                )
            else:
                reason = "ball and feet present but never closer than the threshold"
                example += (
                    f", nearest {stats.nearest_usable_distance_px:.2f} px "
                    f"> tau {threshold:.2f} px"
                )
        add_failure(result.failures, reason, example)

    result.diagnostics.nearest_distance_px = compute_percentiles(window_nearest)
    result.diagnostics.nearest_usable_distance_px = compute_percentiles(
        window_nearest_usable
    )
    result.diagnostics.person_height_px = compute_percentiles(window_heights)

    for e, gt_index in enumerate(result.event_gt_index):
        if gt_index >= 0:
            continue
        add_failure(
            result.failures,
            "false positive (outside every annotation window)",
            describe_event(result.events[e]),
        )

    # --- calibration sweep ---------------------------------------------------
    for ratio in effective.sweep_ratios:
        swept = EvaluationOptions(**vars(effective))
        swept.params = effective.params.replace(contactDistanceRatio=ratio)
        swept_replay = replay(frames, swept.params)
        threshold_ref = detect_module.contact_threshold_px(
            swept.params, 1280, 720, detect_module.F32(1.0e6)
        )
        tolerance_score = score_tolerance(
            ground_truth, swept_replay.events, effective.primary_tolerance_sec, result.fps
        )
        instance_score, _ = score_instances(ground_truth, swept_replay.events, swept)
        result.sweep.append(
            SweepPoint(
                contact_distance_ratio=ratio,
                threshold_px_ref=threshold_ref,
                tolerance_sec=effective.primary_tolerance_sec,
                true_positives=tolerance_score.true_positives,
                false_positives=tolerance_score.false_positives,
                false_negatives=tolerance_score.false_negatives,
                precision=tolerance_score.precision,
                recall=tolerance_score.recall,
                f1=tolerance_score.f1,
                instance_precision=instance_score.precision,
                instance_recall=instance_score.recall,
                instance_f1=instance_score.f1,
            )
        )

    # --- the same run without the rows nobody can judge ----------------------
    if ground_truth.low_quality_rows() > 0:
        result.low_quality = _score_low_quality(
            frames, ground_truth, played.events, effective, result, timestamps
        )
    return result


def _score_low_quality(
    frames: Sequence[FrameEvidence],
    ground_truth: GroundTruth,
    events: Sequence[TouchEvent],
    effective: EvaluationOptions,
    result: EvaluationResult,
    timestamps: Sequence[float],
) -> LowQualityScore:
    clean_ground_truth = without_low_quality(ground_truth)
    clean_events, excluded = drop_low_quality_events(ground_truth, events, effective)

    score = LowQualityScore(
        rows=ground_truth.low_quality_rows(),
        touches=ground_truth.low_quality_touch_count(),
        excluded_events=excluded,
    )
    score.instance, _ = score_instances(clean_ground_truth, clean_events, effective)
    score.instance.suppressed_by_refractory = result.instance.suppressed_by_refractory
    score.instance.suppressed_by_person_refractory = (
        result.instance.suppressed_by_person_refractory
    )
    score.instance.rejected_by_speed_gate = result.instance.rejected_by_speed_gate
    score.instance.rejected_by_ball_gate = result.instance.rejected_by_ball_gate
    for tolerance in effective.tolerances:
        score.tolerances.append(
            score_tolerance(clean_ground_truth, clean_events, tolerance, result.fps)
        )
    score.primary = score_tolerance(
        clean_ground_truth, clean_events, effective.primary_tolerance_sec, result.fps
    )
    score.counts = score_counts(
        clean_ground_truth,
        clean_events,
        result.instance.point_window_sec,
        effective.window_pad_sec,
        effective.point_window_anchor,
    )

    heights: list[float] = []
    distances: list[float] = []
    for row in list(ground_truth.events) + list(ground_truth.no_touch):
        if not row.low_quality:
            continue
        stats = scan_window(
            frames, timestamps, row.start_sec - effective.window_sec,
            row.end_sec + effective.window_sec
        )
        if stats.has_usable_ball:
            score.windows_with_usable_ball += 1
        if stats.has_usable_feet:
            score.windows_with_usable_feet += 1
        if stats.nearest_distance_px < INFINITY:
            heights.append(stats.nearest_person_height_px)
        if stats.nearest_usable_distance_px < INFINITY:
            distances.append(stats.nearest_usable_distance_px)
    score.person_height_px = compute_percentiles(heights)
    score.nearest_usable_distance_px = compute_percentiles(distances)
    return score


def _format_seconds(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}.{int((seconds - total) * 100):02d}"
