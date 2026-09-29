"""Unit tests for the rules, on inputs too small to reach with a real video.

The differential tests prove the port agrees with the C++ on 移动传球.mov; these
cover the branches that single recording never takes - a `no_touch` span, an
`ignore` flag, a CSV without a header, a row that claims two touches, the three
window anchors - so a future change to the rules fails a small test with a name
instead of a 97 row diff.
"""

from __future__ import annotations

import math

import pytest

from touch_eval import detect, gt, metrics
from touch_eval.params import DetectorParams


def evidence(
    frame_index: int,
    timestamp_sec: float,
    distance_px: float,
    person_track_id: int = 1,
    left_foot: bool = True,
    person_height_px: float = 200.0,
    ball_confidence: float = 0.9,
    ball_tracked: bool = True,
    observed_ball: bool = True,
) -> detect.FrameEvidence:
    """One frame with a single candidate, which is all these tests need."""
    candidate = detect.ContactCandidate(
        distance_px=detect.F32(distance_px),
        person_height_px=detect.F32(person_height_px),
        person_track_id=person_track_id,
        left_foot=left_foot,
        foot_slot=0 if left_foot else 1,
        ball_tracked=ball_tracked,
        ball_track_id=1 if ball_tracked else -1,
        ball_confidence=detect.F32(ball_confidence),
    )
    return detect.FrameEvidence(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        frame_width=1280,
        frame_height=720,
        person_count=1,
        ball_count=1,
        observed_ball_count=1 if observed_ball else 0,
        valid_foot_point_count=1,
        persons_with_feet=1,
        candidates=[candidate],
    )


def contact_run(start_frame: int, count: int, distance: float = 10.0) -> list:
    """`count` contact frames at 60 fps, starting at `start_frame`."""
    return [
        evidence(start_frame + step, (start_frame + step) / 60.0, distance)
        for step in range(count)
    ]


def gap_run(start_frame: int, count: int) -> list:
    """Frames with no admissible contact: the ball is far from every foot."""
    return [
        evidence(start_frame + step, (start_frame + step) / 60.0, 400.0)
        for step in range(count)
    ]


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


def point_row(second: float):
    return gt.GroundTruthEvent(index=1, count=1, start_sec=second, end_sec=second)


def test_anchors_read_a_point_row_three_ways() -> None:
    """`00:07S` owns [6.5, 7.5), [6.0, 7.0) or [7.0, 8.0) depending on the reading."""
    row = point_row(7.0)
    windows = {
        anchor: metrics.window_of(row, 1.0, 0.0, anchor)
        for anchor in (metrics.Anchor.CENTRE, metrics.Anchor.END, metrics.Anchor.START)
    }
    assert (windows[metrics.Anchor.CENTRE].from_sec, windows[metrics.Anchor.CENTRE].to_sec) == (6.5, 7.5)
    assert (windows[metrics.Anchor.END].from_sec, windows[metrics.Anchor.END].to_sec) == (6.0, 7.0)
    assert (windows[metrics.Anchor.START].from_sec, windows[metrics.Anchor.START].to_sec) == (7.0, 8.0)
    # Half open: the far edge belongs to the next window, the near one to this.
    assert metrics.within_window(windows[metrics.Anchor.START], 7.0)
    assert not metrics.within_window(windows[metrics.Anchor.START], 8.0)


def test_interval_rows_ignore_the_anchor_and_the_point_window() -> None:
    row = gt.GroundTruthEvent(index=1, count=1, start_sec=10.0, end_sec=11.5)
    for anchor in (metrics.Anchor.CENTRE, metrics.Anchor.END, metrics.Anchor.START):
        window = metrics.window_of(row, 1.0, 0.0, anchor)
        assert (window.from_sec, window.to_sec) == (10.0, 11.5)


def test_the_end_anchor_reads_the_second_the_row_names() -> None:
    """A touch at 9.3 s belongs to the row that writes 10 under `end`."""
    row = point_row(10.0)
    assert metrics.within_window(metrics.window_of(row, 1.0, 0.0, metrics.Anchor.END), 9.3)
    assert not metrics.within_window(
        metrics.window_of(row, 1.0, 0.0, metrics.Anchor.CENTRE), 9.3
    )
    assert not metrics.within_window(
        metrics.window_of(row, 1.0, 0.0, metrics.Anchor.START), 9.3
    )


# ---------------------------------------------------------------------------
# assignment and slots
# ---------------------------------------------------------------------------


def test_a_row_that_claims_two_touches_can_take_two() -> None:
    ground_truth = gt.GroundTruth(
        events=[gt.GroundTruthEvent(index=1, count=2, start_sec=10.0, end_sec=12.0)],
        default_point_window_sec=0.5,
    )
    events = [
        detect.TouchEvent(frame_index=1, timestamp_sec=10.2, person_track_id=1),
        detect.TouchEvent(frame_index=2, timestamp_sec=11.7, person_track_id=1),
        detect.TouchEvent(frame_index=3, timestamp_sec=11.9, person_track_id=2),
    ]
    options = metrics.EvaluationOptions()
    score, match = metrics.score_instances(ground_truth, events, options)
    assert score.capacity == 2
    assert score.true_positives == 2
    assert score.false_negatives == 0
    assert score.false_positives == 1
    assert match.gt_assigned == [2]


def test_assignment_is_optimal_not_greedy() -> None:
    """An event gives up its nearest row when only it can reach the far one.

    Inside a 0.5 s tolerance the 10.05 s event can only take row A, while the
    10.15 s event can take either. Walking the events in order and letting each
    take its nearest free row spends A on the first event and leaves the second
    unmatched; the augmenting path repairs that, so the far row goes to the
    event that has a choice.
    """
    ground_truth = gt.GroundTruth(
        events=[
            gt.GroundTruthEvent(index=1, count=1, start_sec=10.0, end_sec=10.0),
            gt.GroundTruthEvent(index=2, count=1, start_sec=10.6, end_sec=10.6),
        ],
        default_point_window_sec=0.5,
    )
    events = [
        detect.TouchEvent(frame_index=1, timestamp_sec=10.15, person_track_id=1),
        detect.TouchEvent(frame_index=2, timestamp_sec=10.05, person_track_id=1),
    ]
    match = metrics.match_events(ground_truth, events, 0.5)
    assert match.true_positives == 2
    # The 10.15 s event ends up on row B (index 1), which frees row A for 10.05.
    assert match.event_gt == [1, 0]


# ---------------------------------------------------------------------------
# annotations
# ---------------------------------------------------------------------------


def test_csv_without_a_header_uses_the_fixed_layout() -> None:
    ground_truth = gt.parse_touch_csv_text(
        "1,touch,3.00,4.00,2,,,\n", "inline.csv", fps_override=0.0
    )
    assert len(ground_truth.events) == 1
    event = ground_truth.events[0]
    assert (event.start_sec, event.end_sec, event.count) == (3.0, 4.0, 2)
    assert ground_truth.touch_count() == 2


def test_frame_columns_include_both_ends() -> None:
    ground_truth = gt.parse_touch_csv_text(
        "# fps: 60.00\n"
        "index,type,t_start_sec,t_end_sec,t_start_frame,t_end_frame,count\n"
        "1,touch,1.00,1.05,60,63,1\n",
        "inline.csv",
    )
    event = ground_truth.events[0]
    assert event.frames_stated
    assert event.start_sec == pytest.approx(60 / 60.0)
    assert event.end_sec == pytest.approx(64 / 60.0)


def test_ignore_rows_leave_the_score_and_the_others_stay() -> None:
    ground_truth = gt.parse_touch_csv_text(
        "# point_window_sec: 0.50\n"
        "index,type,t_start_sec,t_end_sec,count,ignore\n"
        "1,touch,5.00,5.00,1,no\n"
        "2,touch,9.00,9.00,1,yes\n",
        "inline.csv",
    )
    assert [event.index for event in ground_truth.events] == [1]
    assert [row.index for row in ground_truth.ignored] == [2]
    assert ground_truth.touch_count() == 1


def test_flagged_rows_are_dropped_from_the_second_view() -> None:
    ground_truth = gt.parse_touch_csv_text(
        "# point_window_sec: 0.50\n"
        "index,type,t_start_sec,t_end_sec,count,low_quality_data\n"
        "1,touch,5.00,5.00,1,no\n"
        "2,touch,9.00,9.00,1,yes\n",
        "inline.csv",
    )
    assert ground_truth.low_quality_rows() == 1
    assert ground_truth.low_quality_touch_count() == 1
    clean = metrics.without_low_quality(ground_truth)
    assert [event.index for event in clean.events] == [1]
    # A prediction only the flagged row explains is dropped with it.
    events = [
        detect.TouchEvent(frame_index=1, timestamp_sec=5.0, person_track_id=1),
        # Under the default End anchor the row written 9.0 owns [8.5, 9.0).
        detect.TouchEvent(frame_index=2, timestamp_sec=8.9, person_track_id=1),
    ]
    kept, excluded = metrics.drop_low_quality_events(
        ground_truth, events, metrics.EvaluationOptions()
    )
    assert excluded == 1
    assert [kept_event.timestamp_sec for kept_event in kept] == [5.0]


def test_no_touch_span_gets_its_own_verdict() -> None:
    ground_truth = gt.parse_touch_csv_text(
        "# point_window_sec: 0.50\n"
        "index,type,t_start_sec,t_end_sec,count,foot,player,note\n"
        "1,touch,5.00,5.00,1,,,\n"
        "2,no_touch,20.00,21.00,0,,,\n",
        "inline.csv",
    )
    events = [detect.TouchEvent(frame_index=1, timestamp_sec=20.5, person_track_id=1)]
    options = metrics.EvaluationOptions()
    score, match = metrics.score_instances(ground_truth, events, options)
    assert score.true_positives == 0
    breakdown = metrics.classify_false_positives(
        ground_truth, events, match.event_gt, options
    )
    assert breakdown.inside_no_touch == 1
    assert breakdown.event_verdict == [metrics.Verdict.INSIDE_NO_TOUCH]


def test_legacy_label_counts_and_notes() -> None:
    ground_truth = gt.parse_touch_label_text(
        "00:07S（2次）左脚 - 快到看不清\n00:09S\n", "inline.label"
    )
    assert ground_truth.default_point_window_sec == 1.0
    assert [event.count for event in ground_truth.events] == [2, 1]
    assert ground_truth.touch_count() == 3
    assert ground_truth.events[0].foot_hint == "left"
    # The note starts after the count, so the foot words stay in it.
    assert ground_truth.events[0].note == "左脚 - 快到看不清"
    assert "no (N次) count" in ground_truth.events[1].warnings[0]


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------


def test_a_run_of_contact_frames_becomes_one_event() -> None:
    params = detect_module_params()
    detector = detect.TouchDetector(params)
    events = []
    for frame in contact_run(100, 5):
        events.extend(detector.push_evidence(frame))
    events.extend(detector.flush())
    assert len(events) == 1
    assert events[0].frame_index == 100
    assert events[0].contact_frames == 5
    assert events[0].person_track_id == 1


def test_refractory_merges_two_reports_of_one_touch() -> None:
    params = detect_module_params().replace(refractoryFrames=30)
    detector = detect.TouchDetector(params)
    events = []
    # Two contacts of the same person and foot 13 frames apart, separated by
    # the frames without evidence that close the first cluster (maxGapFrames is
    # 3, and the fourth gap frame is what closes it): one touch, one report.
    sequence = contact_run(100, 3) + gap_run(103, 4) + contact_run(113, 3)
    for frame in sequence:
        events.extend(detector.push_evidence(frame))
    events.extend(detector.flush())
    assert len(events) == 1
    assert detector.suppressed_by_refractory >= 1


def test_a_short_contact_is_not_an_event() -> None:
    params = detect_module_params().replace(minContactFrames=3)
    detector = detect.TouchDetector(params)
    events = []
    for frame in contact_run(100, 2):
        events.extend(detector.push_evidence(frame))
    events.extend(detector.flush())
    assert events == []


def detect_module_params():
    """The detector defaults the app and the evaluator both use."""
    return DetectorParams()


def test_percentiles_round_the_rank_not_interpolate() -> None:
    """`ComputePercentiles()` picks the index nearest to `fraction * (n-1)`."""
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    result = metrics.compute_percentiles(values)
    assert result.samples == 5
    assert result.p50 == 2.0          # rank 2.0
    assert result.p90 == 4.0          # rank 3.6 -> index 4
    assert result.min == 0.0 and result.max == 4.0
    assert math.isclose(result.mean, 2.0)
