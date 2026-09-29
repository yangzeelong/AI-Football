"""Differential tests: this implementation against the C++ reference.

The C++ tools stay the reference implementation until these pass, so each test
reads an artefact the C++ side already wrote - `touch_events.jsonl`,
`touch_report.json` - and requires the Python port to reproduce it. A mismatch
names the first event or field that differs, because "the numbers moved" is not
a useful failure.
"""

from __future__ import annotations

import pytest

from conftest import FRAMES, TAIL_FRAMES, close_enough, read_jsonl
from touch_eval import detect, observations
from touch_eval.params import DetectorParams

EVENT_FIELDS = (
    "frame_index",
    "first_frame_index",
    "timestamp_sec",
    "person_track_id",
    "foot",
    "distance_px",
    "threshold_px",
    "ball_confidence",
    "ball_tracked",
    "ball_track_id",
    "contact_frames",
    "speed_before_px_per_sec",
    "speed_after_px_per_sec",
)


def _mine_as_rows(events) -> list[dict]:
    return [
        {
            "frame_index": event.frame_index,
            "first_frame_index": event.first_frame_index,
            "timestamp_sec": event.timestamp_sec,
            "person_track_id": event.person_track_id,
            "foot": "left" if event.left_foot else "right",
            "distance_px": float(event.distance_px),
            "threshold_px": float(event.threshold_px),
            "ball_confidence": float(event.ball_confidence),
            "ball_tracked": event.ball_tracked,
            "ball_track_id": event.ball_track_id,
            "contact_frames": event.contact_frames,
            "speed_before_px_per_sec": float(event.speed_before_px_per_sec),
            "speed_after_px_per_sec": float(event.speed_after_px_per_sec),
        }
        for event in events
    ]


def test_detector_parameters_round_trip(reference) -> None:
    """The report's parameter dump must read back exactly."""
    params = DetectorParams.parse(reference.params_text)
    assert params.describe() == reference.params_text


@pytest.mark.parity
@pytest.mark.slow
def test_detector_events_match_cpp(reference) -> None:
    """Every field of every event the C++ detector emitted."""
    params = DetectorParams.parse(reference.params_text)

    frames = observations.iter_frames(str(reference.observations))
    if FRAMES:
        frames = _prefix(frames, FRAMES)
    result = detect.replay(frames, params)
    mine = _mine_as_rows(result.events)

    theirs = [
        {key: row[key] for key in EVENT_FIELDS}
        for row in read_jsonl(reference.events)
        if not FRAMES or row["frame_index"] <= FRAMES - TAIL_FRAMES
    ]
    if FRAMES:
        mine = [row for row in mine if row["frame_index"] <= FRAMES - TAIL_FRAMES]

    assert len(mine) == len(theirs), _length_message(mine, theirs)
    for index, (left, right) in enumerate(zip(mine, theirs)):
        for key in EVENT_FIELDS:
            if isinstance(left[key], float):
                if not close_enough(left[key], right[key]):
                    pytest.fail(
                        f"event #{index} field {key}: python {left[key]!r} "
                        f"cpp {right[key]!r}\npython: {left}\ncpp:    {right}"
                    )
            elif left[key] != right[key]:
                pytest.fail(
                    f"event #{index} field {key}: python {left[key]!r} "
                    f"cpp {right[key]!r}\npython: {left}\ncpp:    {right}"
                )


@pytest.mark.parity
@pytest.mark.slow
def test_detector_gate_counters_match_cpp(reference) -> None:
    """The suppression bookkeeping the report quotes."""
    params = DetectorParams.parse(reference.params_text)
    frames = observations.iter_frames(str(reference.observations))
    if FRAMES:
        frames = _prefix(frames, FRAMES)
    result = detect.replay(frames, params)

    primary = reference.report["primary_count_aware"]
    if FRAMES:
        return  # counters are only comparable over the whole recording
    assert result.suppressed_by_refractory == primary["suppressed_by_refractory"]
    assert (
        result.suppressed_by_person_refractory
        == primary["suppressed_by_person_refractory"]
    )
    assert result.rejected_by_speed_gate == primary["rejected_by_speed_gate"]
    assert result.rejected_by_ball_gate == primary["rejected_by_ball_gate"]
    assert len(result.events) == reference.report["predicted_events"]


def _prefix(iterable, count: int):
    for index, item in enumerate(iterable):
        if index >= count:
            return
        yield item


def _length_message(mine: list[dict], theirs: list[dict]) -> str:
    """Point at the first event that is missing on either side."""
    for index in range(min(len(mine), len(theirs))):
        if mine[index]["frame_index"] != theirs[index]["frame_index"]:
            return (
                f"first divergence at event #{index}: python frame "
                f"{mine[index]['frame_index']} cpp frame {theirs[index]['frame_index']}"
            )
    if len(mine) > len(theirs):
        extra = mine[len(theirs)]
        return f"python emitted {len(mine)} events, cpp {len(theirs)}; extra {extra}"
    missing = theirs[len(mine)]
    return f"python emitted {len(mine)} events, cpp {len(theirs)}; missing {missing}"
