"""Differential tests for scoring: this implementation against the C++ report.

The report is the contract. Every number the C++ wrote - the count aware score,
the auxiliary tolerances, the per annotation counts, the false positive split,
the diagnostics, the calibration sweep, the second view without the flagged rows
and the per event verdicts - is compared field by field.

Only the free text `examples` lists are skipped: they are prose built from a
different formatter, and a mismatch there says nothing about the measurement.
"""

from __future__ import annotations

import pytest

from conftest import REFERENCE, ROOT, close_enough, read_jsonl
from touch_eval import cli, gt, metrics, observations, report
from touch_eval.params import DetectorParams

SKIP_KEYS = {"examples"}


def _compare(mine, theirs, path: str, problems: list[str]) -> None:
    if isinstance(theirs, dict) and isinstance(mine, dict):
        for key in sorted(set(mine) | set(theirs)):
            if key in SKIP_KEYS:
                continue
            if key not in mine:
                problems.append(f"{path}.{key}: missing in python")
            elif key not in theirs:
                problems.append(f"{path}.{key}: missing in cpp")
            else:
                _compare(mine[key], theirs[key], f"{path}.{key}", problems)
        return
    if isinstance(theirs, list) and isinstance(mine, list):
        if len(mine) != len(theirs):
            problems.append(f"{path}: length {len(mine)} vs {len(theirs)}")
            return
        for index, (left, right) in enumerate(zip(mine, theirs)):
            _compare(left, right, f"{path}[{index}]", problems)
        return
    if isinstance(theirs, bool) or isinstance(mine, bool):
        if mine != theirs:
            problems.append(f"{path}: python {mine!r} vs cpp {theirs!r}")
        return
    if isinstance(theirs, (int, float)) and isinstance(mine, (int, float)):
        if not close_enough(float(mine), float(theirs)):
            problems.append(f"{path}: python {mine!r} vs cpp {theirs!r}")
        return
    if mine != theirs:
        problems.append(f"{path}: python {mine!r} vs cpp {theirs!r}")


@pytest.mark.parity
@pytest.mark.slow
def test_report_matches_cpp(reference) -> None:
    """The whole report, field by field."""
    theirs = reference.report
    options = metrics.EvaluationOptions(
        params=DetectorParams.parse(theirs["params"]),
        point_window_anchor={"end": metrics.Anchor.END,
                             "centre": metrics.Anchor.CENTRE,
                             "start": metrics.Anchor.START}[theirs["point_window_anchor"]],
        # The frozen reference is the C++ run, whose sweep was 10/30/60/120
        # frames with 30 as the primary: the current defaults are stricter
        # (zero slack), so the reference configuration is spelled out here.
        tolerance_frames=[10, 30, 60, 120],
        primary_tolerance_frames=theirs["primary_tolerance_frames"],
        primary_tolerance_sec=theirs["primary_tolerance_sec"],
    )

    # The frozen annotations, not the live working copy: the reference report
    # was computed from these rows (see conftest).
    ground_truth = gt.parse_annotation_file(str(REFERENCE / "annotations.csv"))
    video = observations.read_metadata(str(reference.observations))
    frames = cli.load_evidence(str(reference.observations), options.params)
    result = metrics.evaluate_frames(frames, ground_truth, options, video)
    mine = report.report_json(
        result,
        ground_truth,
        options,
        options.params,
        theirs["observations"],
        theirs["ground_truth_source"],
        video=video,
    )

    problems: list[str] = []
    # `failures[].examples` and the FP examples are prose: skipped by SKIP_KEYS.
    _compare(mine, theirs, "", problems)
    assert not problems, "\n".join(problems[:40])


@pytest.mark.parity
@pytest.mark.slow
def test_events_file_matches_cpp(reference) -> None:
    """`touch_events.jsonl` is the interface the renderer reads."""
    options = metrics.EvaluationOptions(
        params=DetectorParams.parse(reference.report["params"]),
        point_window_anchor={
            "end": metrics.Anchor.END,
            "centre": metrics.Anchor.CENTRE,
            "start": metrics.Anchor.START,
        }[reference.report["point_window_anchor"]],
    )
    ground_truth = gt.parse_annotation_file(str(REFERENCE / "annotations.csv"))
    frames = cli.load_evidence(str(reference.observations), options.params)
    result = metrics.evaluate_frames(frames, ground_truth, options, None)
    payload = report.report_json(
        result,
        ground_truth,
        options,
        options.params,
        reference.report["observations"],
        reference.report["ground_truth_source"],
    )
    theirs = read_jsonl(reference.events)
    mine = report.event_rows(payload)
    assert len(mine) == len(theirs)
    problems: list[str] = []
    for index, (left, right) in enumerate(zip(mine, theirs)):
        for key in right:
            if key not in left:
                problems.append(f"event {index}: missing {key}")
                continue
            if isinstance(right[key], float):
                if not close_enough(left[key], right[key]):
                    problems.append(f"event {index}.{key}: {left[key]!r} vs {right[key]!r}")
            elif left[key] != right[key]:
                problems.append(f"event {index}.{key}: {left[key]!r} vs {right[key]!r}")
    assert not problems, "\n".join(problems[:40])
