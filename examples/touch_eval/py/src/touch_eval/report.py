"""The report the evaluation writes, as JSON.

A port of `WriteReportJson()`. Every producer of numbers puts them in the same
place as the C++ does, because `tests/test_parity_cpp.py` diffs this file against
the one the C++ wrote: a field that moves or disappears is a failing test, not a
silent change of meaning.
"""

from __future__ import annotations

import numpy as np

from . import metrics
from .gt import GroundTruth
from .observations import VideoMetadata
from .params import DetectorParams

SCHEMA_VERSION = 2


def plain(value):
    """Native Python types all the way down.

    The detector works in `numpy.float32` on purpose, and those values reach the
    report; JSON has no idea what they are, so the boundary converts once,
    instead of every field remembering to.
    """
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _percentiles_short(value: metrics.Percentiles) -> dict:
    return {
        "n": value.samples,
        "mean": value.mean,
        "p50": value.p50,
        "p95": value.p95,
        "max": value.max,
    }


def _percentiles(value: metrics.Percentiles) -> dict:
    return {
        "n": value.samples,
        "mean": value.mean,
        "p50": value.p50,
        "p90": value.p90,
        "p95": value.p95,
        "p99": value.p99,
        "min": value.min,
        "max": value.max,
    }


def _fill_score(score: metrics.ToleranceScore) -> dict:
    return {
        "tolerance_sec": score.tolerance_sec,
        "tolerance_frames": score.tolerance_frames,
        "true_positives": score.true_positives,
        "false_positives": score.false_positives,
        "false_negatives": score.false_negatives,
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "signed_dt_sec": _percentiles_short(score.signed_dt_sec),
        "abs_dt_sec": _percentiles_short(score.abs_dt_sec),
    }


def _fill_instance(score: metrics.InstanceScore, options: metrics.EvaluationOptions) -> dict:
    return {
        "point_window_sec": score.point_window_sec,
        "window_pad_sec": score.window_pad_sec,
        "adjacency_sec": options.adjacency_sec,
        "duplicate_sec": options.duplicate_sec,
        "capacity": score.capacity,
        "true_positives": score.true_positives,
        "false_positives": score.false_positives,
        "false_negatives": score.false_negatives,
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "windows_with_event": score.windows_with_event,
        "windows_exact": score.windows_exact,
        "suppressed_by_refractory": score.suppressed_by_refractory,
        "suppressed_by_person_refractory": score.suppressed_by_person_refractory,
        "rejected_by_speed_gate": score.rejected_by_speed_gate,
        "rejected_by_ball_gate": score.rejected_by_ball_gate,
        "signed_dt_sec": _percentiles_short(score.signed_dt_sec),
        "abs_dt_sec": _percentiles_short(score.abs_dt_sec),
    }


def _fill_counts(counts: metrics.CountScore, with_buckets: bool) -> dict:
    payload = {
        "records": counts.records,
        "exact_records": counts.exact_records,
        "over_records": counts.over_records,
        "under_records": counts.under_records,
        "gt_touches": counts.gt_touches,
        "predicted_in_windows": counts.predicted_in_windows,
        "mean_abs_count_error": counts.mean_abs_count_error,
    }
    if with_buckets:
        payload["buckets"] = [
            {
                "gt_index": bucket.gt_index,
                "start_sec": bucket.start_sec,
                "end_sec": bucket.end_sec,
                "span_from_sec": bucket.span_from_sec,
                "span_to_sec": bucket.span_to_sec,
                "gt_count": bucket.gt_count,
                "predicted_count": bucket.predicted_count,
                "exact": bucket.exact,
            }
            for bucket in counts.buckets
        ]
    return payload


def _event_json(event, gt_index: int, verdict: str) -> dict:
    return {
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
        "gt_index": gt_index,
        "verdict": verdict,
    }

#: `contact_signal.csv`: one row per frame, the decision the detector made on
#: it. The review tools read this to explain a miss without re-running anything
#: ("was the ball there? did it come inside the threshold?").
CONTACT_SIGNAL_COLUMNS = (
    "frame_index", "timestamp_sec", "level", "person_track_id", "foot",
    "distance_px", "threshold_px", "ball_confidence", "ball_track_id",
    "ball_tracked", "observed_balls", "valid_foot_points", "persons_with_feet",
    "persons",
)


def _level_name(level: int) -> str:
    from .detect import ContactLevel

    if level == ContactLevel.CONTACT:
        return "contact"
    if level == ContactLevel.BAND:
        return "band"
    return "none"


def contact_signal_rows(contacts, frames) -> list[list]:
    """The rows of `contact_signal.csv`, one per frame, in frame order.

    A frame without a candidate writes empty cells rather than zeros: "no
    measurement" and "measured zero pixels" are different things, and the
    review tools treat them differently.
    """
    rows: list[list] = []
    for contact in contacts:
        candidate = contact.candidate if contact.has_ball else None
        rows.append([
            contact.frame_index,
            f"{contact.timestamp_sec:.6f}",
            _level_name(contact.level),
            candidate.person_track_id if candidate is not None else "",
            ("left" if candidate.left_foot else "right") if candidate is not None else "",
            f"{float(candidate.distance_px):.4f}" if candidate is not None else "",
            f"{float(contact.threshold_px):.4f}" if contact.has_ball else "",
            f"{float(candidate.ball_confidence):.4f}" if candidate is not None else "",
            candidate.ball_track_id if candidate is not None else "",
            ("yes" if candidate.ball_tracked else "no") if candidate is not None else "",
            contact.observed_ball_count,
            contact.valid_foot_point_count,
            contact.persons_with_feet,
            contact.person_count,
        ])
    return rows


def write_contact_signal(path: str, contacts, frames) -> None:
    import csv as _csv

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = _csv.writer(handle)
        writer.writerow(CONTACT_SIGNAL_COLUMNS)
        writer.writerows(contact_signal_rows(contacts, frames))


def event_rows(payload: dict) -> list[dict]:
    """The rows `touch_events.jsonl` holds, in the C++ file's shape.

    `matched` is what `gt_index >= 0` means, spelled out for the readers that
    only want to know whether the event is supported by an annotation.
    """
    return [
        {"type": "touch", **row, "matched": row["gt_index"] >= 0}
        for row in payload["events"]
    ]


def report_json(
    result: metrics.EvaluationResult,
    ground_truth: GroundTruth,
    options: metrics.EvaluationOptions,
    params: DetectorParams,
    observations_path: str,
    ground_truth_source: str,
    ground_truth_from_json: bool = False,
    video: VideoMetadata | None = None,
) -> dict:
    """Everything the run produced, in the C++ report's shape."""
    video = video or VideoMetadata()
    payload: dict = {
        "schema_version": SCHEMA_VERSION,
        "observations": observations_path,
        "ground_truth_source": ground_truth_source,
        "ground_truth_from_json": ground_truth_from_json,
        "ground_truth": {
            "source_kind": ground_truth.source_kind,
            "no_touch_spans": len(ground_truth.no_touch),
            "events": len(ground_truth.events),
            "touches": ground_truth.touch_count(),
            "ignored_rows": ground_truth.ignored_rows(),
            "ignored_touches": ground_truth.ignored_touch_count(),
            "covered_events": result.gt_events_covered,
        },
        "video": {
            "path": video.video_path,
            "camera_id": video.camera_id,
            "fps": video.fps,
            "width": video.width,
            "height": video.height,
            "stride": video.stride,
        },
        "frames": result.frames,
        "duration_sec": result.duration_sec,
        "ignored": [
            {
                "index": row.index,
                "type": "touch" if row.count > 0 else "no_touch",
                "start_sec": row.start_sec,
                "end_sec": row.end_sec,
                "note": row.note,
            }
            for row in ground_truth.ignored
        ],
        "covered": {
            "from_sec": result.covered_from_sec,
            "to_sec": result.covered_to_sec,
        },
        "partial_run": result.partial_run,
        "predicted_events": result.predicted_events,
        "params": params.describe(),
        "primary_tolerance_sec": result.primary.tolerance_sec,
        "primary_tolerance_frames": result.primary.tolerance_frames,
        "fps": result.fps,
        "tolerances_in_frames": result.tolerances_in_frames,
        "primary_count_aware": _fill_instance(result.instance, options),
        "point_window_anchor": metrics.anchor_name(options.point_window_anchor),
        "false_positive_breakdown": {
            "duplicates": result.false_positives.duplicates,
            "inside_no_touch": result.false_positives.inside_no_touch,
            "in_window_no_slot": result.false_positives.in_window_no_slot,
            "adjacent": result.false_positives.adjacent,
            "unannotated": result.false_positives.unannotated,
            "examples": list(result.false_positives.examples),
        },
        "tolerance_sweep": [_fill_score(score) for score in result.tolerances],
        "primary": _fill_score(result.primary),
        "counts": _fill_counts(result.counts, with_buckets=True),
        "diagnostics": {
            "records": result.diagnostics.records,
            "windows_with_detected_ball": result.diagnostics.with_detected_ball,
            "windows_with_usable_ball": result.diagnostics.with_usable_ball,
            "windows_with_usable_feet": result.diagnostics.with_usable_feet,
            "nearest_distance_px": _percentiles(result.diagnostics.nearest_distance_px),
            "nearest_usable_distance_px": _percentiles(
                result.diagnostics.nearest_usable_distance_px
            ),
            "background_nearest_usable_distance_px": _percentiles(
                result.diagnostics.background_nearest_usable_distance_px
            ),
            "person_height_px": _percentiles(result.diagnostics.person_height_px),
        },
        "failures": [
            {
                "reason": group.reason,
                "count": group.count,
                "examples": list(group.examples),
            }
            for group in result.failures
        ],
        "calibration": [
            {
                "contact_distance_ratio": point.contact_distance_ratio,
                "threshold_px_ref": point.threshold_px_ref,
                "tolerance_sec": point.tolerance_sec,
                "true_positives": point.true_positives,
                "false_positives": point.false_positives,
                "false_negatives": point.false_negatives,
                "precision": point.precision,
                "recall": point.recall,
                "f1": point.f1,
                "instance_precision": point.instance_precision,
                "instance_recall": point.instance_recall,
                "instance_f1": point.instance_f1,
            }
            for point in result.sweep
        ],
        "events": [
            _event_json(
                event,
                result.event_gt_index[index] if index < len(result.event_gt_index) else -1,
                result.false_positives.event_verdict[index]
                if index < len(result.false_positives.event_verdict)
                else "assigned",
            )
            for index, event in enumerate(result.events)
        ],
    }

    if result.has_anchor_alternative:
        payload["anchor_alternative"] = dict(
            _fill_instance(result.anchor_alternative_instance, options),
            anchor=metrics.anchor_name(result.anchor_alternative),
            counts={
                "records": result.anchor_alternative_counts.records,
                "exact_records": result.anchor_alternative_counts.exact_records,
                "over_records": result.anchor_alternative_counts.over_records,
                "under_records": result.anchor_alternative_counts.under_records,
            },
        )

    if result.low_quality.rows > 0:
        low = result.low_quality
        payload["low_quality"] = {
            "rows": low.rows,
            "touches": low.touches,
            "excluded_events": low.excluded_events,
            "scored_events": result.predicted_events - low.excluded_events,
            "primary_count_aware": _fill_instance(low.instance, options),
            "tolerance_sweep": [_fill_score(score) for score in low.tolerances],
            "primary": _fill_score(low.primary),
            "counts": _fill_counts(low.counts, with_buckets=False),
            "windows": {
                "with_usable_ball": low.windows_with_usable_ball,
                "with_usable_feet": low.windows_with_usable_feet,
                "person_height_px": _percentiles(low.person_height_px),
                "nearest_usable_distance_px": _percentiles(
                    low.nearest_usable_distance_px
                ),
            },
        }
    return plain(payload)
