"""Command line entry point: `uv run touch-eval`.

Mirrors the C++ tool's interface for the parts that matter to the business
logic: it reads the same `observations.jsonl`, the same annotations, applies the
same rules and writes the same report, so a run can be swapped between the two
implementations while the differential tests pin them together.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import gt as gt_module
from . import metrics, observations, report
from .detect import FrameEvidence, build_frame_evidence, replay
from .params import DetectorParams

_DETECTOR_FLAGS = {
    "--keypoint_confidence": ("keypointConfidence", float),
    "--ball_min_confidence": ("ballMinConfidence", float),
    "--contact_distance_ratio": ("contactDistanceRatio", float),
    "--person_height_ratio": ("personHeightRatio", float),
    "--min_contact_px": ("minContactPx", float),
    "--max_contact_px": ("maxContactPx", float),
    "--release_ratio": ("releaseRatio", float),
    "--min_contact_frames": ("minContactFrames", int),
    "--max_gap_frames": ("maxGapFrames", int),
    "--min_split_frames": ("minSplitFrames", int),
    "--min_rise_frames": ("minRiseFrames", int),
    "--max_cluster_frames": ("maxClusterFrames", int),
    "--candidate_count": ("candidateCount", int),
    "--speed_window": ("speedWindowSec", float),
    "--refractory_frames": ("refractoryFrames", int),
    "--person_refractory_frames": ("personRefractoryFrames", int),
    "--event_time_mode": ("eventTimeMode", int),
    "--min_speed_change": ("minSpeedChangePxPerSec", float),
    "--min_speed_ratio": ("minSpeedRatio", float),
}

_ANCHORS = {
    "end": metrics.Anchor.END,
    "ending": metrics.Anchor.END,
    "previous": metrics.Anchor.END,
    "整秒末": metrics.Anchor.END,
    "centre": metrics.Anchor.CENTRE,
    "center": metrics.Anchor.CENTRE,
    "居中": metrics.Anchor.CENTRE,
    "start": metrics.Anchor.START,
    "bucket": metrics.Anchor.START,
    "整秒": metrics.Anchor.START,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="touch-eval",
        description=(
            "Score touch detection against annotations, and render the review "
            "material, in Python. The C++ tool remains the reference for the "
            "detector until the parity tests are retired."
        ),
    )
    parser.add_argument("--observations", help="observations.jsonl to score")
    parser.add_argument("--label", help="legacy .label annotation file")
    parser.add_argument("--touches_csv", help="v2 CSV annotation file")
    parser.add_argument("--output_dir", default=".", help="where the report goes")
    parser.add_argument("--dump_gt", help="write the parsed annotations as JSON and stop")
    parser.add_argument(
        "--dump_gt_csv", help="reserved: the v2 CSV re-export is not ported yet"
    )
    parser.add_argument("--fps", type=float, default=0.0, help="override the frame rate")
    parser.add_argument("--max_frames", type=int, default=0, help="stop after N frames")
    parser.add_argument(
        "--window_anchor",
        default="end",
        help="which second a point annotation names: end | centre | start",
    )
    parser.add_argument(
        "--point_window", type=float, default=0.0,
        help="window a point annotation stands for (0 = from the annotation source)",
    )
    parser.add_argument("--window_pad", type=float, default=0.0, help="pad every window")
    parser.add_argument("--adjacency", type=float, default=1.5, help="neighbour blame range")
    parser.add_argument("--duplicate", type=float, default=0.5, help="duplicate range")
    parser.add_argument("--tolerance", type=float, action="append", default=None,
                        help="tolerance in seconds for the auxiliary sweep")
    parser.add_argument("--primary_tolerance", type=float, default=0.0,
                        help="tolerance in seconds for the primary one-to-one score "
                             "(0 = the event must fall inside the span)")
    parser.add_argument("--no_sweep", action="store_true", help="skip the calibration sweep")
    parser.add_argument(
        "--no_csv", action="store_true",
        help="accepted for compatibility: the per frame contact CSV is "
        "not written yet, the report and the events file are",
    )
    parser.add_argument("--quiet", action="store_true", help="no progress on stdout")
    for flag in _DETECTOR_FLAGS:
        parser.add_argument(flag, default=None)
    parser.add_argument("--prefer_tracked_ball", action="store_true", default=None)
    parser.add_argument("--no_prefer_tracked_ball", dest="prefer_tracked_ball",
                        action="store_false")
    parser.add_argument("--require_tracked_ball", action="store_true", default=None)
    return parser


def options_from_args(args: argparse.Namespace) -> metrics.EvaluationOptions:
    changes = {}
    for flag, (field, caster) in _DETECTOR_FLAGS.items():
        value = getattr(args, flag.lstrip("-"))
        if value is not None:
            changes[field] = caster(value)
    if args.prefer_tracked_ball is not None:
        changes["preferTrackedBall"] = bool(args.prefer_tracked_ball)
    if args.require_tracked_ball is not None:
        changes["requireTrackedBall"] = bool(args.require_tracked_ball)

    options = metrics.EvaluationOptions()
    options.params = DetectorParams().replace(**changes) if changes else DetectorParams()
    options.point_window_anchor = _ANCHORS.get(args.window_anchor.lower(), None)
    if options.point_window_anchor is None:
        raise SystemExit(
            f"touch-eval: unknown --window_anchor '{args.window_anchor}' "
            "(end | centre | start)"
        )
    options.point_window_sec = args.point_window
    options.window_pad_sec = args.window_pad
    options.adjacency_sec = args.adjacency
    options.duplicate_sec = args.duplicate
    options.primary_tolerance_sec = args.primary_tolerance
    if args.tolerance:
        options.tolerances = list(args.tolerance)
        options.tolerances_in_seconds = True
    return options


def load_evidence(
    path: str, params: DetectorParams, max_frames: int = 0
) -> list[FrameEvidence]:
    """Read the observations into the evidence the evaluator needs twice."""
    frames: list[FrameEvidence] = []
    for index, frame in enumerate(observations.iter_frames(path)):
        if max_frames and index >= max_frames:
            break
        frames.append(build_frame_evidence(frame, params))
    return frames


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    annotation_path = args.touches_csv or args.label
    if not annotation_path:
        print("touch-eval: one of --label or --touches_csv is required", file=sys.stderr)
        return 2

    ground_truth = gt_module.parse_annotation_file(annotation_path, args.fps)

    if args.dump_gt:
        payload = gt_module.ground_truth_json(ground_truth)
        _write_json(payload, args.dump_gt)
        if not args.quiet:
            print(
                f"annotations: {len(ground_truth.events)} touches rows, "
                f"{ground_truth.touch_count()} touches, "
                f"{len(ground_truth.no_touch)} no_touch spans "
                f"({ground_truth.source_kind})"
            )
        return 0

    if not args.observations:
        print("touch-eval: --observations is required unless --dump_gt", file=sys.stderr)
        return 2

    options = options_from_args(args)
    started = time.time()
    video = observations.read_metadata(args.observations)
    frames = load_evidence(args.observations, options.params, args.max_frames)
    if not args.quiet:
        print(f"observations: {len(frames)} frames in {time.time() - started:.1f} s")

    result = metrics.evaluate_frames(frames, ground_truth, options, video)
    payload = report.report_json(
        result,
        ground_truth,
        options,
        options.params,
        args.observations,
        annotation_path,
        ground_truth_from_json=annotation_path.lower().endswith(".json"),
        video=video,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    _write_json(payload, os.path.join(args.output_dir, "touch_report.json"))
    _write_events(result, payload, os.path.join(args.output_dir, "touch_events.jsonl"))
    if not args.no_csv:
        report.write_contact_signal(
            os.path.join(args.output_dir, "contact_signal.csv"), result.contacts, frames
        )

    if not args.quiet:
        score = result.instance
        print(
            f"count aware: P {score.precision:.3f} R {score.recall:.3f} "
            f"F1 {score.f1:.3f} (TP {score.true_positives} FP {score.false_positives} "
            f"FN {score.false_negatives}, capacity {score.capacity}) "
            f"in {time.time() - started:.1f} s"
        )
    return 0


def _write_json(payload: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _write_events(result, payload: dict, path: str) -> None:
    """`touch_events.jsonl`: the interface the renderer and the app read.

    `type` and `matched` are part of the file as the C++ writes it, so they are
    part of it here too: `matched` is what `gt_index >= 0` means, spelled out.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for row in report.event_rows(payload):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
