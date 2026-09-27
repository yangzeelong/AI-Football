#!/usr/bin/env python3
"""Burn the touch evaluation onto a video so pred and gt can be eyeballed.

The video is meant to be watched, not parsed: every detected touch is shown
with the verdict the evaluator gave it (correct / duplicate / next to an
annotation / unannotated), and every annotation is shown with whether it was
found. The bottom strip is the whole timeline, so a miss or a spurious touch
can be spotted without scrubbing.

    python3 examples/touch_eval/tools/render_touch_review.py \
        --output-dir output/touch_eval_moving_pass

Reads <output-dir>/ground_truth.json and <output-dir>/touch_report.json (both
written by `touch_eval`), plus the base video (<output-dir>/rendered.mp4 when it
exists, otherwise the source video).
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Any

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR, matching the evaluator's verdict names.
COLOR_GT_HIT = (80, 220, 80)
COLOR_GT_MISS = (60, 60, 235)
COLOR_ASSIGNED = (80, 220, 80)
COLOR_DUPLICATE = (0, 165, 255)
COLOR_ADJACENT = (60, 220, 250)
COLOR_UNANNOTATED = (60, 60, 235)
COLOR_NO_TOUCH = (200, 80, 200)
COLOR_TEXT = (240, 240, 240)
COLOR_DIM = (170, 170, 170)

VERDICT_COLORS = {
    "assigned": COLOR_ASSIGNED,
    "duplicate": COLOR_DUPLICATE,
    "adjacent": COLOR_ADJACENT,
    "unannotated": COLOR_UNANNOTATED,
    "inside_no_touch": COLOR_NO_TOUCH,
    "in_window_no_slot": COLOR_DUPLICATE,
}

# What each verdict means, in one short line for the overlay.
VERDICT_LABELS = {
    "assigned": "correct (inside an annotation window)",
    "duplicate": "duplicate (same touch, another frame)",
    "adjacent": "outside the window by a fraction of a second",
    "unannotated": "UNANNOTATED (nothing in the file supports it)",
    "inside_no_touch": "inside a no_touch span",
    "in_window_no_slot": "inside the window, but the annotation claims one touch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/touch_eval_moving_pass")
    parser.add_argument("--video", default=None, help="Base video to draw on.")
    parser.add_argument("--out", default=None, help="Output review MP4.")
    parser.add_argument("--from", dest="from_sec", type=float, default=None)
    parser.add_argument("--to", dest="to_sec", type=float, default=None)
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Keep every Nth frame (the output keeps the source fps).",
    )
    parser.add_argument(
        "--hold", type=float, default=0.40,
        help="Seconds a detected touch stays highlighted.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_negative_spans(ground_truth: dict):
    """Spans the annotator confirmed contain no touch."""
    spans = []
    for event in ground_truth.get("no_touch", []):
        spans.append((float(event["start_sec"]), float(event["end_sec"])))
    return spans


def build_annotations(ground_truth: dict, point_window_sec: float):
    """One record per annotation, with the window it stands for."""
    annotations = []
    for event in ground_truth.get("events", []):
        start = float(event["start_sec"])
        end = float(event["end_sec"])
        if event.get("is_interval"):
            window_from, window_to = start, end
        else:
            half = 0.5 * point_window_sec
            window_from, window_to = start - half, start + half
        annotations.append(
            {
                "index": int(event["index"]),
                "count": int(event.get("count", 1)),
                "from": window_from,
                "to": window_to,
                "at": start,
                "assigned": 0,
                "note": event.get("note", ""),
                "foot": event.get("foot_hint", ""),
            }
        )
    # Kept in file order: the evaluator's gt_index refers to this order.
    return annotations


def build_predictions(report: dict):
    predictions = []
    for event in report.get("events", []):
        predictions.append(
            {
                "frame": int(event["frame_index"]),
                "t": float(event["timestamp_sec"]),
                "gt_index": int(event.get("gt_index", -1)),
                "verdict": event.get("verdict", "assigned"),
                "person": int(event.get("person_track_id", -1)),
                "foot": event.get("foot", ""),
                "distance": float(event.get("distance_px", 0.0)),
                "threshold": float(event.get("threshold_px", 0.0)),
            }
        )
    predictions.sort(key=lambda item: item["t"])
    return predictions


def active_index(times: list[float], t: float) -> int:
    """Index of the last entry at or before `t`, or -1."""
    return bisect.bisect_right(times, t) - 1


def draw_label(image, text, x, y, color, scale=0.55, thickness=1, plate=True):
    (width, height), _ = cv2.getTextSize(text, FONT, scale, thickness)
    if plate:
        cv2.rectangle(image, (x - 6, y - height - 7), (x + width + 6, y + 7),
                      (18, 18, 18), -1)
    cv2.putText(image, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return width


def draw_timeline(image, height, width, annotations, predictions, duration,
                  time_sec, frame_index, negative_spans):
    """Whole-video overview: annotations at the bottom, detections on top."""
    strip_top = height - 34
    overlay = image.copy()
    cv2.rectangle(overlay, (0, strip_top), (width, height), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    def x_of(seconds: float) -> int:
        if duration <= 0.0:
            return 0
        return int(round(width * max(0.0, min(seconds, duration)) / duration))

    # One second grid, every five seconds a brighter line.
    for second in range(0, int(duration) + 1):
        x = x_of(float(second))
        shade = (90, 90, 90) if second % 5 else (140, 140, 140)
        cv2.line(image, (x, strip_top + 2), (x, height - 2), shade, 1)

    # Confirmed negative spans: purple band at the bottom of the strip.
    for span_from, span_to in negative_spans:
        left = x_of(span_from)
        right = max(left + 2, x_of(span_to))
        cv2.rectangle(image, (left, height - 5), (right, height - 1),
                      COLOR_NO_TOUCH, -1)

    # Annotations: green when something was found there, red when it was missed.
    for annotation in annotations:
        x = x_of(annotation["at"])
        colour = COLOR_GT_HIT if annotation["assigned"] > 0 else COLOR_GT_MISS
        top = strip_top + 22 - (3 if annotation["count"] > 1 else 0)
        cv2.line(image, (x, top), (x, height - 4), colour, 2)

    # Detections: colour says which verdict they got.
    for prediction in predictions:
        x = x_of(prediction["t"])
        colour = VERDICT_COLORS.get(prediction["verdict"], COLOR_TEXT)
        cv2.line(image, (x, strip_top + 6), (x, strip_top + 16), colour, 2)

    # Playhead.
    x = x_of(time_sec)
    cv2.line(image, (x, strip_top + 1), (x, height - 1), (255, 255, 255), 1)

    cv2.putText(image,
                "annotations (bottom): green = found, red = missed, purple = "
                "no_touch    detections (top): green = correct, orange = "
                "duplicate, yellow = off-window, purple = in no_touch, "
                "red = unannotated",
                (10, strip_top - 26), FONT, 0.45, COLOR_DIM, 1, cv2.LINE_AA)
    cv2.putText(image, f"frame {frame_index}  {time_sec:7.2f}s", (10, strip_top - 6),
                FONT, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_path = output_dir / "touch_report.json"
    ground_truth_path = output_dir / "ground_truth.json"
    if not report_path.is_file() or not ground_truth_path.is_file():
        print(f"[review] need {report_path} and {ground_truth_path}")
        print("[review] run touch_eval with --dump_gt first")
        return 1

    report = load_json(report_path)
    ground_truth = load_json(ground_truth_path)
    point_window = float(report.get("primary_count_aware", {}).get(
        "point_window_sec", 1.0))
    annotations = build_annotations(ground_truth, point_window)
    negative_spans = build_negative_spans(ground_truth)
    predictions = build_predictions(report)

    # Count what each annotation received, for its hit/partial status.
    for prediction in predictions:
        if 0 <= prediction["gt_index"] < len(annotations):
            annotations[prediction["gt_index"]]["assigned"] += 1

    video_path = Path(args.video) if args.video else output_dir / "rendered.mp4"
    if not video_path.is_file():
        source = report.get("video", {}).get("path")
        if not source or not Path(source).is_file():
            print(f"[review] base video not found: {video_path}")
            return 1
        video_path = Path(source)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[review] cannot open {video_path}")
        return 1
    fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0.0

    out_path = Path(args.out) if args.out else output_dir / "review_pred_vs_gt.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Every Nth frame is written, so the output plays at the original speed.
    out_fps = fps / max(1, args.stride)
    writer = None
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec),
                                 out_fps, (width, height))
        if writer.isOpened():
            print(f"[review] codec={codec} output={out_path}")
            break
        writer.release()
        writer = None
    if writer is None:
        print("[review] no usable video codec")
        return 1

    first_frame = int(args.from_sec * fps) if args.from_sec else 0
    last_frame = int(args.to_sec * fps) if args.to_sec else total
    stride = max(1, args.stride)
    hold = max(0.05, args.hold)
    pred_times = [item["t"] for item in predictions]

    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    frame_index = first_frame
    written = 0
    while frame_index < last_frame:
        ok, frame = capture.read()
        if not ok:
            break
        time_sec = frame_index / fps
        if frame_index % stride == 0:
            # An annotation that is open right now (or about to be, within its
            # own window padded by a quarter second).
            open_annotations = [
                item for item in annotations
                if item["from"] - 0.25 <= time_sec <= item["to"] + 0.25
            ]
            # Detections within the hold window.
            nearby = []
            start = bisect.bisect_left(pred_times, time_sec - hold)
            for index in range(max(0, start), len(predictions)):
                prediction = predictions[index]
                if prediction["t"] > time_sec + hold:
                    break
                nearby.append(prediction)

            # Periphery cue: is something annotated here and was it found?
            if open_annotations:
                hit = any(item["assigned"] > 0 for item in open_annotations)
                cv2.rectangle(frame, (0, 0), (5, height - 1),
                              COLOR_GT_HIT if hit else COLOR_GT_MISS, -1)

            y = 20
            if open_annotations:
                # Show the closest annotation of the ones that are open.
                current = min(open_annotations,
                              key=lambda item: abs(item["at"] - time_sec))
                if current["assigned"] >= current["count"]:
                    colour, status = COLOR_GT_HIT, "found"
                elif current["assigned"] > 0:
                    colour, status = COLOR_ADJACENT, "partly found"
                else:
                    colour, status = COLOR_GT_MISS, "MISSED"
                text = (f"GT #{current['index']}  {current['count']} touch(es)  "
                        f"{current['assigned']}/{current['count']} {status}")
                draw_label(frame, text, width // 3, y, colour)
                y += 34
            for prediction in nearby:
                colour = VERDICT_COLORS.get(prediction["verdict"], COLOR_TEXT)
                # gt_index is 0 based; print the annotation's own number, the
                # same one the GT line above and the report use.
                gt_index = prediction["gt_index"]
                origin = ""
                if 0 <= gt_index < len(annotations):
                    origin = f" -> GT #{annotations[gt_index]['index']}"
                text = (f"det @{prediction['t']:.2f}s  "
                        f"{VERDICT_LABELS.get(prediction['verdict'], '')}{origin}"
                        f"  p{prediction['person']} {prediction['foot']}"
                        f"  d={prediction['distance']:.1f}/{prediction['threshold']:.1f}px")
                draw_label(frame, text, width // 3, y, colour)
                y += 34

            draw_timeline(frame, height, width, annotations, predictions,
                          duration, time_sec, frame_index, negative_spans)
            writer.write(frame)
            written += 1
        frame_index += 1

    capture.release()
    writer.release()
    print(f"[review] wrote {written} frame(s) to {out_path}")
    print(f"[review] annotations: {len(annotations)}  "
          f"detections: {len(predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
