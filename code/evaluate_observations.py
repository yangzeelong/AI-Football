from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import hypot
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate single-view 2D observation quality without ground truth."
    )
    parser.add_argument("--input", required=True, help="Input observation JSONL.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--markdown", default=None, help="Optional Markdown report path.")
    parser.add_argument("--low-conf", type=float, default=0.3, help="Low keypoint confidence threshold.")
    parser.add_argument("--short-track", type=int, default=10, help="Track length below this is short.")
    parser.add_argument("--jitter-gap", type=int, default=1, help="Max frame gap for keypoint motion stats.")
    parser.add_argument("--review-frames", type=int, default=30, help="Max frames to list for review.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata, frames = load_jsonl(args.input)
    report = build_report(
        metadata=metadata,
        frames=frames,
        input_path=args.input,
        low_conf=args.low_conf,
        short_track=args.short_track,
        jitter_gap=args.jitter_gap,
        review_frame_limit=args.review_frames,
    )
    write_json(args.output, report)
    if args.markdown:
        write_markdown(args.markdown, report)


def load_jsonl(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    record_index = 0
    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record_index += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {path}") from exc
            if record_index == 1:
                # 报告脚本只支持带 metadata 的新 schema，旧 JSONL 会在这里直接失败。
                validate_metadata(record, path)
                metadata = record
                continue
            validate_frame_record(record, line_no, path)
            frames.append(record)
    if metadata is None:
        raise ValueError(f"Observation JSONL is empty: {path}")
    return metadata, sorted(frames, key=lambda frame: frame["frame_index"])


def build_report(
    metadata: dict[str, Any],
    frames: list[dict[str, Any]],
    input_path: str | Path,
    low_conf: float,
    short_track: int,
    jitter_gap: int,
    review_frame_limit: int,
) -> dict[str, Any]:
    frame_stats = compute_frame_stats(frames, low_conf)
    person_tracks = compute_person_track_stats(frames, short_track)
    keypoints = compute_keypoint_stats(frames, low_conf, jitter_gap)
    balls = compute_ball_stats(frames)
    quality_score = compute_quality_score(frame_stats, person_tracks, keypoints, balls)
    review_frames = select_review_frames(frames, low_conf, review_frame_limit)

    return {
        "input": str(input_path),
        "metadata": metadata,
        "quality_score": quality_score,
        "frame_stats": frame_stats,
        "person_tracks": person_tracks,
        "keypoints": keypoints,
        "balls": balls,
        "review_frames": review_frames,
        "settings": {
            "low_conf": low_conf,
            "short_track": short_track,
            "jitter_gap": jitter_gap,
            "review_frame_limit": review_frame_limit,
        },
    }


def compute_frame_stats(frames: list[dict[str, Any]], low_conf: float) -> dict[str, Any]:
    total_frames = len(frames)
    person_counts = [len(frame.get("persons", [])) for frame in frames]
    ball_counts = [len(frame.get("balls", [])) for frame in frames]
    frame_keypoint_confidences: list[float] = []
    low_keypoint_frame_count = 0

    for frame in frames:
        keypoints = [
            keypoint
            for person in frame.get("persons", [])
            for keypoint in person.get("keypoints", [])
        ]
        confidences = [
            float(keypoint.get("confidence", 0.0))
            for keypoint in keypoints
            if keypoint.get("x") is not None and keypoint.get("y") is not None
        ]
        if confidences:
            frame_mean = mean(confidences)
            frame_keypoint_confidences.append(frame_mean)
            if frame_mean < low_conf:
                low_keypoint_frame_count += 1

    return {
        "total_frames": total_frames,
        "avg_persons_per_frame": round(mean_or_zero(person_counts), 3),
        "max_persons_per_frame": max(person_counts, default=0),
        "empty_person_frame_ratio": round(ratio(count_equal(person_counts, 0), total_frames), 4),
        "avg_balls_per_frame": round(mean_or_zero(ball_counts), 3),
        "ball_present_frame_ratio": round(ratio(count_greater(ball_counts, 0), total_frames), 4),
        "avg_frame_keypoint_confidence": round(mean_or_zero(frame_keypoint_confidences), 4),
        "low_keypoint_frame_ratio": round(ratio(low_keypoint_frame_count, total_frames), 4),
    }


def compute_person_track_stats(frames: list[dict[str, Any]], short_track: int) -> dict[str, Any]:
    tracks: dict[str, list[int]] = defaultdict(list)
    untracked_persons = 0

    for frame in frames:
        frame_index = int(frame.get("frame_index", 0))
        for person in frame.get("persons", []):
            track_id = person.get("track_id")
            if track_id is None:
                untracked_persons += 1
                continue
            tracks[str(track_id)].append(frame_index)

    track_items = []
    for track_id, track_frames in sorted(tracks.items(), key=lambda item: int(item[0])):
        unique_frames = sorted(set(track_frames))
        gaps = sum(
            max(0, curr - prev - 1)
            for prev, curr in zip(unique_frames, unique_frames[1:])
        )
        track_items.append({
            "track_id": track_id,
            "observed_frames": len(unique_frames),
            "first_frame": unique_frames[0],
            "last_frame": unique_frames[-1],
            "internal_missing_frames": gaps,
            "is_short": len(unique_frames) < short_track,
        })

    lengths = [item["observed_frames"] for item in track_items]
    short_tracks = [item for item in track_items if item["is_short"]]
    return {
        "total_tracks": len(track_items),
        "untracked_person_observations": untracked_persons,
        "avg_track_length": round(mean_or_zero(lengths), 3),
        "short_track_count": len(short_tracks),
        "short_track_ratio": round(ratio(len(short_tracks), len(track_items)), 4),
        "tracks": track_items,
    }


def compute_keypoint_stats(
    frames: list[dict[str, Any]],
    low_conf: float,
    jitter_gap: int,
) -> dict[str, Any]:
    per_keypoint: dict[str, dict[str, Any]] = defaultdict(new_keypoint_bucket)
    all_confidences: list[float] = []
    motions: list[float] = []
    # 以 track_id + keypoint name 连接相邻帧，用于估算关键点逐帧运动/抖动。
    previous_points: dict[tuple[str, str], tuple[int, float, float]] = {}

    for frame in frames:
        frame_index = int(frame.get("frame_index", 0))
        for person in frame.get("persons", []):
            track_id = person.get("track_id")
            for keypoint in person.get("keypoints", []):
                name = str(keypoint.get("name", "unknown"))
                bucket = per_keypoint[name]
                bucket["total"] += 1
                x = keypoint.get("x")
                y = keypoint.get("y")
                confidence = float(keypoint.get("confidence", 0.0))
                if x is None or y is None:
                    bucket["missing"] += 1
                    continue

                bucket["observed"] += 1
                bucket["confidence_sum"] += confidence
                all_confidences.append(confidence)
                if confidence < low_conf:
                    bucket["low_conf"] += 1

                if track_id is None:
                    continue
                key = (str(track_id), name)
                previous = previous_points.get(key)
                if previous:
                    prev_frame, prev_x, prev_y = previous
                    if 0 < frame_index - prev_frame <= jitter_gap:
                        motions.append(hypot(float(x) - prev_x, float(y) - prev_y))
                previous_points[key] = (frame_index, float(x), float(y))

    keypoint_items = []
    for name, bucket in sorted(per_keypoint.items()):
        observed = bucket["observed"]
        total = bucket["total"]
        keypoint_items.append({
            "name": name,
            "total": total,
            "observed": observed,
            "missing_ratio": round(ratio(bucket["missing"], total), 4),
            "low_conf_ratio": round(ratio(bucket["low_conf"], observed), 4),
            "avg_confidence": round(ratio(bucket["confidence_sum"], observed), 4),
        })

    worst_keypoints = sorted(
        keypoint_items,
        key=lambda item: (item["avg_confidence"], -item["missing_ratio"]),
    )[:10]

    return {
        "avg_confidence": round(mean_or_zero(all_confidences), 4),
        "low_conf_ratio": round(
            ratio(sum(1 for value in all_confidences if value < low_conf),
                  len(all_confidences)), 4),
        "missing_ratio": round(
            ratio(sum(item["total"] - item["observed"] for item in keypoint_items),
                  sum(item["total"] for item in keypoint_items)), 4),
        "motion_px_per_frame_mean": round(mean_or_zero(motions), 3),
        "motion_px_per_frame_p95": round(percentile(motions, 95), 3),
        "per_keypoint": keypoint_items,
        "worst_keypoints": worst_keypoints,
    }


def compute_ball_stats(frames: list[dict[str, Any]]) -> dict[str, Any]:
    total_frames = len(frames)
    state_counts: Counter[str] = Counter()
    missing_streak = 0
    max_missing_streak = 0
    tracks: dict[str, list[tuple[int, float, tuple[float, float]]]] = defaultdict(list)

    for frame in frames:
        frame_index = int(frame.get("frame_index", 0))
        timestamp_sec = float(frame.get("timestamp_sec", 0.0))
        balls = frame.get("balls", [])
        if not balls:
            missing_streak += 1
            max_missing_streak = max(max_missing_streak, missing_streak)
            continue

        missing_streak = 0
        for ball in balls:
            state_counts[str(ball.get("state", "observed"))] += 1
            track_id = str(ball.get("track_id", "none"))
            center = ball.get("center", [None, None])
            if center[0] is not None and center[1] is not None:
                tracks[track_id].append((frame_index, timestamp_sec,
                                         (float(center[0]), float(center[1]))))

    speeds = []
    track_items = []
    for track_id, points in sorted(tracks.items()):
        points = sorted(points, key=lambda item: item[0])
        track_items.append({
            "track_id": track_id,
            "observed_frames": len(points),
            "first_frame": points[0][0],
            "last_frame": points[-1][0],
        })
        for prev, curr in zip(points, points[1:]):
            dt = curr[1] - prev[1]
            if dt > 0:
                speeds.append(hypot(curr[2][0] - prev[2][0],
                                    curr[2][1] - prev[2][1]) / dt)

    return {
        "ball_present_frame_ratio": round(
            ratio(sum(1 for frame in frames if frame.get("balls", [])), total_frames), 4),
        "max_missing_streak": max_missing_streak,
        "state_counts": dict(state_counts),
        "state_ratios": {
            state: round(ratio(count, sum(state_counts.values())), 4)
            for state, count in state_counts.items()
        },
        "speed_px_s_mean": round(mean_or_zero(speeds), 3),
        "speed_px_s_p95": round(percentile(speeds, 95), 3),
        "tracks": track_items,
    }


def compute_quality_score(
    frame_stats: dict[str, Any],
    person_tracks: dict[str, Any],
    keypoints: dict[str, Any],
    balls: dict[str, Any],
) -> dict[str, float]:
    # 质量分是无标注粗评分，用于横向比较视频/参数配置，不等价于真实准确率。
    keypoint_score = clamp(keypoints["avg_confidence"] * 100.0)
    person_score = clamp((1.0 - person_tracks["short_track_ratio"]) * 100.0)
    ball_score = clamp(balls["ball_present_frame_ratio"] * 100.0)
    stability_score = clamp(100.0 - keypoints["motion_px_per_frame_p95"] * 2.0)
    total = (
        0.35 * keypoint_score +
        0.25 * person_score +
        0.25 * ball_score +
        0.15 * stability_score
    )
    return {
        "total": round(total, 2),
        "keypoint": round(keypoint_score, 2),
        "person_tracking": round(person_score, 2),
        "ball_tracking": round(ball_score, 2),
        "stability": round(stability_score, 2),
    }


def select_review_frames(
    frames: list[dict[str, Any]],
    low_conf: float,
    limit: int,
) -> list[dict[str, Any]]:
    review = []
    for frame in frames:
        reasons = []
        persons = frame.get("persons", [])
        balls = frame.get("balls", [])
        if not persons:
            reasons.append("no_person")
        if not balls:
            reasons.append("no_ball")

        confidences = [
            float(keypoint.get("confidence", 0.0))
            for person in persons
            for keypoint in person.get("keypoints", [])
            if keypoint.get("x") is not None and keypoint.get("y") is not None
        ]
        if confidences and mean(confidences) < low_conf:
            reasons.append("low_keypoint_confidence")

        if reasons:
            review.append({
                "frame_index": frame.get("frame_index", 0),
                "timestamp_sec": frame.get("timestamp_sec", 0.0),
                "reasons": reasons,
            })
        if len(review) >= limit:
            break
    return review


def write_json(path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(report), encoding="utf-8")


def validate_metadata(record: dict[str, Any], path: str | Path) -> None:
    if record["type"] != "metadata":
        raise ValueError(f"First JSONL record must be metadata: {path}")
    _ = record["schema_version"]
    video = record["video"]
    required = ("video_path", "fps", "width", "height", "frame_count", "stride",
                "camera_id")
    for key in required:
        _ = video[key]


def validate_frame_record(record: dict[str, Any], line_no: int,
                          path: str | Path) -> None:
    if record["type"] != "frame":
        raise ValueError(f"Expected frame record at line {line_no}: {path}")
    _ = record["frame_index"]
    _ = record["timestamp_sec"]
    _ = record["camera_id"]
    _ = record["persons"]
    _ = record["balls"]


def render_markdown(report: dict[str, Any]) -> str:
    frame_stats = report["frame_stats"]
    person_tracks = report["person_tracks"]
    keypoints = report["keypoints"]
    balls = report["balls"]
    score = report["quality_score"]
    video = report["metadata"]["video"]

    lines = [
        "# Single-view Quality Report",
        "",
        "## Summary",
        "",
        f"- input: `{report['input']}`",
        f"- video: `{video['video_path']}`",
        f"- camera: `{video['camera_id']}`",
        f"- fps: `{video['fps']}`",
        f"- resolution: `{video['width']}x{video['height']}`",
        f"- video frames: `{video['frame_count']}`",
        f"- stride: `{video['stride']}`",
        f"- total score: `{score['total']}`",
        f"- frames: `{frame_stats['total_frames']}`",
        f"- avg persons/frame: `{frame_stats['avg_persons_per_frame']}`",
        f"- ball present ratio: `{frame_stats['ball_present_frame_ratio']}`",
        f"- avg keypoint confidence: `{keypoints['avg_confidence']}`",
        "",
        "## Score Breakdown",
        "",
        "| item | score |",
        "| --- | ---: |",
        f"| keypoint | {score['keypoint']} |",
        f"| person tracking | {score['person_tracking']} |",
        f"| ball tracking | {score['ball_tracking']} |",
        f"| stability | {score['stability']} |",
        "",
        "## Person Tracking",
        "",
        f"- total tracks: `{person_tracks['total_tracks']}`",
        f"- avg track length: `{person_tracks['avg_track_length']}`",
        f"- short track count: `{person_tracks['short_track_count']}`",
        f"- short track ratio: `{person_tracks['short_track_ratio']}`",
        "",
        "## Keypoints",
        "",
        f"- low confidence ratio: `{keypoints['low_conf_ratio']}`",
        f"- missing ratio: `{keypoints['missing_ratio']}`",
        f"- motion p95 px/frame: `{keypoints['motion_px_per_frame_p95']}`",
        "",
        "### Worst Keypoints",
        "",
        "| keypoint | avg confidence | missing ratio | low confidence ratio |",
        "| --- | ---: | ---: | ---: |",
    ]

    for item in keypoints["worst_keypoints"]:
        lines.append(
            f"| {item['name']} | {item['avg_confidence']} | "
            f"{item['missing_ratio']} | {item['low_conf_ratio']} |")

    lines.extend([
        "",
        "## Ball",
        "",
        f"- present frame ratio: `{balls['ball_present_frame_ratio']}`",
        f"- max missing streak: `{balls['max_missing_streak']}`",
        f"- speed p95 px/s: `{balls['speed_px_s_p95']}`",
        f"- state counts: `{balls['state_counts']}`",
        "",
        "## Frames To Review",
        "",
    ])

    if not report["review_frames"]:
        lines.append("- none")
    else:
        for frame in report["review_frames"]:
            reasons = ", ".join(frame["reasons"])
            lines.append(
                f"- frame `{frame['frame_index']}` t=`{frame['timestamp_sec']}`: {reasons}"
            )

    lines.append("")
    return "\n".join(lines)


def new_keypoint_bucket() -> dict[str, Any]:
    return {
        "total": 0,
        "observed": 0,
        "missing": 0,
        "low_conf": 0,
        "confidence_sum": 0.0,
    }


def count_equal(values: list[int], expected: int) -> int:
    return sum(1 for value in values if value == expected)


def count_greater(values: list[int], expected: int) -> int:
    return sum(1 for value in values if value > expected)


def mean_or_zero(values: list[float] | list[int]) -> float:
    return mean(values) if values else 0.0


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * percent / 100.0))
    return sorted_values[index]


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


if __name__ == "__main__":
    main()
