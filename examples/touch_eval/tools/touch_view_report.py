#!/usr/bin/env python3
"""Print the same run under the views that are easier to argue about.

`touch_report.txt` is deliberately strict: one prediction per annotated touch is a
hit, so a touch the detector reported twice inside a one second annotation window
costs a false positive. That is the right number to track, but it is not the
number to quote when the question is "does it find the touches at all". This
tool prints both, from the artifacts an evaluation already wrote:

    python3 examples/touch_eval/tools/touch_view_report.py \
        --output-dir output/touch_eval_moving_pass

  event level     every prediction is scored on its own (the primary metric).
  process level   predictions of the same person and foot within `--process-gap`
                  are merged into one touch process first, so reporting a touch
                  three times is one extra process, not three extra detections.

`--coverage` adds a third, many-to-one reading: an annotation counts as found if
any prediction lands within T seconds of it, and a prediction counts as supported
if any annotation is within T. It answers "did it see this touch" and is useful
when hunting for misses, but it is a *diagnostic*: it is not precision and recall,
and the numbers to report are the one-to-one ones above.

Reads <output-dir>/touch_events.jsonl and <output-dir>/ground_truth.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_events(output_dir: Path) -> list[dict]:
    events = []
    with open(output_dir / "touch_events.jsonl", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if "timestamp_sec" in event:
                events.append(event)
    return events


def percent(value: float) -> str:
    return f"{100.0 * value:5.1f}%"


def merge_processes(events: list[dict], gap_sec: float) -> list[dict]:
    """Same person and foot, close in time: one touch process."""
    processes = []
    for event in sorted(events, key=lambda item: item["timestamp_sec"]):
        key = (event.get("person_track_id", -1), event.get("foot", ""))
        last = processes[-1] if processes else None
        if (last is not None and last["key"] == key
                and event["timestamp_sec"] - last["last"] <= gap_sec):
            last["last"] = event["timestamp_sec"]
            last["events"] += 1
            last["hit"] = last["hit"] or event.get("verdict") == "assigned"
        else:
            processes.append({
                "key": key,
                "first": event["timestamp_sec"],
                "last": event["timestamp_sec"],
                "events": 1,
                "hit": event.get("verdict") == "assigned",
            })
    return processes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=".",
                        help="directory holding touch_events.jsonl and "
                             "ground_truth.json")
    parser.add_argument("--process-gap", default="0.5,1.0",
                        help="seconds within which two predictions of the same "
                             "person and foot are one process")
    parser.add_argument("--coverage", default="",
                        help="opt in to the many-to-one coverage diagnostic, e.g. "
                             "--coverage 0.5,1.0 (not precision/recall)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    events = load_events(output_dir)
    with open(output_dir / "ground_truth.json", encoding="utf-8") as handle:
        ground_truth = json.load(handle)
    annotations = ground_truth["events"]
    if not annotations:
        raise SystemExit(f"{output_dir}/ground_truth.json has no touch row")
    if not events:
        raise SystemExit(f"{output_dir}/touch_events.jsonl has no event")

    hits = [event for event in events if event.get("verdict") == "assigned"]
    capacity = sum(max(1, row.get("count", 1)) for row in annotations)

    print(f"predictions {len(events)}, annotations {len(annotations)} "
          f"({capacity} touches claimed)")

    def score(true_positives: int, predicted: int, claimed: int,
              recall_hits: int | None = None) -> str:
        hits_for_recall = true_positives if recall_hits is None else recall_hits
        precision = true_positives / predicted if predicted else 0.0
        recall = hits_for_recall / claimed if claimed else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        return (f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

    print("\n-- event level (the primary metric) --")
    print(f"  {score(len(hits), len(events), capacity)}  "
          f"TP={len(hits)} FP={len(events) - len(hits)} FN={capacity - len(hits)}")

    # Recall always counts annotations: merging two annotated touches into one
    # process must not make the detector look like it missed one.
    claimed_hits = len({event.get("gt_index", -1) for event in hits
                        if event.get("gt_index", -1) >= 0})
    print("\n-- process level (same person and foot merged) --")
    for gap in (float(part) for part in args.process_gap.split(",")):
        processes = merge_processes(events, gap)
        process_hits = sum(1 for process in processes if process["hit"])
        sizes = {}
        for process in processes:
            sizes[process["events"]] = sizes.get(process["events"], 0) + 1
        spread = ", ".join(f"{count}x{size}" for size, count in sorted(sizes.items()))
        print(f"  gap {gap:4.2f}s: {len(events)} predictions -> {len(processes)} "
              f"processes ({spread})")
        print(f"             {score(process_hits, len(processes), capacity, claimed_hits)}"
              f"  (recall still counts annotations)")

    # Off by default: this is a diagnostic for chasing misses, and its two
    # numbers are deliberately not called precision and recall. Only run it when
    # the caller asked for it.
    if not args.coverage.strip():
        return 0
    print("\n-- coverage (many-to-one diagnostic, NOT the metric) --")
    print("  annotations found     = a touch has some prediction within T")
    print("  predictions supported = a prediction has some annotation within T")
    print("  both are one sided: one prediction can support two annotations and "
          "the other way round")
    print("  tol(s)  annotations found      predictions supported   F1")
    for tolerance in (float(part) for part in args.coverage.split(",")):
        found = 0
        readable_found = 0
        readable = 0
        for row in annotations:
            low, high = row["start_sec"] - tolerance, row["end_sec"] + tolerance
            hit = any(low <= event["timestamp_sec"] <= high for event in events)
            found += 1 if hit else 0
            if not row.get("low_quality", False):
                readable += 1
                readable_found += 1 if hit else 0
        supported = 0
        for event in events:
            stamp = event["timestamp_sec"]
            if any(row["start_sec"] - tolerance <= stamp <= row["end_sec"] + tolerance
                   for row in annotations):
                supported += 1
        recall = found / len(annotations) if annotations else 0.0
        precision = supported / len(events) if events else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        print(f"  {tolerance:5.2f}   {found:3d}/{len(annotations)} "
              f"({percent(recall):>6})      {supported:3d}/{len(events)} "
              f"({percent(precision):>6})   {f1:.3f}")
        print(f"          readable rows only {readable_found}/{readable} "
              f"({percent(readable_found / readable if readable else 0.0)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
