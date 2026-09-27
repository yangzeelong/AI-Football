#!/usr/bin/env python3
"""Sweep one detector knob at a time and plot precision against recall.

Nothing is reimplemented here: every point comes from a fresh `touch_eval` run on
the same observations file, so a plotted number is exactly the number the report
prints. A 16037 frame replay takes ~2 s on CPU, so a dozen point sweep is well
under a minute and needs no GPU.

    python3 examples/touch_eval/tools/plot_touch_sweep.py \
        --observations output/touch_eval_moving_pass/observations.jsonl \
        --annotations data/移动传球.touches.csv \
        --output-dir output/touch_eval_moving_pass

    # a knob whose values are not in the preset, or two sweeps in one go
    python3 ... --sweep min_contact_px=16,20,24,28 --sweep window_pad

Writes, per knob, `<output-dir>/sweep_<knob>.csv` (the numbers, one row per
value, both views) and `<output-dir>/pr_<knob>.png` (left: recall vs precision
with iso-F1 guides; right: precision / recall / F1 against the knob).

Which knob is worth sweeping:

  min_contact_px      the real operating point today: at 720p the per person
                      height term is clamped by this floor, so it is what
                      actually decides whether a frame counts as contact.
                      This is the knob that trades precision for recall.
  contact_ratio       scales the frame term (min(W,H) * ratio) and only shows up
                      once it exceeds the floor; a wide sweep of it looks flat
                      for that reason.
  person_height_ratio how far the per person cap can move before the clamps.
  window_pad          how much slack the annotation itself is given. Widening a
                      window can only turn false positives into hits, never the
                      other way round, so its curve is a monotone line and NOT a
                      precision/recall trade-off: treat it as an evaluation
                      policy, not as a detector setting.
  refractory_frames   same person and foot within this many frames is one touch.
  min_contact_frames  contact frames required to accept one touch.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)

# Values used when --sweep only names the knob. Ordered as the sweep runs, which
# is also the order the curve is drawn in.
KNOB_PRESETS = {
    "min_contact_px": [10, 12, 14, 16, 18, 20, 22, 25, 28, 32, 36, 40],
    "contact_ratio": [0.005, 0.01, 0.015, 0.02, 0.03, 0.045, 0.06, 0.08, 0.12],
    "person_height_ratio": [0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20],
    "window_pad": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0],
    "refractory_frames": [0, 6, 12, 18, 21, 24, 31, 40, 60],
    "min_contact_frames": [1, 2, 3, 4, 6, 8],
    "release_ratio": [1.0, 1.2, 1.4, 1.6, 2.0],
}

# Where each knob's effective value can be read back from a report, so the
# current default can be marked without guessing it here.
PARAM_KEYS = {
    "contact_ratio": "contactDistanceRatio",
    "person_height_ratio": "personHeightRatio",
    "min_contact_px": "minContactPx",
    "max_contact_px": "maxContactPx",
    "release_ratio": "releaseRatio",
    "min_contact_frames": "minContactFrames",
    "max_gap_frames": "maxGapFrames",
    "min_split_frames": "minSplitFrames",
    "min_rise_frames": "minRiseFrames",
    "refractory_frames": "refractoryFrames",
    "min_speed_change": "minSpeedChangePxPerSec",
    "min_speed_ratio": "minSpeedRatio",
}

KNOB_NOTES = {
    "min_contact_px": "the operating point: higher = more events, lower precision",
    "contact_ratio": "frame term, in effect once it clears min_contact_px",
    "person_height_ratio": "per person cap for distant players",
    "window_pad": "slack given to the annotation - monotone, not a trade-off",
    "refractory_frames": "same person and foot within N frames = one touch",
    "min_contact_frames": "contact frames required for one touch",
    "release_ratio": "hysteresis: a cluster survives up to enter * ratio",
}


def parse_sweeps(specs: list[str]) -> list[tuple[str, list[float]]]:
    """Turn `--sweep knob` / `--sweep knob=1,2,3` into (knob, values) pairs."""
    sweeps = []
    for spec in specs:
        if "=" in spec:
            name, raw = spec.split("=", 1)
            name = name.strip()
            values = [float(part) for part in raw.split(",") if part.strip()]
        else:
            name = spec.strip()
            if name not in KNOB_PRESETS:
                raise SystemExit(
                    f"unknown knob '{name}'; presets: "
                    + ", ".join(sorted(KNOB_PRESETS))
                )
            values = list(KNOB_PRESETS[name])
        if not values:
            raise SystemExit(f"no value given for '{name}'")
        sweeps.append((name, values))
    return sweeps


def format_value(value: float) -> str:
    """Integers stay integers on the command line, floats keep their digits."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def annotation_args(path: str) -> list[str]:
    """The evaluator picks its parser from the extension; keep that mapping."""
    return ["--touches_csv", path] if path.lower().endswith(".csv") else ["--label", path]


def run_evaluator(args, knob: str | None, value: float, workdir: Path) -> dict:
    """One evaluator run. `knob is None` runs with the built in defaults."""
    command = [
        args.touch_eval,
        "--observations", args.observations,
        "--output_dir", str(workdir),
        "--no_sweep",  # the calibration curve would only slow the sweep down
        "--no_csv",    # per frame contact signal is not needed either
        "--quiet",
    ]
    if knob is not None:
        command += [f"--{knob}", format_value(value)]
    command += annotation_args(args.annotations)
    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0:
        sys.stderr.write(finished.stdout)
        sys.stderr.write(finished.stderr)
        raise SystemExit(f"{knob}={value} failed with {finished.returncode}")
    with open(workdir / "touch_report.json", encoding="utf-8") as handle:
        return json.load(handle)


def score_of(block: dict) -> dict:
    return {
        "tp": block["true_positives"],
        "fp": block["false_positives"],
        "fn": block["false_negatives"],
        "precision": block["precision"],
        "recall": block["recall"],
        "f1": block["f1"],
    }


def collect_point(report: dict) -> dict:
    """Both views of one run: everything, and without the unreadable rows."""
    point = {"all": score_of(report["primary_count_aware"])}
    low_quality = report.get("low_quality", {})
    if low_quality.get("rows", 0) > 0:
        point["clean"] = score_of(low_quality["primary_count_aware"])
        point["clean_rows"] = low_quality["rows"]
        point["clean_excluded_events"] = low_quality["excluded_events"]
    return point


def default_value(report: dict, knob: str) -> float | None:
    """The value the run without an explicit knob would use."""
    if knob == "window_pad":
        return float(report["primary_count_aware"]["window_pad_sec"])
    if knob == "point_window":
        return float(report["primary_count_aware"]["point_window_sec"])
    key = PARAM_KEYS.get(knob)
    if key is None:
        return None
    match = re.search(rf"{key}=([-0-9.eE+]+)", report.get("params", ""))
    return float(match.group(1)) if match else None


def iso_f1(level: float, recalls):
    """Precision for a target F1 at each recall: p = f*r / (2r - f)."""
    return [level * r / (2.0 * r - level) if r > level / 2.0 else None for r in recalls]


def iso_levels(rows: list[dict]) -> list[float]:
    """Guides around the scores that were actually reached.

    A fixed 0.2..0.8 ladder is useless once a detector sits in a narrow band: the
    useful question is which side of the current F1 a point falls on.
    """
    values = sorted(row[view]["f1"] for row in rows
                    for view in ("all", "clean") if view in row)
    middle = values[len(values) // 2]
    steps = [round((middle + offset) * 20.0) / 20.0 for offset in (-0.05, 0.0, 0.05)]
    return [value for value in dict.fromkeys(steps) if 0.05 <= value <= 0.95]


def zoom(values: list[float], low_bound: float = 0.0,
         high_bound: float = 1.02, pad: float = 0.08,
         minimum_span: float = 0.22) -> tuple[float, float]:
    """Frame the data instead of the whole 0..1 square.

    A detector sweep usually moves precision and recall by a few hundredths; on a
    full square those points collapse into one blob and the shape of the curve
    (which is the whole point of the plot) is invisible.
    """
    low = max(low_bound, min(values) - pad)
    high = min(high_bound, max(values) + pad)
    if high - low < minimum_span:
        middle = 0.5 * (low + high)
        low = max(low_bound, middle - minimum_span / 2.0)
        high = min(high_bound, low + minimum_span)
    return low, high


def plot_knob(knob: str, rows: list[dict], default: float | None, out_path: Path,
              meta: dict) -> None:
    labelled = [row for row in rows if row["all"]["tp"] > 0]
    has_clean = any("clean" in row for row in rows)

    context = [f"{meta.get('frames', '?')} frames"]
    if meta.get("touches") is not None:
        context.append(f"{meta['touches']} annotated touches")
    if meta.get("events") is not None:
        context.append(f"{meta['events']} predicted events")
    if meta.get("point_window_sec") is not None:
        context.append(f"point window {meta['point_window_sec']:.2f} s + pad "
                       f"{meta['pad_sec']:.2f} s")

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=150)
    figure.suptitle(
        f"touch detection sweep — {knob}  ({KNOB_NOTES.get(knob, '')})\n"
        + ", ".join(context),
        fontsize=11,
    )

    # --- precision against recall, with iso-F1 guides -----------------------
    recall_values = [row[view]["recall"] for row in labelled
                     for view in ("all", "clean") if view in row]
    precision_values = [row[view]["precision"] for row in labelled
                        for view in ("all", "clean") if view in row]
    x_low, x_high = zoom(recall_values)
    y_low, y_high = zoom(precision_values)

    recalls = [x_low + i * (x_high - x_low) / 200.0 for i in range(201)]
    for level in iso_levels(labelled):
        curve = iso_f1(level, recalls)
        left.plot(recalls, curve, color="#cccccc", linewidth=0.8, zorder=1)
        # Label where the guide leaves the top of the window, else at its left
        # edge: either way the text stays inside the zoomed axes.
        if 2.0 * y_high > level:
            exit_recall = level * y_high / (2.0 * y_high - level)
        else:
            exit_recall = None
        if exit_recall is not None and x_low <= exit_recall <= x_high:
            # Just inside the top edge: a label sitting on it collides with the
            # panel title.
            left.annotate(f"F1={level:.2f}", (exit_recall, y_high), fontsize=7,
                          color="#999999", ha="center", va="top", zorder=2,
                          textcoords="offset points", xytext=(0, -3))
        else:
            p_left = iso_f1(level, [x_low])[0]
            if p_left is not None and y_low <= p_left <= y_high:
                left.annotate(f"F1={level:.2f}", (x_low, p_left), fontsize=7,
                              color="#999999", ha="left", va="bottom", zorder=2,
                              textcoords="offset points", xytext=(4, 3))

    def series(key: str, colour: str, marker: str, label: str):
        points = sorted((row[key]["recall"], row[key]["precision"])
                        for row in labelled)
        left.plot([point[0] for point in points], [point[1] for point in points],
                  color=colour, marker=marker, markersize=5, linewidth=1.4,
                  label=label, zorder=3)

    series("all", "#1f77b4", "o", "all annotations")
    if has_clean:
        series("clean", "#2ca02c", "s", "low quality rows excluded")

    for index, row in enumerate(labelled):
        # Values crowd together, so neighbouring labels alternate sides.
        offset = (6, 5) if index % 2 == 0 else (-16, -11)
        left.annotate(f"{row['value']:g}", (row["all"]["recall"], row["all"]["precision"]),
                      textcoords="offset points", xytext=offset, fontsize=6.5,
                      color="#1f77b4", zorder=4)
        if row["value"] == default:
            left.plot(row["all"]["recall"], row["all"]["precision"], marker="*",
                      markersize=17, color="#d62728", zorder=5,
                      label=f"default ({default:g})")

    left.set_xlim(x_low, x_high)
    left.set_ylim(y_low, y_high)
    left.set_xlabel(f"recall  (axis {x_low:.2f}-{x_high:.2f})")
    left.set_ylabel(f"precision  (axis {y_low:.2f}-{y_high:.2f})")
    left.set_title("precision-recall (label = knob value)")
    left.grid(alpha=0.25)
    left.legend(loc="lower left", fontsize=8)

    # --- precision, recall and F1 against the knob --------------------------
    values = [row["value"] for row in rows]
    for key, style in (("precision", ":"), ("recall", "--"), ("f1", "-")):
        right.plot(values, [row["all"][key] for row in rows], style,
                   color="#1f77b4", marker="o", markersize=3.5,
                   label=f"{key} (all)")
    if has_clean:
        right.plot(values, [row["clean"]["f1"] for row in rows], "-",
                   color="#2ca02c", marker="s", markersize=3.5,
                   label="f1 (low quality excluded)")
    if default is not None:
        right.axvline(default, color="#d62728", linewidth=1.0, linestyle="-.",
                      label=f"default ({default:g})")
    best = max(rows, key=lambda row: row["all"]["f1"])
    right.plot(best["value"], best["all"]["f1"], marker="*", markersize=15,
               color="#ff7f0e", zorder=5,
               label=f"best F1 ({best['value']:g} -> {best['all']['f1']:.3f})")
    right.set_xlabel(knob)
    right.set_ylabel("score")
    right.set_title(f"metrics against {knob}")
    score_values = [row[view][key] for row in rows
                    for view in ("all", "clean") if view in row
                    for key in ("precision", "recall", "f1")]
    right.set_ylim(*zoom(score_values, pad=0.05, minimum_span=0.15))
    right.grid(alpha=0.25)
    right.legend(loc="upper right", fontsize=8)

    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(out_path)
    plt.close(figure)


def read_csv(path: Path) -> list[dict]:
    """Inverse of write_csv, for re-plotting without re-running the sweep."""
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            row = {"value": float(record["value"])}
            for view in ("all", "clean"):
                if record.get(f"tp_{view}", "") == "":
                    continue
                row[view] = {
                    key: (int(record[f"{key}_{view}"]) if key in ("tp", "fp", "fn")
                          else float(record[f"{key}_{view}"]))
                    for key in ("tp", "fp", "fn", "precision", "recall", "f1")
                }
            for key in ("clean_rows", "clean_excluded_events"):
                if record.get(key, "") != "":
                    row[key] = int(record[key])
            rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    columns = ["value"]
    for view in ("all", "clean"):
        for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
            columns.append(f"{key}_{view}")
    columns += ["clean_rows", "clean_excluded_events"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {"value": row["value"]}
            for view in ("all", "clean"):
                for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
                    value = row.get(view, {}).get(key)
                    flat[f"{key}_{view}"] = "" if value is None else value
            flat["clean_rows"] = row.get("clean_rows", "")
            flat["clean_excluded_events"] = row.get("clean_excluded_events", "")
            writer.writerow(flat)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep detector knobs and plot precision against recall.")
    parser.add_argument("--observations", required=True,
                        help="observations.jsonl written by aifootball_app")
    parser.add_argument("--annotations", required=True,
                        help="v2 CSV or legacy .label (the extension picks the parser)")
    parser.add_argument("--output-dir", default=".",
                        help="where the CSV and PNG are written")
    parser.add_argument("--touch-eval", default="build/bin/touch_eval",
                        help="evaluator binary; every point is one run of it")
    parser.add_argument("--sweep", action="append", default=[],
                        help="knob to sweep, optionally 'knob=v1,v2,...'; "
                             "defaults to min_contact_px and window_pad")
    parser.add_argument("--keep-temp", action="store_true",
                        help="leave the per value reports on disk for inspection")
    parser.add_argument("--render-only", action="store_true",
                        help="re-plot the sweep_*.csv already in --output-dir, "
                             "without running the evaluator (for plot tweaks)")
    args = parser.parse_args()

    specs = args.sweep or ["min_contact_px", "window_pad"]
    sweeps = parse_sweeps(specs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        report_path = output_dir / "touch_report.json"
        report = {}
        if report_path.is_file():
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle)
        meta = {}
        if report:
            meta = {
                "frames": f"{report['frames']:,}",
                "touches": report["ground_truth"]["touches"],
                "events": report["predicted_events"],
                "point_window_sec":
                    report["primary_count_aware"]["point_window_sec"],
                "pad_sec": report["primary_count_aware"]["window_pad_sec"],
            }
        for knob, _ in sweeps:
            csv_path = output_dir / f"sweep_{knob}.csv"
            if not csv_path.is_file():
                raise SystemExit(f"no {csv_path} to re-plot; run the sweep first")
            rows = read_csv(csv_path)
            default = default_value(report, knob) if report else None
            plot_knob(knob, rows, default, output_dir / f"pr_{knob}.png", meta)
            print(f"{csv_path} -> {output_dir / f'pr_{knob}.png'}")
        return 0

    temp_root = Path(tempfile.mkdtemp(prefix="touch_sweep_"))
    try:
        # One run without a knob override: it gives the baseline the default
        # markers use, and the metadata that goes in the plot titles.
        baseline_dir = temp_root / "default"
        baseline_dir.mkdir()
        baseline = run_evaluator(args, None, 0.0, baseline_dir)
        meta = {
            "frames": f"{baseline['frames']:,}",
            "touches": baseline["ground_truth"]["touches"],
            "events": baseline["predicted_events"],
            "point_window_sec": baseline["primary_count_aware"]["point_window_sec"],
            "pad_sec": baseline["primary_count_aware"]["window_pad_sec"],
        }
        print(f"baseline (no knob override): "
              f"P={baseline['primary_count_aware']['precision']:.3f} "
              f"R={baseline['primary_count_aware']['recall']:.3f} "
              f"F1={baseline['primary_count_aware']['f1']:.3f}")

        for knob, values in sweeps:
            rows = []
            for index, value in enumerate(values):
                workdir = temp_root / f"{knob}_{index}"
                workdir.mkdir()
                report = run_evaluator(args, knob, value, workdir)
                point = collect_point(report)
                point["value"] = value
                rows.append(point)
                print(f"{knob}={value:<10g} all: TP={point['all']['tp']:<3d} "
                      f"FP={point['all']['fp']:<3d} FN={point['all']['fn']:<3d} "
                      f"P={point['all']['precision']:.3f} "
                      f"R={point['all']['recall']:.3f} "
                      f"F1={point['all']['f1']:.3f}"
                      + (f" | clean F1={point['clean']['f1']:.3f}"
                         if "clean" in point else ""))

            default = default_value(baseline, knob)
            csv_path = output_dir / f"sweep_{knob}.csv"
            png_path = output_dir / f"pr_{knob}.png"
            write_csv(rows, csv_path)
            plot_knob(knob, rows, default, png_path, meta)

            best = max(rows, key=lambda row: row["all"]["f1"])
            print(f"  -> {csv_path}")
            print(f"  -> {png_path}")
            print(f"  best by F1 (all): {knob}={best['value']:g} "
                  f"F1={best['all']['f1']:.3f} (default {default:g})")
    finally:
        if args.keep_temp:
            print(f"per value reports kept in {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
