#!/usr/bin/env python3
"""Render every false positive and false negative of the strict reading.

The number that is reported counts a prediction as a hit only when it falls
inside the span its annotation writes, so its mistakes deserve to be looked at
one by one instead of being summarised away. Two products come out of it:

* **one contact sheet per error** - the frames around it, cut out of
  `rendered.mp4`, with the numbers and the Chinese reason burned in, plus an
  index that names the fix as well;
* **one full-length playback** (`--playback`) - the whole recording at the rate
  it was shot, so a reviewer can pause wherever the note says something: the
  reason at the top centre, and two timelines at the bottom (`pred` over `gt`,
  green = hit, red = false positive, yellow = miss).

    python3 examples/touch_eval/tools/render_pr_errors.py \\
        --output-dir output/touch_eval_moving_pass            # sheets + index
    python3 examples/touch_eval/tools/render_pr_errors.py \\
        --output-dir output/touch_eval_moving_pass --playback --playback-only

`--tolerance 0.5` restores the older +-0.5 s reading; the default is 0, which is
what `touch_report.json` leads with.

Outputs, all inside <output-dir>/pr_review/:

    fp_003_35s_adjacent.png        one contact sheet per error, reason burned in
    fn_gt019_47_00s_never_near.png
    index.md                       the same list in text, each row naming the
                                   image it describes and the fix
    index.csv                      one row per error, UTF-8, for a spreadsheet
    touch_playback.mp4             optional (--playback): the whole recording,
                                   real time, with the note and the timelines

The assignment is recomputed here with the same rule as the C++ report
(maximum cardinality first, closest pairs among the optimal ones) and the
totals are checked against `touch_report.json` before anything is rendered, so
a sheet can never disagree with the printed score.

Chinese text needs a CJK font. The tool looks for one (env `PR_REVIEW_FONT`,
then the usual system paths, then anything fontconfig knows); if the machine has
none, install one first:

    apt-get install -y fonts-wqy-microhei          # or fonts-noto-cjk
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_ENV = "PR_REVIEW_FONT"
FONT_PATHS = (
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
)
FALLBACK_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
COLOURS = {"fp": (255, 96, 96), "fn": (255, 196, 64)}
GREY = (176, 176, 182)
#: The advice line: warm, so it reads as the third kind of text on a sheet
#: (title / reason / evidence) rather than as more evidence.
ADVICE_COLOUR = (255, 214, 102)
REASON_CN = {
    "slot_taken": "窗口内名额已满",
    "adjacent": "贴着标注窗口",
    "inside_no_touch": "落在确认无触球区间",
    "unannotated": "附近无任何标注",
    "contact_but_no_event": "球脚贴到阈值但事件被否",
    "never_near": "窗内球从未近到阈值",
    "no_ball": "窗内没有可用球",
    "no_feet": "窗内没有可用脚点",
    "no_signal": "缺少逐帧信号",
}
_CJK: str | None | bool = False   # False = not looked up yet, None = none found


def _renders_chinese(path: str) -> bool:
    """A font 'supports' Chinese only if the glyph is not the .notdef box."""
    try:
        font = ImageFont.truetype(path, 20)
    except OSError:
        return False

    def draw(character: str) -> bytes:
        image = Image.new("L", (48, 48), 0)
        ImageDraw.Draw(image).text((4, 4), character, font=font, fill=255)
        return image.tobytes()

    return draw("触") != draw("\ue001")


def cjk_font_path() -> str | None:
    """The first font on this machine that can draw Chinese, cached."""
    global _CJK
    if _CJK is not False:
        return _CJK
    candidates: list[str] = []
    from_env = os.environ.get(FONT_ENV, "")
    if from_env:
        candidates.append(from_env)
    candidates.extend(FONT_PATHS)
    for root in ("/usr/share/fonts", str(Path.home() / ".local/share/fonts")):
        if os.path.isdir(root):
            candidates.extend(str(path) for path in sorted(Path(root).rglob("*"))
                              if path.suffix.lower() in (".ttf", ".ttc", ".otf"))
    for path in candidates:
        if os.path.isfile(path) and _renders_chinese(path):
            _CJK = path
            return _CJK
    _CJK = None
    return None


def load_font(size: int):
    path = cjk_font_path()
    if path:
        return ImageFont.truetype(path, size)
    for fallback in FALLBACK_PATHS:          # English only, and it says so
        if os.path.exists(fallback):
            return ImageFont.truetype(fallback, size)
    return ImageFont.load_default()


# ---------------------------------------------------------------- matching --

def distance_to(row: dict, stamp: float) -> float:
    """Distance from a prediction to the annotation (zero inside a span)."""
    if row["start_sec"] <= stamp <= row["end_sec"]:
        return 0.0
    return min(abs(stamp - row["start_sec"]), abs(stamp - row["end_sec"]))


def optimal_match(preference: list[list[int]], right_count: int) -> list[int]:
    """Maximum cardinality bipartite matching (Kuhn), as in TouchMetrics.cpp."""
    match_of_event = [-1] * len(preference)
    match_of_right = [-1] * right_count

    def augment(event: int, seen: set[int]) -> bool:
        for right in preference[event]:
            if right in seen:
                continue
            seen.add(right)
            if match_of_right[right] == -1 or augment(match_of_right[right], seen):
                match_of_right[right] = event
                return True
        return False

    for event in range(len(preference)):
        augment(event, set())
    for right, event in enumerate(match_of_right):
        if event >= 0:
            match_of_event[event] = right
    return match_of_event


def tolerance_view(events: list[dict], annotations: list[dict],
                   tolerance: float) -> tuple[dict[int, int], dict[int, int]]:
    """Strict one-to-one: the pairs it finds, in both directions."""
    preference = []
    for event in events:
        ranked = sorted(
            (distance_to(annotation, event["timestamp_sec"]), row)
            for row, annotation in enumerate(annotations)
            if distance_to(annotation, event["timestamp_sec"]) <= tolerance
        )
        preference.append([row for _, row in ranked])
    matched = optimal_match(preference, len(annotations))
    event_to_gt = {event: row for event, row in enumerate(matched) if row >= 0}
    gt_to_event = {row: event for event, row in event_to_gt.items()}
    return event_to_gt, gt_to_event


# -------------------------------------------------------------- the signal --

class ContactSignal:
    """The per frame contact evidence the evaluator wrote beside the report."""

    NUMERIC = ("frame_index", "timestamp_sec", "person_track_id", "distance_px",
               "threshold_px", "ball_confidence", "ball_track_id", "observed_balls",
               "valid_foot_points", "persons_with_feet", "persons")

    def __init__(self, path: Path):
        self.rows: list[dict] = []
        if not path.is_file():
            return
        with open(path, newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row: dict = {}
                for key, value in raw.items():
                    if key not in self.NUMERIC:
                        row[key] = value
                        continue
                    # An empty numeric cell means "no measurement", not zero: it
                    # must never enter a min() or a max() as a string.
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        row[key] = None
                self.rows.append(row)

    @property
    def usable(self) -> bool:
        return bool(self.rows)

    def describe(self, low: float, high: float) -> dict:
        rows = [row for row in self.rows if low <= row["timestamp_sec"] <= high]
        info = {"frames": len(rows), "usable_ball": False, "usable_feet": False,
                "nearest_px": None, "nearest_tau_px": None, "contact": False,
                "max_ball_confidence": None, "contact_frames": 0,
                "contact_runs": 0, "nearest_sec": None, "contact_secs": []}
        for row in rows:
            if (row.get("observed_balls") or 0) > 0:
                info["usable_ball"] = True
            if (row.get("valid_foot_points") or 0) > 0:
                info["usable_feet"] = True
            confidence = row.get("ball_confidence")
            if confidence is not None:
                best = info["max_ball_confidence"]
                info["max_ball_confidence"] = confidence if best is None \
                    else max(best, confidence)
            distance = row.get("distance_px")
            if distance is None:
                continue
            if info["nearest_px"] is None or distance < info["nearest_px"]:
                info["nearest_px"] = distance
                info["nearest_tau_px"] = row.get("threshold_px")
                info["nearest_sec"] = row["timestamp_sec"]
            tau = row.get("threshold_px")
            if tau and distance <= tau:
                info["contact"] = True
                info["contact_secs"].append(row["timestamp_sec"])
        # Contact frames decide between "too short to be an event" and "looked
        # like one touch but the split/refractory rule merged it away", and the
        # number of runs separates one slow contact from two separate ones.
        info["contact_frames"] = len(info["contact_secs"])
        runs = 0
        previous = None
        for stamp in info["contact_secs"]:
            if previous is None or stamp - previous > 0.05:
                runs += 1
            previous = stamp
        info["contact_runs"] = runs
        return info


# ------------------------------------------------------------- the reasons --

def _events_inside(annotation: dict, events: list[dict]) -> list[dict]:
    """Every prediction inside one annotation span, in time order."""
    inside = [event for event in events
              if annotation["start_sec"] <= event["timestamp_sec"] <= annotation["end_sec"]]
    return sorted(inside, key=lambda item: item["timestamp_sec"])


def false_positive_reason(event: dict, annotations: list[dict], events: list[dict],
                          no_touch: list[dict], gt_to_event: dict[int, int],
                          tolerance: float, adjacency: float,
                          refractory_frames: int = 30) -> dict:
    stamp = event["timestamp_sec"]
    distance, row = min(
        ((distance_to(annotation, stamp), row) for row, annotation in enumerate(annotations)),
        key=lambda item: item[0],
    )
    annotation = annotations[row]
    owner = gt_to_event.get(row)
    span = (annotation["start_sec"], annotation["end_sec"])
    inside = _events_inside(annotation, events)
    claim = annotation.get("count", 1) or 1
    if any(other["start_sec"] <= stamp <= other["end_sec"] for other in no_touch):
        kind = "inside_no_touch"
        note = ("落在标注者确认无触球的区间内（最近标注 gt#%d 差 %.2f s），"
                "但这一帧球离脚只有 %.1f px（τ %.1f px）"
                % (annotation["index"], distance, event["distance_px"],
                   event["threshold_px"] or 0.0))
        advice = ("看这一段的视频：若确实触球了，删掉或收窄那条 no_touch；"
                  "若没有，这是误报，记下来当反例")
    elif distance <= tolerance:
        kind = "slot_taken"
        note = ("落在标注 gt#%d（%.2f–%.2f s，claim %d）的跨度内，但这一行只有 %d 个名额，"
                "已配给 %.2f s 的检测；该跨度内一共检出 %d 次接触%s"
                % (annotation["index"], span[0], span[1], claim, claim,
                   events[owner]["timestamp_sec"] if owner is not None else float("nan"),
                   len(inside),
                   "（t=" + ", ".join("%.2f" % item["timestamp_sec"] for item in inside) + "）"
                   if inside else ""))
        if len(inside) > claim:
            advice = ("该秒里检测到 %d 次接触、这一行只 claim %d 次：若视频确实是两次触球，"
                      "把这一行的 count 改成 %d" % (len(inside), claim, len(inside)))
        else:
            advice = ("同一行的重复上报：看两次接触是否同一次，若是则调不应期（当前 %d 帧）"
                      % refractory_frames)
    elif distance <= adjacency:
        kind = "adjacent"
        slack = "（在跨度外）" if tolerance <= 1e-9 else "（超 ±%.2f s）" % tolerance
        note = ("贴着标注 gt#%d：最近标注 %.2f–%.2f s，本次差 %.2f s%s"
                % (annotation["index"], span[0], span[1], distance, slack))
        if owner is not None:
            note += "；该标注已配给 %.2f s 的检测" % events[owner]["timestamp_sec"]
        advice = ("同一个触球的时刻差：看 %.2f s 附近的视频，若触球确实在 %.2f s，"
                  "把 gt#%d 改写成这个时刻（或改成相邻那一秒）"
                  % (event["timestamp_sec"], event["timestamp_sec"], annotation["index"]))
    else:
        kind = "unannotated"
        note = ("前后 %.1f s 内没有任何标注（最近的 gt#%d 也差 %.2f s），"
                "而这一帧球离脚 %.1f px（τ %.1f px）"
                % (adjacency, annotation["index"], distance, event["distance_px"],
                   event["threshold_px"] or 0.0))
        advice = "看 %.2f s 的视频：若确实有触球，缺的是标注（补一行）；若没有，是误报" % stamp
    return {"kind": kind, "note_cn": note, "advice_cn": advice,
            "nearest_row": row,
            "nearest_gt_index": annotation["index"],
            "nearest_gt_sec": annotation["start_sec"],
            "nearest_gt_distance": distance,
            "contacts_in_span": len(inside),
            "gt_index": event.get("gt_index", -1),
            "verdict": event.get("verdict", "")}


def false_negative_reason(annotation: dict, annotations: list[dict],
                          events: list[dict], event_to_gt: dict[int, int],
                          signal: ContactSignal, tolerance: float,
                          min_contact_frames: int = 2,
                          refractory_frames: int = 30,
                          person_refractory_frames: int = 30) -> dict:
    low, high = annotation["start_sec"] - tolerance, annotation["end_sec"] + tolerance
    info = signal.describe(low, high)
    if not signal.usable:
        kind = "no_signal"
        note = "缺少 contact_signal.csv，无法回看窗口"
    elif not info["usable_ball"]:
        kind = "no_ball"
        note = "窗口内没有检出任何球"
    elif not info["usable_feet"]:
        kind = "no_feet"
        note = "窗口内有球，但没有可用的脚部关键点"
    elif info["contact"]:
        kind = "contact_but_no_event"
        note = ("球脚其实贴进了阈值：最近 %.1f px ≤ τ %.1f px，窗口内 %d 帧接触（%d 段）"
                % (info["nearest_px"], info["nearest_tau_px"] or 0.0,
                   info["contact_frames"], info["contact_runs"]))
    else:
        kind = "never_near"
        note = ("窗口内球离最近脚点最近 %.1f px，始终没进阈值（τ %.1f px），差 %.1f px"
                % (info["nearest_px"] if info["nearest_px"] is not None else -1.0,
                   info["nearest_tau_px"] or 0.0,
                   (info["nearest_px"] - info["nearest_tau_px"])
                   if (info["nearest_px"] is not None and info["nearest_tau_px"])
                   else -1.0))

    distance, event_row = min(
        ((distance_to(annotation, event["timestamp_sec"]), row)
         for row, event in enumerate(events)),
        key=lambda item: item[0],
    )
    owner = event_to_gt.get(event_row)
    nearby_sec = events[event_row]["timestamp_sec"]
    note += "；最近的输出事件在 %.2f s 外（%.2f s）" % (distance, nearby_sec)
    if owner is not None:
        note += "，那个事件配给了 gt#%d" % annotations[owner]["index"]
    frames_apart = abs(round(distance * 60.0))
    if kind == "no_ball" or kind == "no_feet":
        advice = ("这一秒里%s，无法判读：如实承认就给它标 low_quality_data=yes，"
                  "别让它在分母里当漏检"
                  % ("没有可用的球" if kind == "no_ball" else "没有可用的脚点"))
    elif kind == "contact_but_no_event":
        if info["contact_frames"] < min_contact_frames:
            advice = ("窗口里只有 %d 帧接触（min_contact_frames=%d），所以没成事件："
                      "看视频确认后把 --min_contact_frames 降到 %d 试一次"
                      % (info["contact_frames"], min_contact_frames,
                         max(1, info["contact_frames"])))
        elif distance <= 0.5:
            advice = ("最近的输出事件就在 %.2f s（约 %d 帧）外：很可能被不应期吞了"
                      "（同脚 %d 帧 / 同人 %d 帧）。要分清是一次触球被拆开还是真两次，"
                      "用 --refractory_frames 0 --person_refractory_frames 0 重跑这一秒核对"
                      % (distance, frames_apart, refractory_frames,
                         person_refractory_frames))
        else:
            advice = ("球脚确实进了阈值但没出事件（太短或被合并）：看视频确认触球时刻，"
                      "再调 --min_contact_frames / --min_split_frames")
    elif kind == "never_near":
        gap = (info["nearest_px"] - (info["nearest_tau_px"] or 0.0)) \
            if info["nearest_px"] is not None else -1.0
        advice = ("球始终没到脚边（还差 %.0f px）：先看视频确认这一秒到底有没有触球——"
                  "没有就是标注错；有则是球或脚在这一段被追丢了"
                  "（看球 confidence 与脚点是否 interpolated）" % gap)
    else:
        advice = "缺少 contact_signal.csv：先跑一次不带 --no_csv 的评估，才能回看这一秒"
    return {"kind": kind, "note_cn": note, "advice_cn": advice,
            "gt_index": annotation["index"],
            "nearest_event_row": event_row,
            "nearest_event_frame": events[event_row]["frame_index"],
            "nearest_event_owner": (annotations[owner]["index"]
                                    if owner is not None else None),
            "nearest_event_sec": events[event_row]["timestamp_sec"],
            "nearest_event_distance": distance,
            "nearest_px": info["nearest_px"], "nearest_tau_px": info["nearest_tau_px"],
            "contact_frames_in_window": info["contact_frames"],
            "usable_ball": info["usable_ball"], "usable_feet": info["usable_feet"],
            "max_ball_confidence": info["max_ball_confidence"],
            "verdict": "missing"}


def raw_stream_check(args, output_dir: Path) -> tuple[list[dict] | None, str]:
    """Re-run the detector with every gate open.

    A miss that is still a miss there is a detection miss; a miss that exists
    only with the gates on was suppressed by a refractory rule.
    """
    observations = output_dir / "observations.jsonl"
    annotations = Path(args.annotations)
    if not observations.is_file() or not annotations.is_file():
        return None, "未做原始流核对（需要 --touch-eval/--observations/--annotations）"
    work = Path(tempfile.mkdtemp(prefix="touch_pr_review_"))
    command = shlex.split(args.touch_eval) + [
        "--observations", str(observations),
        "--touches_csv", str(annotations), "--output_dir", str(work),
        "--no_sweep", "--no_csv", "--quiet",
        "--refractory_frames", "0", "--person_refractory_frames", "0"]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        with open(work / "touch_report.json", encoding="utf-8") as handle:
            events = json.load(handle)["events"]
    except (subprocess.CalledProcessError, OSError, KeyError,
            json.JSONDecodeError) as error:
        shutil.rmtree(work, ignore_errors=True)
        return None, "原始流核对失败：%s" % error
    shutil.rmtree(work, ignore_errors=True)
    return events, "原始流核对：关掉两个不应期重跑，得到 %d 个事件" % len(events)


# --------------------------------------------------- zooming on the action --

FOOT_KEYPOINTS = ("left_ankle", "right_ankle", "left_heel", "right_heel",
                  "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe")
MIN_CROP_PX = 320.0          # never zoom past this, or the crop turns to mush


def collect_observations(path: Path, wanted: set[int]) -> dict[int, dict]:
    """The person boxes, foot keypoints and ball centres of the shown frames.

    One streaming pass over observations.jsonl keeping only the frames a sheet
    needs: the file itself is a few hundred MB, this slice is a few hundred dicts.
    """
    kept: dict[int, dict] = {}
    if not path.is_file() or not wanted:
        return kept
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if '"frame_index"' not in line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = frame.get("frame_index")
            if index not in wanted:
                continue
            people = []
            for person in frame.get("persons", []):
                box = person.get("bbox") or []
                if len(box) != 4:
                    continue
                feet = [(key["x"], key["y"]) for key in person.get("keypoints", [])
                        if key.get("name") in FOOT_KEYPOINTS
                        and key.get("state") == "observed"]
                people.append({"track_id": person.get("track_id"), "bbox": box,
                               "feet": feet})
            balls = [{"center": ball["center"], "track_id": ball.get("track_id"),
                      "confidence": ball.get("confidence")}
                     for ball in frame.get("balls", []) if ball.get("center")]
            kept[index] = {"persons": people, "balls": balls}
    return kept


BALL_SEARCH_PX = 300.0       # how near a foot a person must be to count as "involved"


def choose_ball(info: dict | None, entry: dict,
                people: list[dict]) -> tuple[float, float] | None:
    """The ball this error is about.

    A frame carries several ball candidates (the detector proposes, the tracker
    confirms), and taking all of them would stretch the crop across the whole
    court. For a false positive the report knows the ball's track id; otherwise
    take the candidate nearest to the feet that matter.
    """
    if info is None or not info["balls"]:
        return None
    wanted = entry.get("ball_track_id")
    if wanted is not None and int(wanted) >= 0:
        for ball in info["balls"]:
            if ball["track_id"] == wanted:
                return ball["center"]
    anchors = [foot for person in people for foot in person["feet"]]
    if not anchors:
        anchors = [[0.5 * (person["bbox"][0] + person["bbox"][2]),
                    0.5 * (person["bbox"][1] + person["bbox"][3])] for person in people]
    if not anchors:
        tracked = [ball for ball in info["balls"]
                   if ball["track_id"] is not None and ball["track_id"] >= 1]
        return (tracked or info["balls"])[0]["center"]
    return min(
        (ball["center"] for ball in info["balls"]),
        key=lambda centre: min((centre[0] - ax) ** 2 + (centre[1] - ay) ** 2
                               for ax, ay in anchors),
    )


def crop_rect(infos: list[dict], entry: dict, size: tuple[int, int],
              aspect: float) -> tuple[int, int, int, int] | None:
    """Box around the action: the legs of the player and the ball.

    A touch is judged by where the ball is relative to a foot, so a full 720p
    frame wastes the whole sheet: this keeps the lower two thirds of the player
    the event names (plus whoever stands next to the ball, for a miss) and the
    ball itself. Returns None (full frame) when neither is known.
    """
    # Who is involved: a false positive names its person, a miss does not, so
    # the people near the ball are the ones worth showing there.
    involved: list[list[dict]] = []
    for info in infos:
        if info is None:
            involved.append([])
            continue
        if entry["type"] == "fp":
            people = [person for person in info["persons"]
                      if person["track_id"] is None
                      or person["track_id"] == entry.get("person_track_id")]
        else:
            ball = choose_ball(info, entry, info["persons"])
            if ball is None:
                people = [person for person in info["persons"] if person["feet"]]
            else:
                people = [
                    person for person in info["persons"] if person["feet"] and
                    abs(0.5 * (person["bbox"][0] + person["bbox"][2]) - ball[0]) <= BALL_SEARCH_PX and
                    abs(0.5 * (person["bbox"][1] + person["bbox"][3]) - ball[1]) <= BALL_SEARCH_PX]
        involved.append(people)

    xs: list[float] = []
    ys: list[float] = []
    for info, people in zip(infos, involved):
        if info is None:
            continue
        for person in people:
            x1, y1, x2, y2 = person["bbox"]
            xs += [x1, x2]
            # Legs and ball decide a touch; the head only makes the crop bigger.
            ys += [y1 + 0.4 * (y2 - y1), y2]
            for foot_x, foot_y in person["feet"]:
                xs.append(foot_x)
                ys.append(foot_y)
        ball = choose_ball(info, entry, people)
        if ball is not None:
            xs.append(ball[0])
            ys.append(ball[1])
    if len(xs) < 2:
        return None
    width, height = size
    x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
    margin = 0.22 * max(x2 - x1, y2 - y1)
    x1, y1, x2, y2 = x1 - margin, y1 - margin, x2 + margin, y2 + margin
    # Take the tile aspect ratio, then refuse to zoom past a usable minimum.
    if (x2 - x1) / max(1e-6, y2 - y1) < aspect:
        wanted = (y2 - y1) * aspect
        centre = 0.5 * (x1 + x2)
        x1, x2 = centre - wanted / 2.0, centre + wanted / 2.0
    else:
        wanted = (x2 - x1) / aspect
        centre = 0.5 * (y1 + y2)
        y1, y2 = centre - wanted / 2.0, centre + wanted / 2.0
    if x2 - x1 < MIN_CROP_PX:
        centre = 0.5 * (x1 + x2)
        x1, x2 = centre - MIN_CROP_PX / 2.0, centre + MIN_CROP_PX / 2.0
    if y2 - y1 < MIN_CROP_PX / aspect:
        centre = 0.5 * (y1 + y2)
        half = MIN_CROP_PX / aspect / 2.0
        y1, y2 = centre - half, centre + half
    span_x, span_y = x2 - x1, y2 - y1
    x1 = max(0.0, min(x1, width - span_x))
    y1 = max(0.0, min(y1, height - span_y))
    return (int(x1), int(y1), int(min(width, x1 + span_x)),
            int(min(height, y1 + span_y)))


def lane_gt_text(annotation: dict, fps: float) -> str:
    """The gt lane: the annotation, with its own frame range and times."""
    start, end = annotation["start_sec"], annotation["end_sec"]
    if annotation.get("frames_stated") and annotation.get("start_frame") is not None:
        frames = "%d-%d" % (annotation["start_frame"], annotation.get("end_frame", -1))
    else:
        frames = "≈%d-%d" % (round(start * fps), round(end * fps))
    return ("gt#%d   frame %s   t=%.2f-%.2f s   claim %d   （窗口 %.2f-%.2f s）"
            % (annotation["index"], frames, start, end, annotation.get("count", 1) or 1,
               start, end))


def lane_pred_text(entry: dict) -> str:
    """The pred lane: the prediction this row is about, or the nearest one."""
    if entry["type"] == "fp":
        return ("frame %d   t=%.3f s   人%d %s   d=%.1f/τ%.1f px   %d 帧接触   球置信度 %.2f   [%s]"
                % (entry["frame"], entry["time_sec"], entry["person_track_id"],
                   entry["foot"], entry["distance_px"], entry["threshold_px"] or 0.0,
                   int(entry["contact_frames"]), entry["ball_confidence"],
                   entry.get("reason", "")))
    distance = entry.get("nearest_event_distance", 0.0)
    if distance > 1e-9:
        return "跨度内没有任何输出事件（最近的一个差 %.2f s）" % distance
    owner = entry.get("nearest_event_owner")
    return ("frame %d   t=%.3f s   偏 %.2f s   %s"
            % (entry.get("nearest_event_frame", 0), entry.get("nearest_event_sec", 0.0),
               distance, ("已配给 gt#%d" % owner) if owner else "未配对"))


# -------------------------------------------------------------- rendering --

def wrap(text: str, font, width: int) -> list[str]:
    """Break a line to `width` pixels; Chinese has no spaces to break on."""
    lines, current = [], ""
    for character in text:
        if font.getlength(current + character) > width and current:
            lines.append(current)
            current = character
        else:
            current += character
    if current:
        lines.append(current)
    return lines


def draw_lines(draw, lines, x: int, y: int, width: int, title_font, body_font) -> int:
    """Draw (text, colour, font) triples top down, wrapped; returns the height."""
    used = y
    for text, colour, font in lines:
        for piece in wrap(text, font, width):
            draw.text((x, used), piece, font=font, fill=colour)
            used += int(font.size * 1.35)
    return used


def frame_offsets(span_sec: float, count: int, fps: float) -> list[int]:
    """`count` frame indices spread over +-span, symmetric around zero."""
    count = max(2, count)
    step = 2.0 * span_sec * fps / (count - 1)
    if count % 2 == 0:                      # an even count cannot contain zero
        return [int(round((index + 0.5 - count / 2.0) * step)) for index in range(count)]
    return [int(round(-span_sec * fps + index * step)) for index in range(count)]


def contact_sheet(cap, frames: list[int], lines, marker: int, columns: int,
                  tile: tuple[int, int], fps: float,
                  rect: tuple[int, int, int, int] | None) -> Image.Image:
    width, height = tile
    title_font, body_font = load_font(21), load_font(18)
    caption_font = load_font(16)
    total_width = width * columns
    measured = 0
    for text, _, font in lines:
        measured += int(font.size * 1.35) * len(wrap(text, font, total_width - 30))
    rows = (len(frames) + columns - 1) // columns
    header = 14 + measured
    canvas = Image.new("RGB", (total_width, header + height * rows), (16, 16, 18))
    draw = ImageDraw.Draw(canvas)
    draw_lines(draw, lines, 14, 8, total_width - 30, title_font, body_font)
    for slot, frame in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        if not ok:
            continue
        if rect is not None:
            x1, y1, x2, y2 = rect
            cropped = image[y1:y2, x1:x2]
            if cropped.size:
                image = cropped
        interpolation = cv2.INTER_AREA if image.shape[1] > width else cv2.INTER_LINEAR
        image = cv2.resize(image, (width, height), interpolation=interpolation)
        x = (slot % columns) * width
        top = header + (slot // columns) * height
        canvas.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (x, top))
        highlight = frame == marker
        draw.rectangle([x, top, x + width - 1, top + height - 1],
                       outline=(255, 96, 96) if highlight else (70, 70, 74),
                       width=3 if highlight else 1)
        draw.text((x + 8, top + 8), "第 %d 帧  %.3fs" % (frame, frame / fps),
                  font=caption_font, fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0))
    return canvas


def banner(image: np.ndarray, caption: str, colour: tuple[int, int, int]) -> np.ndarray:
    """Burn a wrapped Chinese caption onto a BGR frame (OpenCV cannot draw it)."""
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(rgb)
    font = load_font(26)
    pieces = wrap(caption, font, image.shape[1] - 24)[:2]
    band = 10 + len(pieces) * 34
    draw.rectangle([0, 0, image.shape[1], band], fill=(12, 12, 14))
    for index, piece in enumerate(pieces):
        draw.text((12, 6 + index * 34), piece, font=font, fill=(255, 255, 255))
    draw.rectangle([2, 2, image.shape[1] - 3, image.shape[0] - 3], outline=colour, width=4)
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def _timeline(draw, width: int, top: int, height: int, fps: float,
              signal: dict[int, str], current_frame: int,
              spans: list[tuple[float, float]],
              events: list[float], half_sec: float) -> None:
    """The strip along the bottom: per frame contact state, span, prediction.

    Same vocabulary as the app's own strip - a frame is green when the ball is
    inside the touch threshold, amber while it is inside the hysteresis band -
    with two things added that a review needs: where the annotation says the
    touch is, and where the detector says it is.
    """
    left_frame = max(0, int(round(current_frame - half_sec * fps)))
    right_frame = max(left_frame + 1, int(round(current_frame + half_sec * fps)))
    span_frames = float(right_frame - left_frame)
    for frame in range(left_frame, right_frame):
        level = signal.get(frame)
        colour = ((46, 150, 70) if level == "contact"
                  else (150, 122, 44) if level == "band" else (38, 38, 44))
        x0 = int((frame - left_frame) / span_frames * width)
        x1 = int((frame + 1 - left_frame) / span_frames * width)
        draw.rectangle([x0, top, max(x1, x0 + 1), top + height], fill=colour)
    for start_sec, end_sec in spans:
        x0 = int((start_sec * fps - left_frame) / span_frames * width)
        x1 = int((end_sec * fps - left_frame) / span_frames * width)
        if x1 < 0 or x0 > width:
            continue
        draw.rectangle([max(0, x0), top + height - 4, min(width, max(x1, x0 + 2)),
                        top + height], fill=(255, 196, 64))
    for stamp in events:
        x = int((stamp * fps - left_frame) / span_frames * width)
        if 0 <= x <= width:
            draw.rectangle([max(0, x - 1), top, min(width, x + 2), top + height],
                           fill=(255, 96, 96))
    x = int((current_frame - left_frame) / span_frames * width)
    draw.rectangle([max(0, x - 1), top - 3, min(width, x + 2), top + height + 1],
                   fill=(255, 255, 255))


def annotate_video_frame(image: np.ndarray, entry: dict, current_frame: int,
                         fps: float, signal: dict[int, str],
                         spans: list[tuple[float, float]],
                         event_times: list[float]) -> np.ndarray:
    """One clip frame in the review layout.

    Top right: the annotation, whether it was found, what the detector reported,
    then the reason and the fix in Chinese. Bottom: the signal timeline with the
    annotation span and the prediction marked, and the frame number and time.
    """
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
    draw = ImageDraw.Draw(rgb, "RGBA")
    width, height = rgb.size
    margin = 14
    title_font, body_font = load_font(21), load_font(19)
    strip_font = load_font(18)

    strip_height = 18
    strip_top = height - strip_height - 34
    _timeline(draw, width, strip_top, strip_height, fps, signal, current_frame,
              spans, event_times, max(1.0, abs(entry.get("video_span", 1.5))))

    lines = [(entry.get("gt_line", ""), entry["colour"], title_font),
             (entry.get("det_line", ""), (238, 238, 242), body_font),
             ("原因：" + entry.get("note_cn", ""), (238, 238, 242), body_font),
             ("怎么修：" + entry.get("advice_cn", ""), (255, 214, 102), body_font)]
    wrapped: list[tuple[str, tuple, object]] = []
    for label, colour, font in lines:
        if not label:
            continue
        for piece in wrap(label, font, int(width * 0.74)):
            wrapped.append((piece, colour, font))
    block_width = min(max(int(font.getlength(piece)) for piece, _, font in wrapped) + 2 * 12,
                      int(width * 0.78))
    block_height = min(12 + sum(int(font.size * 1.32) for _, _, font in wrapped),
                       int(height * 0.55))
    x0 = width - block_width - 6
    draw.rectangle([x0, 6, width - 6, 6 + block_height], fill=(10, 10, 12, 224))
    y = 12
    for piece, colour, font in wrapped:
        if y + int(font.size * 1.32) > 6 + block_height - 4:
            break
        draw.text((x0 + 10, y), piece, font=font, fill=colour)
        y += int(font.size * 1.32)

    stamp = "第 %d 帧   t=%.3f s" % (current_frame, current_frame / fps)
    draw.rectangle([6, strip_top - 26, 6 + int(strip_font.getlength(stamp)) + 20,
                    strip_top - 2], fill=(10, 10, 12, 210))
    draw.text((14, strip_top - 24), stamp, font=strip_font, fill=(200, 255, 210))
    legend = "绿=接触内  黄=迟滞带  橙=标注跨度  红=检测  白=当前帧"
    legend_x = width - int(strip_font.getlength(legend)) - 14
    draw.rectangle([legend_x - 8, strip_top - 26, width - 6, strip_top - 2],
                   fill=(10, 10, 12, 210))
    draw.text((legend_x, strip_top - 24), legend, font=strip_font, fill=(186, 186, 192))
    draw.rectangle([2, 2, width - 3, height - 3], outline=entry["colour"], width=3)
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def load_signal_levels(path: Path) -> dict[int, str]:
    """`contact_signal.csv` reduced to frame -> level, for the timeline strip."""
    levels: dict[int, str] = {}
    if not path.is_file():
        return levels
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                levels[int(row["frame_index"])] = row.get("level", "none")
            except (TypeError, ValueError, KeyError):
                continue
    return levels


def render(args, output_dir: Path, rows: list[dict], fps: float) -> int:
    video = output_dir / "rendered.mp4"
    if not video.is_file():
        print("[pr-review] 没有 rendered.mp4：只写索引，跳过图片")
        return 0
    review = output_dir / "pr_review"
    review.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    offsets = frame_offsets(args.sheet_span, args.sheet_frames, fps)
    wanted: set[int] = set()
    for entry in rows:
        centre = int(entry["frame"])
        wanted.update(max(0, min(total - 1, centre + offset)) for offset in offsets)
    observations = (collect_observations(output_dir / "observations.jsonl", wanted)
                    if args.zoom else {})
    signal = load_signal_levels(output_dir / "contact_signal.csv")
    print("[pr-review] 自动放大：%s（观测切片 %d 帧）"
          % ("开" if observations else "关", len(observations)))
    writer = None
    annotated = None
    # The frames are written at the rate they were shot: writing 60 fps material
    # into a 30 fps container plays it at half speed, which is not what a review
    # clip is for.
    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
    if args.video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        size = (args.video_width, args.video_height)
        writer = cv2.VideoWriter(str(review / "pr_errors.mp4"), fourcc, source_fps, size)
        # The annotated cut: what the error is, why, and pred over gt, on every
        # frame - so a reviewer can read it while the clip plays.
        annotated = cv2.VideoWriter(str(review / "pr_errors_annotated.mp4"),
                                    fourcc, source_fps, size)
    for entry in rows:
        center = int(entry["frame"])
        frames = [max(0, min(total - 1, center + offset)) for offset in offsets]
        rect = crop_rect([observations.get(frame) for frame in frames], entry,
                         (frame_width, frame_height),
                         args.tile_width / float(args.tile_height)) if observations else None
        sheet = contact_sheet(cap, frames, entry["sheet_lines"], center,
                              args.sheet_columns,
                              (args.tile_width, args.tile_height), fps, rect)
        sheet.save(review / entry["file"])
        if writer is not None:
            span = int(round(args.video_span * fps))
            last = None
            for frame in range(max(0, center - span), min(total - 1, center + span)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                if not ok:
                    continue
                image = cv2.resize(image, (args.video_width, args.video_height),
                                   interpolation=cv2.INTER_AREA)
                writer.write(banner(image, entry["caption"], entry["colour"]))
                last = annotate_video_frame(
                    image, entry, frame, fps, signal,
                    entry.get("gt_spans", []), entry.get("event_times", []))
                annotated.write(last)
            if last is not None:
                # Freeze only when asked: the note can be read while paused, and
                # a still tail makes the clip longer than the moment it shows.
                for _ in range(max(0, args.hold_frames)):
                    annotated.write(last)
            card = cv2.resize(cv2.cvtColor(np.array(sheet), cv2.COLOR_RGB2BGR),
                              (args.video_width, args.video_height))
            for _ in range(24):
                writer.write(card)
    cap.release()
    if writer is not None:
        writer.release()
    if annotated is not None:
        annotated.release()
    return len(rows)


# --------------------------------------------------------------- playback --

class PlaybackWriter:
    """One `write`/`release` for both encoders: an ffmpeg pipe or cv2.

    ffmpeg (x264) is preferred when present: the same codec the app renders with,
    a fifth of the size of mp4v at this resolution and rate. `--playback-crf 0`
    forces the cv2 path, and a machine without ffmpeg falls back to it silently.
    """

    def __init__(self, path: Path, size: tuple[int, int], fps: float, crf: float) -> None:
        self.path = path
        self.size = size
        self.fps = fps
        self.process: "subprocess.Popen | None" = None
        self.writer = None
        ffmpeg = shutil.which("ffmpeg") if crf > 0 else None
        if ffmpeg:
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                       "-f", "rawvideo", "-pix_fmt", "bgr24",
                       "-s", "%dx%d" % size, "-r", "%.6f" % fps, "-i", "-",
                       "-an", "-c:v", "libx264", "-crf", "%.2f" % crf,
                       "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        else:
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          fps, size)

    @property
    def codec(self) -> str:
        return "x264 (ffmpeg)" if self.process is not None else "mp4v (cv2)"

    @property
    def ok(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        return self.writer is not None and self.writer.isOpened()

    def write(self, image: "np.ndarray") -> None:
        if self.process is not None:
            self.process.stdin.write(image.tobytes())
        else:
            self.writer.write(image)

    def release(self) -> None:
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
        elif self.writer is not None:
            self.writer.release()


def render_playback(args, output_dir: Path, events: list[dict], annotations: list[dict],
                    no_touch: list[dict], event_to_gt: dict[int, int],
                    gt_to_event: dict[int, int], fps: float, report: dict,
                    signal: "ContactSignal", tolerance: float) -> int:
    """The whole recording, real time, with the two timelines and the note.

    No clips and no slow motion: one source frame in, one output frame out, at
    the source rate, so pausing anywhere leaves the reviewer on the moment the
    note describes. The marks and the notes are computed once - a frame only
    pastes the timelines and moves the playhead.

    The mark counts are checked against `touch_report.json` before encoding: a
    video whose colours disagree with the report would be worse than no video.
    """
    source = (Path(args.playback_source) if args.playback_source
              else output_dir / "rendered.mp4")
    if not source.is_file():
        print("[pr-review] 没有 %s：跳过整段播放视频" % source)
        return 0
    review = output_dir / "pr_review"
    review.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = args.playback_width or frame_width
    out_height = args.playback_height or frame_height
    stride = max(1, args.stride)
    limit = args.max_frames if args.max_frames > 0 else total

    pred_marks, gt_marks = playback_marks(events, annotations, event_to_gt,
                                          gt_to_event, total, source_fps)
    pred_tp = sum(1 for _, _, kind in pred_marks if kind == "tp")
    gt_tp = sum(1 for _, _, kind in gt_marks if kind == "tp")
    print("[pr-review] 播放视频标记：pred TP %d / FP %d ｜ gt TP %d / FN %d"
          % (pred_tp, len(pred_marks) - pred_tp, gt_tp, len(gt_marks) - gt_tp))
    expected = report.get("primary_count_aware") or {}
    if expected:
        mismatched = (pred_tp != expected["true_positives"]
                      or len(pred_marks) - pred_tp != expected["false_positives"]
                      or gt_tp != expected["true_positives"]
                      or len(gt_marks) - gt_tp != expected["false_negatives"])
        assert not mismatched, (
            "播放视频的标记与 touch_report.json 不一致：pred TP/FP %d/%d、gt TP/FN %d/%d "
            "，报告是 TP%d FP%d FN%d —— 先查 tolerance_view 的口径"
            % (pred_tp, len(pred_marks) - pred_tp, gt_tp, len(gt_marks) - gt_tp,
               expected["true_positives"], expected["false_positives"],
               expected["false_negatives"]))

    lanes, lane_left, lane_track = build_lanes_image(
        (out_width, out_height), total, pred_marks, gt_marks, source_fps)
    lanes_top = out_height - lanes.height
    notes = playback_notes(events, annotations, no_touch, event_to_gt, gt_to_event,
                           signal, args, tolerance, total, source_fps)
    per_frame = lane_track / float(max(1, total))

    target = review / "touch_playback.mp4"
    writer = PlaybackWriter(target, (out_width, out_height), source_fps / stride,
                            args.playback_crf)
    if not writer.ok:
        print("[pr-review] 无法写入 %s" % target, file=sys.stderr)
        return 0
    if writer.codec.startswith("mp4v"):
        print("[pr-review] 播放视频编码：mp4v（机器上没有 ffmpeg CLI，OpenCV 也没有 "
              "H.264 编码器）—— 整段 720p60 约 400 MB，要更小可加 "
              "--stride 2（~200 MB）或 --playback-width 960 --playback-height 540（~230 MB）")
    else:
        print("[pr-review] 播放视频编码：%s" % writer.codec)
    stamp_font = load_font(18)
    written = 0
    frame_index = -1
    while written < limit:
        ok, image = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index >= limit:
            break
        if frame_index % stride:
            continue
        if (out_width, out_height) != (frame_width, frame_height):
            image = cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_AREA)
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
        rgb.paste(lanes, (0, lanes_top))
        draw = ImageDraw.Draw(rgb)

        x = lane_left + int(frame_index * per_frame)
        draw.rectangle([x, lanes_top, x + 1, out_height - 1], fill=(255, 255, 255))

        stamp = "第 %d 帧   t=%.3f s" % (frame_index, frame_index / source_fps)
        plate = [6, lanes_top - 30, 6 + int(stamp_font.getlength(stamp)) + 20,
                 lanes_top - 4]
        draw.rectangle(plate, fill=(10, 10, 12, 214))
        draw.text((14, lanes_top - 28), stamp, font=stamp_font, fill=(200, 255, 210))

        entries = notes.get(frame_index)
        if entries:
            entries.sort(key=lambda item: abs(item[0] - frame_index))
            lines: list = []
            for _, block in entries[:2]:
                lines.extend(block)
            draw_playback_note(draw, (out_width, out_height), lines)
        writer.write(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
        written += 1
        if written % 2000 == 0:
            print("[pr-review] 播放视频：%d/%d 帧" % (written, min(total, limit)))
    cap.release()
    writer.release()
    print("[pr-review] 播放视频：%s（%d 帧 / %.1f s，源 %s）"
          % (target, written, written * stride / source_fps, source))
    return written
#: The vocabulary of the two timelines: green = the pair matched (TP), red = a
#: prediction no annotation supports (FP), yellow = an annotation no prediction
#: filled (FN).
MARK_COLOURS = {"tp": (60, 200, 90), "fp": (255, 80, 80), "fn": (255, 196, 64)}
LANE_BG = (16, 16, 20)
LANE_TRACK = (34, 34, 40)
LANE_LABEL = (236, 236, 240)
TICK_COLOUR = (132, 132, 140)
LANE_ROW = 26
LANE_GAP = 3
TICK_ROW = 16


def playback_marks(events: list[dict], annotations: list[dict],
                   event_to_gt: dict[int, int], gt_to_event: dict[int, int],
                   total_frames: int, fps: float) -> tuple[list, list]:
    """One mark per prediction and per annotation: (start, end, verdict).

    A prediction spans from the first contact frame of its event to the frame it
    reports (the file does not carry the end of the contact, so the mark stops
    at the reported moment - it is a position marker, not a duration claim).
    An annotation spans exactly what it writes, `[S-1, S)` for a whole second.
    """
    def clamp(frame: float) -> int:
        return max(0, min(total_frames - 1, int(frame)))

    pred: list[tuple[int, int, str]] = []
    for index, event in enumerate(events):
        start = clamp(event.get("first_frame_index", event["frame_index"]))
        end = clamp(event["frame_index"])
        if end < start:
            start, end = end, start
        pred.append((start, end, "tp" if index in event_to_gt else "fp"))

    gt: list[tuple[int, int, str]] = []
    for row, annotation in enumerate(annotations):
        if annotation.get("frames_stated") and annotation.get("start_frame") is not None:
            start = clamp(annotation["start_frame"])
            end = clamp(annotation.get("end_frame", annotation["start_frame"]))
        else:
            start = clamp(round(annotation["start_sec"] * fps))
            end = clamp(round(annotation["end_sec"] * fps))
        if end < start:
            start, end = end, start
        gt.append((start, end, "tp" if row in gt_to_event else "fn"))
    return pred, gt


def build_lanes_image(size: tuple[int, int], total_frames: int,
                      pred_marks: list, gt_marks: list,
                      fps: float, tick_seconds: float = 30.0) -> tuple:
    """Both timelines and the time scale, drawn once for the whole video.

    Returns `(image, left, track)`: the track geometry comes back out because the
    playhead of every frame has to be drawn at the same positions.
    """
    width = size[0]
    font = load_font(17)
    height = 2 * LANE_ROW + LANE_GAP + TICK_ROW
    lanes = Image.new("RGB", (width, height), LANE_BG)
    draw = ImageDraw.Draw(lanes)
    left = int(font.getlength("pred")) + 20
    track = max(1, width - left - 14)
    per_frame = track / float(max(1, total_frames))

    for row, (label, marks, label_colour) in enumerate((
            ("pred", pred_marks, MARK_COLOURS["fp"]),
            ("gt", gt_marks, MARK_COLOURS["fn"]))):
        top = row * (LANE_ROW + LANE_GAP)
        draw.text((10, top + 4), label, font=font, fill=label_colour)
        draw.rectangle([left, top + 5, left + track, top + LANE_ROW - 5], fill=LANE_TRACK)
        for start, end, kind in marks:
            x0 = left + int(start * per_frame)
            x1 = left + int((end + 1) * per_frame)
            draw.rectangle([x0, top + 5, max(x1, x0 + 2), top + LANE_ROW - 5],
                           fill=MARK_COLOURS[kind])

    tick_top = 2 * LANE_ROW + LANE_GAP
    duration = total_frames / fps if fps > 0 else 0.0
    second = 0.0
    while second <= duration:
        x = left + int(second * fps * per_frame)
        if x <= left + track:
            draw.rectangle([x, tick_top, x + 1, tick_top + 3], fill=TICK_COLOUR)
            minute, rest = divmod(int(second), 60)
            draw.text((x + 2, tick_top + 2), "%d:%02d" % (minute, rest),
                      font=load_font(14), fill=TICK_COLOUR)
        second += tick_seconds
    return lanes, left, track


def playback_notes(events: list[dict], annotations: list[dict],
                   no_touch: list[dict], event_to_gt: dict[int, int],
                   gt_to_event: dict[int, int], signal: "ContactSignal",
                   args, tolerance: float, total_frames: int,
                   fps: float) -> dict[int, list]:
    """Frame -> the notes to show, each `(centre_frame, [(text, colour), ...])`.

    A prediction is answered by `false_positive_reason` (or, when it matched, by
    the annotation it filled) and an annotation by `false_negative_reason`, so
    the note on the video is the same Chinese sentence the sheets carry.
    """
    notes: dict[int, list] = {}
    margin = max(0, int(args.note_margin))

    def add(first: int, last: int, centre: int, lines: list) -> None:
        for frame in range(max(0, first), min(total_frames, last + 1)):
            notes.setdefault(frame, []).append((centre, lines))

    for index, event in enumerate(events):
        first = int(event.get("first_frame_index", event["frame_index"]))
        last = int(event["frame_index"])
        if last < first:
            first, last = last, first
        if index in event_to_gt:
            annotation = annotations[event_to_gt[index]]
            lines = [("命中：GT #%d（%.2f–%.2f s）由这次检测填上"
                      % (annotation["index"], annotation["start_sec"], annotation["end_sec"]),
                      MARK_COLOURS["tp"])]
            add(first - margin, last + margin, last, lines)
            continue
        reason = false_positive_reason(event, annotations, events, no_touch,
                                       gt_to_event, tolerance, args.adjacency,
                                       args.refractory_frames)
        lines = [("误检 FP（%s）  t=%.3f s  第 %d 帧"
                  % (REASON_CN.get(reason["kind"], reason["kind"]),
                     event["timestamp_sec"], event["frame_index"]), MARK_COLOURS["fp"]),
                 ("原因：%s" % reason["note_cn"], (240, 240, 244)),
                 ("怎么修：%s" % reason["advice_cn"], ADVICE_COLOUR)]
        add(first - margin, last + margin, last, lines)

    for row, annotation in enumerate(annotations):
        if annotation.get("frames_stated") and annotation.get("start_frame") is not None:
            first = int(annotation["start_frame"])
            last = int(annotation.get("end_frame", first))
        else:
            first = int(round(annotation["start_sec"] * fps))
            last = int(round(annotation["end_sec"] * fps))
        if last < first:
            first, last = last, first
        if row in gt_to_event:
            continue
        reason = false_negative_reason(annotation, annotations, events, event_to_gt,
                                       signal, tolerance, args.min_contact_frames,
                                       args.refractory_frames,
                                       args.person_refractory_frames)
        lines = [("漏检 FN（%s）  GT #%d  %.2f–%.2f s  claim %d"
                  % (REASON_CN.get(reason["kind"], reason["kind"]), annotation["index"],
                     annotation["start_sec"], annotation["end_sec"],
                     annotation.get("count", 1) or 1), MARK_COLOURS["fn"]),
                 ("原因：%s" % reason["note_cn"], (240, 240, 244)),
                 ("怎么修：%s" % reason["advice_cn"], ADVICE_COLOUR)]
        add(first - margin, last + margin, (first + last) // 2, lines)
    return notes


def draw_playback_note(draw, size: tuple[int, int], lines: list,
                       margin: int = 14) -> None:
    """The note at the top centre: what this moment is and what to do about it."""
    if not lines:
        return
    width = size[0]
    title_font, body_font = load_font(21), load_font(18)
    wrapped: list[tuple[str, tuple, object]] = []
    for index, (text, colour) in enumerate(lines[:4]):
        font = title_font if index == 0 else body_font
        for piece in wrap(text, font, int(width * 0.86)):
            wrapped.append((piece, colour, font))
    band = 10 + sum(int(font.size * 1.34) for _, _, font in wrapped) + 4
    draw.rectangle([0, 0, width, band], fill=(10, 10, 12, 226))
    y = 8
    for piece, colour, font in wrapped:
        draw.text(((width - font.getlength(piece)) / 2, y), piece, font=font, fill=colour)
        y += int(font.size * 1.34)


# ------------------------------------------------------------------ index --

def write_index(output_dir: Path, tolerance: float, fp_rows: list[dict],
                fn_rows: list[dict], summary: dict, raw_note: str,
                adjacency: float) -> None:
    review = output_dir / "pr_review"
    review.mkdir(exist_ok=True)
    fields = ["type", "time_sec", "frame", "gt_index", "reason", "note",
              "distance_px", "threshold_px", "contact_frames", "ball_confidence",
              "ball_tracked", "person_track_id", "foot", "verdict", "image"]
    with open(review / "index.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in fp_rows + fn_rows:
            writer.writerow({key: entry.get(key, "") for key in fields}
                            | {"image": entry["file"]})

    if tolerance <= 1e-9:
        title = "# 严格口径的 PR 错误清单（逐条渲染 + 原因）"
        rule = ("判据：**容差 0** —— 预测事件必须落在标注写的那段跨度里（整秒行 = `[S−1, S)`，"
                "粗跨度行 = 它写的宽度）才算命中；没落进去的是误检（FP），跨度里没有事件的是漏检（FN）。")
    else:
        title = "# 一对一 ±%.2f s 的 PR 错误清单（逐条渲染 + 原因）" % tolerance
        rule = ("判据：严格一对一（最优一配一，最大基数匹配）。预测与标注距离超过 "
                "%.2f s 就不配对，剩下的分别是误检（FP）与漏检（FN）。" % tolerance)
    lines = [title, "", rule, "",
             "| 数 | 值 |", "| --- | --- |",
             "| 预测事件 | %d |" % summary["events"],
             "| 标注 | %d |" % summary["annotations"],
             "| TP / FP / FN | %d / %d / %d |" % (summary["tp"], summary["fp"], summary["fn"]),
             "| precision / recall / F1 | %.3f / %.3f / %.3f |"
             % (summary["precision"], summary["recall"], summary["f1"]), ""]
    if raw_note:
        lines += ["> %s" % raw_note, ""]
    lines += ["## 误检 FP（%d 个）" % len(fp_rows), "",
              "| # | 时刻 | 帧 | 原因 | 说明与证据 | 怎么修 | 图 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for index, entry in enumerate(fp_rows, 1):
        evidence = ("d=%.1f≤τ=%.1f，%d 帧接触，球置信度 %.2f，%s"
                    % (entry["distance_px"], entry["threshold_px"] or 0.0,
                       int(entry["contact_frames"]), entry["ball_confidence"],
                       "跟踪球" if entry["ball_tracked"] else "候选球"))
        lines.append("| %d | %.2f s | %d | %s | %s | %s | `%s` |"
                     % (index, entry["time_sec"], entry["frame"], entry["kind_cn"],
                        "%s（%s）" % (entry["note_cn"], evidence),
                        entry["advice_cn"], entry["file"]))
    lines += ["", "## 漏检 FN（%d 个）" % len(fn_rows), "",
              "| # | 标注 | 时刻 | 原因 | 说明 | 原始流（关掉门控） | 怎么修 | 图 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for index, entry in enumerate(fn_rows, 1):
        lines.append("| %d | gt#%d | %.2f s | %s | %s | %s | %s | `%s` |"
                     % (index, entry["gt_index"], entry["time_sec"], entry["kind_cn"],
                        entry["note_cn"], entry.get("raw_note", ""),
                        entry["advice_cn"], entry["file"]))
    lines += ["", "原因代号：", "",
              "- FP `slot_taken`：落在标注窗口内，但该标注已被更近的检测配走",
              "- FP `adjacent`：贴着标注窗口（邻接半径 %.1f s 内），窗口里已无名额" % adjacency,
              "- FP `inside_no_touch`：落在标注者确认无触球的区间",
              "- FP `unannotated`：附近 %.1f s 内没有任何标注（无人支持）" % adjacency,
              "- FN `contact_but_no_event`：窗内球脚确实贴到阈值内，但事件被判太短或被合并",
              "- FN `never_near`：窗内球从未近到阈值（对照最近距离与 τ）",
              "- FN `no_ball` / `no_feet`：窗内没有可用球 / 没有可用脚点", "",
              "", "怎么修（按原因）：", "",
              "- FP `slot_taken`：跨度里检出 N 次接触而该行只 claim 1 → 确认是两次就把 count 改成 N；只一次则是重复上报，查不应期",
              "- FP `adjacent`：同一触球的时刻差 → 看视频确认后改写这条标注（或改到相邻那一秒）",
              "- FP `inside_no_touch`：标注说无触球而球脚贴上了 → 看视频，删/收窄 no_touch 或记为反例",
              "- FP `unannotated`：附近没标注而球脚贴上了 → 看视频，缺标注就补一行",
              "- FN `contact_but_no_event`：接触帧 < min_contact_frames → 按视频把它调低；最近的输出在 ≤0.5 s 内 → 很可能是不应期吞了，用 `--refractory_frames 0 --person_refractory_frames 0` 重跑这一秒核对",
              "- FN `never_near`：球没到脚边 → 先分辨「标注错」还是「球/脚被追丢」",
              "- FN `no_ball` / `no_feet`：这一秒不可判读 → 标 `low_quality_data=yes`", "",
              "图上文字是中文；若机器没有中文字体会自动退回英文并提示安装："
              "`apt-get install -y fonts-wqy-microhei`（也可用 `%s` 指定）" % FONT_ENV, ""]
    (review / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------- main --

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="output/touch_eval_moving_pass")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="one-to-one tolerance in seconds (default 0.5)")
    parser.add_argument("--adjacency", type=float, default=1.5,
                        help="how far an annotation still counts as 'next to it'")
    parser.add_argument("--type", choices=("all", "fp", "fn"), default="all")
    parser.add_argument("--sheet-frames", type=int, default=6)
    parser.add_argument("--sheet-columns", type=int, default=3)
    parser.add_argument("--sheet-span", type=float, default=0.2,
                        help="seconds shown on each side of the error")
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=480,
                        help="4:3 by default: a cropped player needs the height")
    parser.add_argument("--video", action="store_true",
                        help="also write pr_review/pr_errors.mp4")
    parser.add_argument("--playback", action="store_true",
                        help="also write pr_review/touch_playback.mp4: the whole "
                             "recording at real speed, with the pred/gt timelines")
    parser.add_argument("--playback-only", action="store_true",
                        help="with --playback: leave the sheets and the clip videos "
                             "alone and only write the playback")
    parser.add_argument("--playback-source", default="",
                        help="video to draw the playback on (default rendered.mp4)")
    parser.add_argument("--playback-width", type=int, default=0,
                        help="playback width in pixels (0 = the source width)")
    parser.add_argument("--playback-height", type=int, default=0,
                        help="playback height in pixels (0 = the source height)")
    parser.add_argument("--playback-crf", type=float, default=20.0,
                        help="playback quality when ffmpeg encodes it (0 = use the "
                             "in-process mp4v writer instead)")
    parser.add_argument("--note-margin", type=int, default=12,
                        help="frames a note lingers before and after its error")
    parser.add_argument("--stride", type=int, default=1,
                        help="playback: keep every Nth frame (the rate follows)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="playback: stop after N source frames (0 = all)")
    parser.add_argument("--hold-frames", type=int, default=0,
                        help="clip videos: frozen frames after each clip (0 = none)")
    parser.add_argument("--video-span", type=float, default=0.5)
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    parser.add_argument("--no-zoom", dest="zoom", action="store_false",
                        help="show the whole frame instead of cropping around the "
                             "player the event names and the ball")
    parser.add_argument("--no-raw-check", action="store_true",
                        help="skip re-running the detector with the gates open")
    parser.add_argument("--touch-eval", default="uv run --project examples/touch_eval/py touch-eval")
    parser.add_argument("--annotations", default="data/移动传球.touches.csv")
    parser.add_argument("--limit", type=int, default=0,
                        help="render only the first N errors per side (0 = all)")
    parser.add_argument("--min_contact_frames", type=int, default=2,
                        help="detector knob the advice quotes (default 2)")
    parser.add_argument("--refractory_frames", type=int, default=30,
                        help="same-foot refractory the advice quotes (default 30)")
    parser.add_argument("--person_refractory_frames", type=int, default=30,
                        help="same-person refractory the advice quotes (default 30)")
    args = parser.parse_args()

    font_path = cjk_font_path()
    print("[pr-review] 中文字体：%s" % (font_path or "未找到（图上将退回英文）"))
    output_dir = Path(args.output_dir)
    with open(output_dir / "touch_report.json", encoding="utf-8") as handle:
        report = json.load(handle)
    with open(output_dir / "ground_truth.json", encoding="utf-8") as handle:
        ground_truth = json.load(handle)
    events = report["events"]
    annotations = ground_truth["events"]
    no_touch = ground_truth.get("no_touch", [])
    fps = float(report.get("fps") or 60.0)
    tolerance = args.tolerance

    event_to_gt, gt_to_event = tolerance_view(events, annotations, tolerance)

    tp = len(event_to_gt)
    precision = tp / len(events) if events else 0.0
    recall = tp / len(annotations) if annotations else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {"events": len(events), "annotations": len(annotations), "tp": tp,
               "fp": len(events) - tp, "fn": len(annotations) - tp,
               "precision": precision, "recall": recall, "f1": f1}
    print("[pr-review] ±%.2f s：TP=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f"
          % (tolerance, tp, summary["fp"], summary["fn"], precision, recall, f1))
    for score in report.get("tolerance_sweep", []):
        if abs(score["tolerance_sec"] - tolerance) < 1e-9:
            if score["true_positives"] != tp:
                print("[pr-review] 警告：这里算出 %d 个 TP，报告写的是 %d"
                      % (tp, score["true_positives"]), file=sys.stderr)
            else:
                print("[pr-review] 与 touch_report.json 一致（TP=%d）" % tp)

    signal = ContactSignal(output_dir / "contact_signal.csv")
    raw_events: list[dict] = []
    raw_note = ""
    if not args.no_raw_check:
        found, raw_note = raw_stream_check(args, output_dir)
        raw_events = found or []
        print("[pr-review] %s" % raw_note)

    fp_rows = []
    if args.type in ("all", "fp"):
        for index, event in enumerate(events):
            if index in event_to_gt:
                continue
            reason = false_positive_reason(event, annotations, events, no_touch,
                                           gt_to_event, tolerance, args.adjacency,
                                           args.refractory_frames)
            entry = dict(event)
            entry.update(reason)
            entry.update({
                "type": "fp", "time_sec": event["timestamp_sec"],
                "tolerance_sec": tolerance,
                "frame": event["frame_index"],
                "kind_cn": REASON_CN.get(reason["kind"], reason["kind"]),
                "reason": reason["kind"],
                "file": "fp_%s_%s.png" % (("%07.2f" % event["timestamp_sec"]).replace(".", "_"),
                                          reason["kind"]),
                "colour": COLOURS["fp"],
                "caption": "误检 FP（%s）t=%.3f s：%s"
                           % (reason["kind"], event["timestamp_sec"], reason["note_cn"]),
            })
            nearest = annotations[reason["nearest_row"]]
            entry["lane_gt"] = lane_gt_text(nearest, fps)
            entry["lane_pred"] = lane_pred_text(entry)
            entry["video_span"] = args.video_span + 0.5
            entry["gt_spans"] = [(nearest["start_sec"], nearest["end_sec"])]
            entry["event_times"] = [event["timestamp_sec"]]
            entry["gt_line"] = ("GT #%d   %d 次触球   命中 %d/%d"
                                % (nearest["index"], nearest.get("count", 1) or 1,
                                   1 if reason.get("nearest_row") is not None
                                   and gt_to_event.get(reason["nearest_row"]) is not None
                                   else 0,
                                   nearest.get("count", 1) or 1))
            entry["det_line"] = ("检测 @%.3f s 第 %d 帧   %s   → 最近 GT #%d  人%d %s  d=%.1f/τ%.1f px"
                                 % (event["timestamp_sec"], event["frame_index"],
                                    REASON_CN.get(reason["kind"], reason["kind"]),
                                    nearest["index"], event["person_track_id"],
                                    event["foot"], event["distance_px"],
                                    event["threshold_px"] or 0.0))
            entry["sheet_lines"] = [
                ("误检 FP   t=%.3fs   第 %d 帧   人 %d %s   [%s]"
                 % (event["timestamp_sec"], event["frame_index"],
                    event["person_track_id"], event["foot"], reason["kind"]),
                 COLOURS["fp"], load_font(21)),
                ("原因：%s" % reason["note_cn"], (240, 240, 240), load_font(18)),
                ("证据：d=%.1f px ≤ τ=%.1f px，%d 帧接触，球置信度 %.2f，%s"
                 % (event["distance_px"], event["threshold_px"] or 0.0,
                    event["contact_frames"], event["ball_confidence"],
                    "跟踪球" if event["ball_tracked"] else "候选球"),
                 GREY, load_font(18)),
                ("怎么修：%s" % reason["advice_cn"], ADVICE_COLOUR, load_font(18)),
            ]
            fp_rows.append(entry)

    fn_rows = []
    if args.type in ("all", "fn"):
        for row, annotation in enumerate(annotations):
            if row in gt_to_event:
                continue
            reason = false_negative_reason(annotation, annotations, events,
                                           event_to_gt, signal, tolerance,
                                           args.min_contact_frames,
                                           args.refractory_frames,
                                           args.person_refractory_frames)
            entry = dict(reason)
            entry.update({"type": "fn", "time_sec": annotation["start_sec"],
                          "tolerance_sec": tolerance,
                          "frame": annotation.get("start_frame", 0) or 0,
                          "kind_cn": REASON_CN.get(reason["kind"], reason["kind"]),
                          "reason": reason["kind"],
                          "distance_px": reason["nearest_px"]
                          if reason["nearest_px"] is not None else "",
                          "threshold_px": reason["nearest_tau_px"] or "",
                          "contact_frames": "",
                          "ball_confidence": reason["max_ball_confidence"]
                          if reason["max_ball_confidence"] is not None else "",
                          "ball_tracked": "", "person_track_id": "", "foot": ""})
            if raw_events:
                nearest = min(distance_to(annotation, event["timestamp_sec"])
                              for event in raw_events)
                entry["raw_note"] = (
                    "原始流在该跨度内有事件 → 这个事件被去重门控压掉了"
                    "（放开门控要付精度代价，见 §4 消融）"
                ) if nearest <= tolerance else \
                    "原始流最近也差 %.2f s → 不是被门控压掉，检测本身没在该时刻出事件" % nearest
            else:
                entry["raw_note"] = ""
            entry["file"] = "fn_gt%03d_%s_%s.png" % (
                annotation["index"], ("%07.2f" % annotation["start_sec"]).replace(".", "_"),
                reason["kind"])
            entry["lane_gt"] = lane_gt_text(annotation, fps)
            entry["lane_pred"] = lane_pred_text(entry)
            entry["video_span"] = args.video_span + 0.5
            entry["gt_spans"] = [(annotation["start_sec"], annotation["end_sec"])]
            entry["event_times"] = ([reason["nearest_event_sec"]]
                                    if reason.get("nearest_event_sec") is not None else [])
            entry["gt_line"] = ("GT #%d   %d 次触球   命中 0/%d"
                                % (annotation["index"], annotation.get("count", 1) or 1,
                                   annotation.get("count", 1) or 1))
            entry["det_line"] = ("检测 %s   → GT #%d @ %.2f s 窗内最近 %.1f px / τ %.1f px"
                                 % ("跨度内无输出" if reason["nearest_event_distance"] > 1e-9
                                    else "最近 @%.2f s 偏 %.2f s" % (
                                        reason["nearest_event_sec"],
                                        reason["nearest_event_distance"]),
                                    annotation["index"], annotation["start_sec"],
                                    reason["nearest_px"]
                                    if reason["nearest_px"] is not None else -1.0,
                                    reason["nearest_tau_px"] or 0.0))
            entry["colour"] = COLOURS["fn"]
            entry["caption"] = "漏检 FN（%s）gt#%d @ %.2f s：%s" % (
                reason["kind"], annotation["index"], annotation["start_sec"],
                reason["note_cn"])
            entry["sheet_lines"] = [
                ("漏检 FN   gt#%d @ %.2fs   窗口 %.2f–%.2fs   [%s]"
                 % (annotation["index"], annotation["start_sec"],
                    annotation["start_sec"] - tolerance,
                    annotation["end_sec"] + tolerance, reason["kind"]),
                 COLOURS["fn"], load_font(21)),
                ("原因：%s" % reason["note_cn"], (240, 240, 240), load_font(18)),
                ("窗口：最近 %.1f px / τ %.1f px，可用球 %s，可用脚 %s，球置信度峰值 %s ｜ %s"
                 % (reason["nearest_px"] if reason["nearest_px"] is not None else -1.0,
                    reason["nearest_tau_px"] or 0.0,
                    "是" if reason["usable_ball"] else "否",
                    "是" if reason["usable_feet"] else "否",
                    "%.2f" % reason["max_ball_confidence"]
                    if reason["max_ball_confidence"] is not None else "无",
                    entry["raw_note"] or "未做原始流核对"),
                 GREY, load_font(18)),
                ("怎么修：%s" % reason["advice_cn"], ADVICE_COLOUR, load_font(18)),
            ]
            fn_rows.append(entry)

    if args.playback_only:
        if not args.playback:
            print("[pr-review] --playback-only 要与 --playback 一起用；这里什么也没做",
                  file=sys.stderr)
            return 0
        # The sheets and the clip videos are left exactly as they are: the whole
        # recording is the deliverable of this run.
        render_playback(args, output_dir, events, annotations, no_touch,
                        event_to_gt, gt_to_event, fps, report, signal, tolerance)
        return 0
    if args.limit:
        fp_rows = fp_rows[:args.limit]
        fn_rows = fn_rows[:args.limit]
    write_index(output_dir, tolerance, fp_rows, fn_rows, summary, raw_note,
                args.adjacency)
    render(args, output_dir, fp_rows + fn_rows, fps)
    if args.playback:
        render_playback(args, output_dir, events, annotations, no_touch,
                        event_to_gt, gt_to_event, fps, report, signal, tolerance)
    print("[pr-review] %d 个 FP + %d 个 FN 已渲染到 %s"
          % (len(fp_rows), len(fn_rows), output_dir / "pr_review"))
    print("[pr-review] 索引：%s" % (output_dir / "pr_review" / "index.md"))
    if args.video:
        print("[pr-review] 视频：%s（原片 + 标题）与 %s（顶部原因、底部 pred/gt 两栏）"
              % (output_dir / "pr_review" / "pr_errors.mp4",
             output_dir / "pr_review" / "pr_errors_annotated.mp4"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
