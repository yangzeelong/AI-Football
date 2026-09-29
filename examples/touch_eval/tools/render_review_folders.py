#!/usr/bin/env python3
"""TP / FP / FN 各出一套接触片，分三个目录，并给一张按时间排的汇总表。

和 `render_pr_errors.py` 的区别在**口径**：那个工具用的是"行级一对一"（每个标注最多
配一个事件），于是多触球行（`count>1`）里的第 2、3 个事件会被算成误检。这里用
`touch_report.json` 对外那个**名额级**口径：一行写 N 次就有 N 个名额，事件配到哪一行
由报告里的 `gt_index` 决定 —— 也就是 TP=79 / FP=18 / FN=18 那一套数。

切帧、放大裁剪、中文字、时间线都复用 `render_pr_errors.py`，所以两类产物看起来是一套
东西。产物：

    <output-dir>/pr_review/tp/tp_gt001_0000_71.png          命中的事件
    <output-dir>/pr_review/fp/fp_00005_82_slot_taken.png    误检
    <output-dir>/pr_review/fn/fn_gt004_0006_00_....png      漏检（含"缺名额"的行）
    <output-dir>/pr_review/timeline.md                      按时间排的汇总表
    <output-dir>/pr_review/timeline.csv                     同一张表的 CSV

渲染前会先拿报告自己的 TP/FP/FN 与分类结果对拍，不一致就直接报错退出，所以图不会和
打印出来的分数打架。

    uv run --project examples/touch_eval/py --extra tools \
        python3 examples/touch_eval/tools/render_review_folders.py \
        --output-dir output/touch_eval_moving_pass-update2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_pr_errors as R  # noqa: E402  同目录的绘制与归因代码
from touch_eval import cli as cli_module  # noqa: E402
from touch_eval import detect as detect_module  # noqa: E402
from touch_eval import gt as gt_module  # noqa: E402
from touch_eval import metrics as metrics_module  # noqa: E402
from touch_eval.params import DetectorParams  # noqa: E402

#: 报告里的判决 -> 这个工具的原因代号（两边的分类逻辑是同一套，只是命名不同）。
VERDICT_KIND = {
    "in_window_no_slot": "slot_taken",
    "adjacent": "adjacent",
    "inside_no_touch": "inside_no_touch",
    "unannotated": "unannotated",
}
#: 报告里的漏检原因 -> 这个工具的原因代号（报告 `failures[].reason` 的原文）。
FAILURE_KIND = {
    "contact frames present but no accepted event (too short or merged)": "contact_but_no_event",
    "ball and feet present but never closer than the threshold": "never_near",
    "no ball detected in the window": "no_ball",
    "ball detected but never admissible (extrapolated or low confidence)": "no_usable_ball",
    "no usable foot keypoints (missing or interpolated)": "no_feet",
}
#: 补上库里没有的代号，其余沿用 `render_pr_errors.REASON_CN`。
REASON_CN = {**R.REASON_CN, "no_usable_ball": "球在但都不可用（外推或低置信）"}
TP_COLOUR = (120, 220, 140)
LQ_COLOUR = (150, 190, 255)   # 低质量数据：蓝，和 tp/fp/fn 都区分得开
TYPES = ("tp", "fp", "fn")


def load_annotations(path: Path) -> tuple[list[dict], list[dict]]:
    """按行序读标注：`rows` 的下标就是报告里 `gt_index` 指的东西（见 `resolve_mapping`）。"""
    parsed = gt_module.parse_annotation_file(str(path))
    rows = [
        {"index": row.index, "start_sec": row.start_sec, "end_sec": row.end_sec,
         "count": row.count or 1, "low_quality": bool(row.low_quality),
         "note": row.note or ""}
        for row in parsed.events
    ]
    no_touch = [{"start_sec": span.start_sec, "end_sec": span.end_sec}
                for span in parsed.no_touch]
    return rows, no_touch


def inside(row: dict, stamp: float) -> bool:
    return row["start_sec"] <= stamp <= row["end_sec"]


def resolve_mapping(rows: list[dict], events: list[dict]) -> tuple[dict[int, int], dict[int, list[int]]]:
    """把报告里的 `gt_index` 归一到 `rows` 的下标，并对拍一次。

    严格口径下配上的事件必然落在该行写出的跨度里，所以"下标"和"行号"两种解释里只有
    一种能让全部配上的事件都落在窗口内 —— 用它钉住含义，顺便证明图没贴错标注。
    """
    by_id = {row["index"]: position for position, row in enumerate(rows)}
    assigned = [(index, event["gt_index"]) for index, event in enumerate(events)
                if event["gt_index"] >= 0]
    if not assigned:
        return {}, {}
    as_position = all(inside(rows[gt], events[index]["timestamp_sec"]) for index, gt in assigned)
    as_row_id = all(gt in by_id and inside(rows[by_id[gt]], events[index]["timestamp_sec"])
                    for index, gt in assigned)
    if as_position == as_row_id:
        raise SystemExit("无法判断 gt_index 的含义（两种解释都%s）"
                         % ("成立" if as_position else "不成立"))
    event_to_position: dict[int, int] = {}
    for index, gt in assigned:
        event_to_position[index] = gt if as_position else by_id[gt]
    position_to_event: dict[int, list[int]] = {}
    for index, position in event_to_position.items():
        position_to_event.setdefault(position, []).append(index)
    positions = "rows 下标" if as_position else "标注行号"
    matches = len(event_to_position)
    print("[folders] gt_index = %s（%d 个配上的事件全部落在对应窗口内）" % (positions, matches))
    return event_to_position, position_to_event


def fn_reason(frames: list, timestamps: list[float], annotation: dict,
              params, window_sec: float, events: list[dict],
              min_contact_frames: int) -> dict:
    """漏检原因：和 `evaluate_frames()` 用同一套判据（`scan_window` + 同一阈值函数）。

    `render_pr_errors.false_negative_reason` 走的是 `contact_signal.csv`，在个别窗口
    上会和评估器差出很远（实测 2 行：评估器 6.23/1.85 px，那份信号给 168/320 px），
    所以这里直接用评估器自己的扫描，保证图和 `touch_report.json` 的分组一致。
    """
    low = annotation["start_sec"] - window_sec
    high = annotation["end_sec"] + window_sec
    stats = metrics_module.scan_window(frames, timestamps, low, high)
    nearest_px = stats.nearest_usable_distance_px
    tau = None
    if not stats.has_detected_ball:
        kind = "no_ball"
        note = "窗口 %.2f-%.2f s 内没有检出任何球" % (low, high)
    elif not stats.has_usable_ball:
        kind = "no_usable_ball"
        note = ("窗口 %.2f-%.2f s 内有球，但都不可用（外推或置信度低），"
                "最近 %.2f px" % (low, high, stats.nearest_distance_px))
    elif not stats.has_usable_feet:
        kind = "no_feet"
        note = ("窗口 %.2f-%.2f s 内有可用球，但没有可用的脚部关键点"
                "（缺失或被插值），最近 %.2f px" % (low, high, stats.nearest_distance_px))
    else:
        tau = float(detect_module.contact_threshold_px(
            params, stats.nearest_frame_width, stats.nearest_frame_height,
            detect_module.F32(stats.nearest_person_height_px)))
        if nearest_px <= tau:
            kind = "contact_but_no_event"
            note = ("球脚其实贴进了阈值：窗口 %.2f-%.2f s 内最近 %.2f px ≤ τ %.2f px"
                    % (low, high, nearest_px, tau))
        else:
            kind = "never_near"
            note = ("窗口 %.2f-%.2f s 内球离最近脚点最近 %.2f px，始终没进阈值"
                    "（τ %.2f px）" % (low, high, nearest_px, tau))
    nearest_event = min((abs(event["timestamp_sec"] - annotation["start_sec"]), event)
                        for event in events)
    if kind == "contact_but_no_event":
        if nearest_event[0] <= 0.5:
            advice = ("最近的输出事件在 %.2f s（约 %d 帧）外，很可能被不应期吞了："
                      "用 --refractory_frames 0 --person_refractory_frames 0 重跑这一秒核对"
                      % (nearest_event[0], round(nearest_event[0] * 60.0)))
        else:
            advice = ("接触帧太少（min_contact_frames=%d）或被合并：看视频确认触球时刻，"
                      "再调 --min_contact_frames" % min_contact_frames)
    elif kind == "never_near":
        advice = ("球始终没到脚边：先看视频确认这一秒到底有没有触球 —— 没有就是标注错；"
                  "有则是球或脚在这一段被追丢了")
    else:
        advice = "这一秒无法判读：如实承认就给它标 low_quality_data=yes，别让它当漏检"
    return {"kind": kind, "note_cn": note, "advice_cn": advice,
            "nearest_px": nearest_px if nearest_px < float("inf") else None,
            "nearest_tau_px": tau, "max_ball_confidence": stats.max_ball_confidence,
            "window_sec": window_sec, "gt_index": annotation["index"]}


def tp_entry(event: dict, annotation: dict, fps: float) -> dict:
    delta = event["timestamp_sec"] - annotation["end_sec"]
    entry = dict(event)
    entry.update({
        "type": "tp", "time_sec": event["timestamp_sec"], "tolerance_sec": 0.0,
        "frame": event["frame_index"], "gt_index": annotation["index"],
        "count": annotation["count"], "low_quality": annotation["low_quality"],
        "kind": "hit", "kind_cn": "命中", "reason": "hit",
        "note_cn": "预测落在该标注写出的跨度内（Δ%+.2f s 相对跨度末尾）" % delta,
        "advice_cn": "无需修改",
        "file": "tp/tp_gt%03d_%07.2f.png" % (annotation["index"], event["timestamp_sec"]),
        "colour": TP_COLOUR,
        "caption": "命中 TP  t=%.3f s  gt#%d  %d 次触球"
                   % (event["timestamp_sec"], annotation["index"], annotation["count"]),
    })
    entry["lane_gt"] = R.lane_gt_text(annotation, fps)
    entry["lane_pred"] = R.lane_pred_text(entry)
    entry["gt_spans"] = [(annotation["start_sec"], annotation["end_sec"])]
    entry["event_times"] = [event["timestamp_sec"]]
    entry["gt_line"] = ("GT #%d   %d 次触球   %.2f-%.2f s"
                        % (annotation["index"], annotation["count"],
                           annotation["start_sec"], annotation["end_sec"]))
    entry["det_line"] = ("检测 @%.3f s 第 %d 帧   人%d %s   d=%.1f/τ%.1f px"
                         % (event["timestamp_sec"], event["frame_index"],
                            event["person_track_id"], event["foot"],
                            event["distance_px"], event["threshold_px"] or 0.0))
    entry["sheet_lines"] = [
        ("命中 TP   t=%.3fs   第 %d 帧   人 %d %s   gt#%d"
         % (event["timestamp_sec"], event["frame_index"], event["person_track_id"],
            event["foot"], annotation["index"]), TP_COLOUR, R.load_font(21)),
        ("标注：gt#%d  %.2f-%.2f s  %d 次触球%s"
         % (annotation["index"], annotation["start_sec"], annotation["end_sec"],
            annotation["count"], "（低质量数据）" if annotation["low_quality"] else ""),
         (240, 240, 240), R.load_font(18)),
        ("偏离跨度末尾 %+.2f s   证据：d=%.1f px ≤ τ=%.1f px，%d 帧接触，球置信度 %.2f，%s"
         % (delta, event["distance_px"], event["threshold_px"] or 0.0,
            event["contact_frames"], event["ball_confidence"],
            "跟踪球" if event["ball_tracked"] else "候选球"), R.GREY, R.load_font(18)),
    ]
    return entry


def fp_entry(reason: dict, event: dict, annotations: list[dict]) -> dict:
    nearest = annotations[reason["nearest_row"]]
    entry = dict(event)
    entry.update(reason)
    entry.update({
        "type": "fp", "time_sec": event["timestamp_sec"], "tolerance_sec": 0.0,
        "frame": event["frame_index"], "gt_index": nearest["index"], "count": "",
        "low_quality": False,
        "kind_cn": R.REASON_CN.get(reason["kind"], reason["kind"]),
        "reason": reason["kind"],
        "file": "fp/fp_%07.2f_%s.png" % (event["timestamp_sec"], reason["kind"]),
        "colour": R.COLOURS["fp"],
        "caption": "误检 FP（%s）t=%.3f s：%s"
                   % (reason["kind"], event["timestamp_sec"], reason["note_cn"]),
    })
    entry["lane_gt"] = R.lane_gt_text(nearest, 0.0)
    entry["lane_pred"] = R.lane_pred_text(entry)
    entry["gt_spans"] = [(nearest["start_sec"], nearest["end_sec"])]
    entry["event_times"] = [event["timestamp_sec"]]
    entry["gt_line"] = "GT #%d   %d 次触球" % (nearest["index"], nearest["count"])
    entry["det_line"] = ("检测 @%.3f s 第 %d 帧   → 最近 GT #%d  d=%.1f/τ%.1f px"
                         % (event["timestamp_sec"], event["frame_index"], nearest["index"],
                            event["distance_px"], event["threshold_px"] or 0.0))
    entry["sheet_lines"] = [
        ("误检 FP   t=%.3fs   第 %d 帧   人 %d %s   [%s]"
         % (event["timestamp_sec"], event["frame_index"], event["person_track_id"],
            event["foot"], reason["kind"]), R.COLOURS["fp"], R.load_font(21)),
        ("原因：%s" % reason["note_cn"], (240, 240, 240), R.load_font(18)),
        ("证据：d=%.1f px ≤ τ=%.1f px，%d 帧接触，球置信度 %.2f，%s"
         % (event["distance_px"], event["threshold_px"] or 0.0, event["contact_frames"],
            event["ball_confidence"], "跟踪球" if event["ball_tracked"] else "候选球"),
         R.GREY, R.load_font(18)),
        ("怎么修：%s" % reason["advice_cn"], R.ADVICE_COLOUR, R.load_font(18)),
    ]
    return entry


def fn_entry(reason: dict, annotation: dict, fps: float, missing: int, matched: int) -> dict:
    entry = dict(reason)
    entry.update({
        "type": "fn", "time_sec": annotation["start_sec"], "tolerance_sec": 0.0,
        "matched": matched, "missing": missing,
        "frame": int(round(annotation["start_sec"] * fps)), "gt_index": annotation["index"],
        "count": annotation["count"], "low_quality": annotation["low_quality"],
        "kind_cn": REASON_CN.get(reason["kind"], reason["kind"]),
        "reason": reason["kind"],
        "distance_px": reason["nearest_px"] if reason["nearest_px"] is not None else "",
        "threshold_px": reason["nearest_tau_px"] or "",
        "contact_frames": "",
        "ball_confidence": reason["max_ball_confidence"]
        if reason["max_ball_confidence"] is not None else "",
        "ball_tracked": "", "person_track_id": "", "foot": "",
        "advice_cn": reason["advice_cn"],
        "file": "fn/fn_gt%03d_%07.2f_%s.png"
                % (annotation["index"], annotation["start_sec"], reason["kind"]),
        "colour": R.COLOURS["fn"],
        "caption": "漏检 FN（%s）gt#%d %.2f s：%s"
                   % (reason["kind"], annotation["index"], annotation["start_sec"],
                      reason["note_cn"]),
    })
    entry["lane_gt"] = R.lane_gt_text(annotation, fps)
    entry["lane_pred"] = R.lane_pred_text(entry)
    entry["gt_spans"] = [(annotation["start_sec"], annotation["end_sec"])]
    entry["event_times"] = []
    entry["note_cn"] = "%s；本行命中 %d/%d" % (reason["note_cn"], matched, annotation["count"])
    entry["gt_line"] = ("GT #%d   %d 次触球   命中 %d/%d"
                        % (annotation["index"], annotation["count"], matched, annotation["count"]))
    entry["det_line"] = "检测：该跨度内没有配上的事件（缺 %d 个名额）" % missing
    entry["sheet_lines"] = [
        ("漏检 FN   gt#%d   %.2f-%.2f s   %d 次触球   命中 %d/%d   [%s]"
         % (annotation["index"], annotation["start_sec"], annotation["end_sec"],
            annotation["count"], matched, annotation["count"], reason["kind"]),
         R.COLOURS["fn"], R.load_font(21)),
        ("原因：%s" % reason["note_cn"], (240, 240, 240), R.load_font(18)),
        ("缺 %d 个名额%s" % (missing, "（低质量数据）" if annotation["low_quality"] else ""),
         R.GREY, R.load_font(18)),
        ("怎么修：%s" % reason["advice_cn"], R.ADVICE_COLOUR, R.load_font(18)),
    ]
    return entry



def fn_rows_of(rows: list[dict], position_to_event: dict[int, list[int]]) -> list:
    """没有填满的标注行（数量应当等于 FN 图的张数）。"""
    return [position for position, row in enumerate(rows)
            if len(position_to_event.get(position, [])) < row["count"]]


def playback_marks(rows: list[dict], position_to_event: dict[int, list[int]],
                   events: list[dict], event_to_position: dict[int, int],
                   total_frames: int, fps: float) -> tuple[list, list]:
    """两条轨道的标记：预测一条、标注一条，按名额级判色。

    - 预测：命中标绿、没配上标红（名额已满的第二个事件同样算命中）。
    - 标注：这一行写出的名额**全部填满**才标绿；一个都没填上、或只填了一部分 → 标黄。
    """
    def clamp(frame: float) -> int:
        return max(0, min(total_frames - 1, int(frame)))

    pred: list[tuple[int, int, str]] = []
    for index, event in enumerate(events):
        start = clamp(event.get("first_frame_index", event["frame_index"]))
        end = clamp(event["frame_index"])
        if end < start:
            start, end = end, start
        pred.append((start, end, "tp" if index in event_to_position else "fp"))

    gt: list[tuple[int, int, str]] = []
    for position, row in enumerate(rows):
        filled = len(position_to_event.get(position, []))
        start = clamp(round(row["start_sec"] * fps))
        end = clamp(round(row["end_sec"] * fps))
        if end < start:
            start, end = end, start
        gt.append((start, end, "tp" if filled >= row["count"] else "fn"))
    return pred, gt


def _entry_span(entry: dict) -> tuple[float, float]:
    span = entry.get("gt_spans") or []
    if span:
        return float(span[0][0]), float(span[0][1])
    moment = float(entry["time_sec"])
    return moment, moment


def playback_notes(entries: list[dict], fps: float, total_frames: int,
                   margin: int) -> dict[int, list]:
    """每一帧要显示的说明：命中的一句话，误检/漏检三行（和接触片上同样的字）。"""
    notes: dict[int, list] = {}

    def add(first: int, last: int, centre: int, lines: list) -> None:
        for frame in range(max(0, first), min(total_frames, last + 1)):
            notes.setdefault(frame, []).append((centre, lines))

    for entry in entries:
        start_sec, end_sec = _entry_span(entry)
        first = int(round(start_sec * fps))
        last = int(round(max(end_sec, entry["time_sec"]) * fps))
        if last < first:
            first, last = last, first
        if entry["type"] == "tp":
            lines = [("命中 TP   GT #%d（%.2f-%.2f s）由这次检测填上"
                      % (entry["gt_index"], start_sec, end_sec), R.MARK_COLOURS["tp"])]
            add(first - margin, last + margin, last, lines)
            continue
        if entry["type"] == "fp":
            head = "误检 FP（%s）  t=%.3f s  第 %d 帧" % (
                entry["kind_cn"], entry["time_sec"], entry["frame"])
            colour = R.MARK_COLOURS["fp"]
        else:
            head = "漏检 FN（%s）  GT #%d  %.2f-%.2f s  %d 次触球  缺 %d 个名额" % (
                entry["kind_cn"], entry["gt_index"], start_sec, end_sec,
                entry["count"], entry["missing"])
            colour = R.MARK_COLOURS["fn"]
        lines = [(head, colour),
                 ("原因：%s" % entry["note_cn"], (240, 240, 244)),
                 ("怎么修：%s" % entry["advice_cn"], R.ADVICE_COLOUR)]
        add(first - margin, last + margin, last, lines)
    return notes


def render_playback(args, output_dir: Path, rows: list[dict], entries: list[dict],
                    position_to_event: dict[int, list[int]],
                    event_to_position: dict[int, int], report: dict, fps: float) -> int:
    """整段实时播放：底部两条轨道 + 顶部说明；编码复用 `PlaybackWriter`。"""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    source = Path(args.playback_source) if args.playback_source else output_dir / "rendered.mp4"
    if not source.is_file():
        print("[folders] 没有 %s，跳过播放视频" % source)
        return 0
    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = args.playback_width or frame_width
    out_height = args.playback_height or frame_height
    stride = max(1, args.stride)

    pred_marks, gt_marks = playback_marks(rows, position_to_event, report["events"],
                                          event_to_position, total, source_fps)
    main = report["primary_count_aware"]
    pred_tp = sum(1 for _, _, kind in pred_marks if kind == "tp")
    gt_full = sum(1 for _, _, kind in gt_marks if kind == "tp")
    print("[folders] 播放视频标记：pred 命中 %d / 误检 %d（报告 %d / %d）｜ 标注 填满 %d / 未填满 %d"
          % (pred_tp, len(pred_marks) - pred_tp,
             main["true_positives"], main["false_positives"],
             gt_full, len(gt_marks) - gt_full))
    assert pred_tp == main["true_positives"], "播放视频的预测标记与报告不一致"
    assert len(pred_marks) - pred_tp == main["false_positives"], "播放视频的误检标记与报告不一致"
    assert len(gt_marks) - gt_full == len(fn_rows_of(rows, position_to_event)), \
        "标注轨道上未填满的行数与 FN 图的张数不一致"

    lanes, lane_left, lane_track = R.build_lanes_image(
        (out_width, out_height), total, pred_marks, gt_marks, source_fps)
    lanes_top = out_height - lanes.height
    notes = playback_notes(entries, source_fps, total, max(0, int(args.note_margin)))
    per_frame = lane_track / float(max(1, total))

    target = output_dir / "pr_review" / "touch_playback.mp4"
    writer = R.PlaybackWriter(target, (out_width, out_height), source_fps / stride,
                              args.playback_crf)
    if not writer.ok:
        print("[folders] 无法写入 %s" % target)
        return 0
    print("[folders] 播放视频编码：%s → %s" % (writer.codec, target))
    stamp_font = R.load_font(18)
    index = -1
    written = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        index += 1
        if index % stride:
            continue
        if (out_width, out_height) != (frame_width, frame_height):
            image = cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_AREA)
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
        rgb.paste(lanes, (0, lanes_top))
        draw = ImageDraw.Draw(rgb)
        x = lane_left + int(index * per_frame)
        draw.rectangle([x, lanes_top, x + 1, out_height - 1], fill=(255, 255, 255))
        stamp = "第 %d 帧   t=%.3f s" % (index, index / source_fps)
        draw.rectangle([6, lanes_top - 30,
                        26 + int(stamp_font.getlength(stamp)), lanes_top - 4],
                       fill=(10, 10, 12))
        draw.text((14, lanes_top - 28), stamp, font=stamp_font, fill=(200, 255, 210))
        shown = notes.get(index)
        if shown:
            shown.sort(key=lambda item: abs(item[0] - index))
            R.draw_playback_note(draw, (out_width, out_height), shown[0][1])
        writer.write(cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR))
        written += 1
    cap.release()
    writer.release()
    print("[folders] 播放视频写完：%d 帧" % written)
    return written


def lq_entry(annotation: dict, matched: int, events_in_span: list[tuple[float, str]],
             stats, tau, fps: float) -> dict:
    """一行低质量标注一张图：落在这一秒里"球离脚最近"的那一刻，带上标注者的备注。"""
    centre = (stats.nearest_timestamp_sec
              if (stats.frames and stats.nearest_timestamp_sec > 0.0)
              else annotation["start_sec"])
    frame = int(round(centre * fps))
    nearest = (stats.nearest_usable_distance_px
               if (stats.has_usable_ball and stats.has_usable_feet)
               else stats.nearest_distance_px)
    if tau is not None and nearest < float("inf"):
        evidence = "最近球脚 %.2f px / τ %.2f px（t=%.2f s）" % (nearest, tau, centre)
    elif nearest < float("inf"):
        evidence = "最近球脚 %.2f px（t=%.2f s，缺可用球或脚点）" % (nearest, centre)
    else:
        evidence = "这一秒内没有可用的球或脚点（t=%.2f s）" % centre
    inside = "；".join("%.2f s %s" % (moment, label) for moment, label in events_in_span) \
        or "该跨度内没有预测事件"
    entry = {
        "type": "lq", "time_sec": annotation["start_sec"], "tolerance_sec": 0.0,
        "gt_from": annotation["start_sec"], "gt_to": annotation["end_sec"],
        "frame": frame, "gt_index": annotation["index"], "count": annotation["count"],
        "matched": matched, "missing": annotation["count"] - matched,
        "low_quality": True, "kind": "low_quality", "kind_cn": "低质量数据",
        "reason": "low_quality", "gt_note": annotation["note"],
        "note_cn": "标注为低质量：%s（本行 %d 次触球，已命中 %d 次）"
                   % (annotation["note"] or "未写备注", annotation["count"], matched),
        "advice_cn": "能判读就按视频改写这一行（或去掉 low_quality_data）；确实无法判读就保留标记，"
                     "它会只出现在第二视图里",
        "distance_px": nearest if nearest < float("inf") else "",
        "threshold_px": tau if tau is not None else "",
        "contact_frames": "", "ball_confidence": "", "ball_tracked": "",
        "person_track_id": "", "foot": "",
        "file": "low_quality/lq_gt%03d_%07.2f.png" % (annotation["index"], centre),
        "colour": LQ_COLOUR,
        "caption": "低质量数据  GT #%d  %.2f-%.2f s  %d 次触球"
                   % (annotation["index"], annotation["start_sec"], annotation["end_sec"],
                      annotation["count"]),
    }
    entry["lane_gt"] = R.lane_gt_text(annotation, fps)
    entry["lane_pred"] = ""
    entry["gt_spans"] = [(annotation["start_sec"], annotation["end_sec"])]
    entry["event_times"] = [moment for moment, _ in events_in_span]
    entry["gt_line"] = ("GT #%d   %d 次触球   命中 %d/%d   low_quality_data=yes"
                        % (annotation["index"], annotation["count"], matched,
                           annotation["count"]))
    entry["det_line"] = "跨度内的事件：%s" % inside
    entry["sheet_lines"] = [
        ("低质量数据   GT #%d   %.2f-%.2f s   %d 次触球   命中 %d/%d"
         % (annotation["index"], annotation["start_sec"], annotation["end_sec"],
            annotation["count"], matched, annotation["count"]), LQ_COLOUR, R.load_font(21)),
        ("标注备注：%s" % (annotation["note"] or "（空）"), (240, 240, 240), R.load_font(18)),
        ("%s" % evidence, R.GREY, R.load_font(18)),
        ("跨度内的事件：%s" % inside, R.GREY, R.load_font(18)),
        ("怎么处理：%s" % entry["advice_cn"], R.ADVICE_COLOUR, R.load_font(18)),
    ]
    return entry


def write_timeline(output_dir: Path, report: dict, entries: list[dict],
                   lq_entries: list[dict] | None = None) -> None:
    review = output_dir / "pr_review"
    fields = ["type", "time_sec", "frame", "gt_index", "count", "matched", "missing",
              "low_quality", "kind_cn", "reason", "distance_px", "threshold_px",
              "note_cn", "image"]
    with open(review / "timeline.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for entry in sorted(entries, key=lambda item: (item["time_sec"], item["type"])):
            writer.writerow({key: entry.get(key, "") for key in fields[:-1]}
                            | {"image": entry["file"]})

    main = report["primary_count_aware"]
    clean = report["low_quality"]["primary_count_aware"]
    counts = {name: sum(1 for entry in entries if entry["type"] == name) for name in TYPES}
    lines = [
        "# 触球 PR 逐条复核（按时间线）",
        "",
        "判据：**名额级主口径**（容差 0）—— 事件落在标注写出的跨度里、且该行的名额没被占满，才算命中。"
        "这里三类图分别是：命中的事件（TP）、没配上标注的事件（FP）、没被填满的名额（FN，含写了 "
        "`count` 次但只命中一部分的行）。",
        "",
        "| 数 | 含 low_quality | 不含 low_quality |",
        "| --- | --- | --- |",
        "| 容量（名额数） | %d | %d |" % (main["capacity"], clean["capacity"]),
        "| TP | %d | %d |" % (main["true_positives"], clean["true_positives"]),
        "| FP | %d | %d |" % (main["false_positives"], clean["false_positives"]),
        "| FN | %d | %d |" % (main["false_negatives"], clean["false_negatives"]),
        "| precision | %.4f | %.4f |" % (main["precision"], clean["precision"]),
        "| recall | %.4f | %.4f |" % (main["recall"], clean["recall"]),
        "| F1 | %.4f | %.4f |" % (main["f1"], clean["f1"]),
        "",
        "图：TP %d 张（`tp/`）、FP %d 张（`fp/`）、FN %d 张（`fn/`），下面是同一批按时间排的清单"
        "（等于 `timeline.csv`）。" % (counts["tp"], counts["fp"], counts["fn"]),
        "",
        "| # | 时间 (s) | 类别 | GT行 | 次数 | 已命中 | 低质量 | 帧 | 原因 | 说明 | 图 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, entry in enumerate(sorted(entries, key=lambda item: (item["time_sec"], item["type"])), 1):
        matched = entry.get("matched", "")
        if entry["type"] == "tp":
            matched = 1
        lines.append("| %d | %.2f | %s | %s | %s | %s | %s | %s | %s | %s | `%s` |"
                     % (index, entry["time_sec"], entry["type"].upper(),
                        entry.get("gt_index", ""), entry.get("count", ""), matched,
                        "是" if entry.get("low_quality") else "",
                        entry.get("frame", ""), entry["kind_cn"],
                        str(entry.get("note_cn", "")).replace("|", "/"), entry["file"]))
    lines += ["", "原因代号与每条建议见 `render_pr_errors.py` 的说明；图上中文需要 CJK 字体"
              "（`PR_REVIEW_FONT` 可指定）。", ""]
    lq_entries = lq_entries or []
    if lq_entries:
        lq_fields = ["gt_index", "start_sec", "end_sec", "count", "matched", "gt_note",
                     "distance_px", "threshold_px", "frame", "image"]
        with open(review / "low_quality.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=lq_fields, extrasaction="ignore")
            writer.writeheader()
            for entry in sorted(lq_entries, key=lambda item: item["gt_index"]):
                writer.writerow({key: entry.get(key, "") for key in lq_fields[:-1]}
                                | {"image": entry["file"]})
        lines += ["", "## 低质量数据（单独一套图：`low_quality/`）", "",
                  "这些行是标注者自己标了 `low_quality_data=yes` 的。图落在这一秒里球离脚最近的"
                  "那一刻，并带上标注者写的备注，方便逐个判断能不能判读：能判读就改写成确定结论，"
                  "判不了就保留标记（它只会出现在第二视图里）。", "",
                  "| GT行 | 起止 (s) | 次数 | 已命中 | 标注备注 | 图 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for entry in sorted(lq_entries, key=lambda item: item["gt_index"]):
            lines.append("| %d | %.2f-%.2f | %d | %d | %s | `%s` |"
                         % (entry["gt_index"], entry["gt_from"], entry["gt_to"],
                            entry["count"], entry["matched"],
                            str(entry.get("gt_note", "")).replace("|", "/"), entry["file"]))
        print("[folders] 低质量清单：%s（%d 行）"
              % (review / "low_quality.csv", len(lq_entries)))
    (review / "timeline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[folders] 汇总表：%s" % (review / "timeline.md"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="output/touch_eval_moving_pass",
                        help="目录里要有 observations.jsonl / touch_report.json / "
                             "contact_signal.csv / rendered.mp4")
    parser.add_argument("--annotations", default="data/移动传球.touches.csv")
    parser.add_argument("--fn-window", type=float, default=0.5,
                        help="判定 FN 原因时把跨度放宽多少秒（和报告一致，默认 0.5）")
    parser.add_argument("--adjacency", type=float, default=1.5)
    parser.add_argument("--min_contact_frames", type=int, default=2)
    parser.add_argument("--refractory_frames", type=int, default=30)
    parser.add_argument("--person_refractory_frames", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="每类只出前 N 张（0 = 全部）")
    parser.add_argument("--playback", action="store_true",
                        help="另外写 pr_review/touch_playback.mp4（整段实时，底部两条轨道）")
    parser.add_argument("--playback-only", action="store_true",
                        help="配合 --playback：不重出接触片，只写播放视频")
    parser.add_argument("--playback-source", default="",
                        help="播放视频的底片（默认 <output-dir>/rendered.mp4）")
    parser.add_argument("--playback-width", type=int, default=0)
    parser.add_argument("--playback-height", type=int, default=0)
    parser.add_argument("--playback-crf", type=float, default=20.0,
                        help="用 ffmpeg 编码时的质量；0 则退回 cv2 的 mp4v")
    parser.add_argument("--stride", type=int, default=1, help="每 N 帧留一帧")
    parser.add_argument("--note-margin", type=int, default=12,
                        help="说明文字在事发前后各停留多少帧")
    parser.add_argument("--sheet-frames", type=int, default=6)
    parser.add_argument("--sheet-columns", type=int, default=3)
    parser.add_argument("--sheet-span", type=float, default=0.2)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=480)
    parser.add_argument("--video", action="store_true",
                        help="另外给每个条目写 pr_review/pr_errors*.mp4 短片")
    parser.add_argument("--video-span", type=float, default=0.5)
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    parser.add_argument("--hold-frames", type=int, default=0)
    parser.add_argument("--no-zoom", dest="zoom", action="store_false",
                        help="显示整帧，而不是裁到事发的人和球附近")
    args = parser.parse_args()

    print("[folders] 中文字体：%s" % (R.cjk_font_path() or "未找到（图上将退回英文）"))
    output_dir = Path(args.output_dir)
    report = json.loads((output_dir / "touch_report.json").read_text(encoding="utf-8"))
    events = report["events"]
    fps = float(report.get("fps") or 60.0)
    params = DetectorParams.parse(report["params"])
    rows, no_touch = load_annotations(Path(args.annotations))

    event_to_position, position_to_event = resolve_mapping(rows, events)
    tp_events = [(index, event) for index, event in enumerate(events)
                 if index in event_to_position]
    fp_events = [(index, event) for index, event in enumerate(events)
                 if index not in event_to_position]
    fn_rows = []
    for position, annotation in enumerate(rows):
        matched = len(position_to_event.get(position, []))
        if matched < annotation["count"]:
            fn_rows.append((position, annotation, annotation["count"] - matched, matched))

    main = report["primary_count_aware"]
    fn_slots = sum(missing for _, _, missing, _ in fn_rows)
    print("[folders] 名额级：TP=%d  FP=%d  FN=%d（%d 行，缺 %d 个名额）"
          % (len(tp_events), len(fp_events), fn_slots, len(fn_rows), fn_slots))
    assert (len(tp_events), len(fp_events), fn_slots) == (
        main["true_positives"], main["false_positives"], main["false_negatives"]), \
        "与 touch_report.json 的主口径不一致，先别渲染"
    assert len(tp_events) + fn_slots == main["capacity"], "容量与 TP+FN 不闭合"

    entries: list[dict] = [
        tp_entry(events[index], rows[position], fps)
        for index, position in sorted(event_to_position.items())
    ]

    for _, event in fp_events:
        # 归因函数要的是"一行一个事件"，而一行可能配了多次；取这一行里离本事件最近的
        # 那个来叙述（"该标注已配给 X s 的检测"）。
        owner = {
            position: min(indices,
                          key=lambda index: abs(events[index]["timestamp_sec"]
                                                - event["timestamp_sec"]))
            for position, indices in position_to_event.items()
        }
        reason = R.false_positive_reason(event, rows, events, no_touch, owner,
                                         0.0, args.adjacency, args.refractory_frames)
        want = VERDICT_KIND.get(event["verdict"])
        if want and reason["kind"] != want:
            raise SystemExit("FP 分类不一致：报告 %s / 本工具 %s"
                             % (event["verdict"], reason["kind"]))
        entries.append(fp_entry(reason, event, rows))

    lq_positions = [position for position, row in enumerate(rows) if row["low_quality"]]
    frames: list = []
    timestamps: list[float] = []
    if fn_rows or lq_positions:
        frames = cli_module.load_evidence(str(output_dir / "observations.jsonl"), params)
        timestamps = [frame.timestamp_sec for frame in frames]
    mine: dict[str, int] = {}
    mine_unmatched: dict[str, int] = {}
    for _, annotation, missing, matched in fn_rows:
        reason = fn_reason(frames, timestamps, annotation, params, args.fn_window,
                           events, args.min_contact_frames)
        mine[reason["kind"]] = mine.get(reason["kind"], 0) + 1
        if matched == 0:
            mine_unmatched[reason["kind"]] = mine_unmatched.get(reason["kind"], 0) + 1
        entries.append(fn_entry(reason, annotation, fps, missing, matched))
    # 报告的 failures 只统计"整行未命中"的行，所以拿这一部分和它对拍。
    report_kinds = {FAILURE_KIND.get(group["reason"], group["reason"]): group["count"]
                    for group in report["failures"]
                    if not group["reason"].startswith("false positive")}
    print("[folders] FN 原因分组：全部 %s（其中整行未命中 %s）/ 报告 %s"
          % (mine, mine_unmatched, report_kinds))
    if mine_unmatched != report_kinds:
        raise SystemExit("FN 原因分组与报告不一致：本工具 %s / 报告 %s"
                         % (mine_unmatched, report_kinds))

    full_entries = list(entries)  # 播放视频用全量，--limit 只影响接触片
    lq_entries: list[dict] = []
    for position in lq_positions:
        annotation = rows[position]
        stats = metrics_module.scan_window(
            frames, timestamps, annotation["start_sec"] - args.fn_window,
            annotation["end_sec"] + args.fn_window)
        tau = None
        if stats.has_usable_ball and stats.has_usable_feet:
            tau = float(detect_module.contact_threshold_px(
                params, stats.nearest_frame_width, stats.nearest_frame_height,
                detect_module.F32(stats.nearest_person_height_px)))
        in_span = []
        for index, event in enumerate(events):
            moment = event["timestamp_sec"]
            if annotation["start_sec"] <= moment <= annotation["end_sec"]:
                verdict = event["verdict"]
                in_span.append((moment,
                                "命中" if verdict == "assigned"
                                else R.REASON_CN.get(VERDICT_KIND.get(verdict, verdict),
                                                     verdict)))
        matched = len(position_to_event.get(position, []))
        lq_entries.append(lq_entry(annotation, matched, sorted(in_span), stats, tau, fps))
    print("[folders] 低质量行 %d 行（其中 %d 行同时也是漏检行）"
          % (len(lq_entries), sum(1 for entry in lq_entries if entry["missing"] > 0)))

    if args.limit:
        keep: list[dict] = []
        for name in TYPES:
            keep += [entry for entry in entries if entry["type"] == name][:args.limit]
        entries = keep

    review = output_dir / "pr_review"
    for name in TYPES + ("low_quality",):
        (review / name).mkdir(parents=True, exist_ok=True)
    if not args.playback_only:
        written = R.render(args, output_dir, entries + lq_entries, fps)
        write_timeline(output_dir, report, entries, lq_entries)
        print("[folders] 写完 %d 张图：TP %d / FP %d / FN %d + 低质量 %d"
              % (written, *[sum(1 for e in entries if e["type"] == n) for n in TYPES],
                 len(lq_entries)))
    if args.playback or args.playback_only:
        render_playback(args, output_dir, rows, full_entries, position_to_event,
                        event_to_position, report, fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
