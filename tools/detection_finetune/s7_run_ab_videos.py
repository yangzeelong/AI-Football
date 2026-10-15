#!/usr/bin/env python3
"""多视频检测级 A/B：同一批视频上对比多个模型版本（默认 v1 vs v1-frozen-step2）。

为什么单独一个脚本（而不是直接用 `code/app.py`）
------------------------------------------------
* `code/app.py` 走的是**完整管线**（tracker + 姿态 + 渲染 + observations.jsonl），
  而 `FrameObservation2D` 只有 persons/balls，**锥框不落盘**（只在渲染视频里画）→
  拿它统计"锥被判成球"根本拿不到锥。所以这里在**检测层**直接统计（`label == cone`）。
* 视频**只解码一遍**，两个版本在同一帧上依次推理 → 8 个视频 × 2 版本不必解码两次。
* 复用 `s5_eval_render.load_model/dets_for`：前者加载 rfdetr 权重，后者把模型自带的
  类别名（person/ball/cone）映射回本项目 id（1/2/3），避免"名字写错整类静默消失"。

视频上没有 GT，所以"锥被判成球"只能用**代理指标**：
    同一个预测球框，若与同帧某个预测锥框 IoU ≥ `--overlap-iou`（默认 0.5），
    就记为一次"球∩锥"重叠 —— 这正是管线里"球候选与锥框高度重叠则抑制"能救回的那批误检。
真实 GT 级指标（mAP、精确率/召回）请看 `s5_eval_detector.py`（固定评测集，有标注）。

用法
----
    source env.sh
    # 默认：脚本内置的 4 段视频 × (v1, v1-frozen-step2)
    $PY s7_run_ab_videos.py
    # 指定视频（逗号分隔 / glob / 目录都行）
    $PY s7_run_ab_videos.py --videos "data/移动传球.mov,data/颠球.mov"
    $PY s7_run_ab_videos.py --videos-glob "$AI_FOOTBALL_DATA_DIR/20260726/*.mov" --stride 10
    # 只看前 200 帧、置信度 0.25、并把逐帧计数存 CSV
    $PY s7_run_ab_videos.py --max-frames 200 --frames-csv
"""
from __future__ import annotations

import argparse
import csv
import glob as globmod
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
#: 工作区根 = <本文件>/../../..（…/Work）；用 AI_FOOTBALL_WORKSPACE 覆盖。
#: 解析视频相对路径时按 (CWD, WORKSPACE, <repo>) 依次尝试。
WORK = Path(os.environ.get("AI_FOOTBALL_WORKSPACE") or HERE.parents[2]).resolve()
import s5_eval_render as rd                                            # noqa: E402

DEFAULT_VERSIONS = [
    "v1=%s" % (HERE / "runs/v1/checkpoint_best_ema.pth"),
    "v1-frozen-step2=%s" % (HERE / "runs/v1-frozen-step2/checkpoint_best_ema.pth"),
]
DEFAULT_VIDEOS = [
    "data/移动传球.mov", "data/颠球.mov", "data/绕障碍.mov", "data/射门1-1080p60.mov",
]
PERSON, BALL, CONE = 1, 2, 3


def parse_versions(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit("--version 要写 TAG=权重路径，收到：%s" % item)
        tag, path = item.split("=", 1)
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (HERE / p).resolve()
        if not p.is_file():
            raise SystemExit("权重不存在：%s" % p)
        out.append((tag.strip(), p))
    return out


def resolve_videos(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.videos:
        paths += [Path(v.strip()) for v in args.videos.split(",") if v.strip()]
    if args.videos_glob:
        paths += [Path(p) for p in sorted(globmod.glob(args.videos_glob, recursive=True))]
    if args.videos_dir:
        d = Path(args.videos_dir)
        paths += sorted(p for p in d.rglob("*") if p.suffix.lower() in (".mov", ".mp4", ".avi", ".mkv"))
    if not paths:
        paths = [Path(p) for p in DEFAULT_VIDEOS]
    fixed: list[Path] = []
    for p in paths:
        for base in (Path.cwd(), WORK, HERE.parents[1]):
            cand = p if p.is_absolute() else (base / p)
            if cand.is_file():
                fixed.append(cand)
                break
        else:
            print("  跳过（找不到文件）：%s" % p, file=sys.stderr)
    seen, uniq = set(), []
    for p in fixed:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_video(cap: cv2.VideoCapture, models: list[tuple[str, Any]], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """一段视频上跑所有版本（只解码一遍）。返回 {版本: 指标}。"""
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    stats: dict[str, dict[str, Any]] = {}
    for tag, _ in models:
        stats[tag] = {"frames": 0, "boxes": {PERSON: 0, BALL: 0, CONE: 0},
                      "frames_with_ball": 0, "frames_with_cone": 0, "frames_with_both": 0,
                      "frames_with_overlap": 0, "overlap_boxes": 0, "ball_only_frames": 0,
                      "inside_boxes": 0, "frames_with_inside": 0,
                      "elapsed": 0.0, "rows": []}
    frame_idx = -1
    processed = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        processed += 1
        for tag, model in models:
            t0 = time.perf_counter()
            det = rd.dets_for(model, frame, args.threshold)
            dt = time.perf_counter() - t0
            st = stats[tag]
            labels = det["labels"]
            n_person = labels.count(PERSON)
            n_ball = labels.count(BALL)
            n_cone = labels.count(CONE)
            st["frames"] += 1
            st["elapsed"] += dt
            st["boxes"][PERSON] += n_person
            st["boxes"][BALL] += n_ball
            st["boxes"][CONE] += n_cone
            st["frames_with_ball"] += 1 if n_ball else 0
            st["frames_with_cone"] += 1 if n_cone else 0
            st["frames_with_both"] += 1 if (n_ball and n_cone) else 0
            st["ball_only_frames"] += 1 if (n_ball and not n_cone) else 0
            # 两种"锥被判成球"的代理，严格程度不同（视频上没 GT，只能这样近似）：
            #   overlaps = 球框与锥框 IoU ≥ 阈值的框数（松；球滚到锥旁边也会命中）
            #   inside   = 球框**中心落在锥框内**的框数（严；更接近"这个锥被当成了球"）
            overlaps = inside = 0
            if n_ball and n_cone:
                ball_boxes = [b for b, lb in zip(det["boxes"], labels) if lb == BALL]
                cone_boxes = [b for b, lb in zip(det["boxes"], labels) if lb == CONE]
                for bb in ball_boxes:
                    if any(iou(bb, cb) >= args.overlap_iou for cb in cone_boxes):
                        overlaps += 1
                    ccx, ccy = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
                    if any(cb[0] <= ccx <= cb[2] and cb[1] <= ccy <= cb[3] for cb in cone_boxes):
                        inside += 1
            st["overlap_boxes"] += overlaps
            st["frames_with_overlap"] += 1 if overlaps else 0
            st["inside_boxes"] += inside
            st["frames_with_inside"] += 1 if inside else 0
            if args.frames_csv:
                st["rows"].append([frame_idx, round(frame_idx / src_fps, 3),
                                   n_person, n_ball, n_cone, overlaps, inside])
        if processed % 60 == 0:
            print("      ...%d 帧" % processed, flush=True)
        if args.max_frames and processed >= args.max_frames:
            break
    return stats


METRICS: list[tuple[str, str, Any]] = [
    ("处理帧数", "frames", lambda v: v),
    ("人框总数", "boxes", lambda v: v[PERSON]),
    ("球框总数", "boxes", lambda v: v[BALL]),
    ("锥框总数", "boxes", lambda v: v[CONE]),
    ("有球的帧", "frames_with_ball", lambda v: v),
    ("有锥的帧", "frames_with_cone", lambda v: v),
    ("球锥同帧", "frames_with_both", lambda v: v),
    ("**球∩锥重叠框数（误检代理）**", "overlap_boxes", lambda v: v),
    ("球∩锥重叠帧数", "frames_with_overlap", lambda v: v),
    ("**球心落在锥框内（严格代理）**", "inside_boxes", lambda v: v),
    ("球心落锥框内的帧", "frames_with_inside", lambda v: v),
    ("只有球没有锥的帧", "ball_only_frames", lambda v: v),
    ("平均每帧耗时(ms)", "elapsed", lambda v: v),      # 取值见 metric_value：这里存的是总秒数
]


def metric_value(stats_one: dict[str, Any], key: str) -> float | int:
    """取某个指标的值。`elapsed` 存的是**总耗时秒数**，要除以帧数才是"每帧毫秒"。"""
    if key == "elapsed":
        return 1000.0 * float(stats_one["elapsed"]) / max(1, int(stats_one["frames"]))
    return stats_one[key]


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", help="逗号分隔的视频路径（相对 Work / 当前目录 / 仓库根都行）")
    ap.add_argument("--videos-glob", help="glob，如 'data/20260726/*.mov'")
    ap.add_argument("--videos-dir", help="目录（递归找 mov/mp4/avi/mkv）")
    ap.add_argument("--version", action="append", default=None, metavar="TAG=权重",
                    help="可重复；默认 v1 与 v1-frozen-step2")
    ap.add_argument("--resolution", type=int, default=960)
    ap.add_argument("--threshold", type=float, default=0.25, help="检测置信度阈值（部署默认 0.25）")
    ap.add_argument("--stride", type=int, default=6, help="每 N 帧取 1 帧（60fps 下 6 ≈ 10 帧/秒）")
    ap.add_argument("--max-frames", type=int, default=240, help="每段视频最多处理多少采样帧（0=全部）")
    ap.add_argument("--overlap-iou", type=float, default=0.5, help="球框与锥框算重叠的 IoU 阈值")
    ap.add_argument("--frames-csv", action="store_true", help="逐帧计数存 CSV")
    ap.add_argument("--out-dir", default=str(HERE / "runs" / "videos_ab"))
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.max_frames == 0:
        a.max_frames = None

    versions = parse_versions(a.version or DEFAULT_VERSIONS)
    videos = resolve_videos(a)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not videos:
        raise SystemExit("没有可跑的视频")

    print("版本：%s" % " | ".join("%s(%s)" % (t, p.name) for t, p in versions))
    print("视频：%d 段，stride=%d，最多 %s 采样帧，阈值 %.2f"
          % (len(videos), a.stride, a.max_frames or "全部", a.threshold))
    models: list[tuple[str, Any]] = []
    for tag, path in versions:
        print("  加载 %s …" % tag, flush=True)
        models.append((tag, rd.load_model(path, a.resolution)))

    all_stats: dict[str, dict[str, dict[str, Any]]] = {}
    src_meta: dict[str, dict[str, float]] = {}
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print("  !! 打不开，跳过：%s" % video, file=sys.stderr)
            cap.release()
            continue
        src_meta[video.name] = {"fps": cap.get(cv2.CAP_PROP_FPS) or 60.0,
                                "w": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                "h": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                "total": cap.get(cv2.CAP_PROP_FRAME_COUNT)}
        print("\n[%s] %dx%d %.0ffps 共 %d 帧"
              % (video.name, src_meta[video.name]["w"], src_meta[video.name]["h"],
                 src_meta[video.name]["fps"], src_meta[video.name]["total"]), flush=True)
        stats = run_video(cap, models, a)
        cap.release()
        all_stats[video.name] = stats
        for tag, _ in versions:
            st = stats[tag]
            print("   %-18s 帧 %d  球 %d  锥 %d  重叠框 %d  每帧 %.1fms"
                  % (tag, st["frames"], st["boxes"][BALL], st["boxes"][CONE],
                     st["overlap_boxes"], 1000.0 * st["elapsed"] / max(1, st["frames"])))
            if a.frames_csv:
                csv_path = out_dir / ("per_frame_%s_%s.csv" % (tag, video.stem))
                with csv_path.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["frame", "time_sec", "person", "ball", "cone",
                                "ball_cone_overlap", "ball_center_in_cone"])
                    w.writerows(st["rows"])

    tags = [t for t, _ in versions]
    lines = ["# 多视频检测级 A/B —— %s" % " vs ".join(tags), "",
             "阈值 %.2f，stride %d，每段最多 %s 采样帧；视频上无 GT，故用「球∩锥重叠」作误检代理"
             % (a.threshold, a.stride, a.max_frames or "全部"), ""]
    for video, stats in all_stats.items():
        meta = src_meta.get(video, {})
        lines += ["## %s（%dx%d %.0ffps）" % (video, meta.get("w", 0), meta.get("h", 0), meta.get("fps", 0)),
                  "", "| 指标 | " + " | ".join(tags) + " |", "| --- |" + " --- |" * len(tags)]
        for label, key, get in METRICS:
            vals = []
            for tag in tags:
                raw = metric_value(stats[tag], key)
                v = get(raw)
                vals.append("%.1f" % v if isinstance(v, float) else str(v))
            lines.append("| %s | %s |" % (label, " | ".join(vals)))
        f_i, l_i = stats[tags[0]]["inside_boxes"], stats[tags[-1]]["inside_boxes"]
        f_o, l_o = stats[tags[0]]["overlap_boxes"], stats[tags[-1]]["overlap_boxes"]
        lines += ["", "**变化（%s → %s）**：球心落锥框内 %d → %d（%s）；球∩锥重叠 %d → %d（%s）" % (
            tags[0], tags[-1], f_i, l_i,
            "%.1f%%" % (100.0 * (l_i - f_i) / f_i) if f_i else "—", f_o, l_o,
            "%.1f%%" % (100.0 * (l_o - f_o) / f_o) if f_o else "—"), ""]
    report = out_dir / "summary.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "version", "metric", "value"])
        for video, stats in all_stats.items():
            for tag in tags:
                for label, key, get in METRICS:
                    w.writerow([video, tag, label, get(metric_value(stats[tag], key))])
    print("\n汇总 → %s\n       %s" % (report, out_dir / "summary.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
