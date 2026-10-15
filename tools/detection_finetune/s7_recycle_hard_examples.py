#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""误检帧回收闭环：跑视频 → 挑出「锥被当球」的误检帧 → 导出待标注图 + 预标注 → 回灌数据集。

为什么不复用 `s7_run_ab_videos.py`：那个脚本只出「球∩锥重叠」的**计数报表**，不落图、
不登记 manifest、也产不出能喂给 S3 的待标包。本脚本把同一条判定链延伸到「可回收」：
选帧 → 出图 → 写 COCO 预标注 → 登 manifest → 交回 S3/S4。

两个判据（可同时生效，命中次数就是这一帧的难度分）：

* ``cross``  —— **A 判球、B 判锥**（默认 A=`base` COCO 模型、B=`v1-frozen-step2`）。
  这正是「锥形桶被基础模型当成足球」的那批帧，也是闭环最想捞回来的样本。
  （COCO base 没有 cone 类，它自己内部永远不可能「球∩锥」，所以必须跨模型比。）
* ``within`` —— 同一模型内「球框与锥框高度重叠 / 球心落在锥框内」：三分类模型自己的残留歧义。

选帧不是"全要"，而是**按段取代表帧**：候选帧按间隔分段，每段按难度分取前 N 帧，
再过一遍 16×16 dHash 去重（8×8 会把真实变化当重复，见坑清单），最后受 `--max-frames` 约束。

**安全闸门（默认拒绝）**：落在 `valid`/`test` 的源视频是固定评测集，回收进它会污染评测
→ 脚本直接报错退出，要覆盖得显式 `--allow-eval-source`。

用法::

    source env.sh
    # 先看会挑出什么（不写任何文件）
    $PY s7_recycle_hard_examples.py --video $AI_FOOTBALL_DATA_DIR/移动传球.mov --dry-run
    # 真回收（出图 + 预标注 + 登 manifest）
    $PY s7_recycle_hard_examples.py --video $AI_FOOTBALL_DATA_DIR/移动传球.mov
    # 回灌：把导出的预标注合并进 GT（按类别名映射，S3 的同一套往返）
    $PY s3_label_review_roundtrip.py --import <work/recycle/<stem>__recycle_preannot.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import cv2

import base_layout
import s1_data_extract_frames as s1
import s5_eval_render as rd
import s7_run_ab_videos as ab

HERE = Path(__file__).resolve().parent
#: 注意：`ab.parse_versions` 把相对路径解析到**本脚本目录**（不是仓库根），所以这里写 `../..`。
DEFAULT_VERSIONS = (
    "base=../../models/rfdetr/rf-detr-small.pth",
    "v1-frozen-step2=runs/v1-frozen-step2/checkpoint_best_ema.pth",
)
WORK_RECYCLE = base_layout.WORK_DIR / "recycle"
CONTAIN_EPS = 1.0          # 判定"球心落在锥框内"时的边界容差（像素）


# --- 判据 ---------------------------------------------------------------------


def box_contains_point(box: Sequence[float], x: float, y: float, eps: float = CONTAIN_EPS) -> bool:
    """box 是 xyxy；点 (x, y) 是否落在框内（含 eps 容差）。"""
    return (box[0] - eps) <= x <= (box[2] + eps) and (box[1] - eps) <= y <= (box[3] + eps)


def frame_signals(dets_by_tag: dict[str, dict[str, Any]], iou_thr: float,
                  contain_center: bool) -> list[dict[str, Any]]:
    """返回这一帧命中的判据列表（空列表 = 干净帧）。

    每项：``{"kind", "a", "b", "iou"/"contain", "ball", "cone"}``，``a``/``b`` 是版本 tag。
    """
    hits: list[dict[str, Any]] = []
    tags = sorted(dets_by_tag)
    # ① 同一模型内：球框与锥框重叠 / 球心落锥框内
    for tag in tags:
        det = dets_by_tag[tag]
        for ball in det["ball"]:
            for cone in det["cone"]:
                if ab.iou(ball, cone) >= iou_thr:
                    hits.append({"kind": "within", "a": tag, "b": tag,
                                 "why": "iou>=%.2f" % iou_thr, "ball": ball, "cone": cone})
                    break
                if contain_center:
                    cx, cy = (ball[0] + ball[2]) / 2.0, (ball[1] + ball[3]) / 2.0
                    if box_contains_point(cone, cx, cy):
                        hits.append({"kind": "within", "a": tag, "b": tag,
                                     "why": "ball_center_in_cone", "ball": ball, "cone": cone})
                        break
    # ② 跨模型：A 判球、B 判锥（A != B）
    for a in tags:
        for b in tags:
            if a == b:
                continue
            for ball in dets_by_tag[a]["ball"]:
                for cone in dets_by_tag[b]["cone"]:
                    if ab.iou(ball, cone) >= iou_thr:
                        hits.append({"kind": "cross", "a": a, "b": b,
                                     "why": "iou>=%.2f" % iou_thr, "ball": ball, "cone": cone})
                        break
    return hits


def segment_pick(cands: list[dict[str, Any]], gap: int, per_segment: int) -> list[dict[str, Any]]:
    """候选帧按间隔分段，每段按难度分降序取前 per_segment 帧。"""
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: c["frame"])
    segments: list[list[dict[str, Any]]] = [[cands[0]]]
    for cur in cands[1:]:
        if cur["frame"] - segments[-1][-1]["frame"] <= gap:
            segments[-1].append(cur)
        else:
            segments.append([cur])
    picked: list[dict[str, Any]] = []
    for seg in segments:
        for item in sorted(seg, key=lambda c: (-c["score"], c["frame"]))[:max(1, per_segment)]:
            picked.append(item)
    return picked


def dhash_ok(frame: np.ndarray, seen: list[int], size: int, min_delta: int) -> bool:
    """与本次已选帧比 dHash；任何一帧太像就丢弃（size 必须 ≥16，8×8 会误杀真实变化）。"""
    h = s1.dhash(frame, size)
    for prev in seen:
        if s1.hamming(h, prev) < min_delta:
            return False
    seen.append(h)
    return True


# --- 单视频 -------------------------------------------------------------------


def scan_video(path: Path, models: list[tuple[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """抽帧扫描；返回（候选帧列表，统计）。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit("打不开视频：%s" % path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cands: list[dict[str, Any]] = []
    scanned = 0
    width = height = 0
    limit = total if args.max_scan_frames <= 0 else min(total, args.max_scan_frames)
    for fi in range(0, limit, max(1, args.stride)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        scanned += 1
        dets_by_tag: dict[str, dict[str, Any]] = {}
        for tag, model in models:
            det = rd.dets_for(model, frame, args.threshold)
            dets_by_tag[tag] = {
                "person": [b for b, l in zip(det["boxes"], det["labels"]) if l == ab.PERSON],
                "ball": [b for b, l in zip(det["boxes"], det["labels"]) if l == ab.BALL],
                "cone": [b for b, l in zip(det["boxes"], det["labels"]) if l == ab.CONE],
            }
        hits = frame_signals(dets_by_tag, args.overlap_iou, args.contain_center)
        if hits:
            cands.append({"frame": fi, "time_sec": round(fi / fps, 3) if fps else 0.0,
                          "score": len(hits), "hits": hits[:8],
                          "counts": {t: {k: len(v) for k, v in d.items()} for t, d in dets_by_tag.items()}})
    cap.release()
    stats = {"video": str(path), "fps": round(fps, 2), "frames_total": total,
             "frames_scanned": scanned, "resolution": "%dx%d" % (width, height),
             "candidate_frames": len(cands), "stride": max(1, args.stride)}
    return cands, stats


def export_frames(path: Path, picked: list[dict[str, Any]], stem: str, split: str,
                  plan_why: str, scene: str, view: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把选中的帧写成 jpg + manifest 行 + COCO 预标注。返回（manifest 行，COCO dict）。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit("打不开视频：%s" % path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    rel_prefix = base_layout.RAW_VIDEO_DIR.relative_to(base_layout.DATASET_ROOT)
    out_dir = base_layout.DATASET_ROOT / rel_prefix / stem
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    coco: dict[str, Any] = {
        "info": {"description": "recycle hard examples (%s)" % stem,
                 "date_created": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "licenses": [],
        "categories": base_layout.categories(),
        "images": [],
        "annotations": [],
    }
    rows: list[dict[str, Any]] = []
    ann_id = 0
    img_id = 0
    for item in sorted(picked, key=lambda c: c["frame"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame"])
        ok, frame = cap.read()
        if not ok:
            print("  !! 读不到第 %d 帧，跳过" % item["frame"], file=sys.stderr)
            continue
        h, w = frame.shape[:2]
        rel_image = "%s/%s/%s__f%06d__t%07.2f.jpg" % (rel_prefix, stem, stem, item["frame"],
                                                      item["frame"] / fps if fps else 0.0)
        if not args.dry_run:
            target = base_layout.DATASET_ROOT / rel_image
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality]):
                raise RuntimeError("写图失败（父目录在？）：%s" % target)
            if target.stat().st_size == 0:
                raise RuntimeError("写出的图为空：%s" % target)
        kinds = sorted({h_["kind"] for h_ in item["hits"]})
        rows.append({
            "image": rel_image, "source": "raw_video", "source_video": str(path),
            "frame_index": item["frame"], "time_sec": item["time_sec"],
            "width": w, "height": h, "scene": scene, "view": view, "split": split,
            "is_synthetic": "no", "ai_background": "no",
            "is_pseudolabel": "no", "license": "自有素材",
            "note": "recycled:%s score=%d %s" % ("+".join(kinds) or "-", item["score"], plan_why),
        })
        img_id += 1
        coco["images"].append({"id": img_id, "file_name": rel_image, "width": w, "height": h})
        # 预标注：把**最会出问题的那一侧**画上——cross 用 A（base 的球框，人工多半改成锥），
        # within 用同模型。取该帧第一条命中的 ball/cone，供人工增删改。
        for h_ in item["hits"][:1]:
            for box, cid in ((h_["ball"], ab.BALL), (h_["cone"], ab.CONE)):
                x1, y1, x2, y2 = box
                ann_id += 1
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cid,
                    "bbox": [round(x1, 2), round(y1, 2), round(max(0.0, x2 - x1), 2),
                             round(max(0.0, y2 - y1), 2)],
                    "area": round(max(0.0, x2 - x1) * max(0.0, y2 - y1), 2),
                    "iscrowd": 0,
                    "attributes": {"source": "recycle:%s" % h_["kind"], "why": h_["why"],
                                   "a": h_["a"], "b": h_["b"]},
                })
    cap.release()
    return rows, coco


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="append", required=True,
                    help="源视频路径（可重复；绝对路径或相对 Work / DATA_ROOT / 当前目录）")
    ap.add_argument("--version", action="append", default=None, metavar="TAG=权重",
                    help="模型版本（可重复）。默认 base + v1-frozen-step2（跨模型判据需要两个）")
    ap.add_argument("--resolution", type=int, default=960)
    ap.add_argument("--threshold", type=float, default=0.25, help="检测置信度阈值（部署默认 0.25）")
    ap.add_argument("--stride", type=int, default=6, help="每 N 帧扫一帧（60fps 下 6 = 10 帧/秒）")
    ap.add_argument("--max-scan-frames", type=int, default=0, help="最多扫多少帧（0=全片）")
    ap.add_argument("--overlap-iou", type=float, default=0.5, help="球框与锥框算重叠的 IoU 阈值")
    ap.add_argument("--no-contain-center", dest="contain_center", action="store_false",
                    help="不把「球心落在锥框内」当命中（默认会当）")
    ap.add_argument("--segment-gap", type=int, default=30, help="相邻候选帧间隔 ≤ 该值算同一段")
    ap.add_argument("--per-segment", type=int, default=1, help="每段取几帧代表")
    ap.add_argument("--max-frames", type=int, default=120, help="每段视频最多回收多少帧")
    ap.add_argument("--dhash-size", type=int, default=16, help="dHash 边长（必须 ≥16，8×8 会误杀）")
    ap.add_argument("--min-dhash-delta", type=int, default=6, help="汉明距离小于它就当重复")
    ap.add_argument("--quality", type=int, default=92, help="jpg 质量")
    ap.add_argument("--split", default="", help="写进 manifest 的 split（默认取 split_policy）")
    ap.add_argument("--allow-eval-source", action="store_true",
                    help="允许从 valid/test 源视频回收（会污染固定评测集，默认拒绝）")
    ap.add_argument("--dry-run", action="store_true", help="只打印会挑出什么，不写任何文件")
    args = ap.parse_args(argv)

    version_items = args.version or list(DEFAULT_VERSIONS)
    versions = ab.parse_versions(version_items)
    print("模型：%s" % "，".join("%s=%s" % (t, p) for t, p in versions))
    missing = [str(p) for _, p in versions if not Path(p).is_file()]
    if missing:
        raise SystemExit("权重不存在：%s" % "；".join(missing))
    if len(versions) == 1:
        print("  提示：只给了 1 个模型 → 只会用 within（同模型内球∩锥）判据；"
              "跨模型判据需要 2 个（base 判球 + 三分类判锥）。")

    policy = s1.load_policy()
    known = s1.existing_images()
    # dry-run 也要真推理（否则永远挑不出帧，"预览"就成了假绿），它只是不写文件。
    # 先把目标与 split 定下来并过安全闸门——**放在加载权重之前**，免得白等十几秒才被拒。
    targets: list[dict[str, Any]] = []
    for raw in args.video:
        path = Path(raw).expanduser()
        if not path.is_file():
            path = base_layout.DATA_ROOT / raw
        if not path.is_file():
            raise SystemExit("找不到视频：%s" % raw)
        try:
            plan = policy.get(str(path.relative_to(base_layout.DATA_ROOT)))
        except ValueError:
            plan = None
        if plan is None:
            # 策略表没列到：按 s1 的默认（train / unknown），但必须说清楚
            plan_split, scene, view, why = (args.split or "train"), "unknown", "unknown", "not-in-split-policy"
            print("  !! %s 不在 split_policy.json 里 → 默认 split=train、scene/view=unknown" % path.name)
        else:
            plan_split, scene, view, why = plan.split, plan.scene, plan.view, plan.why
        split = args.split or plan_split
        if split != "train" and not args.allow_eval_source:
            raise SystemExit(
                "%s 属于固定评测集（split=%s）→ 回收进它会污染评测，已拒绝。"
                "确要覆盖请显式 --allow-eval-source（或给它换一段不属于评测的视频）。" % (path.name, split))
        targets.append({"path": path, "stem": path.stem, "split": split,
                        "scene": scene, "view": view, "why": why})

    # dry-run 也要真推理（否则永远挑不出帧，预览就成了假绿），它只是不写文件。
    models: list[tuple[str, Any]] = []
    for tag, ckpt in versions:
        models.append((tag, rd.load_model(ckpt, args.resolution)))
    print("加载模型完成（dry-run：只推理不落盘）" if args.dry_run else "加载模型完成")

    WORK_RECYCLE.mkdir(parents=True, exist_ok=True)
    report: list[str] = ["# 误检帧回收（锥被当球）", "",
                         "生成时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"), "",
                         "判据：`cross`（A 判球、B 判锥）+ `within`（同模型球∩锥 / 球心落锥框内），"
                         "IoU 阈值 %.2f；选帧按段取代表 + %d×%d dHash 去重（汉明 < %d 视为重复）。"
                         % (args.overlap_iou, args.dhash_size, args.dhash_size, args.min_dhash_delta), ""]
    all_rows: list[dict[str, Any]] = []
    for target in targets:
        path, stem = target["path"], target["stem"]
        split, scene, view, why = target["split"], target["scene"], target["view"], target["why"]

        cands, stats = scan_video(path, models, args)
        picked = segment_pick(cands, args.segment_gap, args.per_segment)[:max(0, args.max_frames)]
        # dHash 去重（需要真读到帧）
        kept: list[dict[str, Any]] = []
        if picked:
            cap = cv2.VideoCapture(str(path))
            seen: list[int] = []
            for item in picked:
                cap.set(cv2.CAP_PROP_POS_FRAMES, item["frame"])
                ok, frame = cap.read()
                if not ok:
                    continue
                if dhash_ok(frame, seen, args.dhash_size, args.min_dhash_delta):
                    kept.append(item)
            cap.release()
        dup = len(picked) - len(kept)

        rows, coco = export_frames(path, kept, stem, split, why, scene, view, args)
        fresh = [r for r in rows if r["image"] not in known]
        for r in fresh:
            known.add(r["image"])
        if not args.dry_run and fresh:
            s1.append_manifest(fresh)
        all_rows.extend(fresh)

        preannot_out = WORK_RECYCLE / ("%s__recycle_preannot.json" % stem)
        if not args.dry_run:
            preannot_out.write_text(json.dumps(coco, ensure_ascii=False, indent=1) + "\n",
                                    encoding="utf-8")
            (WORK_RECYCLE / ("%s.json" % stem)).write_text(
                json.dumps({"stats": stats, "split": split, "scene": scene, "view": view,
                            "picked": [{"frame": c["frame"], "time_sec": c["time_sec"],
                                        "score": c["score"],
                                        "hits": [{"kind": h_["kind"], "a": h_["a"], "b": h_["b"],
                                                  "why": h_["why"]} for h_ in c["hits"]]}
                                       for c in kept]},
                           ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        report += ["## %s（split=%s，%s %s）" % (stem, split, stats["resolution"], "%g fps" % stats["fps"]),
                   "",
                   "| 项 | 值 |", "| --- | --- |",
                   "| 全片帧数 / 扫描帧数 | %d / %d（stride=%d） |" % (stats["frames_total"], stats["frames_scanned"], stats["stride"]),
                   "| 命中候选帧 | %d |" % stats["candidate_frames"],
                   "| 分段取代表 | %d |" % len(picked),
                   "| dHash 去重丢弃 | %d |" % dup,
                   "| **本次回收帧** | **%d** |" % len(fresh),
                   "| 预标注 | `%s` |" % preannot_out.relative_to(base_layout.DATASET_ROOT),
                   ""]
        print("  %-24s 候选 %4d → 代表 %3d → 去重后 %3d → 新登记 %3d  （split=%s）"
              % (stem, stats["candidate_frames"], len(picked), len(kept), len(fresh), split))

    report += ["## 下一步", "",
               "```bash",
               "$PY s3_label_review_roundtrip.py --import %s/<视频>__recycle_preannot.json" % WORK_RECYCLE.name,
               "$PY s4_train_finetune.py --config train_configs/full_960.yaml   # 回灌后重训",
               "```", ""]
    if not args.dry_run:
        out = base_layout.REPORTS_DIR / "recycle_hard_examples.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(report) + "\n", encoding="utf-8")
        print("\n→ 报告 %s" % out)
        print("→ 预标注目录 %s" % WORK_RECYCLE)
    else:
        print("\n（dry-run：未写任何文件）")
    print("合计回收 %d 帧" % len(all_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
