#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测级 A/B：base（COCO）vs 微调（person/ball/cone）同帧对照。**只读数据**，产物写仓库 runs/。

输出三块：
1. 总览 COCO 12 项；
2. **按类 × 目标大小**（all/small/medium/large）—— 看"小目标 mAP 低"卡在哪个类；
3. **主指标**：锥被判成球（预测 ball 落在 cone GT 上且未命中任何 ball GT）的帧数/框数，
   外加 ball 帧级 P/R 与"无球帧却出球"比例。
类别映射**按名字**（person / ball·sports ball / cone·traffic cone），模型其它类丢弃。

    $PY s5_eval_detector.py --split test \\
        --model base=models/rfdetr/rf-detr-small.pth \\
        --model v1=runs/v1/checkpoint_best_ema.pth
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import base_layout

OUR = {1: ("person",), 2: ("ball", "sports ball", "soccer ball"), 3: ("cone", "traffic cone")}
NAME = {1: "person", 2: "ball", 3: "cone"}
IDS = (1, 2, 3)
AREAS = (("all", 0.0, 1e10), ("small", 0.0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2),
         ("large", 96 ** 2, 1e10))
CONE_IOU, BALL_IOU = 0.30, 0.50


def our_id_of(class_names: Any) -> dict[int, int]:
    """{模型id: 名字} → {模型id: 本项目 1/2/3}，按名字匹配。"""
    out: dict[int, int] = {}
    for raw, name in dict(class_names).items():
        low = str(name).strip().lower()
        for oid, aliases in OUR.items():
            if low in aliases:
                out[int(raw)] = oid
    return out


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_gt(split: str) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    path = base_layout.DATASET_DIR / split / "_annotations.coco.json"
    if not path.exists():
        raise SystemExit("找不到 %s（先跑 s3_label_convert_dataset.py）" % path)
    data = json.loads(path.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann)
    return data, by_image


def infer(model: Any, gt: dict[str, Any], split: str, limit: int | None, thr:
          float) -> list[dict[str, Any]]:
    from PIL import Image

    mapping = our_id_of(model.class_names)
    if not mapping:
        raise SystemExit("模型类别名对不上 person/ball/cone：%s" % model.class_names)
    images = sorted(gt["images"], key=lambda i: i["file_name"])[: limit or None]
    results, t0 = [], time.time()
    for i, img in enumerate(images, 1):
        with Image.open(base_layout.DATASET_DIR / split / img["file_name"]) as fh:
            det = model.predict(fh.convert("RGB"), threshold=thr)
        for box, score, label in zip(det.xyxy, det.confidence, det.class_id):
            oid = mapping.get(int(label))
            if oid is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            results.append({"image_id": img["id"], "category_id": oid,
                            "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                            "score": float(score)})
        if i % 25 == 0:
            print("    已推理 %d/%d（%.0fs）" % (i, len(images), time.time() - t0), flush=True)
    return results


def coco_metrics(gt: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    out: dict[str, Any] = {"overall": {}, "per_class": {}, "per_class_area": {}}
    if not results:
        return out
    g = COCO()
    g.dataset = gt
    g.createIndex()
    ev = COCOeval(g, g.loadRes(results), iouType="bbox")
    ev.params.maxDets = [1, 10, 100]

    def run(cats: list[int] | None = None, area: tuple[float, float] | None = None) -> dict[str, float]:
        import contextlib
        import io

        ev.params.catIds = cats or list(IDS)
        ev.params.areaRng = [list(area)] if area else [[0.0, 1e10]]
        # 注意：summarize() 只认 label == 'all'，限制某一档面积要靠 areaRng 本身，
        # 不能把 label 改成别的（否则 stats 全是 -1）
        ev.params.areaRngLbl = ["all"]
        # pycocotools 2.0.11 的 summarize() 不收 print_fn → 直接吞掉它的打印（每次 re-eval 都会刷屏）
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        return {"mAP@.5:.95": round(float(ev.stats[0]), 4), "mAP@.5": round(float(ev.stats[1]), 4),
                "mAP@.75": round(float(ev.stats[2]), 4), "AR@100": round(float(ev.stats[8]), 4)}

    out["overall"] = run()
    for cid in IDS:
        if not any(a["category_id"] == cid for a in gt["annotations"]):
            zero = {"mAP@.5:.95": 0.0, "mAP@.5": 0.0, "mAP@.75": 0.0, "AR@100": 0.0}
            out["per_class"][NAME[cid]] = dict(zero)
            out["per_class_area"][NAME[cid]] = {lab: dict(zero) for lab, _, _ in AREAS}
            continue
        out["per_class"][NAME[cid]] = run([cid])
        out["per_class_area"][NAME[cid]] = {lab: run([cid], (lo, hi)) for lab, lo, hi in AREAS}
    return out


def scene_of_images() -> dict[str, str]:
    """从 manifest.csv 读 {file_name: scene}（按文件名匹配，路径前缀差异都能对上）。"""
    import csv as _csv

    if not base_layout.MANIFEST_CSV.exists():
        return {}
    out: dict[str, str] = {}
    with base_layout.MANIFEST_CSV.open(encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            name = str(row.get("image", "")).replace("\\", "/").split("/")[-1]
            if name:
                out[name] = str(row.get("scene", "") or "unknown")
    return out



def box_prf(images: list[dict[str, Any]], by_image: dict[int, list[dict[str, Any]]],
            preds: dict[int, list[dict[str, Any]]], cat: int, iou_thr: float = 0.5,
            score_thr: float = 0.25) -> tuple[float, float, float]:
    """按类的框级 P/R/F1（贪心 IoU 匹配）。

    为什么不能只看帧级：本评测集上"帧级 ball P/R"在四个模型上完全相同——17 个无球帧里
    **每个模型都至少出一个球框**，帧级口径只看"有没有"，区分不了"出几个/错多少"。
    框级口径才会随误检数量变化，才是选部署版该看的数。
    """
    tp = fp = fn = 0
    for img in images:
        gts = [a for a in by_image.get(img["id"], []) if a["category_id"] == cat]
        ps = [r for r in preds.get(img["id"], [])
              if r["category_id"] == cat and float(r.get("score", 1.0)) >= score_thr]
        used: set[int] = set()
        for g in gts:
            best, best_i = 0.0, -1
            for i, pr in enumerate(ps):
                if i in used:
                    continue
                v = iou(g["bbox"], pr["bbox"])
                if v > best:
                    best, best_i = v, i
            if best_i >= 0 and best >= iou_thr:
                used.add(best_i)
                tp += 1
            else:
                fn += 1
        fp += len(ps) - len(used)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def behavior(gt: dict[str, Any], by_image: dict[int, list[dict[str, Any]]],
             results: list[dict[str, Any]], limit: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preds: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in results:
        preds[r["image_id"]].append(r)
    images = sorted(gt["images"], key=lambda i: i["file_name"])[: limit or None]
    ca_frames = ca_boxes = with_cone = tp = fp = fn = empty_fp = 0
    ca_boxes_dup_cone = ca_frames_dup_cone = 0
    rows = []
    for img in images:
        anns = by_image.get(img["id"], [])
        g_cone = [a for a in anns if a["category_id"] == 3]
        g_ball = [a for a in anns if a["category_id"] == 2]
        p_all = preds.get(img["id"], [])
        p_ball = [r for r in p_all if r["category_id"] == 2]
        # 每个预测球框：找 IoU 最大的那个 GT，看它到底是谁 → 定性为"球/锥/背景"
        hit, wrong = [], []
        for p in p_ball:
            best_iou, best_cat = 0.0, 0
            for g in anns:
                v = iou(p["bbox"], g["bbox"])
                if v > best_iou:
                    best_iou, best_cat = v, g["category_id"]
            if best_cat == 2 and best_iou >= BALL_IOU:
                hit.append(p)
            elif best_cat == 3 and best_iou >= CONE_IOU:
                wrong.append(p)
        p_cone = [r for r in p_all if r["category_id"] == 3]
        dup = [p for p in wrong if any(iou(p["bbox"], c["bbox"]) >= 0.5 for c in p_cone)]
        with_cone += 1 if g_cone else 0
        ca_frames += 1 if wrong else 0
        ca_boxes += len(wrong)
        ca_boxes_dup_cone += len(dup)
        ca_frames_dup_cone += 1 if dup else 0
        if g_ball and hit:
            tp += 1
        elif g_ball:
            fn += 1
            fp += 1 if p_ball else 0
        elif p_ball:
            fp += 1
            empty_fp += 1
        rows.append({"image": img["file_name"],
                     "gt_person": sum(1 for a in anns if a["category_id"] == 1),
                     "gt_ball": len(g_ball), "gt_cone": len(g_cone),
                     "pred_person": sum(1 for r in p_all if r["category_id"] == 1),
                     "pred_ball": len(p_ball),
                     "pred_cone": sum(1 for r in p_all if r["category_id"] == 3),
                     "ball_hit": int(bool(hit)), "cone_as_ball": len(wrong),
                     "max_ball_score": round(max((p["score"] for p in p_ball), default=0.0), 3)})
    scenes = scene_of_images()
    by_scene: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_scene[scenes.get(row["image"].split("/")[-1], "unknown")].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        cone_frames = sum(1 for r in group if r["gt_cone"])
        wrong_frames = sum(1 for r in group if r["cone_as_ball"])
        gb = sum(1 for r in group if r["gt_ball"])
        hit = sum(1 for r in group if r["gt_ball"] and r["ball_hit"])
        return {"frames": len(group), "cone_as_ball_frames": wrong_frames,
                "cone_as_ball_boxes": sum(r["cone_as_ball"] for r in group),
                "frames_with_cone_gt": cone_frames,
                "cone_as_ball_frame_ratio": round(wrong_frames / cone_frames, 3) if cone_frames else 0.0,
                "ball_frame_recall": round(hit / gb, 3) if gb else 0.0}

    n = len(images)
    # 框级 P/R/F1（按类）：帧级口径在本题不区分模型（17 个无球帧上各模型都至少出一个球框），
    # 框级才会随误检数量变化，是选部署版该看的数。
    box: dict[str, float] = {}
    for cat, name in ((1, "person"), (2, "ball"), (3, "cone")):
        pr, rc, f1 = box_prf(images, by_image, preds, cat)
        box["%s_box_P" % name] = pr
        box["%s_box_R" % name] = rc
        box["%s_box_F1" % name] = f1
    m = {"frames": n, "frames_with_cone_gt": with_cone, "cone_as_ball_frames": ca_frames,
         "cone_as_ball_boxes": ca_boxes,
         "cone_as_ball_frame_ratio": round(ca_frames / with_cone, 4) if with_cone else 0.0,
         # 这些框同时有 cone 预测 → 管线里"球候选与 cone 框重叠则抑制"能直接救回
         "cone_as_ball_boxes_with_cone_pred": ca_boxes_dup_cone,
         "cone_as_ball_frames_with_cone_pred": ca_frames_dup_cone,
         "ball_frame_tp": tp, "ball_frame_fp": fp, "ball_frame_fn": fn,
         "ball_frame_precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
         "ball_frame_recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
         "ball_frame_f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else 0.0,
         "empty_ball_fp_frames": empty_fp,
         "empty_ball_fp_ratio": round(empty_fp / n, 4) if n else 0.0,
         **box,
         "by_scene": {scene: summarize(group) for scene, group in sorted(by_scene.items())}}
    return m, rows


def run_model(tag: str, ckpt: str, split: str, res: int, limit: int | None, thr: float,
              out_dir: Path) -> dict[str, Any]:
    from rfdetr import RFDETRSmall
    import torch

    gt, by_image = load_gt(split)
    print("  [%s] %s（resolution=%d，%d 图）" % (tag, ckpt, res, len(gt["images"])))
    model = RFDETRSmall(pretrain_weights=ckpt, resolution=res, device="cuda")
    try:
        results = infer(model, gt, split, limit, thr)
    finally:
        del model
        torch.cuda.empty_cache()
    metrics = {"tag": tag, "checkpoint": ckpt, "split": split, "resolution": res,
               "threshold": thr, "n_predictions": len(results)}
    metrics["coco"] = coco_metrics(gt, results)
    metrics["behavior"], rows = behavior(gt, by_image, results, limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("metrics_%s.json" % tag)).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out_dir / ("per_frame_%s.csv" % tag)).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["image"])
        w.writeheader()
        w.writerows(rows)
    o, b = metrics["coco"]["overall"], metrics["behavior"]
    print("  [%s] mAP@.5=%.3f  mAP@.5:.95=%.3f | 锥判成球 %d 帧 | ball 帧级 P/R=%.3f/%.3f"
          % (tag, o["mAP@.5"], o["mAP@.5:.95"], b["cone_as_ball_frames"],
             b["ball_frame_precision"], b["ball_frame_recall"]))
    return metrics


def per_frame_diff(allm: list[dict[str, Any]], out_dir: Path) -> Path | None:
    """两张逐帧表相减：哪些帧从"锥判成球"变好了、哪些变差、哪些球丢了。"""
    if len(allm) < 2:
        return None
    a, b = allm[0]["tag"], allm[-1]["tag"]
    fa, fb = out_dir / ("per_frame_%s.csv" % a), out_dir / ("per_frame_%s.csv" % b)
    if not (fa.exists() and fb.exists()):
        return None
    import csv as _csv

    def load(path: Path) -> dict[str, dict[str, str]]:
        with path.open(encoding="utf-8") as fh:
            return {r["image"]: r for r in _csv.DictReader(fh)}

    rows_a, rows_b = load(fa), load(fb)
    out_rows = []
    for image in sorted(set(rows_a) & set(rows_b)):
        ra, rb = rows_a[image], rows_b[image]
        before, after = int(ra["cone_as_ball"]), int(rb["cone_as_ball"])
        out_rows.append({"image": image,
                         "gt_ball": ra["gt_ball"], "gt_cone": ra["gt_cone"],
                         "%s_pred_ball" % a: ra["pred_ball"], "%s_pred_cone" % a: ra["pred_cone"],
                         "%s_cone_as_ball" % a: before,
                         "%s_pred_ball" % b: rb["pred_ball"], "%s_pred_cone" % b: rb["pred_cone"],
                         "%s_cone_as_ball" % b: after,
                         "delta_cone_as_ball": after - before,
                         "ball_hit_%s" % a: ra["ball_hit"], "ball_hit_%s" % b: rb["ball_hit"],
                         "ball_lost": int(ra["ball_hit"] == "1" and rb["ball_hit"] == "0")})
    path = out_dir / ("per_frame_diff_%s_vs_%s.csv" % (a, b))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    improved = sum(1 for r in out_rows if r["delta_cone_as_ball"] < 0)
    worse = sum(1 for r in out_rows if r["delta_cone_as_ball"] > 0)
    lost = sum(r["ball_lost"] for r in out_rows)
    print("\n逐帧差异：%s → %s 变好 %d 帧 / 变差 %d 帧 / 球丢检 %d 帧 → %s"
          % (a, b, improved, worse, lost, path))
    return path


def report(allm: list[dict[str, Any]], out_dir: Path) -> Path:
    tags = [m["tag"] for m in allm]
    L = ["# 检测级 A/B（split=%s，resolution=%s）—— 数据只读" % (allm[0]["split"], allm[0]["resolution"]),
         "", "| 指标 | " + " | ".join(tags) + " |", "| --- |" + " --- |" * len(tags)]

    def row(label: str, fn: Any) -> None:
        vals = []
        for m in allm:
            v = fn(m)
            if isinstance(v, float) and v < 0:
                vals.append("—")
            else:
                vals.append("%.3f" % v if isinstance(v, float) else str(v))
        L.append("| %s | %s |" % (label, " | ".join(vals)))

    for k in ("mAP@.5", "mAP@.5:.95", "mAP@.75", "AR@100"):
        row("总览 " + k, lambda m, k=k: m["coco"]["overall"][k])
    L += ["", "## 按类 × 目标大小（mAP@.5:.95）", "| 类 / 大小 | " + " | ".join(tags) + " |",
          "| --- |" + " --- |" * len(tags)]
    for cls in ("person", "ball", "cone"):
        for area in ("all", "small", "medium", "large"):
            row("%s / %s" % (cls, area),
                lambda m, c=cls, a=area: m["coco"]["per_class_area"][c][a]["mAP@.5:.95"])
    # mAP@.5 单独一栏：部署侧常用单阈值，且小目标差距在 .5 上比 .5:.95 更可读
    L += ["", "## 按类 × 目标大小（mAP@.5）", "| 类 / 大小 | " + " | ".join(tags) + " |",
          "| --- |" + " --- |" * len(tags)]
    for cls in ("person", "ball", "cone"):
        for area in ("all", "small"):
            row("%s / %s" % (cls, area),
                lambda m, c=cls, a=area: m["coco"]["per_class_area"][c][a]["mAP@.5"])
    L += ["", "## 主指标（锥被判成球）与 ball 帧级行为",
          "| 指标 | " + " | ".join(tags) + " |", "| --- |" + " --- |" * len(tags)]
    row("帧数", lambda m: m["behavior"]["frames"])
    row("含锥 GT 的帧", lambda m: m["behavior"]["frames_with_cone_gt"])
    row("**锥被判成球的帧数**", lambda m: m["behavior"]["cone_as_ball_frames"])
    row("锥被判成球的框数", lambda m: m["behavior"]["cone_as_ball_boxes"])
    row("锥被判成球帧占比", lambda m: m["behavior"]["cone_as_ball_frame_ratio"])
    row("其中「同帧也预测出锥」（抑制规则可救）", lambda m: m["behavior"]["cone_as_ball_frames_with_cone_pred"])
    row("抑制后残留的硬误检帧", lambda m: m["behavior"]["cone_as_ball_frames"]
        - m["behavior"]["cone_as_ball_frames_with_cone_pred"])
    row("ball 帧级 P", lambda m: m["behavior"]["ball_frame_precision"])
    row("ball 帧级 R", lambda m: m["behavior"]["ball_frame_recall"])
    row("ball 帧级 F1", lambda m: m["behavior"]["ball_frame_f1"])
    row("无球帧却出球（占比）", lambda m: m["behavior"]["empty_ball_fp_ratio"])
    L += ["", "## 框级 P / R / F1（IoU 0.5，score ≥ 0.25 —— 帧级口径在本题不区分模型，见代码注释）",
          "| 类 / 指标 | " + " | ".join(tags) + " |", "| --- |" + " --- |" * len(tags)]
    for cls in ("person", "ball", "cone"):
        for k, label in (("P", "精确率"), ("R", "召回"), ("F1", "F1")):
            key = "%s_box_%s" % (cls, k)
            row("%s %s" % (cls, label), lambda m, key=key: m["behavior"].get(key, -1.0))
    scenes = sorted({s for m in allm for s in m["behavior"].get("by_scene", {})})
    if scenes:
        L += ["", "## 按场景分层（锥被判成球 / ball 帧级召回）",
              "| 场景 | 指标 | " + " | ".join(tags) + " |",
              "| --- | --- |" + " --- |" * len(tags)]
        for scene in scenes:
            for key, label in (("cone_as_ball_frames", "锥判成球帧"),
                               ("frames_with_cone_gt", "含锥帧"),
                               ("ball_frame_recall", "ball 帧召回")):
                vals = []
                for m in allm:
                    v = m["behavior"].get("by_scene", {}).get(scene, {}).get(key)
                    vals.append("—" if v is None else ("%.3f" % v if isinstance(v, float) else str(v)))
                L.append("| %s | %s | %s |" % (scene, label, " | ".join(vals)))
    if len(allm) >= 2:
        f, l = allm[0]["behavior"], allm[-1]["behavior"]
        d = f["cone_as_ball_frames"] - l["cone_as_ball_frames"]
        L += ["", "**主指标变化（%s → %s）**：锥被判成球 %d → %d 帧（%+.1f%%）；ball 帧级 R %+.3f；"
                  "无球帧却出球 %+.3f" % (tags[0], tags[-1], f["cone_as_ball_frames"],
                                          l["cone_as_ball_frames"],
                                          -100.0 * d / f["cone_as_ball_frames"] if f["cone_as_ball_frames"] else 0.0,
                                          l["ball_frame_recall"] - f["ball_frame_recall"],
                                          l["empty_ball_fp_ratio"] - f["empty_ball_fp_ratio"])]
    per_frame_diff(allm, out_dir)
    path = out_dir / "compare.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    (out_dir / "compare.json").write_text(json.dumps(allm, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    return path


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检测级 A/B（base vs 微调）")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--model", action="append", required=True, metavar="TAG=PATH")
    ap.add_argument("--resolution", type=int, default=960)
    ap.add_argument("--threshold", type=float, default=0.05, help="收集预测的下限（AP 靠排序，别设高）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output-dir", default=str(Path(__file__).parent / "runs" / "eval_ab"))
    a = ap.parse_args(list(argv) if argv is not None else None)
    specs = []
    for item in a.model:
        if "=" not in item:
            raise SystemExit("--model 要写 TAG=PATH：%s" % item)
        tag, path = item.split("=", 1)
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).parent / path
        if not p.exists():
            raise SystemExit("checkpoint 不存在：%s" % p)
        specs.append((tag, str(p)))
    print("评测 split=%s resolution=%d thr=%.2f 模型=%s" % (a.split, a.resolution, a.threshold,
                                                          [t for t, _ in specs]))
    out = Path(a.output_dir)
    allm = [run_model(t, p, a.split, a.resolution, a.limit or None, a.threshold, out)
            for t, p in specs]
    path = report(allm, out)
    print("\n对比报告 → %s\n" % path)
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
