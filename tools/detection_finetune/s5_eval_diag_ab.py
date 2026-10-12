#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回归诊断：v1 的 person/ball 到底"真掉点"还是"框变松 / GT 自评偏高"。

为什么要单独一个脚本：A/B 表里只看得到 mAP（聚合指标），分不清三种原因：
  (a) 漏检变多（recall 掉）；(b) 框变松（IoU 掉，recall 还在）；(c) GT 本身偏向 base（自评）。
本脚本按类给出：GT 数 / 预测数 / 命中数@0.5 / 命中数@0.75 / 平均 IoU / 漏检数，
并可选生成**三方对照图**（GT 红 · base 绿 · v1 蓝）供人工一眼判断。

只读数据；图与报告写到 `--out-dir`（默认仓库 runs/eval_ab/diag/）。

    $PY s5_eval_diag_ab.py --split test \\
        --model base=models/rfdetr/rf-detr-small.pth \\
        --model v1=runs/v1/checkpoint_best_ema.pth --overlays 6
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import s5_eval_detector as ed
import base_layout

CLASS_NAME = {1: "person", 2: "ball", 3: "cone"}


def stats_for(gt: dict[str, Any], by_image: dict[int, list[dict[str, Any]]],
              results: list[dict[str, Any]], limit: int | None) -> dict[str, Any]:
    """按类统计：召回@0.5 / @0.75、平均 IoU、漏检、预测数。"""
    preds: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in results:
        preds[r["image_id"]].append(r)
    images = sorted(gt["images"], key=lambda i: i["file_name"])[: limit or None]
    out: dict[str, Any] = {}
    for cid, name in CLASS_NAME.items():
        n_gt = hit50 = hit75 = missed = 0
        ious: list[float] = []
        n_pred = 0
        for img in images:
            gts = [a for a in by_image.get(img["id"], []) if a["category_id"] == cid]
            ps = [r for r in preds.get(img["id"], []) if r["category_id"] == cid]
            n_gt += len(gts)
            n_pred += len(ps)
            for g in gts:
                best = max((ed.iou(p["bbox"], g["bbox"]) for p in ps), default=0.0)
                ious.append(best)
                if best >= 0.5:
                    hit50 += 1
                if best >= 0.75:
                    hit75 += 1
                if best < 0.5:
                    missed += 1
        ious.sort()
        # 再按 GT 面积分档（<32²/中/大）看"小目标是否被漏"，避免聚合数字掩盖
        for label, lo, hi in (("small", 0.0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2),
                              ("large", 96 ** 2, 1e10)):
            g_all = h50 = 0
            acc: list[float] = []
            for img in images:
                gts = [a for a in by_image.get(img["id"], [])
                       if a["category_id"] == cid and lo <= a.get("area", 0) < hi]
                ps = [r for r in preds.get(img["id"], []) if r["category_id"] == cid]
                g_all += len(gts)
                for g in gts:
                    best = max((ed.iou(p["bbox"], g["bbox"]) for p in ps), default=0.0)
                    acc.append(best)
                    h50 += best >= 0.5
            if g_all:
                out.setdefault(name + "_by_size", {})[label] = {
                    "gt": g_all, "recall@0.5": round(h50 / g_all, 3),
                    "mean_best_iou": round(sum(acc) / len(acc), 3)}
        out[name] = {"gt": n_gt, "pred": n_pred,
                     "recall@0.5": round(hit50 / n_gt, 3) if n_gt else 0.0,
                     "recall@0.75": round(hit75 / n_gt, 3) if n_gt else 0.0,
                     "missed@0.5": missed,
                     "mean_best_iou": round(sum(ious) / len(ious), 3) if ious else 0.0,
                     "median_best_iou": round(ious[len(ious) // 2], 3) if ious else 0.0,
                     "iou_lt_0.75_share": round(sum(1 for v in ious if 0.5 <= v < 0.75) / n_gt, 3)
                     if n_gt else 0.0}
    return out


def draw_compare(split: str, gt: dict[str, Any], by_image: dict[int, list[dict[str, Any]]],
                 per_model: list[tuple[str, list[dict[str, Any]]]], out_dir: Path,
                 n: int) -> list[Path]:
    """三方对照图：GT 红框 / 各模型绿·蓝·橙框，左上角烧文件名，便于人工判谁准。"""
    import cv2
    import numpy as np

    colors = [(0, 200, 0), (255, 120, 0), (0, 165, 255), (200, 0, 200)]
    preds = [{r["image_id"]: collections.defaultdict(list) for r in []} for _ in per_model]
    preds = []
    for _, results in per_model:
        by_img: dict[int, dict[int, list[dict[str, Any]]]] = collections.defaultdict(
            lambda: collections.defaultdict(list))
        for r in results:
            by_img[r["image_id"]][r["category_id"]].append(r)
        preds.append(by_img)

    # 选"人框最多"的 N 张：正是要判 person 的场景
    images = sorted(gt["images"], key=lambda i: i["file_name"])
    ranked = sorted(images, key=lambda i: -sum(1 for a in by_image.get(i["id"], [])
                                               if a["category_id"] == 1))
    paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in ranked[:n]:
        path = base_layout.DATASET_DIR / split / img["file_name"]
        canvas = cv2.imread(str(path))
        if canvas is None:
            continue
        panels: list[Any] = []
        for index, (tag, _) in enumerate(per_model):
            panel = canvas.copy()
            for cid, color in ((3, (0, 0, 255)), (1, colors[index % len(colors)]),
                               (2, (255, 0, 0))):
                for r in preds[index].get(img["id"], {}).get(cid, []):
                    if r["score"] < 0.35:
                        continue
                    x, y, w, h = (int(v) for v in r["bbox"])
                    cv2.rectangle(panel, (x, y), (x + w, y + h), color, 2)
            cv2.putText(panel, "pred: %s" % tag, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            panels.append(panel)
        gt_panel = canvas.copy()
        for a in by_image.get(img["id"], []):
            x, y, w, h = (int(v) for v in a["bbox"])
            color = {1: (0, 255, 255), 2: (255, 0, 0), 3: (0, 0, 255)}[a["category_id"]]
            cv2.rectangle(gt_panel, (x, y), (x + w, y + h), color, 2)
        cv2.putText(gt_panel, "GT (yellow=person red=cone blue=ball)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        panels = [gt_panel] + panels
        strip = np.hstack([cv2.resize(p, (panels[0].shape[1], panels[0].shape[0])) for p in panels])
        out = out_dir / ("overlay_%s" % Path(img["file_name"]).name)
        cv2.imwrite(str(out), strip)
        paths.append(out)
    return paths


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="person/ball 回归诊断 + 三方对照图")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", action="append", required=True, metavar="TAG=PATH")
    ap.add_argument("--resolution", type=int, default=960)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overlays", type=int, default=0, help="生成几张三方对照图")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "runs" / "eval_ab" / "diag"))
    a = ap.parse_args(list(argv) if argv is not None else None)

    gt, by_image = ed.load_gt(a.split)
    limit = a.limit or None
    per_model: list[tuple[str, list[dict[str, Any]]]] = []
    stats: dict[str, Any] = {}
    for item in a.model:
        tag, path = item.split("=", 1)
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).parent / path
        print("[%s] 推理中 %s" % (tag, p))
        from rfdetr import RFDETRSmall

        model = RFDETRSmall(pretrain_weights=str(p), resolution=a.resolution, device="cuda")
        try:
            results = ed.infer(model, gt, a.split, limit, a.threshold)
        finally:
            import torch
            del model
            torch.cuda.empty_cache()
        per_model.append((tag, results))
        stats[tag] = stats_for(gt, by_image, results, limit)

    tags = list(stats)
    print("\n## 按类诊断（split=%s，GT 数 / 预测数 / 召回 / 平均 IoU）" % a.split)
    print("| 类 | 指标 | " + " | ".join(tags) + " |")
    print("| --- | --- |" + " --- |" * len(tags))
    for name in ("person", "ball", "cone"):
        for key in ("gt", "pred", "recall@0.5", "recall@0.75", "missed@0.5",
                    "mean_best_iou", "iou_lt_0.75_share"):
            print("| %s | %s | %s |" % (name, key, " | ".join(str(stats[t][name][key])
                                                              for t in tags)))
    print("\n## 按目标大小分层（recall@0.5 / 平均 IoU）")
    print("| 类 | 档 | " + " | ".join(tags) + " |")
    print("| --- | --- |" + " --- |" * len(tags))
    for name in ("person", "ball", "cone"):
        for label in ("small", "medium", "large"):
            vals = []
            for t_ in tags:
                got = stats[t_].get(name + "_by_size", {}).get(label)
                vals.append("—" if not got else "%d 框 · R %.3f · IoU %.3f"
                            % (got["gt"], got["recall@0.5"], got["mean_best_iou"]))
            print("| %s | %s | %s |" % (name, label, " | ".join(vals)))
    if len(tags) >= 2:
        print("\n## 判读")
        for name in ("person", "ball", "cone"):
            a0, a1 = stats[tags[0]][name], stats[tags[-1]][name]
            d_rec = a1["recall@0.5"] - a0["recall@0.5"]
            d_iou = a1["mean_best_iou"] - a0["mean_best_iou"]
            verdict = ("漏检变多（真掉点）" if d_rec < -0.02
                       else "框变松（定位退化）" if d_iou < -0.02
                       else "基本持平")
            print("  %-7s 召回 %+.3f｜平均 IoU %+.3f → %s" % (name, d_rec, d_iou, verdict))

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ("diag_%s.json" % a.split)).write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.overlays:
        paths = draw_compare(a.split, gt, by_image, per_model, out_dir, a.overlays)
        print("\n三方对照图（%d 张）→ %s" % (len(paths), out_dir))
        for p in paths:
            print("   ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
