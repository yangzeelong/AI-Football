#!/usr/bin/env python3
"""数据体检：类别/尺度/分辨率/来源/空标注 分层统计，产出 Markdown + JSON 报告。

体检报告是训练前的"体检单"，也是交付报告里的数据台账；它同时做三件检查：
  * 三个 split 的 `categories` 必须与 `classes.json` 对拍（`base_layout.assert_split_annotations`）；
  * 合成样本占比不得超过 30%（计划里的硬约束）；
  * `test` 集里不许出现来源不明确的采集样本（许可风险）。

用法：
    python s3_label_dataset_stats.py                    # 体检当前数据集
    python s3_label_dataset_stats.py --json            # 只输出 JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import base_layout

#: 目标在 960 输入下的像素尺寸分箱（部署侧 letterbox 到 960 方图后的大小）
SIZE_BUCKETS = ((0, 12), (12, 24), (24, 48), (48, 96), (96, 192), (192, 10 ** 6))


def size_bucket(w: float, h: float) -> str:
    longest = max(w, h)
    for lo, hi in SIZE_BUCKETS:
        if lo <= longest < hi:
            return f"{lo}-{hi if hi < 10 ** 6 else 'inf'}px"
    return "unknown"


def collect() -> dict[str, Any]:
    manifest = {r["image"]: r for r in csv.DictReader(base_layout.MANIFEST_CSV.open(encoding="utf-8"))}
    report: dict[str, Any] = {"splits": {}, "totals": {}}
    per_class_total: Counter = Counter()
    per_split_class: dict[str, Counter] = {}
    for split in base_layout.SPLITS:
        ann_path = base_layout.split_annotations(split)
        if not ann_path.exists():
            report["splits"][split] = {"images": 0, "note": "缺 _annotations.coco.json"}
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        id2name = base_layout.id_to_name()
        class_counts: Counter = Counter()
        size_hist: Counter = Counter()
        resolution: Counter = Counter()
        source_hist: Counter = Counter()
        scene_hist: Counter = Counter()
        empty_ball = 0
        per_image_classes: dict[int, set[int]] = defaultdict(set)
        for ann in data["annotations"]:
            name = id2name[ann["category_id"]]
            class_counts[name] += 1
            per_class_total[name] += 1
            _, _, w, h = ann["bbox"]
            size_hist[size_bucket(w, h)] += 1
            per_image_classes[ann["image_id"]].add(ann["category_id"])
        for img in data["images"]:
            resolution[f"{img['width']}x{img['height']}"] += 1
            # 清单里存的是"相对 ball_cone_v1 的原始路径"（raw/… 或 synthetic/…），
            # 而 split 里的 file_name 是 images/<basename> → 必须用 source_image 反查
            row = manifest.get(img.get("source_image") or img["file_name"], {})
            source_hist[row.get("source", "unknown")] += 1
            scene_hist[row.get("scene", "unknown")] += 1
            classes = per_image_classes.get(img["id"], set())
            if base_layout.name_to_id()["ball"] not in classes and base_layout.name_to_id()["cone"] not in classes:
                empty_ball += 1
        per_split_class[split] = class_counts
        report["splits"][split] = {
            "images": len(data["images"]), "annotations": len(data["annotations"]),
            "class_counts": dict(class_counts), "size_hist": dict(size_hist),
            "resolution": dict(resolution), "source": dict(source_hist),
            "scene": dict(scene_hist), "images_without_ball_or_cone": empty_ball,
        }
    report["totals"] = {
        "images": sum(v.get("images", 0) for v in report["splits"].values()),
        "annotations": sum(v.get("annotations", 0) for v in report["splits"].values()),
        "class_counts": dict(per_class_total),
        "per_split_class": {k: dict(v) for k, v in per_split_class.items()},
    }
    # 检查项
    checks: list[dict[str, str]] = []
    cats = base_layout.assert_split_annotations() if report["totals"]["images"] else {}
    if cats:
        checks.append({"name": "三目录类别对拍 classes.json", "status": "pass",
                       "detail": json.dumps(cats, ensure_ascii=False)})
    synthetic = sum(sum(v.get("source", {}).get(src, 0) for src in base_layout.COMPOSITE_SOURCES)
                    for v in report["splits"].values())
    total = max(1, report["totals"]["images"])
    ratio = synthetic / total
    checks.append({"name": "合成样本占比（ai_cut+ai_bg_cut）<= 30%", "status": "pass" if ratio <= 0.30 else "fail",
                   "detail": "%.1f%%（%d/%d）" % (ratio * 100, synthetic, total)})
    test_sources = report["splits"].get("test", {}).get("source", {})
    risky = {k: v for k, v in test_sources.items() if k in {"crawl", "ai_stress"}}
    checks.append({"name": "test 集无许可风险来源", "status": "pass" if not risky else "warn",
                   "detail": json.dumps(risky or {}, ensure_ascii=False)})
    if not report["splits"].get("valid", {}).get("images"):
        checks.append({"name": "valid 非空", "status": "fail", "detail": "0 图"})
    # 过小框哨兵：bootstrap 预标注里"静止锥/碟被当球"的误检通常 3–5 px；S3 复核后应明显下降
    report["tiny_boxes_md"] = tiny_box_report(report)
    tiny_total = sum(int(part.split("|")[2].strip() or 0)
                     for part in report["tiny_boxes_md"].splitlines()
                     if part.startswith("|") and len(part.split("|")) > 3)
    checks.append({"name": "过小框（<%.0fpx@960）哨兵" % MIN_TINY_PX,
                   "status": "warn" if tiny_total else "pass",
                   "detail": "共 %d 个（明细见表）；S3 人工复核后应显著下降，仍高说明还有误检没清" % tiny_total})
    report["checks"] = checks
    return report


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# ball_cone_v1 数据体检", ""]
    lines.append("| split | 图 | 框 | person | ball | cone | 无球无锥的图 | 分辨率 | 来源 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for split, info in report["splits"].items():
        if not info.get("images"):
            lines.append(f"| {split} | 0 | 0 | - | - | - | - | - | - |")
            continue
        cc = info["class_counts"]
        lines.append("| %s | %d | %d | %d | %d | %d | %d | %s | %s |" % (
            split, info["images"], info["annotations"], cc.get("person", 0), cc.get("ball", 0),
            cc.get("cone", 0), info["images_without_ball_or_cone"],
            ", ".join(f"{k}×{v}" for k, v in info["resolution"].items()),
            ", ".join(f"{k}×{v}" for k, v in info["source"].items())))
    lines += ["", "## 目标在 960 输入下的像素尺寸分布", "",
              "| split | " + " | ".join(f"{lo}-{hi if hi < 10**6 else '∞'}px"
                                        for lo, hi in SIZE_BUCKETS) + " |",
              "| --- |" + " --- |" * len(SIZE_BUCKETS)]
    for split, info in report["splits"].items():
        if not info.get("images"):
            continue
        hist = info["size_hist"]
        keys = [f"{lo}-{hi if hi < 10 ** 6 else 'inf'}px" for lo, hi in SIZE_BUCKETS]
        lines.append("| %s | %s |" % (split, " | ".join(str(hist.get(k, 0)) for k in keys)))
    lines += ["", "## 检查项", "", "| 检查 | 结果 | 明细 |", "| --- | --- | --- |"]
    for check in report["checks"]:
        mark = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[check["status"]]
        lines.append(f"| {check['name']} | {mark} {check['status']} | {check['detail']} |")
    if report.get("tiny_boxes_md"):
        lines += ["", "## 过小框哨兵（<%.0fpx @960）" % MIN_TINY_PX, "",
                  "预标注噪声（静止锥/碟被当球）通常 3–5 px；这一节用来判断 S3 复核清干净没有。", "",
                  "| split | 过小框 | 按类别 |", "| --- | --- | --- |",
                  report["tiny_boxes_md"]]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ball_cone_v1 数据体检")
    parser.add_argument("--json", action="store_true", help="只打印 JSON")
    parser.add_argument("--out", default=None, help="Markdown 输出路径（默认 reports/dataset_stats.md）")
    return parser.parse_args(argv)


#: 小于这个尺寸（960 输入下的像素）的框在体检里单列——bootstrap 预标注里那些"静止锥/碟被当球"
#: 的误检通常就 3–5 px；S3 复核后这个数应显著下降（复核对不上就说明还有误检没清理）
MIN_TINY_PX = 5.0


def tiny_box_report(report: dict[str, Any]) -> str:
    """统计各 split 里 <MIN_TINY_PX 的框（按类别），用于判断预标注噪声清干净没有。"""
    lines = []
    for split in base_layout.SPLITS:
        path = base_layout.split_annotations(split)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in data.get("categories", [])}
        size = {img["id"]: (img.get("width") or 1280, img.get("height") or 720)
                for img in data.get("images", [])}
        counts: dict[str, int] = {}
        total = 0
        for ann in data.get("annotations", []):
            width, height = size.get(ann["image_id"], (1280, 720))
            factor = 960.0 / max(width, height)
            if min(ann["bbox"][2] * factor, ann["bbox"][3] * factor) < MIN_TINY_PX:
                total += 1
                key = names.get(ann["category_id"], "?")
                counts[key] = counts.get(key, 0) + 1
        lines.append("| %s | %d | %s |" % (split, total,
                                          ", ".join("%s %d" % kv for kv in sorted(counts.items())) or "-"))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = collect()
    base_layout.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (base_layout.REPORTS_DIR / "dataset_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    markdown = to_markdown(report)
    out = Path(args.out) if args.out else base_layout.REPORTS_DIR / "dataset_stats.md"
    out.write_text(markdown, encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        print(markdown)
        print("→ %s\n→ %s" % (out, base_layout.REPORTS_DIR / "dataset_stats.json"))
    failed = [c for c in report["checks"] if c["status"] == "fail"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
