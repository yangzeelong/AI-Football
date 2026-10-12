#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""标注规范对照示例图（`spec_*`）+ 分布外压力集（`stress_*`）——**从真实帧派生**。

背景：原本这两块打算用 `[skill:多模态内容生成]` 出 AI 原图，但云端服务已停用
（`buddy-cloud.py image` 返回"当前版本的图片和视频功能已停止服务，请勿重试"），
所以改成从**自有视频帧**派生——对标注规范来说真实帧本来就更合适（尺度/光照/场景就是训练数据的分布），
对压力集则按"难例类型"从全库里挑：

* `--spec`：在真实帧上把 GT 框画出来（person 蓝 / ball 绿 / cone 橙）+ ASCII 说明条，产出
  `sources/spec_examples/spec_<NN>_<类型>.jpg` + `SPEC_INDEX.md`（中文逐图说明，供规范文档引用）。
  只进文档、**不进训练**（manifest `source=spec_example`、`split=none`）。
* `--stress`：按难度分桶挑帧（`dark` 低照度 / `blur` 运动模糊 / `tiny_target` 远景小目标 /
  `crowded` 人群密集 / `bright` 强逆光），产出 `sources/stress_set/cohort.json`（含每张的入选原因与指标）
  + `stress_sheet.jpg` 联系表。压力集**只是队列**，原地引用数据集里的帧（不复制、不产生新标注），
  因此不会重复计入数据台账。

用法::

    $PY s2c_data_spec_stress.py --spec --count 6
    $PY s2c_data_spec_stress.py --stress --per-bucket 8 --stress-count 30
    $PY s2c_data_spec_stress.py --list
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

import base_layout

#: 框颜色（BGR）
COLORS = {1: (245, 139, 76), 2: (124, 196, 60), 3: (35, 166, 245)}
NAME_ZH = {1: "人", 2: "球", 3: "锥"}


#: 小于这个尺寸（960 输入下的像素）的框视为标注噪声——bootstrap 预标注里那些"静止锥/碟被当球"
#: 的误检通常 3–5 px，人工复核会删掉；示例图不该拿它们当示范。
MIN_BOX_PX = 5.0


def load_coco(split: str, annotations: str = "auto") -> tuple[dict[str, Any], str]:
    """标注来源：auto = 有 `work/annotations/review.json` 就用它（S3 人工结果），否则用数据集 COCO。"""
    if annotations != "auto":
        path = Path(annotations)
    elif base_layout.gt_annotation_path(fallback_preannot=False):
        path = base_layout.gt_annotation_path(fallback_preannot=False)
    else:
        path = base_layout.split_annotations(split)
    if not path.exists():
        raise SystemExit("缺少标注文件 %s（先跑 s3_label_convert_dataset.py 或用 --annotations 指定）" % path)
    print("标注来源：%s%s" % (path, "" if "review" in path.name else "  ← 仍是 bootstrap 预标注，"
                             "S3 人工复核后建议重跑本脚本"))
    return json.loads(path.read_text(encoding="utf-8")), str(path)


def image_path(split: str, file_name: str) -> Path | None:
    """数据集里的图（`images/<名>`）。"""
    for base in (base_layout.split_dir(split) / file_name, base_layout.split_images(split) / Path(file_name).name):
        if base.exists():
            return base
    return None


def scale_to_960(width: int, height: int, box: Sequence[float]) -> tuple[float, float]:
    """把框换算到 960 输入下的像素尺寸（等比缩放 + 灰边，只用比例）。"""
    factor = 960.0 / max(width, height)
    return box[2] * factor, box[3] * factor


def frame_stats(image: np.ndarray, anns: Sequence[dict[str, Any]]) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    balls = [a for a in anns if a["category_id"] == 2]
    people = [a for a in anns if a["category_id"] == 1]
    cones = [a for a in anns if a["category_id"] == 3]
    tiny = [min(scale_to_960(width, height, a["bbox"])) for a in balls] or [999.0]
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "min_ball_px": round(float(min(tiny)), 1),
        "n_person": len(people), "n_ball": len(balls), "n_cone": len(cones),
    }


# ---------------------------------------------------------------- 规范示例

def draw_spec(image: np.ndarray, anns: Sequence[dict[str, Any]], caption: str) -> np.ndarray:
    """画框 + 说明条（文字用 ASCII：cv2 内置字体不支持中文）。"""
    canvas = image.copy()
    for ann in anns:
        x, y, w, h = (int(round(v)) for v in ann["bbox"])
        color = COLORS.get(ann["category_id"], (200, 200, 200))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        label = {1: "person", 2: "ball", 3: "cone"}.get(ann["category_id"], "obj")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x, max(0, y - th - 4)), (x + tw + 4, y), color, -1)
        cv2.putText(canvas, label, (x + 2, max(th, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (20, 20, 20), 1, cv2.LINE_AA)
    bar = 26
    out = np.zeros((canvas.shape[0] + bar, canvas.shape[1], 3), np.uint8)
    out[:bar] = (32, 36, 42)
    cv2.putText(out, caption[:150], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1,
                cv2.LINE_AA)
    out[bar:] = canvas
    return out


def pick_recipe(coco: dict[str, Any], per_image: dict[int, list[dict[str, Any]]],
                id_to_image: dict[int, dict[str, Any]]) -> list[tuple[str, int, list[dict[str, Any]]]]:
    """按"示例配方"挑图：三类齐全 / 含锥+球 / 小目标 / 合成样本。"""
    rows = []
    for image_id, anns in per_image.items():
        info = id_to_image[image_id]
        classes = {a["category_id"] for a in anns}
        width = info.get("width") or 1280
        height = info.get("height") or 720
        balls = [min(scale_to_960(width, height, a["bbox"])) for a in anns if a["category_id"] == 2]
        rows.append({
            "image_id": image_id, "file_name": info["file_name"], "anns": anns,
            "classes": classes, "min_ball": min(balls) if balls else 999.0,
            "n_cone": sum(1 for a in anns if a["category_id"] == 3),
            "n_ball": sum(1 for a in anns if a["category_id"] == 2),
            #: 该帧里最小框（960 输入像素）——小于 MIN_BOX_PX 基本是 bootstrap 噪声，示例要避开
            "min_any": min([min(scale_to_960(width, height, a["bbox"])) for a in anns] or [999.0]),
        })
    # 丢掉含"<5px 噪声框"的帧（bootstrap 的锥/碟误检），示例必须干净
    noisy = [r for r in rows if r["min_any"] < MIN_BOX_PX]
    rows = [r for r in rows if r["min_any"] >= MIN_BOX_PX]
    if noisy:
        print("  先剔除 %d 张含 <%.0fpx 噪声框的帧（多为静止锥/碟被误检成球）" % (len(noisy), MIN_BOX_PX))
    picked: list[tuple[str, int, list[dict[str, Any]]]] = []
    used: set[int] = set()

    def take(recipe: str, key, reverse: bool = True, need_cones: bool = False,
             need_balls: bool = False) -> None:
        for row in sorted(rows, key=key, reverse=reverse):
            if row["image_id"] in used:
                continue
            if need_cones and row["n_cone"] == 0:
                continue
            if need_balls and row["n_ball"] == 0:
                continue
            used.add(row["image_id"])
            picked.append((recipe, row["image_id"], row["anns"]))
            return

    take("all_three_classes", lambda r: (len(r["classes"]), r["n_cone"]))          # 三类齐全
    take("cone_and_ball", lambda r: r["n_cone"] + r["n_ball"],
         need_cones=True, need_balls=True)                                        # 锥+球同帧
    take("tiny_targets", lambda r: -r["min_ball"] if r["min_ball"] <= 14 else -999)  # 小目标 6–14px
    take("synth_sample", lambda r: r["n_cone"], need_cones=True)                   # 合成/派生样本
    for row in rows:                                                               # 兜底补足
        if len(picked) >= 6 or row["image_id"] in used:
            continue
        used.add(row["image_id"])
        picked.append(("extra", row["image_id"], row["anns"]))
    return picked


def cmd_spec(args: argparse.Namespace) -> int:
    base_layout.SPEC_EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    coco, ann_path = load_coco(args.split, args.annotations)
    per_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        per_image.setdefault(ann["image_id"], []).append(ann)
    id_to_image = {img["id"]: img for img in coco["images"]}
    picked = pick_recipe(coco, per_image, id_to_image)[: args.count]

    index_lines = ["# 标注规范对照示例（`sources/spec_examples/`）", "",
                   "标注来源：`%s`（%s）" % (ann_path,
                   "S3 人工复核结果" if "review" in ann_path else "bootstrap 预标注 —— 复核后请重跑"),
                   "",
                   "由 `s2c_data_spec_stress.py --spec` 从**真实帧**派生（云端图片生成服务已停用）。"
                   "框色约定：人=蓝、球=绿、锥=橙；这些图只用于规范说明，`source=spec_example`、"
                   "`split=none`，**不进训练**。", "",
                   "| 文件 | 示例类型 | 张数/要点 | 来源帧 |", "| --- | --- | --- | --- |"]
    rows: list[dict[str, Any]] = []
    made: list[str] = []
    for i, (recipe, image_id, anns) in enumerate(picked, 1):
        info = id_to_image[image_id]
        path = image_path(args.split, info["file_name"])
        if path is None:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        stats = frame_stats(image, anns)
        caption = ("spec %02d | %s | person %d ball %d cone %d | min_ball %.0fpx@960 | %s"
                   % (i, recipe, stats["n_person"], stats["n_ball"], stats["n_cone"],
                      stats["min_ball_px"], Path(info["file_name"]).name))
        name = "spec_%02d_%s.jpg" % (i, recipe)
        target = base_layout.SPEC_EXAMPLE_DIR / name
        if not cv2.imwrite(str(target), draw_spec(image, anns, caption), [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError("写示例图失败：%s" % target)
        made.append(name)
        rows.append({
            "image": str(target.relative_to(base_layout.DATASET_ROOT)), "source": "spec_example",
            "source_video": info["file_name"], "frame_index": "", "time_sec": "",
            "width": image.shape[1], "height": image.shape[0],
            "scene": recipe, "view": "spec", "split": "none",
            "is_synthetic": "no", "ai_background": "no", "is_pseudolabel": "no",
            "license": "自有素材（派生）", "note": "规范示例：%s" % caption,
        })
        index_lines.append("| `%s` | %s | 人 %d / 球 %d / 锥 %d；最小球 %.0fpx@960 | `%s` |"
                           % (name, recipe, stats["n_person"], stats["n_ball"], stats["n_cone"],
                              stats["min_ball_px"], Path(info["file_name"]).name))
        print("  + %-32s %s（人%d 球%d 锥%d）" % (name, recipe, stats["n_person"],
                                                stats["n_ball"], stats["n_cone"]))

    (base_layout.SPEC_EXAMPLE_DIR / "SPEC_INDEX.md").write_text("\n".join(index_lines) + "\n",
                                                          encoding="utf-8")
    if rows:
        with base_layout.MANIFEST_CSV.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(base_layout.MANIFEST_FIELDS))
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in base_layout.MANIFEST_FIELDS})
    print("规范示例 %d 张 → %s（source=spec_example / split=none，不进训练；索引 SPEC_INDEX.md）"
          % (len(made), base_layout.SPEC_EXAMPLE_DIR.relative_to(base_layout.DATASET_ROOT)))
    print("文档里引用：`sources/spec_examples/%s`" % (made[0] if made else "…"))
    return 0


# ---------------------------------------------------------------- 压力集

BUCKETS = (
    ("dark", "低照度（室内暗光/夜场）", lambda s: s["brightness"], False),
    ("blur", "运动模糊（快门慢/快速移动）", lambda s: s["sharpness"], False),
    ("tiny_target", "远景小目标（球在 960 下 <12px）", lambda s: s["min_ball_px"], False),
    ("crowded", "人群密集（person 多、遮挡重）", lambda s: s["n_person"], True),
    ("bright", "强光/逆光高对比", lambda s: s["contrast"], True),
)


def cmd_stress(args: argparse.Namespace) -> int:
    base_layout.STRESS_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    used: set[str] = set()
    sheets: list[tuple[str, str]] = []
    for split in ("test", "valid", "train"):
        coco = load_coco(split)
        per_image: dict[int, list[dict[str, Any]]] = {}
        for ann in coco["annotations"]:
            per_image.setdefault(ann["image_id"], []).append(ann)
        stats_of: dict[int, dict[str, float]] = {}
        for img in coco["images"]:
            path = image_path(split, img["file_name"])
            if path is None:
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            stats_of[img["id"]] = frame_stats(image, per_image.get(img["id"], []))
        for bucket, why, key, reverse in BUCKETS:
            ranked = sorted(stats_of.items(), key=lambda kv: key(kv[1]), reverse=reverse)
            taken = 0
            for image_id, stats in ranked:
                if taken >= args.per_bucket or len(entries) >= args.stress_count:
                    break
                if bucket == "tiny_target" and stats["min_ball_px"] > 12:
                    break
                info = next(i for i in coco["images"] if i["id"] == image_id)
                if info["file_name"] in used:
                    continue
                used.add(info["file_name"])
                taken += 1
                entries.append({"image": info["file_name"], "split": split, "bucket": bucket,
                                "reason": why, "metrics": stats,
                                "picked_at": datetime.now().isoformat(timespec="seconds")})
                path = image_path(split, info["file_name"])
                sheets.append((str(path) if path else "", "%s | %s" % (bucket, stats["min_ball_px"])))
            if taken:
                print("  %-12s %2d 张（%s）" % (bucket, taken, why))

    cohort = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "note": "分布外压力集：原地引用数据集里的帧（不复制、不新增标注），用于单独评测鲁棒性",
        "buckets": [{"key": b, "reason": w} for b, w, _, _ in BUCKETS],
        "entries": entries,
    }
    (base_layout.STRESS_DIR / "cohort.json").write_text(json.dumps(cohort, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    width = 320
    tiles = []
    for path, label in sheets[:60]:
        image = cv2.imread(path) if path else None
        tile = np.zeros((width // 2 + 20, width, 3), np.uint8)
        if image is not None:
            scale = width / image.shape[1]
            thumb = cv2.resize(image, (width, max(1, int(image.shape[0] * scale))))
            tile[:min(thumb.shape[0], tile.shape[0] - 20)] = thumb[:tile.shape[0] - 20]
        cv2.putText(tile, label, (4, tile.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
        cv2.imwrite(str(base_layout.STRESS_DIR / "stress_sheet.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    ledger = not base_layout.LEDGER_CSV.exists()
    with base_layout.LEDGER_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(base_layout.LEDGER_FIELDS))
        if ledger:
            writer.writeheader()
        writer.writerow({"name": "stress_cohort", "url": "internal", "fetched_at":
                         datetime.now().isoformat(timespec="seconds"), "license": "自有素材",
                         "redistributable": "no", "n_images": len(entries),
                         "note": "压力集队列（引用数据集帧，不新增标注）：%s"
                                 % "；".join("%s %d" % (b, sum(1 for e in entries if e["bucket"] == b))
                                             for b, _, _, _ in BUCKETS)})
    print("压力集 %d 帧 → %s（cohort.json + stress_sheet.jpg；原地引用，不进训练）"
          % (len(entries), base_layout.STRESS_DIR.relative_to(base_layout.DATASET_ROOT)))
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    specs = sorted(base_layout.SPEC_EXAMPLE_DIR.glob("spec_*.jpg")) if base_layout.SPEC_EXAMPLE_DIR.exists() else []
    print("规范示例：%d 张（%s）" % (len(specs), base_layout.SPEC_EXAMPLE_DIR))
    for path in specs:
        print("  %s" % path.name)
    cohort = base_layout.STRESS_DIR / "cohort.json"
    if cohort.exists():
        data = json.loads(cohort.read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        for entry in data["entries"]:
            counts[entry["bucket"]] = counts.get(entry["bucket"], 0) + 1
        print("压力集：%d 帧 %s（%s）" % (len(data["entries"]), counts, cohort.parent))
    else:
        print("压力集：还没生成（跑 --stress）")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="标注规范示例图 + 压力集（真实帧派生）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec", action="store_true", help="生成规范对照示例图")
    group.add_argument("--stress", action="store_true", help="挑难帧生成压力集队列")
    group.add_argument("--list", action="store_true", help="看现有产物")
    parser.add_argument("--count", type=int, default=6, help="示例图张数")
    parser.add_argument("--split", default="train", choices=list(base_layout.SPLITS), help="示例取自哪个 split")
    parser.add_argument("--annotations", default="auto",
                        help="auto（默认：优先 work/annotations/GT.json）或显式标注 JSON 路径")
    parser.add_argument("--per-bucket", type=int, default=8, help="压力集每类最多几帧")
    parser.add_argument("--stress-count", type=int, default=30, help="压力集总帧数上限")
    args = parser.parse_args(list(argv) if argv is not None else None)
    base_layout.ensure_layout()
    if args.spec:
        return cmd_spec(args)
    if args.stress:
        return cmd_stress(args)
    return cmd_list(args)


if __name__ == "__main__":
    sys.exit(main())
