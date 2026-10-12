#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""person 保障③：混入 COCO val2017 里的人体样本（按需，默认不跑）。

只在"微调后 person mAP50 掉点 > 1 个点"时才需要（决策门见方案 §8）。做三件事：

1. 从 `instances_val2017.json` 里挑**含 person** 的图（可按 person 框面积/数量筛），
   只保留 person 框（球/锥不标——COCO 没有这两个类的可用标注）；
2. 下载对应图片到 `sources/coco_person/coco_person_<id>.jpg`（COCO 图 URL 固定为
   `http://images.cocodataset.org/val2017/<id>.jpg`；下载走 **curl 断点续传**，
   因为 pip/urllib 在大文件上容易被掐断——本轮实测过）；
3. 写 `annotations.json`（COCO，只有 person）+ manifest（`source=coco_person`、`split=train`）
   + 台账（`ledger.csv`：许可 = COCO 标注 CC-BY-4.0 / 图片各自 Flickr 许可、`redistributable=no`）。

**合规**：只作内部训练用途，不外发、不再分发；台账记 URL 与时间可追溯。

用法::

    $PY s4b_train_person_coco.py --ping                                 # 先测可达性
    $PY s4b_train_person_coco.py --download-annotations                 # 拉 instances_val2017.json（241MB zip）
    $PY s4b_train_person_coco.py --instances-json /path/instances_val2017.json \\
        --max-images 400 --min-person-px 24 --target-ratio 0.5
    $PY s4b_train_person_coco.py --stats
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2

import base_layout

#: COCO val2017 资源
IMAGE_URL = "http://images.cocodataset.org/val2017/%012d.jpg"
ANNOTATIONS_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
#: person 类别号（COCO）
COCO_PERSON_ID = 1
COCO_DIR = base_layout.COCO_PERSON_DIR
COCO_ANNOTATIONS = COCO_DIR / "annotations.json"


def curl_download(url: str, target: Path, timeout: int = 300) -> bool:
    """用 curl 断点续传下载（比 urllib 稳；pip 的下载器就是在这里吃过亏）。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sSL", "--retry", "10", "--retry-all-errors", "--retry-delay", "3",
           "-C", "-", "-o", str(target), url]
    return subprocess.run(cmd, timeout=timeout, check=False).returncode == 0 and target.exists()


def cmd_ping(_: argparse.Namespace) -> int:
    ok = 0
    for url in (IMAGE_URL % 397133, ANNOTATIONS_ZIP):
        result = subprocess.run(["curl", "-sSI", "--max-time", "20", "-o", "/dev/null", "-w",
                                 "%{http_code}", url], capture_output=True, text=True, check=False)
        code = result.stdout.strip() or "?"
        print("  %-60s HTTP %s" % (url[:60], code))
        ok += code.startswith("2")
    print("可达 %d/2；不可达就别硬跑（先去能访问的机器上准备好 json 与图片）" % ok)
    return 0 if ok else 1


def cmd_download_annotations(args: argparse.Namespace) -> int:
    cache = Path(args.cache_dir)
    zip_path = cache / "annotations_trainval2017.zip"
    if not zip_path.exists():
        print("下载 %s（241MB，curl 断点续传）…" % zip_path.name)
        if not curl_download(ANNOTATIONS_ZIP, zip_path, timeout=1800):
            print("下载失败；可手动下载后放到 %s" % zip_path)
            return 1
    target = cache / "instances_val2017.json"
    cmd = ["unzip", "-o", "-j", str(zip_path), "annotations/instances_val2017.json", "-d", str(cache)]
    if subprocess.run(cmd, check=False).returncode != 0:
        print("解压失败（缺 unzip？）：%s" % " ".join(cmd))
        return 1
    print("标注就绪：%s（%.0f MB）" % (target, target.stat().st_size / 1e6))
    return 0


def load_person_images(instances: Path, min_px: float, min_count: int,
                       max_images: int) -> dict[int, list[list[float]]]:
    """从 instances_val2017.json 里挑含 person 的图，返回 {image_id: [bbox...]}（已换算为 960 输入像素）。"""
    data = json.loads(instances.read_text(encoding="utf-8"))
    name_of = {img["id"]: img["file_name"] for img in data["images"]}
    size_of = {img["id"]: (img["width"], img["height"]) for img in data["images"]}
    per_image: dict[int, list[list[float]]] = {}
    for ann in data["annotations"]:
        if ann.get("category_id") != COCO_PERSON_ID or ann.get("iscrowd"):
            continue
        image_id = ann["image_id"]
        width, height = size_of.get(image_id, (640, 480))
        factor = 960.0 / max(width, height)
        x, y, w, h = ann["bbox"]
        if min(w * factor, h * factor) < min_px:
            continue
        per_image.setdefault(image_id, []).append([float(x), float(y), float(w), float(h)])
    picked = {k: v for k, v in per_image.items()
              if len(v) >= min_count and k in name_of}
    # 优先挑 person 多、框大的图（信息量高）
    order = sorted(picked, key=lambda k: (-len(picked[k]),
                                          -max(b[2] * b[3] for b in picked[k])))
    return {k: picked[k] for k in order[:max_images]}


def cmd_mix(args: argparse.Namespace) -> int:
    if not args.instances_json or not Path(args.instances_json).exists():
        print("需要 --instances-json（先跑 --download-annotations）")
        return 2
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    picked = load_person_images(Path(args.instances_json), args.min_person_px,
                                args.min_person_count, args.max_images)
    print("从 instances_val2017.json 选到 %d 张含 person 的图（person 框合计 %d）"
          % (len(picked), sum(len(v) for v in picked.values())))

    coco = {"info": {"description": "COCO val2017 person 子集（内部训练用）",
                     "date_created": datetime.now().isoformat(timespec="seconds")},
            "licenses": [], "categories": base_layout.categories(), "images": [], "annotations": []}
    if COCO_ANNOTATIONS.exists():
        coco = json.loads(COCO_ANNOTATIONS.read_text(encoding="utf-8"))
    have = {img["file_name"] for img in coco["images"]}
    next_image_id = max((img["id"] for img in coco["images"]), default=0) + 1
    next_ann_id = max((a["id"] for a in coco["annotations"]), default=0) + 1

    rows, added, failed = [], 0, 0
    start = time.time()
    for image_id, boxes in picked.items():
        name = "coco_person_%012d.jpg" % image_id
        rel = str((COCO_DIR / name).relative_to(base_layout.DATASET_ROOT))
        target = COCO_DIR / name
        if rel in have or target.exists():
            continue
        if added >= args.max_images:
            break
        if not curl_download(IMAGE_URL % image_id, target, timeout=120):
            failed += 1
            print("  下载失败 %s" % name)
            continue
        image = cv2.imread(str(target))
        if image is None:
            failed += 1
            target.unlink(missing_ok=True)
            continue
        coco["images"].append({"id": next_image_id, "file_name": rel,
                               "width": image.shape[1], "height": image.shape[0]})
        for x, y, w, h in boxes:
            x, y = max(0.0, x), max(0.0, y)
            w, h = min(w, image.shape[1] - x), min(h, image.shape[0] - y)
            if w <= 0 or h <= 0:
                continue
            coco["annotations"].append({"id": next_ann_id, "image_id": next_image_id,
                                        "category_id": base_layout.name_to_id()["person"],
                                        "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                                        "area": round(w * h, 2), "iscrowd": 0,
                                        "attributes": {"source": "coco_val2017",
                                                       "coco_image_id": image_id}})
            next_ann_id += 1
        rows.append({
            "image": rel, "source": "coco_person", "source_video": IMAGE_URL % image_id,
            "frame_index": "", "time_sec": "", "width": image.shape[1], "height": image.shape[0],
            "scene": "coco", "view": "unknown", "split": "train",
            "is_synthetic": "no", "ai_background": "no", "is_pseudolabel": "no",
            "license": "COCO 标注 CC-BY-4.0 / 图片 Flickr 各自许可（仅内部训练）",
            "note": "COCO val2017 人体子集，%d 个 person 框" % len(boxes),
        })
        added += 1
        if added % 50 == 0:
            print("  已入库 %d 张（%.0fs）" % (added, time.time() - start))

    COCO_ANNOTATIONS.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    if rows:
        with base_layout.MANIFEST_CSV.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(base_layout.MANIFEST_FIELDS))
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in base_layout.MANIFEST_FIELDS})
        exists = base_layout.LEDGER_CSV.exists()
        with base_layout.LEDGER_CSV.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(base_layout.LEDGER_FIELDS))
            if not exists:
                writer.writeheader()
            writer.writerow({"name": "coco_person", "url": ANNOTATIONS_ZIP,
                             "fetched_at": datetime.now().isoformat(timespec="seconds"),
                             "license": "COCO 标注 CC-BY-4.0 / 图片 Flickr 各自许可",
                             "redistributable": "no", "n_images": len(rows),
                             "note": "仅内部训练用途；来源 %s" % (IMAGE_URL % 0).rsplit("/", 1)[0]})
    print("\n新增 %d 张（失败 %d）→ %s；累计 %d 图 / %d 框"
          % (added, failed, COCO_DIR.relative_to(base_layout.DATASET_ROOT),
             len(coco["images"]), len(coco["annotations"])))
    print("下一步：$PY s3_label_convert_dataset.py --link（会自动并入 coco_person）")
    if args.target_ratio:
        report_ratio(args.target_ratio)
    return 0


def report_ratio(target: float) -> None:
    """按当前数据集算 person 实例占比，并给出达到目标还需要多少 person 框。"""
    total_person = total = 0
    for split in base_layout.SPLITS:
        path = base_layout.split_annotations(split)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for ann in data["annotations"]:
            total += 1
            if ann["category_id"] == base_layout.name_to_id()["person"]:
                total_person += 1
    if not total:
        return
    ratio = total_person / total
    print("当前 person 实例占比 %.1f%%（目标 %.0f%%）" % (ratio * 100, target * 100))
    if ratio < target:
        need = (target * total - total_person) / (1 - target)
        print("  还差约 %d 个 person 实例（≈ %d 张平均 %d 人/图的 COCO 图）"
              % (round(need), round(need / 2.0), 2))


def cmd_stats(_: argparse.Namespace) -> int:
    if not COCO_ANNOTATIONS.exists():
        print("还没有 COCO person 子集（按需再跑，见 --help）")
        return 0
    data = json.loads(COCO_ANNOTATIONS.read_text(encoding="utf-8"))
    print("COCO person 子集：%d 图 / %d 框" % (len(data["images"]), len(data["annotations"])))
    report_ratio(0.5)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="person 保障③：COCO val2017 人体子集（按需）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ping", action="store_true", help="测 COCO 资源可达性")
    group.add_argument("--download-annotations", action="store_true", help="下载并解出 instances_val2017.json")
    group.add_argument("--mix", action="store_true", help="按 instances json 挑图并入库")
    group.add_argument("--stats", action="store_true", help="看当前子集与 person 占比")
    parser.add_argument("--instances-json", default="", help="instances_val2017.json 路径")
    parser.add_argument("--cache-dir", default=str(base_layout.DATA_ROOT / "_coco_cache"),
                        help="下载缓存目录（默认 <DATA_DIR>/_coco_cache）")
    parser.add_argument("--max-images", type=int, default=400, help="最多入库多少张")
    parser.add_argument("--min-person-px", type=float, default=24.0,
                        help="person 框短边下限（960 输入像素）")
    parser.add_argument("--min-person-count", type=int, default=1, help="单图最少几个 person")
    parser.add_argument("--target-ratio", type=float, default=0.5,
                        help="person 实例占比目标（0=不算）")
    args = parser.parse_args(list(argv) if argv is not None else None)
    base_layout.ensure_layout()
    if args.ping:
        return cmd_ping(args)
    if args.download_annotations:
        return cmd_download_annotations(args)
    if args.stats:
        return cmd_stats(args)
    return cmd_mix(args)


if __name__ == "__main__":
    sys.exit(main())
