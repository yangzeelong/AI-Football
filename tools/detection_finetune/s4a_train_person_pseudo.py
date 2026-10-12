#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""person 保障①：用 base 模型给**新抽的帧**打 person 伪标签（零人工）。

为什么需要：微调后类别空间变小（person/ball/cone），person 数据量直接影响掉点风险。
自有视频里还有大量没抽过的帧，用 base 模型（COCO 权重，person=class 1）过一遍，
只保留"高置信 person 框"的帧，作为 person 数据的廉价扩量。

产物规则（命名表）：`sources/pseudo_person/pseudo_person_<视频>__f######__t####.##.jpg`
+ `annotations.json`（COCO，只有 person 框）+ manifest（`source=pseudo_person`、
`is_pseudolabel=yes`、`split=train`）。伪标签**不参与** test/valid，避免污染固定评测集；
`s3_label_dataset_stats.py` 会把 `pseudo_person` 单列统计。

幂等：同视频同帧号（±0.5s）已登记过的帧直接跳过，可反复跑直到 `--count` 攒够。

用法::

    $PY s4a_train_person_pseudo.py --count 200 --stride-sec 2 --conf 0.6   # 抽 200 帧高置信 person
    $PY s4a_train_person_pseudo.py --stats
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch

import base_layout

#: COCO 里 person 的类别号（base 权重是 COCO 80 类）
COCO_PERSON_ID = 1
#: 伪标签目录
PSEUDO_DIR = base_layout.PSEUDO_PERSON_DIR
PSEUDO_ANNOTATIONS = PSEUDO_DIR / "annotations.json"
#: 太小的 person 框没训练价值（960 输入下的像素）
MIN_PERSON_PX = 24.0
#: 单帧最多保留几个 person（人群里几十个会拖慢训练，且伪标签质量下降）
MAX_PERSON_PER_FRAME = 12


def load_manifest() -> list[dict[str, str]]:
    with base_layout.MANIFEST_CSV.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def taken_frames(rows: Sequence[dict[str, str]]) -> dict[str, list[float]]:
    """{视频: [已抽时间点]}（含 raw_video 与 pseudo_person 两类来源）。"""
    taken: dict[str, list[float]] = {}
    for row in rows:
        video = str(row.get("source_video") or "")
        if row.get("source") == "raw_video":
            video = str(Path(row["image"]).parent)
        if not video:
            continue
        try:
            taken.setdefault(video, []).append(float(row.get("time_sec") or 0))
        except ValueError:
            continue
    return taken


def train_videos() -> list[Path]:
    """只从 split=train 的视频抽（test/valid 的整段视频留给固定评测集，防泄漏）。

    策略表结构：`{"note": ..., "defaults": {...}, "videos": {<相对路径>: {...}}}`
    """
    policy = json.loads((Path(__file__).parent / "split_policy.json").read_text(encoding="utf-8"))
    defaults = policy.get("defaults", {})
    root = base_layout.DATA_ROOT
    videos = []
    for name, meta in policy.get("videos", {}).items():
        if (meta or {}).get("split", defaults.get("split")) != "train":
            continue
        path = root / name
        if path.exists():
            videos.append(path)
    return videos


def empty_coco() -> dict[str, Any]:
    return {"info": {"description": "ball_cone_v1 person 伪标签（base 模型）",
                     "date_created": datetime.now().isoformat(timespec="seconds")},
            "licenses": [], "categories": base_layout.categories(), "images": [], "annotations": []}


def cmd_label(args: argparse.Namespace) -> int:
    from rfdetr import RFDETRSmall

    PSEUDO_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    taken = taken_frames(rows)
    coco = json.loads(PSEUDO_ANNOTATIONS.read_text(encoding="utf-8")) \
        if PSEUDO_ANNOTATIONS.exists() else empty_coco()
    next_image_id = max((img["id"] for img in coco["images"]), default=0) + 1
    next_ann_id = max((a["id"] for a in coco["annotations"]), default=0) + 1

    weights = Path(__file__).parent.parent.parent / "models/rfdetr/rf-detr-small.pth"
    print("加载 base 权重：%s" % weights)
    model = RFDETRSmall(pretrain_weights=str(weights))
    try:
        model.optimize_for_inference()
    except Exception as exc:                                     # noqa: BLE001
        print("  （optimize_for_inference 跳过：%s）" % exc)
    torch.set_grad_enabled(False)

    videos = train_videos()
    print("train 视频 %d 段；目标新增 %d 帧（stride %.1fs，conf>=%.2f）"
          % (len(videos), args.count, args.stride_sec, args.conf))
    new_rows: list[dict[str, Any]] = []
    made = 0
    start = time.time()
    for video in videos:
        if made >= args.count:
            break
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 60
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(fps * args.stride_sec)))
        key = str(video.parent.relative_to(base_layout.DATA_ROOT))
        skip = taken.get(key, [])
        taken_in_video = 0
        for fi in range(0, n_frames, step):
            if made >= args.count or taken_in_video >= args.max_per_video:
                break
            t_sec = fi / fps
            if any(abs(t_sec - prev) < 0.5 for prev in skip):     # 幂等：抽过的不再抽
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            det = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=args.conf)
            ids = np.asarray(det.class_id)
            boxes = np.asarray(det.xyxy)
            confs = np.asarray(det.confidence)
            factor = 960.0 / max(frame.shape[1], frame.shape[0])
            keep = []
            for cid, box, conf in zip(ids, boxes, confs):
                if int(cid) != COCO_PERSON_ID:
                    continue
                w = (box[2] - box[0]) * factor
                h = (box[3] - box[1]) * factor
                if min(w, h) < MIN_PERSON_PX:
                    continue
                keep.append((box, float(conf)))
            if not keep:
                continue
            keep.sort(key=lambda item: -item[1])
            keep = keep[:MAX_PERSON_PER_FRAME]
            name = "pseudo_person_%s__f%06d__t%07.2f.jpg" % (video.stem, fi, t_sec)
            target = PSEUDO_DIR / name
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError("写图失败：%s" % target)
            rel = str(target.relative_to(base_layout.DATASET_ROOT))
            coco["images"].append({"id": next_image_id, "file_name": rel,
                                   "width": frame.shape[1], "height": frame.shape[0]})
            for box, conf in keep:
                coco["annotations"].append({
                    "id": next_ann_id, "image_id": next_image_id,
                    "category_id": base_layout.name_to_id()["person"],
                    "bbox": [round(float(box[0]), 2), round(float(box[1]), 2),
                             round(float(box[2] - box[0]), 2), round(float(box[3] - box[1]), 2)],
                    "area": round(float((box[2] - box[0]) * (box[3] - box[1])), 2), "iscrowd": 0,
                    "attributes": {"source": "pseudo_label", "confidence": round(conf, 3)}})
                next_ann_id += 1
            new_rows.append({
                "image": rel, "source": "pseudo_person", "source_video": key,
                "frame_index": fi, "time_sec": round(t_sec, 2),
                "width": frame.shape[1], "height": frame.shape[0],
                "scene": "raw_video", "view": "pseudo", "split": "train",
                "is_synthetic": "no", "ai_background": "no", "is_pseudolabel": "yes",
                "license": "自有素材（伪标签）",
                "note": "base 模型 person 伪标签 %d 个（conf>=%.2f）" % (len(keep), args.conf),
            })
            skip.append(t_sec)
            taken_in_video += 1
            made += 1
            next_image_id += 1
            if made % 20 == 0:
                print("  已 %d/%d 帧（%.0fs）" % (made, args.count, time.time() - start))
        cap.release()
        print("  %-22s 新增 %2d 帧（累计 %d）" % (video.name[:22], taken_in_video, made))

    PSEUDO_ANNOTATIONS.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    written = 0
    if new_rows:
        with base_layout.MANIFEST_CSV.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(base_layout.MANIFEST_FIELDS))
            for row in new_rows:
                writer.writerow({k: row.get(k, "") for k in base_layout.MANIFEST_FIELDS})
        written = len(new_rows)
    if written:
        exists = base_layout.LEDGER_CSV.exists()
        with base_layout.LEDGER_CSV.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(base_layout.LEDGER_FIELDS))
            if not exists:
                writer.writeheader()
            writer.writerow({"name": "pseudo_person", "url": "internal",
                             "fetched_at": datetime.now().isoformat(timespec="seconds"),
                             "license": "自有素材（base 模型伪标签）", "redistributable": "no",
                             "n_images": written,
                             "note": "base 模型 person 伪标签，conf>=%.2f，仅进 train" % args.conf})
    print("\n新增 %d 帧 → %s（累计 %d 图 / %d person 框；用时 %.0fs）"
          % (made, PSEUDO_DIR.relative_to(base_layout.DATASET_ROOT), len(coco["images"]),
             len(coco["annotations"]), time.time() - start))
    print("下一步：$PY s3_label_convert_dataset.py --link（会自动并入 pseudo_person）→ s3_label_dataset_stats.py 看 person 占比")
    return 0


def cmd_stats(_: argparse.Namespace) -> int:
    if not PSEUDO_ANNOTATIONS.exists():
        print("还没有伪标签产物（跑 --count N）")
        return 0
    data = json.loads(PSEUDO_ANNOTATIONS.read_text(encoding="utf-8"))
    per_video: dict[str, int] = {}
    for image in data["images"]:
        video = Path(image["file_name"]).name.split("__")[0]
        per_video[video] = per_video.get(video, 0) + 1
    print("person 伪标签：%d 图 / %d 框" % (len(data["images"]), len(data["annotations"])))
    for video, count in sorted(per_video.items(), key=lambda kv: -kv[1]):
        print("  %-30s %d 帧" % (video[:30], count))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="person 保障①：base 模型伪标签扩量")
    parser.add_argument("--count", type=int, default=200, help="目标新增帧数")
    parser.add_argument("--stride-sec", type=float, default=2.0, help="抽帧步长（秒）")
    parser.add_argument("--conf", type=float, default=0.6, help="person 置信度阈值")
    parser.add_argument("--max-per-video", type=int, default=25, help="单视频最多取几帧")
    parser.add_argument("--stats", action="store_true", help="看伪标签概况")
    args = parser.parse_args(list(argv) if argv is not None else None)
    base_layout.ensure_layout()
    return cmd_stats(args) if args.stats else cmd_label(args)


if __name__ == "__main__":
    sys.exit(main())
