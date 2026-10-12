#!/usr/bin/env python3
"""用 base（COCO 80 类）模型对抽好的帧做半自动预标注。

产出三样东西，供人工只做"增删改 + 改类"：

1. `annotations/preannot.json`：COCO 格式预标注。person→1、ball→2 直接给；
   **疑似锥桶的球框**仍然写成 ball(2)，但带 `cone_candidate: true` 与候选分数，
   人工确认后把 category 改成 3（cone）。
2. `reports/cone_candidates.csv`：锥桶候选清单（按分数排序），带"跨帧静止性"与
   颜色/纹理特征，便于从高到低核对。
3. `reports/montage_<split>_*.jpg`：三色抽查拼图（球=绿、锥候选=红、人=灰），
   图上烧注文件名、时间、置信度。

锥桶候选的判据（锥/碟在本项目里的表现）：**固定机位下同一像素位置被反复检成
"球"**（真实足球会移动、会消失），辅以颜色集中度与低纹理（足球是黑白花纹、纹理高）。

用法：
    python s2_data_preannotate.py --resolution 960 --threshold 0.25
    python s2_data_preannotate.py --limit 12 --out-prefix debug      # 快速试跑
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

import base_layout

#: base 模型（COCO）里与本任务相关的类别号
COCO_PERSON, COCO_BALL = 1, 37


@dataclass
class Detection:
    image: str            # 相对 ball_cone_v1
    frame_index: int
    time_sec: float
    cls: int
    conf: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def size(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return x2 - x1, y2 - y1


@dataclass
class Cluster:
    """同一视频内、同一像素位置反复出现的球框（锥/碟候选）。"""

    video: str
    cx: float
    cy: float
    members: list[Detection] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.members)

    @property
    def spread(self) -> float:
        pts = np.array([m.center for m in self.members])
        return float(np.hypot(pts[:, 0] - pts[:, 0].mean(), pts[:, 1] - pts[:, 1].mean()).mean())

    @property
    def size(self) -> tuple[float, float]:
        ws = [m.size[0] for m in self.members]
        hs = [m.size[1] for m in self.members]
        return float(np.mean(ws)), float(np.mean(hs))

    @property
    def conf(self) -> float:
        return float(np.mean([m.conf for m in self.members]))

    @property
    def best(self) -> Detection:
        return max(self.members, key=lambda m: m.conf)


def patch_features(image: np.ndarray, xyxy: Sequence[float]) -> dict[str, float]:
    """检测框内的颜色/纹理特征：锥/碟颜色集中、纹理低；足球是黑白花纹、纹理高。"""
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    h, w = image.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return {"sat": 0.0, "val": 0.0, "texture": 0.0}
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    sat = float(hsv[..., 1].mean()) / 255.0
    val = float(hsv[..., 2].mean()) / 255.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {"sat": round(sat, 3), "val": round(val, 3), "texture": round(texture, 1)}


def build_clusters(dets: Iterable[Detection], radius_px: float = 25.0,
                   per_image_max: int = 1) -> list[Cluster]:
    """按中心距离贪心聚类（高置信优先），把"同一位置的重复球框"聚成一簇。

    同一帧可能对同一物体吐多个框（base 在阈 0.25 下常见），所以**每帧每簇最多记
    `per_image_max` 个成员**——否则"出现次数"会超过采样帧数，静止性被高估。
    """
    clusters: list[Cluster] = []
    for det in sorted(dets, key=lambda d: -d.conf):
        cx, cy = det.center
        w, h = det.size
        radius = max(radius_px, 1.2 * max(w, h))
        for cluster in clusters:
            if abs(cluster.cx - cx) < radius and abs(cluster.cy - cy) < radius:
                if sum(1 for m in cluster.members if m.image == det.image) >= per_image_max:
                    break
                cluster.members.append(det)
                pts = np.array([m.center for m in cluster.members])
                cluster.cx, cluster.cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                break
        else:
            clusters.append(Cluster(video="", cx=cx, cy=cy, members=[det]))
    return clusters


def nms_per_class(dets: Sequence[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """按类别做 IoU NMS（重复框只留最高置信的一个）。"""
    kept: list[Detection] = []
    for cls in sorted({d.cls for d in dets}):
        group = sorted((d for d in dets if d.cls == cls), key=lambda d: -d.conf)
        while group:
            best, rest = group[0], group[1:]
            kept.append(best)
            group = [d for d in rest if iou(best.xyxy, d.xyxy) < iou_threshold]
    return kept


def iou(a: Sequence[float], b: Sequence[float]) -> float:
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
    return inter / max(1e-9, area_a + area_b - inter)


def cone_score(cluster: Cluster, samples: int, feats: dict[str, float],
               inside_person: bool = False, min_size: float = 8.0,
               max_size: float = 120.0) -> float:
    """候选分数（0-1）：静止性强、位置集中、框尺寸合理、颜色集中/纹理低。"""
    if samples <= 0:
        return 0.0
    distinct = len({m.image for m in cluster.members})
    persistence = min(1.0, distinct / samples)          # 必须钳位：重复框会让比值 > 1
    w, h = cluster.size
    size_ok = 1.0 if min_size <= max(w, h) <= max_size else 0.0
    concentration = 1.0 - min(1.0, cluster.spread / max(12.0, max(w, h)))
    colorfulness = min(1.0, feats["sat"] * 1.6) if feats["sat"] > 0.25 else min(1.0, feats["val"])
    smooth = 1.0 if feats["texture"] < 300 else 1.0 / (1.0 + feats["texture"] / 300.0)
    score = (0.45 * persistence + 0.25 * concentration + 0.15 * size_ok
             + 0.10 * colorfulness + 0.05 * smooth)
    if inside_person:                    # 落在人框里：更可能是随身物品/人体部位
        score *= 0.6
    return round(min(1.0, score), 3)


def annotate_tile(image: np.ndarray, dets: Sequence[Detection], candidates: set[int],
                  tile_w: int) -> np.ndarray:
    """缩到 tile_w 后在**缩略图坐标系**里画框。

    若在原始 1080p 上画 1px 框再缩到 480 宽，线宽只剩 0.25px，肉眼看不见
    （踩过一次）；所以先缩放、再按同一比例画框，锥候选额外套一个红色圈标，
    否则 8px 的小锥在缩略图上根本找不到。
    """
    scale = tile_w / image.shape[1]
    tile = cv2.resize(image, (tile_w, max(1, int(round(image.shape[0] * scale)))))
    for i, det in enumerate(dets):
        x1, y1, x2, y2 = (int(round(v * scale)) for v in det.xyxy)
        w, h = max(4, x2 - x1), max(4, y2 - y1)
        if det.cls == COCO_PERSON:
            cv2.rectangle(tile, (x1, y1), (x2, y2), (150, 150, 150), 1)
            continue
        if i in candidates:
            color, label = (0, 0, 255), "cone?"
            radius = int(max(9, 1.8 * max(w, h)))     # 圈标：小目标也看得见
            cv2.circle(tile, ((x1 + x2) // 2, (y1 + y2) // 2), radius, (0, 0, 255), 1)
        else:
            color, label = (0, 200, 0), "ball"
        cv2.rectangle(tile, (x1, y1), (x2, y2), color, 2)
        ty = y1 - 4 if y1 > 12 else min(tile.shape[0] - 2, y2 + 12)
        cv2.putText(tile, "%s %.2f" % (label, det.conf), (max(0, x1), ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return tile


def write_montages(frames: list[tuple[str, np.ndarray, float]], out_dir: Path, split: str,
                   cols: int = 4, tile_w: int = 640) -> list[Path]:
    """把标注结果拼成抽查图（每张 4×3 = 12 帧），文件名烧在图上；返回所有写出的文件。"""
    tiles = []
    for name, tile, t in frames:
        cv2.putText(tile, "%s t=%.1fs" % (Path(name).name, t), (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        raise ValueError("没有可拼的帧")
    per_sheet = cols * 3
    sheets = []
    for start in range(0, len(tiles), per_sheet):
        chunk = tiles[start:start + per_sheet]
        height = max(t.shape[0] for t in chunk)
        chunk = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0)) for t in chunk]
        while len(chunk) % cols:
            chunk.append(np.zeros_like(chunk[0]))
        rows = [np.hstack(chunk[i:i + cols]) for i in range(0, len(chunk), cols)]
        sheets.append(np.vstack(rows))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, sheet in enumerate(sheets, 1):
        path = out_dir / f"montage_{split}_{i:02d}of{len(sheets):02d}.jpg"
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise RuntimeError(f"写拼图失败：{path}")
        written.append(path)
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="base 模型半自动预标注 + 锥桶候选清单")
    parser.add_argument("--resolution", type=int, default=960, help="推理分辨率（部署为 960）")
    parser.add_argument("--positional-encoding-size", type=int, default=0,
                        help="0=用包默认（32）；改它必须同时有匹配的预训练权重，否则加载报错")
    parser.add_argument("--threshold", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="同类 NMS 的 IoU 阈值（0=不做）")
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 帧（0=全部）")
    parser.add_argument("--min-persistence", type=float, default=0.3,
                        help="锥候选的跨帧出现率下限")
    parser.add_argument("--out-prefix", default="preannot", help="预标注文件名前缀")
    parser.add_argument("--montage-splits", default="test",
                        help="为哪些 split 出拼图（默认只 test，省时间）")
    parser.add_argument("--tile-width", type=int, default=640, help="拼图每格宽度（像素）")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from rfdetr import RFDETRSmall

    weights = Path(__file__).resolve().parents[2] / "models/rfdetr/rf-detr-small.pth"
    if not weights.exists():
        print("!! 找不到 base 权重:", weights)
        return 2
    #: 分辨率只改 resolution，**不要**跟着改 positional_encoding_size：
    #: base 权重的位置编码是 32²=1025，传 60²=3601 会在 `main.py:130` 的
    #: `load_state_dict(strict=False)` 上直接 size mismatch 报错（strict=False 只忽略
    #: 缺/多键，形状不符照样抛）。960 输入靠 DINOv2 的窗口注意力 + 插值，
    #: 既有 `rf-detr-small-960-*` engine 就是这条路径建出来的。
    kwargs: dict[str, Any] = {"pretrain_weights": str(weights), "device": "cuda",
                             "resolution": args.resolution}
    if args.positional_encoding_size:
        kwargs["positional_encoding_size"] = args.positional_encoding_size
    model = RFDETRSmall(**kwargs)
    torch.set_grad_enabled(False)
    print("base 模型就绪: resolution=%d positional_encoding_size=%s"
          % (kwargs.get("resolution", 512), kwargs.get("positional_encoding_size", 32)))

    splits = [s for s in args.splits.split(",") if s in base_layout.SPLITS]
    rows = [r for r in csv.DictReader(base_layout.MANIFEST_CSV.open(encoding="utf-8"))
            if r["split"] in splits and r["source"] == "raw_video"]
    if args.limit:
        rows = rows[:args.limit]
    print("待预标注帧: %d（splits=%s）" % (len(rows), splits))

    # 1) 推理
    per_image: dict[str, list[Detection]] = {}
    images_meta: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        path = base_layout.DATASET_ROOT / row["image"]
        image = cv2.imread(str(path))
        if image is None:
            print("  !! 读图失败:", path)
            continue
        height, width = image.shape[:2]
        det = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), threshold=args.threshold)
        cids = np.asarray(det.class_id).astype(int)
        confs = np.asarray(det.confidence)
        boxes = np.asarray(det.xyxy)
        raw = [Detection(image=row["image"], frame_index=int(row["frame_index"]),
                         time_sec=float(row["time_sec"]), cls=int(cids[j]),
                         conf=float(confs[j]), xyxy=tuple(float(v) for v in boxes[j]))
               for j in range(len(cids)) if int(cids[j]) in (COCO_PERSON, COCO_BALL)]
        #: base 在低阈值下会对同一物体吐多个框，必须 NMS；否则"球框数"虚高数倍
        dets = nms_per_class(raw, args.nms_iou) if args.nms_iou > 0 else raw
        per_image[row["image"]] = dets
        images_meta[row["image"]] = {"file_name": row["image"], "width": width, "height": height,
                                     "split": row["split"], "source_video": row["source_video"]}
        if i % 50 == 0 or i == len(rows):
            print("  推理 %d/%d（%.1f 帧/秒，已 %.0fs）" % (i, len(rows), i / (time.time() - t0 or 1),
                                                          time.time() - t0))

    # 2) 跨帧聚类找锥/碟候选（按视频分组）
    by_video: dict[str, list[Detection]] = {}
    for image, dets in per_image.items():
        video = images_meta[image]["source_video"]
        by_video.setdefault(video, []).extend(d for d in dets if d.cls == COCO_BALL)
    samples_per_video = {v: sum(1 for r in rows if r["source_video"] == v) for v in by_video}

    candidates: list[dict[str, Any]] = []
    cand_image_indices: dict[str, set[int]] = {}
    for video, dets in by_video.items():
        samples = max(1, samples_per_video.get(video, 1))
        for cluster in build_clusters(dets):
            if cluster.n < 2:
                continue
            w, h = cluster.size
            if w < 6 or h < 6 or max(w, h) > 200:
                continue  # 太小/太大都不是锥/碟候选
            best = cluster.best
            image = cv2.imread(str(base_layout.DATASET_ROOT / best.image))
            feats = patch_features(image, best.xyxy) if image is not None else {"sat": 0.0, "val": 0.0, "texture": 0.0}
            #: 候选中心落在任何 person 框内 → 更可能是随身物品/人体部位，降权标记
            inside_person = any(d.cls == COCO_PERSON and
                                d.xyxy[0] <= cluster.cx <= d.xyxy[2] and
                                d.xyxy[1] <= cluster.cy <= d.xyxy[3]
                                for d in per_image[best.image])
            score = cone_score(cluster, samples, feats, inside_person=inside_person)
            distinct = len({m.image for m in cluster.members})
            persistence = round(min(1.0, distinct / samples), 3)
            if persistence < args.min_persistence:
                continue
            candidates.append({
                "video": video, "image": best.image, "frame_index": best.frame_index,
                "time_sec": best.time_sec, "cx": round(cluster.cx, 1), "cy": round(cluster.cy, 1),
                "w": round(w, 1), "h": round(h, 1), "conf": round(cluster.conf, 3),
                "distinct_frames": distinct, "samples": samples,
                "persistence": persistence, "spread_px": round(cluster.spread, 1),
                "inside_person": int(inside_person),
                "sat": feats["sat"], "val": feats["val"], "texture": feats["texture"],
                "score": score, "xyxy": [round(v, 1) for v in best.xyxy],
            })
            idx = [i for i, d in enumerate(per_image[best.image]) if d is best]
            if idx:
                cand_image_indices.setdefault(best.image, set()).add(idx[0])
    candidates.sort(key=lambda c: -c["score"])

    # 3) 落盘：COCO 预标注 + 候选清单
    coco = {"info": {"description": "ball_cone_v1 base 预标注（球/人自动，锥候选待人工改类）",
                     "date_created": time.strftime("%Y-%m-%dT%H:%M:%S")},
            "licenses": [], "categories": base_layout.categories(), "images": [], "annotations": []}
    ann_id = 0
    for image_id, (image, meta) in enumerate(sorted(images_meta.items()), 1):
        meta = dict(meta)
        meta["id"] = image_id
        coco["images"].append({k: meta[k] for k in ("id", "file_name", "width", "height")})
        for det in per_image[image]:
            if det.cls not in (COCO_PERSON, COCO_BALL):
                continue
            x1, y1, x2, y2 = det.xyxy
            ann_id += 1
            coco["annotations"].append({
                "id": ann_id, "image_id": image_id,
                # 数据集 category_id 是 0-based（RF-DETR 契约），与 base 模型的 COCO id 不同
                "category_id": (base_layout.name_to_id()["person"] if det.cls == COCO_PERSON
                                else base_layout.name_to_id()["ball"]),
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "area": round((x2 - x1) * (y2 - y1), 2), "iscrowd": 0,
                "attributes": {"source": "base_model", "confidence": round(det.conf, 3),
                               "base_class_id": det.cls},
            })
    out = base_layout.ANNOTATION_DIR / f"{args.out_prefix}.json"
    out.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")

    cand_csv = base_layout.REPORTS_DIR / f"{args.out_prefix}_cone_candidates.csv"
    with cand_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0].keys()) if candidates else
                                ["video", "image", "score"])
        writer.writeheader()
        writer.writerows(candidates)

    # 4) 三色拼图（默认只出 test，省时间）
    time_of = {r["image"]: float(r["time_sec"]) for r in rows}
    sheets: list[str] = []
    for split in [s for s in args.montage_splits.split(",") if s]:
        frames = []
        for image, meta in sorted(images_meta.items()):
            if meta["split"] != split or image not in per_image:
                continue
            img = cv2.imread(str(base_layout.DATASET_ROOT / image))
            if img is None:
                continue
            frames.append((image,
                           annotate_tile(img, per_image[image],
                                         cand_image_indices.get(image, set()),
                                         args.tile_width),
                           time_of.get(image, 0.0)))
        if frames:
            sheets.extend(str(p) for p in write_montages(frames, base_layout.REPORTS_DIR, split))

    print("-" * 88)
    print("预标注      → %s（%d 图 / %d 框）" % (out, len(coco["images"]), len(coco["annotations"])))
    print("锥桶候选清单 → %s（%d 个候选，其中 score>=0.6 的 %d 个）"
          % (cand_csv, len(candidates), sum(1 for c in candidates if c["score"] >= 0.6)))
    for path in sheets:
        print("三色抽查拼图 →", path)
    size_hist: dict[str, int] = {}
    for cand in candidates[:200]:
        key = "%dx%d" % (round(cand["w"] / 10) * 10, round(cand["h"] / 10) * 10)
        size_hist[key] = size_hist.get(key, 0) + 1
    print("候选框尺寸分布（前 200，按 10px 分箱）:", dict(sorted(size_hist.items(), key=lambda kv: -kv[1])[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
