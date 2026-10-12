#!/usr/bin/env python3
"""合成锥桶：从真实帧抠素材 → 粘贴到背景 → 自动精确 GT（S2.5）。

三步 + 两个附加能力：

* `--extract`：读 `reports/preannot_cone_candidates.csv` 的高分候选，用 **GrabCut** 从原帧抠出
  锥/碟 RGBA，存到 `synthetic/cutouts/`，并写 `cutouts.json`（尺寸/圆度/饱和度/锐度/来源）。
* `--compose`：把素材随机缩放粘贴到背景上（真实帧 **或 AI 生成背景**），
  **GT 由粘贴几何直接算出，是精确的**；同时断言：
  ① 不与任何 person 框重叠（IOU < 0.05）；② 完全落在图内；③ 尺寸在
  `[0.6×原生, 3×原生]` 内且 ≤120px（避免把 10px 的素材放大成糊团）；
  ④ 不与已贴的锥重叠（IOU < 0.3）。
* `--qa`：出抽检拼图（贴了框），对应人工校验里"合成帧抽检 10%"。
* `--ablate`：出"有/无合成"两版数据集（给 S8 的来源消融用）。

硬约束：合成 ≤ 训练集 30%（`--max-ratio`，超出即停止并报数）；合成图只从 **train** 背景生成，
避免把评测段（test）的像素带进训练（默认 `--allow-test-backgrounds` 关闭）。

用法：
    python s2a_data_compose_cones.py --extract --min-score 0.7
    python s2a_data_compose_cones.py --compose --n 400 --backgrounds mixed --max-ratio 0.30
    python s2a_data_compose_cones.py --qa --n 24
    python s2a_data_compose_cones.py --ablate
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

import base_layout
from s1_data_extract_frames import append_manifest

CONE_ID = base_layout.name_to_id()["cone"]
PERSON_ID = base_layout.name_to_id()["person"]
CUTOUT_META = base_layout.CUTOUT_META
#: 两个合成桶的标注文件（ai_cut=真实背景，ai_bg_cut=AI 背景）
COMPOSITE_ANNOTATIONS = (base_layout.AI_CUT_ANNOTATIONS, base_layout.AI_BG_CUT_ANNOTATIONS)
#: 素材与合成图在 ball_cone_v1 下的相对前缀（写进 JSON 与 manifest，供跨目录引用）
CUTOUT_REL = str(base_layout.CUTOUT_DIR.relative_to(base_layout.DATASET_ROOT))
COMPOSITE_IMAGE_RELS = tuple(str(d.relative_to(base_layout.DATASET_ROOT)) for d in
                             (base_layout.AI_CUT_IMAGE_DIR, base_layout.AI_BG_CUT_IMAGE_DIR))
SEED = 20261008
#: 粘贴时的目标色相（OpenCV H，0–179）：红/橙/黄/绿/青/蓝/紫 —— 覆盖真实标志锥的常见颜色
PALETTE_HUES = (0, 15, 30, 60, 90, 120, 150)


@dataclass
class Cutout:
    """一个抠好的锥/碟素材。"""

    path: str
    width: int
    height: int
    source_image: str
    score: float
    alpha_ratio: float
    sat: float
    sharpness: float
    method: str = "grabcut"
    license: str = "自有素材"


# --- 抠素材 -------------------------------------------------------------------


def _tidy_alpha(alpha: np.ndarray) -> np.ndarray | None:
    """只留最大连通域 + 轻微内缩羽化；全空返回 None。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    alpha = np.where(labels == biggest, 255, 0).astype(np.uint8)
    alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return alpha if alpha.max() > 0 else None


def _crop_with_pad(image: np.ndarray, box: Sequence[float], pad: float
                   ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
    px, py = bw * pad, bh * pad
    cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    cx2, cy2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return None
    rect = (int(x1 - cx1), int(y1 - cy1), int(max(1, bw)), int(max(1, bh)))
    return crop, rect


def extract_rgba(image: np.ndarray, box: Sequence[float], pad: float = 0.30
                 ) -> tuple[np.ndarray, str] | None:
    """三级降级抠素材：GrabCut → Otsu 颜色分割 → 硬框羽化。

    小目标（7–12 px 的小锥/碟）在 GrabCut 下经常抠出空掩码，但对"锥素材"来说
    alpha 的精度只影响观感、不影响 GT（GT 是粘贴几何），所以必须有兜底：
    ① 高对比目标用 Otsu 距离分割；② 再不行就用框本身（1px 羽化）当 alpha。
    """
    prepped = _crop_with_pad(image, box, pad)
    if prepped is None:
        return None
    crop, rect = prepped
    # ① GrabCut
    try:
        mask = np.zeros(crop.shape[:2], np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        alpha = _tidy_alpha(alpha)
    except cv2.error:
        alpha = None
    method = "grabcut"
    # ② Otsu：以边界环的中位色当背景，按色距离分割
    if alpha is None:
        border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]).reshape(-1, 3)
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(crop.astype(np.float32) - bg, axis=2)
        norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        alpha = _tidy_alpha(binary)
        method = "otsu"
    # ③ 硬框羽化（永远能成功）
    if alpha is None:
        alpha = np.full(crop.shape[:2], 255, np.uint8)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        method = "box"
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = alpha
    ys, xs = np.where(alpha > 8)
    if len(xs) < 4:
        return None
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1], method


def cutout_stats(rgba: np.ndarray) -> tuple[float, float, float]:
    alpha = rgba[..., 3]
    ratio = float((alpha > 8).mean())
    hsv = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2HSV)
    sat = float(hsv[..., 1][alpha > 8].mean()) / 255.0 if (alpha > 8).any() else 0.0
    gray = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return round(ratio, 3), round(sat, 3), round(sharp, 1)


def cmd_extract(args: argparse.Namespace) -> int:
    cand_path = Path(args.candidates)
    if not cand_path.is_absolute():
        cand_path = base_layout.DATASET_ROOT / cand_path
    rows = [r for r in csv.DictReader(cand_path.open(encoding="utf-8"))
            if float(r["score"]) >= args.min_score]
    print("候选 %d 个（score >= %.2f）" % (len(rows), args.min_score))
    base_layout.CUTOUT_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Cutout] = []
    dropped: list[str] = []
    for i, row in enumerate(sorted(rows, key=lambda r: -float(r["score"])), 1):
        image = cv2.imread(str(base_layout.DATASET_ROOT / row["image"]))
        if image is None:
            continue
        box = json.loads(row["xyxy"]) if isinstance(row["xyxy"], str) else row["xyxy"]
        extracted = extract_rgba(image, box)
        if extracted is None:
            print("  #%02d 抠图失败（%s）" % (i, row["image"]))
            continue
        rgba, method = extracted
        ratio, sat, sharp = cutout_stats(rgba)
        side = min(rgba.shape[:2])
        longest = max(rgba.shape[:2])
        # 质量闸门：宁缺勿滥——把"球状 / 地板碎片 / 带背景的硬框 / 灰糊糊的小块"挡在训练集之外
        aspect = longest / max(1, side)
        reason = None
        if side < args.min_side:
            reason = "过小 %dx%d" % (rgba.shape[1], rgba.shape[0])
        elif method == "box" and side > args.max_box_side:
            reason = "硬框带背景 %dx%d" % (rgba.shape[1], rgba.shape[0])
        elif ratio < args.min_alpha_ratio:
            reason = "不实心 alpha%.2f" % ratio
        elif aspect > args.max_aspect:
            reason = "过扁 %.1f:1" % aspect
        elif sat < args.min_sat:
            reason = "低饱和%.2f" % sat
        elif sat < 0.35 and sharp > args.pattern_sharp:
            reason = "图案化(疑似足球) sat%.2f tex%.0f" % (sat, sharp)
        elif sat < 0.15 and longest >= args.ball_side_guard:
            reason = "疑似足球 %dpx" % longest
        if reason:
            dropped.append(reason)
            print("  #%02d 丢弃：%s" % (i, reason))
            continue
        name = "cone_%03d_s%.2f_%s.png" % (i, float(row["score"]), method)
        cv2.imwrite(str(base_layout.CUTOUT_DIR / name), rgba)
        saved.append(Cutout(path="%s/%s" % (CUTOUT_REL, name), width=rgba.shape[1], height=rgba.shape[0],
                            source_image=row["image"], score=float(row["score"]),
                            alpha_ratio=ratio, sat=sat, sharpness=sharp, method=method))
    CUTOUT_META.write_text(json.dumps([c.__dict__ for c in saved], ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print("素材库 → %s（%d 个可用 / %d 个候选，丢弃 %d）"
          % (base_layout.CUTOUT_DIR, len(saved), len(rows), len(dropped)))
    if dropped:
        from collections import Counter as _Counter
        print("  丢弃原因:", dict(_Counter(r.split("（")[0].split("%")[0][:6] for r in dropped)))
    if saved:
        sizes = sorted((c.width * c.height) for c in saved)
        print("  素材像素面积: min %d / 中位 %d / max %d | 方法分布 %s"
              % (sizes[0], sizes[len(sizes) // 2], sizes[-1],
                 {m: sum(1 for c in saved if c.method == m) for m in {c.method for c in saved}}))
    return 0 if saved else 1


# --- 粘贴合成 -----------------------------------------------------------------


def load_person_boxes() -> dict[str, list[list[float]]]:
    """从人工 GT（`GT.json`；旧名 `review.json` 也认，没有才退 `preannot.json`）取每图 person 框，用于粘贴避让。"""
    path = base_layout.gt_annotation_path() or base_layout.PREANNOT_JSON
    if not path.exists():
        return {}
    coco = json.loads(path.read_text(encoding="utf-8"))
    name_of = {img["id"]: img["file_name"] for img in coco["images"]}
    boxes: dict[str, list[list[float]]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] == PERSON_ID:
            boxes.setdefault(name_of[ann["image_id"]], []).append(ann["bbox"])
    return boxes


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix) * max(0.0, iy2 - iy)
    return inter / max(1e-9, aw * ah + bw * bh - inter)


def choose_backgrounds(rows: list[dict[str, str]], want: str, bg_dir: Path,
                       allow_test: bool) -> list[tuple[str, str, str, str]]:
    """返回 (图片相对路径, split, 是否 AI 背景, 场景)。"""
    out: list[tuple[str, str, str, str]] = []
    if want in ("real", "mixed"):
        for row in rows:
            if row.get("source") != "raw_video":
                continue
            if row["split"] != "train" and not allow_test:
                continue
            out.append((row["image"], row["split"], "no", row.get("scene", "unknown")))
    if want in ("ai", "mixed"):
        for path in sorted(bg_dir.glob("*")):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                out.append((str(path.relative_to(base_layout.DATASET_ROOT)), "train", "yes", "ai_background"))
    return out


def compose_one(background: np.ndarray, cutouts: list[Cutout], person_boxes: list[list[float]],
                rng: random.Random, per_image: int, hue_jitter: int = 25,
                min_y_ratio: float = 0.30, max_y_ratio: float = 0.95,
                max_region_tex: float = 800.0,
                paste_min_px: int = 8, paste_max_px: int = 120,
                ) -> tuple[np.ndarray, list[list[float]], list[str]]:
    """在一次背景上贴 `per_image` 个锥；返回 (图, bbox 列表, 素材 id 列表)。"""
    h, w = background.shape[:2]
    canvas = background.copy()
    placed: list[list[float]] = []
    used: list[str] = []
    for _ in range(per_image):
        for _try in range(60):
            cut: Cutout = rng.choice(cutouts)
            sprite = cv2.imread(str(base_layout.DATASET_ROOT / cut.path), cv2.IMREAD_UNCHANGED)
            if sprite is None or sprite.shape[2] != 4:
                continue
            native = max(sprite.shape[:2])
            # 粘贴尺寸（sprite 长边像素）。实测：原先是 8–120px 均匀 → 以"中大"为主，
            # 而评测集里**小锥（<32px 面积档）占 116/133**。所以给出范围旋钮：
            #   --paste-max-px 36 → 专做小目标（≈1–5% 帧高），正是小目标 mAP 那一档。
            lo = max(6, int(paste_min_px))
            hi = max(lo + 1, int(paste_max_px))
            target = rng.randint(lo, hi)
            scale = target / float(native)
            new_w, new_h = max(4, int(sprite.shape[1] * scale)), max(4, int(sprite.shape[0] * scale))
            sprite = cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            # 亮度抖动，避免"太干净"
            gain = rng.uniform(0.88, 1.12)
            sprite[..., :3] = np.clip(sprite[..., :3].astype(np.float32) * gain, 0, 255).astype(np.uint8)
            if hue_jitter > 0:
                # 调色板重映射：现实里标志锥有黄/红/绿/蓝多种，只做小幅抖动会永远停在蓝色
                hsv = cv2.cvtColor(sprite[..., :3], cv2.COLOR_BGR2HSV).astype(np.int16)
                hsv[..., 0] = (rng.choice(PALETTE_HUES) + rng.randint(-hue_jitter, hue_jitter)) % 180
                sprite[..., :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            y_lo = int(min_y_ratio * h)
            y_hi = int(max_y_ratio * h)
            x = rng.randint(0, max(0, w - new_w - 1))
            y = rng.randint(y_lo, max(y_lo, y_hi - new_h))
            box = [float(x), float(y), float(new_w), float(new_h)]
            if x + new_w > w or y + new_h > h:                      # ② 必须完全在图内
                continue
            if any(iou_xywh(box, pb) > 0.05 for pb in person_boxes):  # ① 避让人体框
                continue
            if any(iou_xywh(box, other) > 0.30 for other in placed):   # ④ 不与已贴的锥重叠
                continue
            # ⑤ 局部必须"平"：墙面/幕布/球网纹理重、地板平滑 → 挡掉"锥贴在墙上"的悬空样本
            roi_check = background[y:y + new_h, x:x + new_w]
            if roi_check.size and float(cv2.Laplacian(cv2.cvtColor(roi_check, cv2.COLOR_BGR2GRAY),
                                                     cv2.CV_64F).var()) > max_region_tex:
                continue
            roi = canvas[y:y + new_h, x:x + new_w].astype(np.float32)
            alpha = (sprite[..., 3:4].astype(np.float32) / 255.0)
            canvas[y:y + new_h, x:x + new_w] = np.clip(
                sprite[..., :3].astype(np.float32) * alpha + roi * (1 - alpha), 0, 255).astype(np.uint8)
            placed.append(box)
            used.append(Path(cut.path).name)
            break
    return canvas, placed, used


def cmd_compose(args: argparse.Namespace) -> int:
    if not CUTOUT_META.exists():
        print("!! 还没有素材库，先跑 --extract")
        return 2
    cutouts = [Cutout(**c) for c in json.loads(CUTOUT_META.read_text(encoding="utf-8"))]
    if not cutouts:
        print("!! 素材库为空")
        return 2
    rows = list(csv.DictReader(base_layout.MANIFEST_CSV.open(encoding="utf-8")))
    real_train = sum(1 for r in rows if r.get("source") == "raw_video" and r["split"] == "train")
    max_synth = int(args.max_ratio / (1 - args.max_ratio) * real_train)
    existing = sum(1 for r in rows if r.get("is_synthetic") == "yes")
    budget = max(0, max_synth - existing)
    n = min(args.n, budget)
    print("真实 train 帧 %d → 合成上限 %d（%.0f%%），已有 %d，本次最多 %d"
          % (real_train, max_synth, args.max_ratio * 100, existing, n))
    if n <= 0:
        print("已达上限，停止（要更多合成样本请先提高 --max-ratio 或增加真实帧）")
        return 0

    backgrounds = choose_backgrounds(rows, args.backgrounds, Path(args.bg_dir), args.allow_test_backgrounds)
    if not backgrounds:
        print("!! 没有可用背景（real 需 train 且 source=raw_video；ai 需 %s 里有图）" % args.bg_dir)
        return 2
    # AI 背景池通常只有几张 → 允许**重复使用**（同一背景配不同素材/位置/颜色仍是不同样本），
    # 同时用 --ai-share 控制"AI 背景占合成样本的比例"，避免整批合成图都长一个样。
    real_bgs = [b for b in backgrounds if b[2] == "no"]
    ai_bgs = [b for b in backgrounds if b[2] == "yes"]
    n_ai = int(round(n * args.ai_share)) if ai_bgs else 0
    if ai_bgs and not real_bgs and n_ai < n:
        n_ai = n
    work: list[tuple[str, str, str, str]] = []
    for i in range(n_ai):
        work.append(ai_bgs[i % len(ai_bgs)])
    if real_bgs:
        for i in range(n - len(work)):
            work.append(real_bgs[i % len(real_bgs)])
    elif len(work) < n:
        for i in range(n - len(work)):
            work.append(ai_bgs[i % len(ai_bgs)])
    random.Random(args.seed).shuffle(work)   # 独立随机源：此处 rng 还没建
    print("背景分配：AI %d 张 / 真实 %d 张（AI 池 %d 张、真实池 %d 张，可重复使用）"
          % (sum(1 for w in work if w[2] == "yes"), sum(1 for w in work if w[2] == "no"),
             len(ai_bgs), len(real_bgs)))
    rng = random.Random(args.seed)
    person_boxes = load_person_boxes()
    if not person_boxes:
        print("!! 读不到 person 框（缺 GT.json/preannot.json）→ 无法保证避让，拒绝合成")
        return 2

    # 两个"桶"：ai_cut = 真实背景粘贴；ai_bg_cut = AI 背景粘贴（各自独立目录 + 标注 + source 值）
    buckets = {
        "ai_cut": (base_layout.AI_CUT_IMAGE_DIR, base_layout.AI_CUT_ANNOTATIONS),
        "ai_bg_cut": (base_layout.AI_BG_CUT_IMAGE_DIR, base_layout.AI_BG_CUT_ANNOTATIONS),
    }
    state: dict[str, dict[str, Any]] = {}
    for key, (image_dir, ann_path) in buckets.items():
        coco = {"images": [], "annotations": [], "categories": base_layout.categories()}
        if ann_path.exists():
            coco = json.loads(ann_path.read_text(encoding="utf-8"))
        state[key] = {
            "coco": coco, "images": image_dir, "annotations": ann_path,
            "next_image_id": max([img["id"] for img in coco["images"]], default=0) + 1,
            "next_ann_id": max([a["id"] for a in coco["annotations"]], default=0) + 1,
        }
    manifest_rows: list[dict[str, Any]] = []
    made = 0
    for key, (image_dir, _) in buckets.items():
        image_dir.mkdir(parents=True, exist_ok=True)
    for bpath, split, is_ai, scene in work:
        if made >= n:
            break
        bg = cv2.imread(str(base_layout.DATASET_ROOT / bpath))
        if bg is None:
            continue
        boxes = person_boxes.get(bpath, [])
        per_image = rng.randint(1, args.per_image_max)
        canvas, placed, used = compose_one(
            bg, cutouts, boxes, rng, per_image, args.hue_jitter,
            args.min_y_ratio, args.max_y_ratio, args.max_region_tex,
            args.paste_min_px, args.paste_max_px)
        if not placed:
            continue
        key = "ai_bg_cut" if is_ai == "yes" else "ai_cut"
        bucket = state[key]
        prefix = str(bucket["images"].relative_to(base_layout.DATASET_ROOT))
        # 文件名前缀 = 来源键（ai_cut_* / ai_bg_cut_*），与目录名、manifest.source 三者一致。
        # 序号用「本桶已有张数 + 本次已生成数」：`made` 是**本次调用的局部计数**，多次调用
        # （例如先跑小尺度批次、再跑原分布批次）会从 0 重来 → 撞名覆盖上一批的图，
        # 标注却累积两份（实测触发 dataset_convert 的"同一张图出现多次"断言）。
        idx = len(bucket["coco"]["images"]) + made
        name = "%s_%s_%03d.jpg" % (key, Path(bpath).stem[:24].replace(" ", "_"), idx)
        rel = "%s/%s" % (prefix, name)
        if (base_layout.DATASET_ROOT / rel).exists():
            raise RuntimeError("合成图撞名（同桶序号重复）：%s" % rel)
        if not cv2.imwrite(str(base_layout.DATASET_ROOT / rel), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError("写合成图失败：%s" % rel)
        image_id = bucket["next_image_id"]
        bucket["next_image_id"] += 1
        bucket["coco"]["images"].append({"id": image_id, "file_name": rel,
                                         "width": canvas.shape[1], "height": canvas.shape[0]})
        for box in placed:
            bucket["coco"]["annotations"].append({"id": bucket["next_ann_id"], "image_id": image_id,
                                                  "category_id": CONE_ID,
                                                  "bbox": [round(v, 2) for v in box],
                                                  "area": round(box[2] * box[3], 2), "iscrowd": 0,
                                                  "attributes": {"source": key, "by": "paste"}})
            bucket["next_ann_id"] += 1
        manifest_rows.append({
            # source_video 存"背景图来源"（合成样本没有真实视频帧号，背景路径才是可追溯的关键）
            "image": rel, "source": key, "source_video": bpath,
            "frame_index": "", "time_sec": "", "width": canvas.shape[1], "height": canvas.shape[0],
            "scene": scene, "view": key, "split": split,
            "is_synthetic": "yes", "ai_background": is_ai, "is_pseudolabel": "no",
            "license": "自有合成（AI 背景）" if key == "ai_bg_cut" else "自有合成（真实背景）",
            "note": "粘贴素材 %s" % ",".join(used),
        })
        made += 1
    written = append_manifest(manifest_rows)
    print("合成完成：%d 张" % made)
    for key, bucket in state.items():
        bucket["annotations"].write_text(json.dumps(bucket["coco"], ensure_ascii=False),
                                         encoding="utf-8")
        print("  %-9s → %s（累计 %d 图 / %d 锥框，source=%s）"
              % (key, bucket["annotations"].relative_to(base_layout.DATASET_ROOT),
                 len(bucket["coco"]["images"]), len(bucket["coco"]["annotations"]), key))
    print("  manifest 新增 %d 行" % written)
    return 0


def load_composites() -> dict[str, Any]:
    """合并 ai_cut 与 ai_bg_cut 两桶标注（抽检/消融都要看全量合成样本）。"""
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    categories = base_layout.categories()
    for path in COMPOSITE_ANNOTATIONS:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        images += data.get("images", [])
        annotations += data.get("annotations", [])
        categories = data.get("categories") or categories
    return {"images": images, "annotations": annotations, "categories": categories}


def cmd_qa(args: argparse.Namespace) -> int:
    coco = load_composites()
    if not coco["images"]:
        print("!! 还没有合成样本，先跑 --compose")
        return 2
    by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    rng = random.Random(args.seed)
    picks = rng.sample(coco["images"], min(args.n, len(coco["images"])))
    tiles = []
    for img in picks:
        canvas = cv2.imread(str(base_layout.DATASET_ROOT / img["file_name"]))
        if canvas is None:
            continue
        scale = 480.0 / canvas.shape[1]
        tile = cv2.resize(canvas, (480, max(1, int(canvas.shape[0] * scale))))
        for ann in by_image.get(img["id"], []):
            x, y, w, h = ann["bbox"]
            cv2.rectangle(tile, (int(x * scale), int(y * scale)), (int((x + w) * scale), int((y + h) * scale)),
                          (0, 0, 255), 2)
        cv2.putText(tile, Path(img["file_name"]).name[:38], (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return 1
    cols = 4
    # 背景尺寸不一（真实 1280x720 / AI 1280x677）→ 先补齐高度再拼，否则 hstack 会报维度不一致
    max_h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, max_h - t.shape[0], 0, max_h and 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
             for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows_ = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    out = base_layout.REPORTS_DIR / "composite_qa.jpg"
    cv2.imwrite(str(out), np.vstack(rows_), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("抽检拼图 → %s（%d 张，红框=自动 GT）" % (out, len(picks)))
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """出"有/无合成"两版数据集（只按 train 区分，valid/test 保持纯真实）。"""
    out_root = Path(args.out_root)
    synth = load_composites()
    if not synth["images"]:
        print("!! 还没有合成样本，先跑 --compose")
        return 2
    for variant in ("no_synth", "with_synth"):
        for split in base_layout.SPLITS:
            (out_root / variant / split / "images").mkdir(parents=True, exist_ok=True)
        for split in base_layout.SPLITS:
            data = json.loads(base_layout.split_annotations(split).read_text(encoding="utf-8"))
            images = [img for img in data["images"]
                      if not str(img.get("source_image", "")).startswith(COMPOSITE_IMAGE_RELS)]
            ids = {img["id"] for img in images}
            annots = [a for a in data["annotations"] if a["image_id"] in ids]
            if variant == "with_synth" and split == "train":
                offset = max(ids, default=0)
                id_map = {}
                for img in synth["images"]:
                    new_id = offset + img["id"]
                    id_map[img["id"]] = new_id
                    images.append({"id": new_id, "file_name": img["file_name"],
                                   "width": img["width"], "height": img["height"],
                                   "source_image": img["file_name"]})
                annots += [{**a, "image_id": id_map[a["image_id"]]} for a in synth["annotations"]]
            for img in images:
                src = base_layout.DATASET_ROOT / img["file_name"]
                dst = out_root / variant / split / "images" / Path(img["file_name"]).name
                if not dst.exists():
                    dst.symlink_to(src)
            payload = {"info": {"description": "ablation %s" % variant}, "licenses": [],
                       "categories": base_layout.categories(), "images": images, "annotations": annots}
            (out_root / variant / split / "_annotations.coco.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            print("  %-10s %-5s 图 %4d 框 %5d" % (variant, split, len(images), len(annots)))
    print("消融数据集 → %s（用 s4_train_finetune.py --dataset-dir 指过去）" % out_root)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合成锥桶：抠素材 / 粘贴 / 抽检 / 消融")
    sub = parser.add_argument_group("通用")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--qa", action="store_true")
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--candidates", default=str(base_layout.REPORTS_DIR / "preannot_cone_candidates.csv"))
    parser.add_argument("--min-score", type=float, default=0.7, help="抠素材时的候选分数下限")
    parser.add_argument("--min-side", type=int, default=6, help="素材最短边下限（小锥/碟只有 7–12px）")
    parser.add_argument("--min-alpha-ratio", type=float, default=0.45, help="alpha 实心度下限")
    parser.add_argument("--min-sat", type=float, default=0.22,
                        help="最低饱和度：低饱和多半是足球/白碟/地板，宁缺勿滥")
    parser.add_argument("--max-box-side", type=int, default=8,
                        help="硬框降级法只在短边 <= 此值时才允许（否则把背景也贴进去）")
    parser.add_argument("--ball-side-guard", type=int, default=14,
                        help="低饱和且长边 >= 此值 → 判为疑似足球，丢弃")
    parser.add_argument("--max-aspect", type=float, default=2.0,
                        help="长宽比上限：锥/碟接近方形，白色长条这类线条碎片要挡掉")
    parser.add_argument("--pattern-sharp", type=float, default=350,
                        help="低饱和 + 纹理高于此值 → 判为图案化物体（足球黑白花纹）")
    parser.add_argument("--n", type=int, default=400, help="合成张数（--qa 时为拼图张数）")
    parser.add_argument("--backgrounds", choices=("real", "ai", "mixed"), default="real")
    parser.add_argument("--bg-dir", default=str(base_layout.AI_GEN_BG_DIR))
    parser.add_argument("--per-image-max", type=int, default=4, help="每张背景最多贴几个锥")
    parser.add_argument("--ai-share", type=float, default=0.25,
                        help="合成样本里用 AI 背景的比例（0=全真实背景；需要 ai_backgrounds 目录有图）")
    parser.add_argument("--hue-jitter", type=int, default=25,
                        help="粘贴时的色相抖动（度，180 制）；让蓝/黄/红/绿标志锥都覆盖到")
    parser.add_argument("--max-ratio", type=float, default=0.30, help="合成样本占训练集上限")
    parser.add_argument("--paste-min-px", type=int, default=8,
                        help="粘贴 sprite 长边下限（px）；小目标专项用 8 + --paste-max-px 36")
    parser.add_argument("--paste-max-px", type=int, default=120,
                        help="粘贴 sprite 长边上限（px）；默认 120 ≈ 720p 的 17%% 帧高")
    parser.add_argument("--min-y-ratio", type=float, default=0.30, help="粘贴区上边界（避开天花板/墙）")
    parser.add_argument("--max-y-ratio", type=float, default=0.95, help="粘贴区下边界")
    parser.add_argument("--max-region-tex", type=float, default=800.0,
                        help="粘贴处局部拉普拉斯方差上限：墙面/幕布/球网纹理重，挡掉悬空粘贴")
    parser.add_argument("--allow-test-backgrounds", action="store_true",
                        help="允许用 test 段的帧当背景（默认禁止，避免把评测段像素带进训练）")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-root", default="/tmp/ds_ablate")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.extract:
        return cmd_extract(args)
    if args.compose:
        return cmd_compose(args)
    if args.qa:
        return cmd_qa(args)
    if args.ablate:
        return cmd_ablate(args)
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
