#!/usr/bin/env python3
"""标注回收 → 统一 COCO 三分类 → roboflow 三目录（train/valid/test），带对拍与泄漏检查。

输入：
  * `annotations/preannot.json`：base 模型的预标注（人/球自动给出，锥候选仍是 ball）
  * 可选 `annotations/review.json`：人工改过的完整 COCO（用任何标注工具导出后转换），
    或 `--cones-csv` + `--cone-min-score` 把"锥桶候选"**自动**改成 cone(3)（仅用于
    bootstrap 冒烟，正式训练前必须人工过一遍，产物会标 `cone_source=auto_candidate`）。

输出：
  ball_cone_v1/{train,valid,test}/images/*.jpg + _annotations.coco.json

必须守住的四条：
  1. `categories` 与 `classes.json` **逐项对拍**（`base_layout.assert_categories`），错一位就是"球消失"；
  2. **split 按整段视频划分**，同一 `source_video` 不得跨 split（同源泄漏）；
  3. 同一图片不得出现在多个 split；
  4. bbox 必须落在图内且面积 > 0（越界框会污染训练）。

用法：
    python s3_label_convert_dataset.py --link                                   # 只回收 base 预标注
    python s3_label_convert_dataset.py --cone-min-score 0.6 --link               # bootstrap：自动改类
    python s3_label_convert_dataset.py --annotations annotations/review.json     # 用人工改过的标注
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import base_layout


def load_manifest() -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(base_layout.MANIFEST_CSV.open(encoding="utf-8")))
    if not rows:
        raise SystemExit("manifest.csv 为空，先跑 s1_data_extract_frames.py")
    return {row["image"]: row for row in rows}


def load_candidates(path: Path, min_score: float) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [r for r in csv.DictReader(handle) if float(r["score"]) >= min_score]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix), max(0.0, iy2 - iy)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / max(1e-9, aw * ah + bw * bh - inter)


def apply_cone_candidates(coco: dict[str, Any], candidates: Sequence[dict[str, Any]],
                          min_iou: float = 0.5) -> int:
    """把锥桶候选对应的 ball 框改成 cone(3)，并打上来源标记。"""
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_id_of = {img["file_name"]: img["id"] for img in coco["images"]}
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
    changed = 0
    cone_id = base_layout.name_to_id()["cone"]
    ball_id = base_layout.name_to_id()["ball"]
    for cand in candidates:
        image_id = image_id_of.get(cand["image"])
        if image_id is None:
            continue
        # 候选 CSV 里的 xyxy 是 (x1,y1,x2,y2)，而 bbox/IoU 统一用 (x,y,w,h)
        raw = json.loads(cand["xyxy"]) if isinstance(cand["xyxy"], str) else list(cand["xyxy"])
        target = [float(raw[0]), float(raw[1]), float(raw[2]) - float(raw[0]),
                  float(raw[3]) - float(raw[1])]
        best, best_iou = None, min_iou
        for ann in by_image.get(image_id, []):
            if ann["category_id"] != ball_id:
                continue
            score = iou(ann["bbox"], target)
            if score >= best_iou:
                best, best_iou = ann, score
        if best is not None:
            best["category_id"] = cone_id
            best.setdefault("attributes", {})["cone_source"] = "auto_candidate"
            best["attributes"]["candidate_score"] = cand.get("score")
            changed += 1
    return changed


def sanitize(coco: dict[str, Any]) -> tuple[int, int]:
    """丢掉面积<=0 或完全越界的框；把轻微越界的框裁回图内。返回 (修复, 丢弃)。"""
    size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}
    kept, repaired, dropped = [], 0, 0
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        width, height = size[ann["image_id"]]
        if w <= 1 or h <= 1:
            dropped += 1
            continue
        nx1, ny1 = max(0.0, x), max(0.0, y)
        nx2, ny2 = min(width, x + w), min(height, y + h)
        if nx2 - nx1 <= 1 or ny2 - ny1 <= 1:
            dropped += 1
            continue
        if (nx1, ny1, nx2, ny2) != (x, y, x + w, y + h):
            repaired += 1
        ann["bbox"] = [round(nx1, 2), round(ny1, 2), round(nx2 - nx1, 2), round(ny2 - ny1, 2)]
        ann["area"] = round((nx2 - nx1) * (ny2 - ny1), 2)
        kept.append(ann)
    coco["annotations"] = kept
    return repaired, dropped


def build_split(coco: dict[str, Any], images: list[dict[str, Any]], out_dir: Path,
                link: bool, split: str) -> dict[str, Any]:
    """按 split 落盘图片与 COCO json，并做类别对拍。

    `file_name` **必须**写成 `images/<文件名>`：roboflow 加载器按
    `dataset_dir/<split>/file_name` 找图，而 manifest 里的路径带着 `raw/<视频>/`
    前缀，直接沿用会 `FileNotFoundError`（踩过一次）。原路径另存到
    `source_image` 字段里便于溯源。
    """
    out_images = base_layout.split_images(split)
    out_images.mkdir(parents=True, exist_ok=True)
    ids = {img["id"] for img in images}
    payload_images = []
    for img in images:
        source_rel = img["file_name"]
        basename = Path(source_rel).name
        src = base_layout.DATASET_ROOT / source_rel
        dst = out_images / basename
        if not dst.exists():
            if link:
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)
        payload_images.append({**img, "file_name": f"images/{basename}",
                               "source_image": source_rel})
    payload = {
        "info": {"description": "ball_cone_v1 三分类（person/ball/cone）",
                 "date_created": coco["info"].get("date_created", ""), "split": split},
        "licenses": [{"id": 1, "name": "自有素材", "url": ""}],
        "categories": base_layout.categories(),
        "images": payload_images,
        "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
    }
    ann_path = base_layout.split_annotations(split)
    ann_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    base_layout.assert_categories(base_layout.categories_from_coco(ann_path), source=str(ann_path))
    return {"split": split, "images": len(images), "annotations": len(payload["annotations"]),
            "path": str(ann_path)}


def assert_no_leak(manifest: dict[str, dict[str, str]], wanted: Iterable[str]) -> None:
    """同源视频不得跨 split；同一图片不得重复。"""
    video_splits: dict[str, set[str]] = defaultdict(set)
    seen: dict[str, str] = {}
    for image in wanted:
        row = manifest[image]
        video_splits[row["source_video"]].add(row["split"])
        if image in seen:
            raise ValueError(f"同一张图在待建数据集里出现多次（通常是合成/兜底数据与人工复核结果重复）：{image}")
        seen[image] = row["split"]
    bad = {v: sorted(s) for v, s in video_splits.items() if len(s) > 1}
    if bad:
        raise ValueError(f"同源视频跨 split（会泄漏）：{bad}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="标注回收 → roboflow 三目录")
    # 默认取人工 GT（GT.json；老目录的 review.json 也认），没有才退到机器预标注
    parser.add_argument("--annotations", default=str(base_layout.gt_annotation_path()),
                        help="默认：GT.json（人工复核）→ review.json（旧名）→ preannot.json")
    parser.add_argument("--cones-csv", default=str(base_layout.REPORTS_DIR / "preannot_cone_candidates.csv"))
    parser.add_argument("--cone-min-score", type=float, default=0.0,
                        help=">0 时把该分数以上的锥候选自动改成 cone（bootstrap 用）")
    parser.add_argument("--link", action="store_true", help="用软链接代替复制（省磁盘）")
    parser.add_argument("--no-merge-synth", dest="merge_synth", action="store_false",
                        help="不并入 work/synthetic/annotations_synth.json（做'无合成'消融时用）")
    parser.add_argument("--no-merge-person", dest="merge_person", action="store_false",
                        help="不并入 person 兜底数据（伪标签 / COCO 人体子集）")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest()
    ann_path = Path(args.annotations)
    if not ann_path.is_absolute():
        ann_path = base_layout.DATASET_ROOT / ann_path
    if not ann_path.exists():
        print("!! 找不到标注文件:", ann_path)
        return 2
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    print("标注来源: %s（%d 图 / %d 框）" % (ann_path, len(coco["images"]), len(coco["annotations"])))

    auto_changed = 0
    if args.cone_min_score > 0:
        cands = load_candidates(Path(args.cones_csv), args.cone_min_score)
        auto_changed = apply_cone_candidates(coco, cands)
        print("自动改类（score>=%.2f）: %d 个候选 → cone" % (args.cone_min_score, auto_changed))

    # 合成集（S2.5）单独存放，这里并进主标注：split 由 manifest 决定（合成图固定进 train）
    # 两桶都要并：ai_cut（真实背景）+ ai_bg_cut（AI 背景）
    if args.merge_synth:
        for comp_path in (base_layout.AI_CUT_ANNOTATIONS, base_layout.AI_BG_CUT_ANNOTATIONS):
            if not comp_path.exists():
                print("（缺 %s，跳过该桶；先跑 s2a_data_compose_cones.py --compose）"
                      % comp_path.relative_to(base_layout.DATASET_ROOT))
                continue
            synth = json.loads(comp_path.read_text(encoding="utf-8"))
            # 人工复核结果优先：已经在 coco 里的同名图（来自 review.json）不再并入，否则同图进两次，
            # assert_no_leak 会直接报"图片重复"（实测就卡在这）
            have_names = {img["file_name"] for img in coco["images"]}
            skipped = [img for img in synth["images"] if img["file_name"] in have_names]
            if skipped:
                keep_ids = {img["id"] for img in synth["images"] if img["file_name"] not in have_names}
                synth["images"] = [img for img in synth["images"] if img["id"] in keep_ids]
                synth["annotations"] = [a for a in synth["annotations"] if a["image_id"] in keep_ids]
                print("  %s：跳过 %d 张已在人工复核里的图（人工结果优先）"
                      % (comp_path.parent.name, len(skipped)))
            next_id = max((img["id"] for img in coco["images"]), default=0) + 1
            id_map: dict[int, int] = {}
            for img in synth["images"]:
                id_map[img["id"]] = next_id
                coco["images"].append({"id": next_id, "file_name": img["file_name"],
                                       "width": img["width"], "height": img["height"]})
                next_id += 1
            next_ann = max((a["id"] for a in coco["annotations"]), default=0) + 1
            for ann in synth["annotations"]:
                coco["annotations"].append({**ann, "id": next_ann, "image_id": id_map[ann["image_id"]]})
                next_ann += 1
            print("并入 %s: %d 图 / %d 框（只进 train）"
                  % (comp_path.parent.name, len(synth["images"]), len(synth["annotations"])))

    # person 兜底数据：伪标签（①）与 COCO 人体子集（③）各带自己的 annotations.json，同样并进来
    # （split 仍由 manifest 决定，所以只会在 train 出现）
    if args.merge_person:
        for label, person_path in (("pseudo_person", base_layout.PSEUDO_PERSON_DIR / "annotations.json"),
                                   ("coco_person", base_layout.COCO_PERSON_DIR / "annotations.json")):
            if not person_path.exists():
                print("（缺 %s，跳过 %s）" % (person_path.relative_to(base_layout.DATASET_ROOT), label))
                continue
            extra = json.loads(person_path.read_text(encoding="utf-8"))
            have_names = {img["file_name"] for img in coco["images"]}
            keep_ids = {img["id"] for img in extra["images"] if img["file_name"] not in have_names}
            if len(keep_ids) != len(extra["images"]):
                print("  %s：跳过 %d 张重复图" % (label, len(extra["images"]) - len(keep_ids)))
            extra["images"] = [img for img in extra["images"] if img["id"] in keep_ids]
            extra["annotations"] = [a for a in extra["annotations"] if a["image_id"] in keep_ids]
            next_id = max((img["id"] for img in coco["images"]), default=0) + 1
            id_map: dict[int, int] = {}
            for img in extra["images"]:
                id_map[img["id"]] = next_id
                coco["images"].append({"id": next_id, "file_name": img["file_name"],
                                       "width": img["width"], "height": img["height"]})
                next_id += 1
            next_ann = max((a["id"] for a in coco["annotations"]), default=0) + 1
            for ann in extra["annotations"]:
                coco["annotations"].append({**ann, "id": next_ann, "image_id": id_map[ann["image_id"]]})
                next_ann += 1
            print("并入 %s: %d 图 / %d 框" % (label, len(extra["images"]), len(extra["annotations"])))

    repaired, dropped = sanitize(coco)
    print("框清洗: 裁回图内 %d，丢弃 %d" % (repaired, dropped))

    # 只保留 manifest 里登记过的图，并带上 split
    # 人工判废（S3 里删成 0 框 / 合成抽检不合格）→ 整张图不进任何 split
    excluded_path = base_layout.excluded_path()
    excluded = set()
    if excluded_path.exists():
        excluded = {line.strip() for line in excluded_path.read_text(encoding="utf-8").split() if line.strip()}
    if excluded:
        before, before_ann = len(coco["images"]), len(coco["annotations"])
        keep_ids = {img["id"] for img in coco["images"] if Path(img["file_name"]).name not in excluded}
        coco["images"] = [img for img in coco["images"] if img["id"] in keep_ids]
        coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in keep_ids]
        print("按判废清单剔除 %d 图 / %d 框（判废清单 %d 条）"
              % (before - len(coco["images"]), before_ann - len(coco["annotations"]), len(excluded)))
    known = [img for img in coco["images"] if img["file_name"] in manifest]
    unknown = len(coco["images"]) - len(known)
    if unknown:
        print("!! 有 %d 张图不在 manifest 里，已跳过" % unknown)
    assert_no_leak(manifest, [img["file_name"] for img in known])
    # 落盘到 <split>/images/<basename> 后必须还是唯一名（否则会互相覆盖）
    basenames = Counter(Path(img["file_name"]).name for img in known)
    clashes = [n for n, c in basenames.items() if c > 1]
    if clashes:
        raise ValueError(f"图片文件名冲突（会覆盖）：{clashes[:5]}")

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for img in known:
        by_split[manifest[img["file_name"]]["split"]].append(img)

    stats = []
    for split in base_layout.SPLITS:
        images = by_split.get(split, [])
        if not images:
            print("!! split %s 没有图（roboflow 布局要求三目录都存在）" % split)
            stats.append({"split": split, "images": 0, "annotations": 0})
            continue
        if args.dry_run:
            stats.append({"split": split, "images": len(images), "annotations": 0})
            continue
        stats.append(build_split(coco, images, base_layout.split_dir(split), args.link, split))

    print("-" * 84)
    for row in stats:
        print("  %-6s 图 %4d 框 %5d  %s" % (row["split"], row["images"], row["annotations"],
                                          row.get("path", "(dry-run)")))
    counts = Counter()
    for ann in coco["annotations"]:
        counts[base_layout.id_to_name()[ann["category_id"]]] += 1
    print("类别实例数:", dict(counts), "| 自动标为 cone 的:", auto_changed)
    if not args.dry_run:
        (base_layout.REPORTS_DIR / "dataset_convert_last.json").write_text(
            json.dumps({"args": vars(args), "stats": stats, "class_counts": dict(counts),
                        "auto_cones": auto_changed, "repaired": repaired, "dropped": dropped},
                       ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
