#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S3 人工校验往返：待标包导出 → GUI（Label Studio / CVAT）改 → 回收 + 差异报告。

设计要点（都是实操踩过的）：

* **基线 = `work/annotations/preannot.json`（person/ball 自动框）+ 锥候选（score>=`--cone-min-score`）**
  → 未复核的帧保留基线，复核过的帧用人工结果覆盖；这样 review.json 永远是"全量、可直接训练"的。
* **只按类别名映射**（`classes.json` 是唯一真源）：Label Studio / CVAT 导出时都会**重排 category id**
  （常变成 0-based），按 id 映射必错。
* **删除语义**：某张图人工删成 0 框 → 记进 `work/annotations/excluded.txt`，`s3_label_convert_dataset.py` 会跳过该图
  （合成图抽检不合格时用它）。
* 导出两种产物：`<name>.zip`（images/ + preannot.json + 标注规范 + batch.json，给 CVAT / 手工）
  和 `ls_tasks.json`（Label Studio 任务，图片用 local-files URL，不复制不打包）。

用法::

    # 1) 出待标包（按视频分组，一批 40 张）
    $PY s3_label_review_roundtrip.py --export --split train --source raw_video --limit 40
    $PY s3_label_review_roundtrip.py --export --split all --source ai_cut,ai_bg_cut --synth-sample 0.1

    # 2) Label Studio：起服务 → 建项目(Labels: person/ball/cone) → 导入 ls_tasks.json → 改 → Export COCO
    #    （CVAT：task 上传 zip 里的 images/ → Upload annotations → COCO 1.0 → 改 → Export COCO 1.0）
    $PY s3_label_review_roundtrip.py --ls-tasks <batch>              # 打印导入用的 curl 命令

    # 3) 回收
    $PY s3_label_review_roundtrip.py --import <导出.json> --batch <batch>
    $PY s3_label_review_roundtrip.py --status                        # 看各批次进度
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime
from urllib.parse import quote
from pathlib import Path
from typing import Any, Iterable, Sequence

import base_layout

#: 批次目录（reports/review_batches/<name>/）
BATCH_ROOT = base_layout.REVIEW_BATCH_DIR
#: 回收后的全量标注（s3_label_convert_dataset.py 优先用它）
REVIEW_JSON = base_layout.GT_JSON          # 写入口径固定为 GT.json（权威人工标注）
#: 人工判定"该图作废"的清单（s3_label_convert_dataset.py 会跳过）
EXCLUDED_TXT = base_layout.GT_EXCLUDED_TXT
#: 已复核帧台账（相对路径清单；recall 用，避免靠猜 review.json 里的标记）
REVIEWED_JSON = base_layout.GT_REVIEWED_JSON
#: 差值报告
DIFF_MD = base_layout.REPORTS_DIR / "review_diff.md"
DIFF_CSV = base_layout.REPORTS_DIR / "review_diff.csv"
#: bbox 判定"移动"的 IoU 阈值（同一类别下低于它算移动而非原样保留）
MOVE_IOU = 0.5


# ---------------------------------------------------------------- 基础工具

def read_manifest() -> list[dict[str, str]]:
    with base_layout.MANIFEST_CSV.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_baseline(cone_min_score: float) -> dict[str, Any]:
    """基线 COCO。

    优先级：`work/annotations/review.json`（你当前的标注状态）→ `preannot.json`（bootstrap）。
    用 review.json 当基线，二次复核才不会把已经清掉的误检又显示出来。
    **只有在退回 bootstrap 基线时**才应用锥候选自动改类（否则会覆盖人工结论）。
    """
    use_review = REVIEW_JSON.exists()
    path = REVIEW_JSON if use_review else (base_layout.ANNOTATION_DIR / "preannot.json")
    if not path.exists():
        raise SystemExit("缺少 %s，先跑 s2_data_preannotate.py" % path)
    if use_review:
        print("基线 = 当前标注状态 %s（增量复核；不再应用锥候选自动改类）"
              % path.relative_to(base_layout.DATASET_ROOT))
        cone_min_score = 0.0
    coco = json.loads(path.read_text(encoding="utf-8"))
    # 候选改类复用 dataset_convert 的实现（两处各写一份会漂：实测 47 vs 74）
    from s3_label_convert_dataset import apply_cone_candidates, load_candidates
    changed = 0
    candidates = base_layout.REPORTS_DIR / "preannot_cone_candidates.csv"
    if candidates.exists() and cone_min_score > 0:
        changed = apply_cone_candidates(coco, load_candidates(candidates, cone_min_score))
    # 合成样本的 GT 是粘贴几何算出来的，本就在 work/ai_cut|ai_bg_cut/annotations.json 里，
    # 并进基线后合成抽检批次才有框可核对（早先漏了 → 抽检批次 0 框）
    next_image = max((img["id"] for img in coco["images"]), default=0) + 1
    next_ann = max((a["id"] for a in coco["annotations"]), default=0) + 1
    added_images = added_anns = 0
    for comp in (base_layout.AI_CUT_ANNOTATIONS, base_layout.AI_BG_CUT_ANNOTATIONS):
        if not comp.exists():
            continue
        data = json.loads(comp.read_text(encoding="utf-8"))
        remap: dict[int, int] = {}
        for img in data.get("images", []):
            remap[img["id"]] = next_image
            coco["images"].append({**img, "id": next_image})
            next_image += 1
            added_images += 1
        for ann in data.get("annotations", []):
            coco["annotations"].append({**ann, "id": next_ann,
                                        "image_id": remap[ann["image_id"]]})
            next_ann += 1
            added_anns += 1
    if added_images:
        print("  基线并入合成样本 %d 图 / %d 框（GT 来自粘贴几何）" % (added_images, added_anns))
    print("基线：%d 图 / %d 框（其中脚本自动改类 cone %d 个，阈值 %.2f）"
          % (len(coco["images"]), len(coco["annotations"]), changed, cone_min_score))
    return coco


def empty_coco() -> dict[str, Any]:
    return {"info": {"description": "ball_cone_v1 人工校验（review）",
                     "date_created": datetime.now().isoformat(timespec="seconds")},
            "licenses": [], "categories": base_layout.categories(),
            "images": [], "annotations": []}


def find_image(rel: str) -> Path | None:
    """把各种写法的图片路径归一到磁盘真实文件。"""
    raw = rel.split("?d=")[-1].split("&")[0]
    raw = raw.replace("file://", "").lstrip("/")
    cands = [raw, raw.split("images/", 1)[-1]]
    for cand in cands:
        for base in (base_layout.DATASET_ROOT, base_layout.DATASET_DIR, base_layout.DATASET_ROOT / "dataset"):
            path = (base / cand)
            if path.exists() and path.is_file():
                return path
        for split in base_layout.SPLITS:
            path = base_layout.split_images(split) / Path(cand).name
            if path.exists() and path.is_file():
                return path
    return None


def normalize_rel(rel: str, manifest: dict[str, dict[str, str]]) -> str:
    """归一成 manifest 里的 key（sources/... 或 work/...）。"""
    name = Path(rel.split("?d=")[-1].split("&")[0]).name
    if rel in manifest:
        return rel
    for key, row in manifest.items():
        if Path(key).name == name or Path(row.get("image", "")).name == name:
            return key
    return name


def bbox_xywh(box: Sequence[float]) -> list[float]:
    """COCO/多数工具是 [x,y,w,h]；宽高非正时按 [x1,y1,x2,y2] 兜底转换。"""
    x1, y1, a, b = (float(v) for v in box[:4])
    if a <= 0 or b <= 0:
        if a > x1 and b > y1:
            return [x1, y1, a - x1, b - y1]
        raise ValueError("bbox 无法解析（宽高非正）：%s" % (box,))
    return [x1, y1, a, b]


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------- 增量合并

def human_from_review() -> dict[str, list[dict[str, float]]]:
    """从现有 `GT.json` 取"已复核"的图及其人工框。

    为什么需要：`--import` 是**逐批**跑的，每次都会用基线重建 `review.json`；
    如果只按"当前批次"跳过基线，之前批次的成果会被基线覆盖（实测会丢）。
    这里把所有已复核的图（含 0 框的判废图）都取出来，合并时原样带回。
    """
    out: dict[str, list[dict[str, float]]] = {}
    if REVIEW_JSON.exists():
        data = json.loads(REVIEW_JSON.read_text(encoding="utf-8"))
        id_to_rel = {img["id"]: img["file_name"] for img in data["images"]}
        for ann in data.get("annotations", []):
            if str(ann.get("attributes", {}).get("review")) == "human":
                out.setdefault(id_to_rel[ann["image_id"]], []).append(
                    {"category": ann["category_id"], "bbox": [float(v) for v in ann["bbox"]]})
    for rel in reviewed_images():          # 0 框的判废图也要保留"已复核"语义
        out.setdefault(rel, [])
    return out


def append_image(merged: dict[str, Any], rel: str, items: list[dict[str, float]],
                 by_rel: dict[str, dict[str, str]], source_of: dict[str, str]) -> None:
    """把一张图及其框追加进 merged（自动分配 id）。"""
    next_image = max((img["id"] for img in merged["images"]), default=0) + 1
    merged["images"].append({"id": next_image, "file_name": rel,
                             "width": int(by_rel.get(rel, {}).get("width") or 0),
                             "height": int(by_rel.get(rel, {}).get("height") or 0)})
    for item in items:
        x, y, w, h = (float(v) for v in item["bbox"])
        merged["annotations"].append({
            "id": max((a["id"] for a in merged["annotations"]), default=0) + 1,
            "image_id": next_image, "category_id": int(item["category"]),
            "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2), "iscrowd": 0,
            "attributes": {"source": "human_review", "review": "human",
                           "origin": source_of.get(rel, "")}})


# ---------------------------------------------------------------- 导出

def batched_images() -> set[str]:
    """已经进过任何批次的图（避免多次 --export 拿到同一批帧）。"""
    out: set[str] = set()
    for meta_path in BATCH_ROOT.glob("*/batch.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out |= {item["image"] for item in meta.get("images", [])}
    return out


MIN_TINY_PX = 5.0


def tiny_images(baseline: dict[str, Any]) -> set[str]:
    """基线里仍含 <MIN_TINY_PX（960 输入像素）框的图（与 dataset_stats 过小框哨兵同口径）。"""
    size = {img["id"]: (img.get("width") or 1280, img.get("height") or 720)
            for img in baseline["images"]}
    name = {img["id"]: img["file_name"] for img in baseline["images"]}
    out: set[str] = set()
    for ann in baseline["annotations"]:
        width, height = size.get(ann["image_id"], (1280, 720))
        factor = 960.0 / max(width, height)
        if min(ann["bbox"][2] * factor, ann["bbox"][3] * factor) < MIN_TINY_PX:
            out.add(name[ann["image_id"]])
    return out


def pick_images(manifest: list[dict[str, str]], split: str, sources: set[str],
                limit: int | None, synth_sample: float, include_reviewed: bool,
                done: set[str]) -> list[dict[str, str]]:
    """按 split/来源挑帧；合成类按 `synth_sample` 抽检。同源组内保持顺序（视频连续）。"""
    picked: list[dict[str, str]] = []
    seen_synth = 0
    for row in manifest:
        key = row["image"]
        if key in done and not include_reviewed:
            continue
        if split != "all" and row.get("split") != split:
            continue
        source = str(row.get("source", ""))
        if sources and source not in sources:
            continue
        if row.get("is_synthetic") == "yes":
            seen_synth += 1
            if synth_sample <= 0:
                continue
            # 每 1/synth_sample 张取 1 张（确定性抽检，便于复算）
            if int(round(1 / synth_sample)) <= 0 or seen_synth % int(round(1 / synth_sample)):
                continue
        picked.append(row)
        if limit and len(picked) >= limit:
            break
    return picked


def cmd_export(args: argparse.Namespace) -> int:
    manifest = read_manifest()
    rows = {row["image"]: row for row in manifest}
    baseline = load_baseline(args.cone_min_score)
    if args.source:
        sources = {s.strip() for s in args.source.split(",") if s.strip()}
    else:
        sources = set()
    # --focus tiny 是"二次复核已复核过的图"，必须允许重导（否则会被 batched_images() 全跳过）
    if args.focus == "tiny":
        # 二次复核：这些图既在之前的批次里、也已被标记"已复核"，两个跳过条件都要关掉
        args.reuse = True
        args.include_reviewed = True
        print("--focus tiny：自动打开 --reuse + --include-reviewed（本来就是复核过的图）")
    done = reviewed_images() | (set() if args.include_reviewed or args.reuse
                               else batched_images())
    picked = pick_images(manifest, args.split, sources, args.limit, args.synth_sample,
                         args.include_reviewed, done)
    if args.focus == "tiny":
        keep = tiny_images(baseline)
        before = len(picked)
        picked = [row for row in picked if row["image"] in keep]
        print("--focus tiny：%d → %d 图（只保留仍含 <%.0fpx 框的图）"
              % (before, len(picked), MIN_TINY_PX))
    if not picked:
        print("没有可导出的帧（检查 --split/--source/是否已全部复核）")
        return 2

    name = args.batch_name or ("batch_%s" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir = BATCH_ROOT / name
    if out_dir.exists() and not args.overwrite:
        print("批次已存在：%s（用 --overwrite 覆盖，或换 --batch-name）" % out_dir)
        return 2
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    base_image_by_rel = {img["file_name"]: img for img in baseline["images"]}
    batch = empty_coco()
    id_map: dict[int, int] = {}
    for index, row in enumerate(picked, 1):
        rel = row["image"]
        src = find_image(rel)
        if src is None:
            print("!! 找不到图片，跳过：%s" % rel)
            continue
        base = base_image_by_rel.get(rel) or base_image_by_rel.get("images/" + Path(rel).name)
        batch["images"].append({"id": index, "file_name": rel, "width": int(row.get("width") or 0),
                                "height": int(row.get("height") or 0)})
        if base:
            id_map[base["id"]] = index
    for ann in baseline["annotations"]:
        new_id = id_map.get(ann["image_id"])
        if new_id is None:
            continue
        batch["annotations"].append({**ann, "id": len(batch["annotations"]) + 1, "image_id": new_id})

    meta = {"batch": name, "created": datetime.now().isoformat(timespec="seconds"),
            "split": args.split, "sources": sorted(sources) or "all",
            "synth_sample": args.synth_sample, "cone_min_score": args.cone_min_score,
            "images": [{"image": img["file_name"],
                        "source": rows.get(img["file_name"], {}).get("source", ""),
                        "split": rows.get(img["file_name"], {}).get("split", ""),
                        "is_synthetic": rows.get(img["file_name"], {}).get("is_synthetic", "no")}
                       for img in batch["images"]],
            "status": "exported"}

    # COCO（给 CVAT / 手工改）
    (out_dir / "preannot.json").write_text(json.dumps(batch, ensure_ascii=False), encoding="utf-8")
    # 标注规范
    spec_src = Path(__file__).parent / "docs" / "annotation_spec.md"
    if spec_src.exists():
        shutil.copy2(spec_src, out_dir / "annotation_spec.md")
    # Label Studio 任务（图片走 local-files URL，不复制）
    tasks = ls_tasks(batch, rows)
    (out_dir / "ls_tasks.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    # 待标包 zip（含 images/，给 CVAT / 离线标注）
    with zipfile.ZipFile(out_dir / ("%s.zip" % name), "w", zipfile.ZIP_DEFLATED) as zf:
        for img in batch["images"]:
            src = find_image(img["file_name"])
            if src is not None:
                zf.write(src, "images/%s" % Path(img["file_name"]).name)
        for extra in ("preannot.json", "annotation_spec.md"):
            path = out_dir / extra
            if path.exists():
                zf.write(path, extra)
        zf.writestr("batch.json", json.dumps(meta, ensure_ascii=False, indent=1))
    (out_dir / "batch.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    cones = sum(1 for a in batch["annotations"] if a["category_id"] == base_layout.name_to_id()["cone"])
    print("批次 %s：%d 图 / %d 框（cone %d）" % (name, len(batch["images"]), len(batch["annotations"]), cones))
    print("  → %s" % out_dir)
    print("  → %s.zip（images/ + preannot.json + 规范，CVAT 上传用）" % name)
    print("  → ls_tasks.json（Label Studio 导入用；先 `--ls-tasks %s` 看 curl）" % name)
    print("  合成帧抽检比例：%s；来源：%s" % (args.synth_sample, meta["sources"]))
    return 0


def ls_tasks(batch: dict[str, Any], rows: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """把 COCO 批次转成 Label Studio 任务（含 predictions，人工只需改错的地方）。"""
    names = {c["id"]: c["name"] for c in batch["categories"]}
    doc_root = str(base_layout.DATASET_ROOT)
    by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in batch["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    tasks = []
    for img in batch["images"]:
        w = img.get("width") or 1280
        h = img.get("height") or 720
        results = []
        for ann in by_image.get(img["id"], []):
            x, y, bw, bh = ann["bbox"]
            results.append({
                "id": "%d_%d" % (img["id"], ann["id"]),
                "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                "original_width": w, "original_height": h,
                "image_rotation": 0,
                "value": {"x": 100.0 * x / w, "y": 100.0 * y / h,
                          "width": 100.0 * bw / w, "height": 100.0 * bh / h,
                          "rotation": 0, "rectanglelabels": [names[ann["category_id"]]]},
            })
        tasks.append({
            # 图片走 local-files：路径里可能有空格/中文（"1V2 2/..."），必须 URL 编码，否则 LS 打不开
            "data": {"image": "/data/local-files/?d=%s" % quote(img["file_name"]),
                     "image_key": img["file_name"],
                     "source": rows.get(img["file_name"], {}).get("source", ""),
                     "split": rows.get(img["file_name"], {}).get("split", "")},
            "predictions": ([{"model_version": "preannot", "result": results}]
                            if results else []),
            "meta": {"doc_root": doc_root},
        })
    return tasks


def cmd_ls_tasks(args: argparse.Namespace) -> int:
    out_dir = BATCH_ROOT / args.batch
    tasks_path = out_dir / "ls_tasks.json"
    if not tasks_path.exists():
        print("没有 %s，先 --export" % tasks_path)
        return 2
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    print("批次 %s：%d 个任务" % (args.batch, len(tasks)))
    print("\n① 起服务（本机 headless，用 ssh 端口转发访问）：")
    print("   LABEL_STUDIO_LOCAL_FILES_SOURCES=%s \\" % base_layout.DATASET_ROOT)
    print("   LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%s \\" % base_layout.DATASET_ROOT)
    print("   .venv-labelstudio/bin/label-studio start --port 8080 --no-browser")
    print("   本地另开终端：ssh -L 8080:localhost:8080 <user>@<host>  → 浏览器打开 http://localhost:8080")
    print("\n② 页面里：Create Project → 名称随意 → Labeling Setup 加 3 个 RectangleLabels：")
    print("   person / ball / cone（**名字必须完全一致**，大小写敏感）→ Save")
    print("\n③ 导入任务（含预标注），把 <TOKEN> 换成 Settings → Access Token：")
    print("   curl -H 'Authorization: Token <TOKEN>' -F 'file=@%s' \\" % tasks_path)
    print("     http://localhost:8080/api/projects/<PROJECT_ID>/import")
    print("\n④ 校完：Export → COCO（或 JSON）→ 回来跑：")
    print("   $PY s3_label_review_roundtrip.py --import <导出文件> --batch %s" % args.batch)
    return 0


# ---------------------------------------------------------------- Label Studio 自动建项目

LS_LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="#4C8BF5"/>
    <Label value="ball" background="#3CC47C"/>
    <Label value="cone" background="#F5A623"/>
  </RectangleLabels>
</View>
"""


def _ls_request(url: str, method: str, path: str, token: str, payload: dict[str, Any] | None = None,
                timeout: int = 120) -> Any:
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url.rstrip("/") + path, data=data, method=method,
                                headers={"Authorization": "Token %s" % token,
                                         "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:300]
        raise SystemExit("Label Studio API %s %s 失败：HTTP %s %s" % (method, path, exc.code, detail))
    except urllib.error.URLError as exc:
        raise SystemExit("连不上 Label Studio（%s）：%s\n先按 --ls-tasks 里的命令把服务起起来"
                         % (url, exc.reason))


def cmd_ls_setup(args: argparse.Namespace) -> int:
    """建项目（3 个标签：person/ball/cone）+ 导入各批次任务，一条命令到位。"""
    batches = [b.strip() for b in args.batch.split(",") if b.strip()]
    if not batches:
        raise SystemExit("用 --batch a,b,c 指定要导入的批次（或 --batch all）")
    if batches == ["all"]:
        batches = sorted(p.name for p in BATCH_ROOT.iterdir() if (p / "ls_tasks.json").exists())
    projects = _ls_request(args.ls_url, "GET", "/api/projects", args.ls_token)
    found = [p for p in (projects.get("results", projects) if isinstance(projects, dict) else projects)
             if p.get("title") == args.project_name]
    if found:
        project = found[0]
        print("复用已有项目 #%s %s" % (project["id"], project["title"]))
    else:
        project = _ls_request(args.ls_url, "POST", "/api/projects", args.ls_token,
                              {"title": args.project_name, "label_config": LS_LABEL_CONFIG})
        print("已建项目 #%s %s（标签：person / ball / cone）" % (project["id"], project["title"]))

    total = 0
    for name in batches:
        tasks_path = BATCH_ROOT / name / "ls_tasks.json"
        if not tasks_path.exists():
            print("  跳过 %s（没有 ls_tasks.json，先 --export）" % name)
            continue
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        resp = _ls_request(args.ls_url, "POST", "/api/projects/%s/import" % project["id"],
                           args.ls_token, tasks, timeout=600)
        task_ids = resp.get("task_ids") or resp.get("tasks") or []
        total += len(task_ids) if task_ids else len(tasks)
        print("  导入 %-16s %d 任务" % (name, len(task_ids) if task_ids else len(tasks)))
    print("\n项目地址：%s/projects/%s/  （复核完点 Export → COCO，再跑 --import）"
          % (args.ls_url.rstrip("/"), project["id"]))
    print("累计导入 %d 个任务" % total)
    return 0


# ---------------------------------------------------------------- 导入

def parse_export(path: Path, manifest: dict[str, dict[str, str]]) -> dict[str, list[dict[str, float]]]:
    """读 GUI 导出（COCO 或 Label Studio 原生 JSON），返回 {归一图名: [{category,bbox}]}。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    name_to_id = base_layout.name_to_id()
    out: dict[str, list[dict[str, float]]] = {}

    if isinstance(data, dict) and "categories" in data and "images" in data:   # ---- COCO
        id_to_name = {c["id"]: str(c["name"]).strip().lower() for c in data["categories"]}
        url_to_rel: dict[Any, str] = {}
        for img in data["images"]:
            rel = img.get("file_name") or img.get("coco_url") or img.get("path") or ""
            url_to_rel[img.get("id")] = normalize_rel(rel, manifest)
            out.setdefault(url_to_rel[img.get("id")], [])
        for ann in data.get("annotations", []):
            label = id_to_name.get(ann.get("category_id"))
            if label not in name_to_id:
                raise SystemExit("未知类别名 %r（导出里的 categories 必须叫 person/ball/cone）；"
                                 "检查标注工具里的标签是否与 classes.json 一致" % label)
            key = url_to_rel.get(ann.get("image_id"))
            if key is None:
                continue
            out[key].append({"category": name_to_id[label], "bbox": bbox_xywh(ann["bbox"])})
        return out

    if isinstance(data, list):                                                  # ---- Label Studio 原生
        for task in data:
            info = task.get("data", {})
            rel = normalize_rel(str(info.get("image_key") or info.get("image") or ""), manifest)
            items: list[dict[str, float]] = []
            for block in (task.get("annotations") or task.get("predictions") or []):
                for res in block.get("result", []):
                    if res.get("type") != "rectanglelabels":
                        continue
                    labels = res["value"].get("rectanglelabels") or []
                    label = str(labels[0]).strip().lower() if labels else ""
                    if label not in name_to_id:
                        raise SystemExit("未知标签 %r（必须 person/ball/cone）" % label)
                    w = res.get("original_width") or info.get("width") or 1280
                    h = res.get("original_height") or info.get("height") or 720
                    val = res["value"]
                    items.append({"category": name_to_id[label],
                                  "bbox": [val["x"] / 100.0 * w, val["y"] / 100.0 * h,
                                           val["width"] / 100.0 * w, val["height"] / 100.0 * h]})
            out[rel] = items
        return out
    raise SystemExit("无法识别导出格式（既不是 COCO 也不是 Label Studio JSON）：%s" % path)


def reviewed_images() -> set[str]:
    """已复核过的帧（读 `work/annotations/reviewed.json` 台账）。"""
    if not REVIEWED_JSON.exists():
        return set()
    return set(json.loads(REVIEWED_JSON.read_text(encoding="utf-8")))


def mark_reviewed(keys: Iterable[str]) -> None:
    merged = reviewed_images() | set(keys)
    REVIEWED_JSON.write_text(json.dumps(sorted(merged), ensure_ascii=False, indent=1),
                             encoding="utf-8")


def cmd_import(args: argparse.Namespace) -> int:
    manifest = read_manifest()
    by_rel = {row["image"]: row for row in manifest}
    baseline = load_baseline(args.cone_min_score)
    human = parse_export(Path(args.import_file), by_rel)
    if not human:
        print("导出里没有可用的框（是不是导出了空任务？）")
        return 2

    baseline_by_rel: dict[str, list[dict[str, Any]]] = {}
    id_to_rel = {img["id"]: img["file_name"] for img in baseline["images"]}
    for ann in baseline["annotations"]:
        baseline_by_rel.setdefault(id_to_rel[ann["image_id"]], []).append(ann)

    diff_rows: list[dict[str, Any]] = []
    merged = empty_coco()
    human_keys = set()
    for rel, items in sorted(human.items()):
        if rel not in by_rel and not find_image(rel):
            print("!! 导出里的图不在 manifest 中，跳过：%s" % rel)
            continue
        human_keys.add(rel)
        base = baseline_by_rel.get(rel, [])
        counts = Counter()
        matched = set()
        for item in items:
            best, best_iou = None, 0.0
            for index, ann in enumerate(base):
                if index in matched:
                    continue
                score = iou_xywh(item["bbox"], ann["bbox"])
                if score > best_iou:
                    best, best_iou = index, score
            if best is None or best_iou < 0.1:
                counts["added"] += 1
                diff_rows.append({"image": rel, "change": "added", "from": "",
                                  "to": base_layout.id_to_name()[item["category"]],
                                  "iou": round(best_iou, 2)})
            else:
                matched.add(best)
                old = base[best]
                if old["category_id"] != item["category"]:
                    counts["relabeled"] += 1
                    diff_rows.append({"image": rel, "change": "relabeled",
                                      "from": base_layout.id_to_name()[old["category_id"]],
                                      "to": base_layout.id_to_name()[item["category"]],
                                      "iou": round(best_iou, 2)})
                elif best_iou < MOVE_IOU:
                    counts["moved"] += 1
                    diff_rows.append({"image": rel, "change": "moved", "from": "",
                                      "to": base_layout.id_to_name()[item["category"]],
                                      "iou": round(best_iou, 2)})
                else:
                    counts["kept"] += 1
        for index, ann in enumerate(base):
            if index not in matched:
                counts["deleted"] += 1
                diff_rows.append({"image": rel, "change": "deleted",
                                  "from": base_layout.id_to_name()[ann["category_id"]], "to": "",
                                  "iou": 0.0})
        diff_rows.append({"image": rel, "change": "summary",
                          "from": " ".join("%s=%d" % (k, counts[k]) for k in
                                           ("kept", "relabeled", "added", "deleted", "moved")),
                          "to": "", "iou": 0.0})

    # 合并写 review.json：**增量**——之前批次已复核的图原样带回，本批用新结果，其余保留基线
    source_of = {row["image"]: row.get("source", "") for row in manifest}
    previous = human_from_review()
    carry = {rel: boxes for rel, boxes in previous.items() if rel not in human_keys}
    merged = empty_coco()
    merge_baseline_into(merged, baseline, skip=human_keys | set(carry))
    for rel in sorted(carry):
        append_image(merged, rel, carry[rel], by_rel, source_of)
    if carry:
        print("  带回此前已复核 %d 图（增量合并，不会被基线覆盖）" % len(carry))
    excluded: list[str] = []
    for rel in sorted(human_keys):
        items = human[rel]
        if not items:
            excluded.append(Path(rel).name)
        append_image(merged, rel, items, by_rel, source_of)

    REVIEW_JSON.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_JSON.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    mark_reviewed(human_keys)
    if excluded:
        old = set(EXCLUDED_TXT.read_text(encoding="utf-8").split()) if EXCLUDED_TXT.exists() else set()
        EXCLUDED_TXT.write_text("\n".join(sorted(old | set(excluded))) + "\n", encoding="utf-8")

    acc = getattr(args, "_accumulate", None)
    if acc is None:
        write_diff(diff_rows, human_keys, args, excluded)
    else:                                   # --import-all：攒起来最后写一份汇总
        acc["rows"].extend(diff_rows)
        acc["keys"] |= set(human_keys)
        acc["excluded"].extend(excluded)
        acc["batches"].append(args.batch)
    print("回收 %d 图 → %s（累计 %d 图 / %d 框）"
          % (len(human_keys), REVIEW_JSON.relative_to(base_layout.DATASET_ROOT),
             len(merged["images"]), len(merged["annotations"])))
    if excluded:
        print("  作废 %d 图 → %s（dataset_convert 会跳过）"
              % (len(excluded), EXCLUDED_TXT.relative_to(base_layout.DATASET_ROOT)))
    if acc is None:                          # --import-all 时只写一份汇总，这里不重复打印
        print("  差异报告 → %s / %s" % (DIFF_MD.relative_to(base_layout.DATASET_ROOT),
                                     DIFF_CSV.relative_to(base_layout.DATASET_ROOT)))
    update_batch_status(args.batch, len(human_keys))
    return 0


def merge_baseline_into(merged: dict[str, Any], baseline: dict[str, Any], skip: set[str]) -> None:
    """把基线里"未复核"的帧原样搬进 merged（从 1 重新编号）。"""
    id_to_rel = {img["id"]: img["file_name"] for img in baseline["images"]}
    by_image: dict[str, list[dict[str, Any]]] = {}
    for ann in baseline["annotations"]:
        by_image.setdefault(id_to_rel[ann["image_id"]], []).append(ann)
    for index, img in enumerate([i for i in baseline["images"] if i["file_name"] not in skip], 1):
        merged["images"].append({"id": index, "file_name": img["file_name"],
                                 "width": img["width"], "height": img["height"]})
        for ann in by_image.get(img["file_name"], []):
            merged["annotations"].append({**ann, "id": len(merged["annotations"]) + 1,
                                          "image_id": index})


def cmd_import_all(args: argparse.Namespace) -> int:
    """把所有 `<batch>_edited.json` 依次回收（各自增量合并，不会互相覆盖）。"""
    files = sorted(base_layout.REVIEW_BATCH_DIR.glob("*/*_edited.json"))
    if not files:
        print("没有找到任何 <batch>_edited.json；先在界面上点“保存”")
        return 2
    print("发现 %d 个已保存批次：" % len(files))
    for path in files:
        print("  %s" % path.relative_to(base_layout.DATASET_ROOT))
    args._accumulate = {"rows": [], "keys": set(), "excluded": [], "batches": []}
    rc = 0
    for path in files:
        batch = path.parent.name
        print("\n—— 回收 %s ——" % batch)
        args.import_file = str(path)
        args.batch = batch
        try:
            rc |= cmd_import(args)
        except SystemExit as exc:                      # 单批失败不阻塞其它批
            print("  跳过（%s）" % exc)
            rc = 1
    acc = args._accumulate
    args.batch = "%d 批汇总（%s）" % (len(acc["batches"]), "、".join(acc["batches"]))
    write_diff(acc["rows"], acc["keys"], args, acc["excluded"])
    totals = Counter(row["change"] for row in acc["rows"] if row["change"] != "summary")
    print("\n=== 全部回收完成 ===")
    print("  复核帧 %d | 改类 %d · 新增 %d · 删除 %d · 移动 %d"
          % (len(acc["keys"]), totals["relabeled"], totals["added"], totals["deleted"],
             totals["moved"]))
    if acc["excluded"]:
        print("  判废 %d 图（进 excluded.txt，不进任何 split）" % len(acc["excluded"]))
    print("  汇总差异报告 → reports/review_diff.md / review_diff.csv")
    print("接着跑：$PY s3_label_convert_dataset.py --link && $PY s3_label_dataset_stats.py")
    return rc


def write_diff(rows: list[dict[str, Any]], human_keys: set[str], args: argparse.Namespace,
               excluded: list[str]) -> None:
    DIFF_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DIFF_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "change", "from", "to", "iou"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    totals = Counter(row["change"] for row in rows if row["change"] != "summary")
    lines = ["# S3 人工校验差异报告", "",
             "- 批次：`%s`" % args.batch, "- 导出文件：`%s`" % args.import_file,
             "- 复核帧数：**%d**" % len(human_keys),
             "- 变化：改类 **%d** · 新增 **%d** · 删除 **%d** · 移动 **%d**（IoU<%.1f）"
             % (totals["relabeled"], totals["added"], totals["deleted"], totals["moved"], MOVE_IOU), ""]
    if excluded:
        lines += ["- 作废（0 框）**%d** 张：%s" % (len(excluded), "、".join(excluded[:10])), ""]
    per_image = Counter(row["image"] for row in rows if row["change"] != "summary")
    lines += ["## 变化最多的图", "", "| 图 | 变化数 |", "| --- | --- |"]
    for image, count in per_image.most_common(15):
        lines.append("| `%s` | %d |" % (image, count))
    lines += ["", "逐条明细见 `review_diff.csv`。"]
    DIFF_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_batch_status(batch: str | None, count: int) -> None:
    if not batch:
        return
    meta_path = BATCH_ROOT / batch / "batch.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["status"] = "imported"
    meta["imported"] = count
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 状态

def cmd_status(_: argparse.Namespace) -> int:
    print("批次目录：%s" % BATCH_ROOT)
    if not BATCH_ROOT.exists() or not any(BATCH_ROOT.iterdir()):
        print("  （还没有批次，先 --export）")
    for meta_path in sorted(BATCH_ROOT.glob("*/batch.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print("  %-28s %-9s 图 %3d  %s" % (meta["batch"], meta.get("status", "?"),
                                          len(meta.get("images", [])), meta.get("created", "")))
    reviewed = reviewed_images()
    print("已复核帧总数：%d" % len(reviewed))
    if REVIEW_JSON.exists():
        data = json.loads(REVIEW_JSON.read_text(encoding="utf-8"))
        counts = Counter(base_layout.id_to_name()[a["category_id"]] for a in data["annotations"])
        print("GT.json：%d 图 / %d 框 %s" % (len(data["images"]), len(data["annotations"]),
                                                dict(counts)))
    if EXCLUDED_TXT.exists():
        print("作废图：%d 张（%s）" % (len(EXCLUDED_TXT.read_text(encoding="utf-8").split()),
                                    EXCLUDED_TXT.relative_to(base_layout.DATASET_ROOT)))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S3 人工校验往返（导出待标包 / 回收 + 差异报告）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", action="store_true", help="导出待标包")
    group.add_argument("--ls-tasks", metavar="BATCH", help="打印 Label Studio 导入步骤（含 curl）")
    group.add_argument("--ls-setup", action="store_true",
                       help="建 Label Studio 项目并导入各批次（需服务已启动 + token）")
    group.add_argument("--import", dest="import_file", metavar="EXPORT_JSON",
                       help="回收 GUI 导出结果")
    group.add_argument("--import-all", action="store_true",
                       help="回收所有批次已保存的 `<batch>_edited.json`（增量合并，一条命令）")
    group.add_argument("--status", action="store_true", help="看批次与复核进度")
    parser.add_argument("--split", default="train", help="train|valid|test|all（默认 train）")
    parser.add_argument("--source", default="", help="逗号分隔：raw_video,ai_cut,ai_bg_cut...")
    parser.add_argument("--limit", type=int, default=40, help="本批最多多少图（0=不限）")
    parser.add_argument("--focus", default="all", choices=["all", "tiny"],
                        help="tiny=只导出仍含 <5px 框的图（配合 dataset_stats 哨兵做精准清理）")
    parser.add_argument("--synth-sample", type=float, default=0.0,
                        help="合成帧抽检比例（如 0.1 = 每 10 张取 1 张）")
    parser.add_argument("--batch-name", default="", help="批次名（默认按时间戳）")
    parser.add_argument("--batch", default="", help="回收/查看时指定批次名")
    parser.add_argument("--cone-min-score", type=float, default=0.6,
                        help="基线里把锥候选自动改 cone 的分数阈值（0=不改）")
    parser.add_argument("--include-reviewed", action="store_true", help="允许重复导出已复核帧")
    parser.add_argument("--overwrite", action="store_true", help="批次同名时覆盖")
    parser.add_argument("--ls-url", default="http://localhost:8080", help="Label Studio 地址")
    parser.add_argument("--ls-token", default="aifootball-token", help="Label Studio API Token")
    parser.add_argument("--project-name", default="ball_cone_v1 人工校验",
                        help="Label Studio 项目名（已存在则复用）")
    parser.add_argument("--reuse", action="store_true",
                        help="允许重复导出已进过批次的帧（默认跳过，保证批次互不重合）")
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_layout.ensure_layout()
    if args.export:
        args.limit = args.limit or None
        return cmd_export(args)
    if args.ls_tasks:
        args.batch = args.ls_tasks
        return cmd_ls_tasks(args)
    if args.ls_setup:
        return cmd_ls_setup(args)
    if args.import_file:
        return cmd_import(args)
    if args.import_all:
        return cmd_import_all(args)
    return cmd_status(args)


if __name__ == "__main__":
    sys.exit(main())
