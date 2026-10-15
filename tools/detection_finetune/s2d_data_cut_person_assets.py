#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人体素材库（多视频 + 高质量闸门）：扫多段视频 → 过闸门 → 去重 → 存 PNG + 台账 + 拼图。

为什么要单独一个脚本
--------------------
`work/cutouts_person/`（s5_eval_perf_persons 的 `--cutouts-only`）当时只为"合成测速帧"服务，
闸门只挡退化小框（短边 < 8 px），于是**大量远景小人混进素材库**；这些小人贴回原框原位虽然还算
"原像素"，但一放大到拼图里就是几十像素的糊块，既看不出质量也没法复用。本脚本把"人体素材"
当成一等产物来做，**只收高质量人体**。

七道闸门（都可关/可调，拒绝理由全部记进 index.json 便于调参）
------------------------------------------------------------
1. **框高** `--min-height`（默认 120 px）：主体门，直接砍掉远景小人；
2. **框面积** `--min-area`（默认 8000 px²）；
3. **形态** `--aspect-min/--aspect-max`（默认 1.05–4.5，高宽比）：人直立约 2–3，
   <1.05 多为横躺误检，>4.5 多为杆状误检；
4. **截断** `--edge-margin`（默认 2 px）：框贴画面边 → 人也出画了，属残缺；
5. **重叠** `--max-overlap-iou`（默认 0.30）：与同帧其他人框重叠超阈值 → 框里不止一个人；
6. **清晰度** `--min-sharp`（默认 60，拉普拉斯方差）→ 糊/拖影的不要；
7. **去重** `--dedupe-iou`（默认 0.50）：跨帧同一个人的相邻框只留**最清晰**的那张。

融合裁剪
--------
沿框外扩 `--pad`（默认 0.08）后**整块裁下**（不做像素级抠像——人像 GrabCut 失败率高，
实测 86 个里 53 个掉到 Otsu，抠出来全是白块条纹）。素材用途是"贴回原框原位合成受控人数帧"，
原位贴 → 像素与原帧逐点一致 → **无矩形接缝**。若要挪位/换背景，另需分割模型出掩码。

用法::

    source env.sh
    $PY s2d_data_cut_person_assets.py                          # 自动挑 6 段人多视频
    $PY s2d_data_cut_person_assets.py --video-limit 10 --min-height 160
    $PY s2d_data_cut_person_assets.py --videos 移动传球.mov,颠球.mov
    $PY s2d_data_cut_person_assets.py --list-videos             # 只看会自动挑哪些
    $PY s2d_data_cut_person_assets.py --no-sharp-gate           # 关掉清晰度闸门

产物：`<data>/ball_cone_v1/work/cutouts_person_hq/{*.png, index.json, contact_sheet.jpg}`；
报告 `<data>/ball_cone_v1/reports/person_assets.md`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import base_layout
import s5_eval_render as rd

HERE = Path(__file__).resolve().parent
OUT_DIR = base_layout.WORK_DIR / "cutouts_person_hq"
REPORT_MD = base_layout.REPORTS_DIR / "person_assets.md"
DEFAULT_CHECKPOINT = HERE / "runs" / "v1-frozen-step2" / "checkpoint_best_ema.pth"
VIDEO_EXT = (".mov", ".mp4", ".mkv", ".avi")
# 自动挑视频时的偏好词（人多、距离近的场景优先）
PREFER_KEYS = ("比赛", "训练", "传球", "颠球", "1V2", "带球", "绕障碍")


def discover_videos(root: Path, limit: int, min_mb: float) -> list[Path]:
    """自动挑源视频：先按"含人多场景关键词"，再按体积降序（体积≈信息量），取前 limit 个。"""
    found = [p for p in root.rglob("*")
             if p.suffix.lower() in VIDEO_EXT and p.stat().st_size / 1e6 >= min_mb]

    def rank(p: Path) -> tuple[int, float]:
        rel = str(p.relative_to(root))
        return (0 if any(k in rel for k in PREFER_KEYS) else 1, -p.stat().st_size)

    return sorted(found, key=rank)[:limit]


def resolve_videos(args: argparse.Namespace) -> list[Path]:
    """把 `--videos`（auto / 逗号分隔路径或文件名 / glob）解析成真实文件列表。"""
    if args.videos.strip().lower() == "auto":
        return discover_videos(base_layout.DATA_ROOT, args.video_limit, args.min_video_mb)
    out: list[Path] = []
    for raw in (x.strip() for x in args.videos.split(",")):
        if not raw:
            continue
        cand = Path(raw)
        if cand.is_file():
            out.append(cand)
            continue
        # 允许只写文件名（在 DATA_ROOT 下递归找）或 glob
        hits = sorted(base_layout.DATA_ROOT.rglob(raw)) if any(c in raw for c in "*?") else \
            sorted(base_layout.DATA_ROOT.rglob(Path(raw).name))
        if not hits:
            raise SystemExit("找不到视频：%s（可传绝对/相对路径、文件名或 glob）" % raw)
        out.extend(hits)
    return out


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两个 xyxy 框的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def sharpness(image: np.ndarray) -> float:
    """拉普拉斯方差（越大越清晰）。参考值：清晰人像 > 100，拖影/失焦 < 40。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_from_box(image: np.ndarray, box: Sequence[float], pad: float
                  ) -> tuple[np.ndarray, list[int]]:
    """按框外扩 pad 后整块裁下，返回 (crop_bgr, 裁剪框 xyxy)。"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    px, py = bw * pad, bh * pad
    cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    cx2, cy2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    return image[cy1:cy2, cx1:cx2], [cx1, cy1, cx2, cy2]


def gate(box: Sequence[float], score: float, shape: Sequence[int], others: list[list[float]],
         crop: np.ndarray, a: argparse.Namespace) -> str | None:
    """七道闸门；通过返回 None，否则返回拒绝理由（带数值，便于调参）。"""
    h, w = int(shape[0]), int(shape[1])
    bw, bh = float(box[2]) - float(box[0]), float(box[3]) - float(box[1])
    if score < a.min_conf:
        return "low-conf(%.2f)" % score
    if bh < a.min_height:
        return "too-short(h=%d)" % int(bh)
    if bw * bh < a.min_area:
        return "too-small(a=%d)" % int(bw * bh)
    aspect = bh / max(1.0, bw)
    if not (a.aspect_min <= aspect <= a.aspect_max):
        return "aspect(%.2f)" % aspect
    m = a.edge_margin
    if box[0] <= m or box[1] <= m or box[2] >= w - m or box[3] >= h - m:
        return "truncated"
    if a.max_overlap_iou > 0 and others:
        worst = max(iou(box, o) for o in others)
        if worst >= a.max_overlap_iou:
            return "overlap(%.2f)" % worst
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 4:
        return "crop-degenerate"
    if a.min_sharp > 0:
        sharp = sharpness(crop)
        if sharp < a.min_sharp:
            return "blur(%.0f)" % sharp
    return None


def scan_video(path: Path, model: Any, a: argparse.Namespace
               ) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """扫一段视频，返回 (通过闸门的素材条目, 拒绝理由计数, 实际扫描帧数) 并落盘 PNG。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit("打不开视频：%s" % path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, a.scan_stride)
    if a.scan_max <= 0:
        idxs = range(0, total, stride)
    else:
        # 在**整段视频上均匀铺开**采样 --- 只扫开头 10 秒会让人物外观/机位高度重复
        step = max(stride, total // max(1, a.scan_max))
        idxs = range(0, total, step)
    kept: list[dict[str, Any]] = []      # 已保留（含 box/sharp 用于去重）
    reasons: dict[str, int] = {}
    scanned = 0
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        scanned += 1
        det = rd.dets_for(model, frame, a.threshold)
        cands = [(list(map(float, b)), float(s)) for b, s, l in
                 zip(det["boxes"], det["scores"], det["labels"]) if l == 1]
        cands.sort(key=lambda c: -(c[0][3] - c[0][1]))          # 高个优先（先占配额）
        boxes = [c[0] for c in cands]
        for i, (box, score) in enumerate(cands):
            if len(kept) >= a.max_per_video:
                break
            others = [b for j, b in enumerate(boxes) if j != i]
            crop, cbox = crop_from_box(frame, box, a.pad)
            reason = gate(box, score, frame.shape, others, crop, a)
            # 去重：与已保留的（同视频跨帧）IoU 高 → 只留更清晰的那张
            if reason is None and a.dedupe_iou > 0:
                dup = max(range(len(kept)), key=lambda k: iou(box, kept[k]["box_xyxy"]), default=-1)
                if dup >= 0 and iou(box, kept[dup]["box_xyxy"]) >= a.dedupe_iou:
                    new_sharp, old_sharp = sharpness(crop), kept[dup]["sharpness"]
                    if new_sharp <= old_sharp:
                        reason = "dup(keep %.0f>%.0f)" % (old_sharp, new_sharp)
                    else:
                        (OUT_DIR / kept[dup]["file"]).unlink(missing_ok=True)
                        reasons["dup(replaced)"] = reasons.get("dup(replaced)", 0) + 1
                        kept.pop(dup)
            if reason is not None:
                key = reason.split("(")[0]
                reasons[key] = reasons.get(key, 0) + 1
                continue
            bh = float(box[3]) - float(box[1])
            sharp = sharpness(crop)
            # 文件名带框左上角坐标 → 保证唯一。**不加会踩坑**：同帧两个人框高与清晰度都相同时
            # 文件名会撞车，后者覆盖前者的 PNG，之后前者被去重删掉 → 索引指向不存在的文件。
            name = "%s_f%06d_x%04d_y%04d_h%03d_s%03d.png" % (
                path.stem, fi, int(box[0]) % 10000, int(box[1]) % 10000,
                int(bh), min(999, int(sharp)))
            rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            rgba[..., 3] = 255                                  # 不透明：贴回原框原位即原像素
            if not cv2.imwrite(str(OUT_DIR / name), rgba):
                raise RuntimeError("写素材失败：%s" % (OUT_DIR / name))
            kept.append({
                "file": name, "source": path.name, "source_path": str(path), "frame_index": fi,
                "box_xyxy": [round(v, 1) for v in box], "crop_xyxy": cbox, "score": round(score, 3),
                "box_h": int(bh), "box_px": int((box[2] - box[0]) * bh),
                "aspect": round(bh / max(1.0, box[2] - box[0]), 2),
                "sharpness": round(sharp, 1),
                "png_wh": [int(rgba.shape[1]), int(rgba.shape[0])],
            })
    cap.release()
    return kept, reasons, scanned


def write_sheet(entries: list[dict[str, Any]], path: Path, cols: int = 14, cell: int = 128) -> None:
    """拼图（灰底、按比例缩略，图上烧注框高与清晰度）——一眼判断素材质量。

    读不出来的素材**要报出来**（静默 `continue` 会掩盖"索引与磁盘不一致"这种真 bug）。
    """
    if not entries:
        return
    rows = (len(entries) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 40, np.uint8)
    broken = 0
    for i, entry in enumerate(entries):
        img = cv2.imread(str(OUT_DIR / entry["file"]), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            broken += 1
            print("    [拼图] 读不出素材：%s" % entry["file"])
            continue
        h, w = img.shape[:2]
        scale = min((cell - 14) / max(1, h), (cell - 4) / max(1, w))
        th, tw = max(1, int(h * scale)), max(1, int(w * scale))
        small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        x0 = c * cell + (cell - tw) // 2
        y0 = r * cell + (cell - th) // 2 + 2
        sheet[y0:y0 + th, x0:x0 + tw] = small[..., :3]
        cv2.putText(sheet, "h%d s%d" % (entry["box_h"], int(entry["sharpness"])),
                    (c * cell + 2, r * cell + cell - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (200, 220, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def write_index(entries: list[dict[str, Any]], videos: list[Path], scannes: dict[str, int],
                reasons: dict[str, int], a: argparse.Namespace) -> dict[str, Any]:
    """写 `index.json`：素材明细 + 每视频产量 + 全局拒绝理由分布 + 本次闸门参数。"""
    per_video: dict[str, int] = {}
    for e in entries:
        per_video[e["source"]] = per_video.get(e["source"], 0) + 1
    doc = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_assets": len(entries),
        "videos": [str(v) for v in videos],
        "frames_scanned": scannes,
        "per_video": per_video,
        "reject_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "gates": {
            "min_conf": a.min_conf, "min_height": a.min_height, "min_area": a.min_area,
            "aspect": [a.aspect_min, a.aspect_max], "edge_margin": a.edge_margin,
            "max_overlap_iou": a.max_overlap_iou, "min_sharp": a.min_sharp,
            "dedupe_iou": a.dedupe_iou, "pad": a.pad,
            "threshold": a.threshold, "resolution": a.resolution,
            "scan_stride": a.scan_stride, "max_per_video": a.max_per_video,
        },
        "assets": entries,
    }
    (OUT_DIR / "index.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return doc


def write_report(doc: dict[str, Any], path: Path) -> None:
    """一页 markdown 报告：每视频产量、拒绝理由、闸门参数、素材尺寸/清晰度分布。"""
    assets = doc["assets"]
    g = doc["gates"]
    lens = sorted(e["box_h"] for e in assets)
    sharps = sorted(e["sharpness"] for e in assets)
    q = lambda xs, p: xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0
    lines = [
        "# 人体素材库（高质量闸门）", "",
        "生成时间：%s" % doc["generated_at"], "",
        "## 一、产量", "",
        "**共 %d 张**，来自 %d 段视频。" % (doc["n_assets"], len(doc["videos"])), "",
        "| 视频 | 素材数 | 扫描帧数 |", "| --- | ---: | ---: |",
    ]
    for v in doc["videos"]:
        name = Path(v).name          # doc["videos"] 存的是字符串（要能进 JSON）
        lines.append("| `%s` | %d | %d |" % (name, doc["per_video"].get(name, 0),
                                             doc["frames_scanned"].get(name, 0)))
    lines += ["", "## 二、拒绝理由（调闸门看这里）", "", "| 理由 | 次数 |", "| --- | ---: |"]
    for k, n in doc["reject_reasons"].items():
        lines.append("| `%s` | %d |" % (k, n))
    lines += ["", "## 三、闸门参数", "",
              "框高 ≥ %d px、面积 ≥ %d px²、高宽比 %.2f–%.2f、贴边 ≤ %d px、"
              "同帧重叠 IoU < %.2f、清晰度 ≥ %.0f、去重 IoU ≥ %.2f、外扩 %.0f%%"
              % (g["min_height"], g["min_area"], g["aspect"][0], g["aspect"][1], g["edge_margin"],
                 g["max_overlap_iou"], g["min_sharp"], g["dedupe_iou"], g["pad"] * 100), "",
              "## 四、质量分布", "",
              "| 指标 | P10 | 中位 | P90 | 最大 |", "| --- | ---: | ---: | ---: | ---: |",
              "| 框高 (px) | %d | %d | %d | %d |" % (q(lens, 0.1), q(lens, 0.5), q(lens, 0.9),
                                                    lens[-1] if lens else 0),
              "| 清晰度 | %d | %d | %d | %d |" % (q(sharps, 0.1), q(sharps, 0.5), q(sharps, 0.9),
                                                  int(sharps[-1]) if sharps else 0), "",
              "拼图：`work/cutouts_person_hq/contact_sheet.jpg`（缩略图上烧注了框高 h 与清晰度 s）", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default="auto",
                    help="源视频：auto（默认，自动挑 %d 段）｜逗号分隔的路径/文件名/glob" % 6)
    ap.add_argument("--video-limit", type=int, default=6, help="--videos auto 时挑几段")
    ap.add_argument("--min-video-mb", type=float, default=80, help="auto 时忽略小于该体积的视频")
    ap.add_argument("--list-videos", action="store_true", help="只列出会挑哪些视频然后退出")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="检测权重（默认三分类部署版）")
    ap.add_argument("--resolution", type=int, default=960)
    ap.add_argument("--threshold", type=float, default=0.35, help="检测置信度阈值")
    ap.add_argument("--scan-stride", type=int, default=30, help="扫描步长（帧）")
    ap.add_argument("--scan-max", type=int, default=30,
                    help="每段视频**最多采样多少帧**（在整段上均匀铺开；0=按步长全片扫）")
    # --- 质量闸门 ---
    ap.add_argument("--min-height", type=int, default=120, help="框高下限 px（主闸门，砍远景小人）")
    ap.add_argument("--min-area", type=int, default=8000, help="框面积下限 px²")
    ap.add_argument("--min-conf", type=float, default=0.55, help="检测置信度下限")
    ap.add_argument("--aspect-min", type=float, default=1.05, help="高宽比下限")
    ap.add_argument("--aspect-max", type=float, default=4.5, help="高宽比上限")
    ap.add_argument("--edge-margin", type=int, default=2, help="框贴边多少 px 算截断（丢弃）")
    ap.add_argument("--max-overlap-iou", type=float, default=0.30,
                    help="与同帧其他人框 IoU ≥ 该值算遮挡（框里不止一人），0=关掉")
    ap.add_argument("--min-sharp", type=float, default=300.0,
                    help="清晰度（拉普拉斯方差）下限，0=关掉。高清人像实测在 1e3 量级，"
                         "300 只是挡真正的糊帧/拖影（阈值再高要按报告里的 P10 定）")
    ap.add_argument("--no-sharp-gate", dest="min_sharp", action="store_const", const=0.0)
    ap.add_argument("--dedupe-iou", type=float, default=0.50, help="跨帧同一人只留最清晰，0=不去重")
    ap.add_argument("--pad", type=float, default=0.08, help="框外扩比例")
    ap.add_argument("--max-per-video", type=int, default=60, help="每段视频的素材上限")
    ap.add_argument("--clean", action="store_true", help="先清空输出目录")
    args = ap.parse_args(argv)

    videos = resolve_videos(args)
    if not videos:
        raise SystemExit("没找到任何源视频（--videos / --video-limit / --min-video-mb 检查一下）")
    if args.list_videos:
        print("会自动挑这 %d 段：" % len(videos))
        for v in videos:
            print("  %-58s %6.0f MB" % (v, v.stat().st_size / 1e6))
        return 0
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit("找不到权重：%s" % ckpt)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_layout.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for old in OUT_DIR.iterdir():
            if old.is_file():
                old.unlink()
        print("已清空 %s" % OUT_DIR)

    print("源视频 %d 段：" % len(videos))
    for v in videos:
        print("  - %s（%.0f MB）" % (v.name, v.stat().st_size / 1e6))
    print("闸门：框高 ≥ %d px、面积 ≥ %d、高宽比 %.2f–%.2f、清晰度 ≥ %.0f、去重 IoU %.2f"
          % (args.min_height, args.min_area, args.aspect_min, args.aspect_max,
             args.min_sharp, args.dedupe_iou))
    model = rd.load_model(ckpt, args.resolution)

    entries: list[dict[str, Any]] = []
    all_reasons: dict[str, int] = {}
    scannes: dict[str, int] = {}
    for v in videos:
        kept, reasons, scanned = scan_video(v, model, args)
        entries.extend(kept)
        scannes[v.name] = scanned
        for k, n in reasons.items():
            all_reasons[k] = all_reasons.get(k, 0) + n
        lens = sorted(e["box_h"] for e in kept)
        med = lens[len(lens) // 2] if lens else 0
        print("  %-46s 收 %3d 张（扫 %3d 帧，框高中位 %d px）"
              % (v.name, len(kept), scanned, med), flush=True)

    # 索引 vs 磁盘 一致性自检：文件名撞车/去重误删都会在这里现形（不静默放过）
    on_disk = {p.name for p in OUT_DIR.glob("*.png")}
    in_index = {e["file"] for e in entries}
    if missing := sorted(in_index - on_disk):
        raise SystemExit("索引里有 %d 个素材在磁盘上不存在（首个：%s）" % (len(missing), missing[0]))
    if extra := sorted(on_disk - in_index):
        print("  提示：磁盘上有 %d 个未被索引的 PNG（首个：%s），将被删掉" % (len(extra), extra[0]))
        for name in extra:
            (OUT_DIR / name).unlink()
    if len(entries) != len(in_index):
        raise SystemExit("索引里出现重复文件名（%d 条 / %d 个文件）" % (len(entries), len(in_index)))

    write_sheet(entries, OUT_DIR / "contact_sheet.jpg")
    doc = write_index(entries, videos, scannes, all_reasons, args)
    write_report(doc, REPORT_MD)
    lens = sorted(e["box_h"] for e in entries)
    sharps = sorted(e["sharpness"] for e in entries)
    print("\n合计 **%d 张**人体素材，来自 %d 段视频" % (len(entries), len(videos)))
    print("  框高：中位 %d px（最小 %d / 最大 %d）"
          % (lens[len(lens) // 2] if lens else 0, lens[0] if lens else 0, lens[-1] if lens else 0))
    print("  清晰度：中位 %.0f（最小 %.0f）"
          % (sharps[len(sharps) // 2] if sharps else 0, sharps[0] if sharps else 0))
    print("  拒绝理由 TOP5：%s" % "，".join("%s=%d" % (k, n) for k, n
                                        in list(doc["reject_reasons"].items())[:5]) or "无")
    print("  目录：%s" % OUT_DIR)
    print("  报告：%s" % REPORT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
