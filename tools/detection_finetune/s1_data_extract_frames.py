#!/usr/bin/env python3
"""按策略表从原始视频抽帧，写 manifest.csv（保留原始分辨率与帧号）。

设计要点：

* **不缩放**：源分辨率原样保存（720p/1080p 混着进数据集）。部署侧是
  `letterbox → 960 方图`（`RFDetrDetectorInfer::ComputeResize`），所以源分辨率
  不改变模型看到的目标像素数；真正决定目标大小的是取景/距离，因此抽帧只按
  场景分层、不去统一分辨率。
* **去重**：相邻帧用 dHash 汉明距离筛掉几乎一样的帧，避免"同一秒抽 20 张"。
* **split 按整段视频划分**（读 split_policy.json），同一来源整组进同一 split，
  避免同源泄漏；`test` 集同时是检测级 A/B 的固定评测集，绝不进训练。
* **可复现**：帧号由 stride 直接算出，不做随机采样；`--seed` 只影响多窗口模式的起点。

用法：
    python s1_data_extract_frames.py --dry-run                      # 只看会抽多少帧
    python s1_data_extract_frames.py --priority high,normal --limit-total 600
    python s1_data_extract_frames.py --videos 绕障碍.mov 颠球.mov     # 只抽指定视频
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

import base_layout

POLICY_JSON = Path(__file__).resolve().parent / "split_policy.json"
VIDEO_ROOT = base_layout.DATA_ROOT


@dataclass
class VideoPlan:
    """一段视频的抽帧计划。"""

    rel_path: str
    split: str
    scene: str
    view: str
    priority: str
    why: str = ""

    @property
    def path(self) -> Path:
        return VIDEO_ROOT / self.rel_path

    @property
    def stem(self) -> str:
        return "__".join(Path(self.rel_path).with_suffix("").parts)


def load_policy(path: Path | None = None) -> dict[str, VideoPlan]:
    raw = json.loads((path or POLICY_JSON).read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    plans: dict[str, VideoPlan] = {}
    for rel, cfg in raw["videos"].items():
        plans[rel] = VideoPlan(
            rel_path=rel,
            split=str(cfg.get("split", defaults.get("split", "train"))),
            scene=str(cfg.get("scene", defaults.get("scene", "unknown"))),
            view=str(cfg.get("view", defaults.get("view", "unknown"))),
            priority=str(cfg.get("priority", defaults.get("priority", "normal"))),
            why=str(cfg.get("why", "")),
        )
    # 策略表没列到的视频（新素材）走 defaults，split 留 train 并警告
    for path in sorted(VIDEO_ROOT.rglob("*.mov")):
        rel = str(path.relative_to(VIDEO_ROOT))
        if rel not in plans and "ball_cone_v1" not in Path(rel).parts:
            plans[rel] = VideoPlan(rel_path=rel, **{
                k: defaults.get(k, v) for k, v in
                (("split", "train"), ("scene", "unknown"), ("view", "unknown"), ("priority", "normal"))})
    return plans


def dhash(image: np.ndarray, size: int = 16) -> int:
    """dHash（默认 16×16 = 256 位）：用于近重复帧检测。

    网格必须足够细：8×8 只有 64 位，固定机位下"人走了几步"这种真实变化也可能
    落在汉明距离 6 以内，会把有效帧当重复砍掉（实测 30 帧只剩 2 帧）。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def plan_frames(n_frames: int, fps: float, stride_sec: float, per_video_max: int,
                start_sec: float, end_sec: float) -> list[int]:
    """均匀 stride 抽帧；时间窗 [start_sec, end_sec] 内均匀取，最多 per_video_max 帧。"""
    if n_frames <= 0 or fps <= 0:
        return []
    first = int(round(start_sec * fps))
    last = int(round((end_sec if end_sec > 0 else n_frames / fps) * fps))
    last = min(last, n_frames - 1)
    if last <= first:
        return []
    step = max(1, int(round(stride_sec * fps)))
    idxs = list(range(first, last + 1, step))
    if len(idxs) > per_video_max:  # 等间隔取前 per_video_max 个（保持覆盖整段）
        picks = np.linspace(0, len(idxs) - 1, per_video_max)
        idxs = [idxs[int(round(p))] for p in picks]
    return sorted(set(idxs))


def existing_images() -> set[str]:
    if not base_layout.MANIFEST_CSV.exists():
        return set()
    with base_layout.MANIFEST_CSV.open(encoding="utf-8") as handle:
        return {row["image"] for row in csv.DictReader(handle)}


def append_manifest(rows: Iterable[dict[str, object]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    exists = base_layout.MANIFEST_CSV.exists() and base_layout.MANIFEST_CSV.stat().st_size > 0
    with base_layout.MANIFEST_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(base_layout.MANIFEST_FIELDS))
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in base_layout.MANIFEST_FIELDS})
    return len(rows)


def sample_video(plan: VideoPlan, stride_sec: float, per_video_max: int, start_sec: float,
                 end_sec: float, quality: int, min_dhash_delta: int, dhash_size: int,
                 dry_run: bool, already: set[str]) -> tuple[list[dict[str, object]], dict[str, object]]:
    """抽一段视频，返回 (manifest 行, 统计)。"""
    if not plan.path.exists():
        return [], {"video": plan.rel_path, "error": f"文件不存在 {plan.path}"}
    cap = cv2.VideoCapture(str(plan.path))
    if not cap.isOpened():
        return [], {"error": f"无法打开 {plan.path}"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = plan_frames(n, fps, stride_sec, per_video_max, start_sec, end_sec)
    out_dir = base_layout.RAW_VIDEO_DIR / plan.stem
    rows: list[dict[str, object]] = []
    kept, dup, missing = 0, 0, 0
    last_hash: int | None = None
    width = height = 0
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            missing += 1
            continue
        height, width = frame.shape[:2]
        if min_dhash_delta > 0:
            h = dhash(frame, dhash_size)
            if last_hash is not None and hamming(h, last_hash) < min_dhash_delta:
                dup += 1
                continue
            last_hash = h
        #: 相对 ball_cone_v1 的路径前缀由 layout 决定（现在是 sources/video/），
        #: 且必须带前缀，否则 imwrite 会因父目录不存在静默失败、manifest 却照样记账（踩过一次）。
        rel_image = "%s/%s/%s__f%06d__t%07.2f.jpg" % (
            base_layout.RAW_VIDEO_DIR.relative_to(base_layout.DATASET_ROOT), plan.stem, plan.stem, fi, fi / fps)
        if rel_image in already:
            continue
        if not dry_run:
            target = base_layout.DATASET_ROOT / rel_image
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                raise RuntimeError(f"写图失败（父目录在？）：{target}")
            if target.stat().st_size == 0:
                raise RuntimeError(f"写出的图为空：{target}")
        rows.append({
            "image": rel_image,
            "source": "raw_video",
            "source_video": plan.rel_path,
            "frame_index": fi,
            "time_sec": round(fi / fps, 3),
            "width": width,
            "height": height,
            "scene": plan.scene,
            "view": plan.view,
            "split": plan.split,
            "is_synthetic": "no",
            "is_pseudolabel": "no",
            "license": "自有素材",
            "note": plan.why,
        })
        kept += 1
    cap.release()
    stats = {"video": plan.rel_path, "split": plan.split, "scene": plan.scene,
             "resolution": f"{width}x{height}", "fps": round(fps, 2), "frames": n,
             "planned": len(idxs), "kept": kept, "dup_skipped": dup, "read_failed": missing}
    return rows, stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按策略表抽帧并登记 manifest")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="只抽这些视频（相对 /home/hx1/yzl/Work/data 的路径，支持子串匹配）")
    parser.add_argument("--priority", default="high,normal",
                        help="只抽这些优先级（逗号分隔，high,normal,low；all=全部）")
    parser.add_argument("--split", default=None, help="只抽某个 split（train/valid/test）")
    parser.add_argument("--stride-sec", type=float, default=3.0, help="抽帧步长（秒）")
    parser.add_argument("--per-video-max", type=int, default=40, help="每段视频最多抽多少帧")
    parser.add_argument("--limit-total", type=int, default=600, help="本次最多抽多少帧（0=不限）")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float, default=0.0, help="0=到结尾")
    parser.add_argument("--quality", type=int, default=92, help="JPEG 质量")
    parser.add_argument("--min-dhash-delta", type=int, default=6,
                        # 注意：argparse 的 help 会做 % 格式化，百分号必须写成 %%（否则 --help 直接崩）
                        help="相邻保留帧的 dHash 最小汉明距离（0=不去重；默认 6/256 位≈2.3%%）")
    parser.add_argument("--dhash-size", type=int, default=16, help="dHash 网格边长（位宽=size²）")
    parser.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan_all = load_policy()
    wanted = set(args.priority.split(",")) if args.priority != "all" else None
    plans = [p for p in plan_all.values() if wanted is None or p.priority in wanted]
    if args.split:
        plans = [p for p in plans if p.split == args.split]
    if args.videos:
        plans = [p for p in plans if any(v in p.rel_path for v in args.videos)]
    plans.sort(key=lambda p: (p.split, p.rel_path))

    print("策略表: %s" % POLICY_JSON)
    print("待抽帧的视频 %d 段（stride %.1fs，每段上限 %d 帧，总上限 %s）"
          % (len(plans), args.stride_sec, args.per_video_max,
             args.limit_total or "不限"))
    existing = existing_images()
    total = 0
    all_rows: list[dict[str, object]] = []
    stats_list: list[dict[str, object]] = []
    for plan in plans:
        budget = args.per_video_max
        if args.limit_total:
            budget = min(budget, max(0, args.limit_total - total))
        if budget == 0:
            break
        rows, stats = sample_video(plan, args.stride_sec, budget, args.start_sec, args.end_sec,
                                   args.quality, args.min_dhash_delta, args.dhash_size,
                                   args.dry_run, existing)
        stats_list.append(stats)
        all_rows.extend(rows)
        total += len(rows)
        if "error" in stats:
            print("  !! %-46s %s" % (plan.rel_path[:46], stats["error"]))
        else:
            print("  %-4s %-46s %-11s 计划%3d 实收%3d 去重%3d"
                  % (plan.split, plan.rel_path[:46], stats["resolution"],
                     stats["planned"], stats["kept"], stats["dup_skipped"]))

    print("-" * 96)
    by_split: dict[str, int] = {}
    for row in all_rows:
        by_split[str(row["split"])] = by_split.get(str(row["split"]), 0) + 1
    print("合计 %d 帧 %s" % (total, by_split))
    res_hist: dict[str, int] = {}
    for row in all_rows:
        key = f"{row['width']}x{row['height']}"
        res_hist[key] = res_hist.get(key, 0) + 1
    print("分辨率分布（源分辨率原样保存，不缩放）: %s" % res_hist)
    if args.dry_run:
        print("dry-run：未落盘")
        return 0
    written = append_manifest(all_rows)
    (base_layout.REPORTS_DIR / "sample_frames_last.json").write_text(
        json.dumps({"args": vars(args), "stats": stats_list}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("已登记 manifest %d 行 → %s" % (written, base_layout.MANIFEST_CSV))
    print("抽帧统计 → %s" % (base_layout.REPORTS_DIR / "sample_frames_last.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
