#!/usr/bin/env python3
"""AI 生成球场背景：提示词包 → （用 [skill:多模态内容生成] 出图）→ 回收校验 → 台账（S2.6）。

**AI 图在本方案里的定位**：只当**合成背景**用——图上本来没有待标物体，把真实抠出的锥素材
贴上去，GT 仍然是几何精确的；换来的价值是场景多样性（黄昏/强光/湿滑/陌生场地），
缓解"合成图背景全是自家球馆 → 过拟合"。AI 原图（自带球/人/锥）只用于规范示例与压力集。

`--validate` 的四道闸门（任一不过就剔除，并把原因计数）：
① 尺寸/长宽比合法；② 清晰度（拉普拉斯方差）达标；③ 用 base 模型确认**没有球/人/锥**
（背景自带待标物体 = 造出错标签）；④ dHash 去重（AI 图常有近似重复）。
通过的图登记 `manifest.csv`（`source=ai_background`）并写 `ledger.csv`（生成工具/时间/提示词）。

用法：
    python s2b_data_gen_backgrounds.py --prompts                 # 出提示词包（含每张的目标文件名）
    # 用 [skill:多模态内容生成] 按提示词出图，落到 synthetic/ai_backgrounds/
    python s2b_data_gen_backgrounds.py --validate                # 校验 + 登记 + 台账
    python s2b_data_gen_backgrounds.py --spec-and-stress         # 规范示例图与压力集的提示词包
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import base_layout
from s1_data_extract_frames import append_manifest, dhash, hamming

#: 三类提示词包分别落文件（早先共用一个文件名，结果后写的覆盖了前一个）
PROMPTS_JSON = base_layout.PROMPT_DIR / "prompts_ai_background.json"
SPEC_PROMPTS_JSON = base_layout.PROMPT_DIR / "prompts_ai_spec.json"
STRESS_PROMPTS_JSON = base_layout.PROMPT_DIR / "prompts_ai_stress.json"
#: 背景池：每张都要求"空场地、无球无人无器材"，只换光照/场地/机位
BACKGROUND_PROMPTS: tuple[dict[str, str], ...] = (
    {"id": "bg_01", "scene": "indoor_gym", "prompt":
     "Empty indoor sports hall with light blue walls and a smooth light-brown wooden floor, "
     "painted blue court lines, seen from a fixed wide sideline camera at about 3 meters height, "
     "even overhead fluorescent lighting, no people, no balls, no training equipment, no text"},
    {"id": "bg_02", "scene": "indoor_gym", "prompt":
     "Empty indoor futsal court with glossy green floor and white markings, dark blue wall panels, "
     "wide fixed camera view from the side, cool white LED lighting, strong floor reflections, "
     "absolutely no people or sports balls or cones visible, no watermarks"},
    {"id": "bg_03", "scene": "outdoor_pitch", "prompt":
     "Empty outdoor artificial turf football pitch at golden hour, low warm sunlight from the left, "
     "long shadows of the goal post on the green turf, white line markings, blurred fence in the "
     "background, wide fixed camera from the sideline, no players, no ball, no cones, no text"},
    {"id": "bg_04", "scene": "outdoor_pitch", "prompt":
     "Empty outdoor grass football field on an overcast afternoon, slightly wet grass with soft "
     "specular highlights, distant trees and a small stand out of focus, wide fixed sideline camera, "
     "flat grey light, no people, no ball, no equipment, no watermark"},
    {"id": "bg_05", "scene": "outdoor_pitch", "prompt":
     "Empty concrete futsal court at dusk under floodlights, bluish ambient light with warm lamp "
     "pools, chain-link fence around the court, wide fixed camera view, no players, no ball, "
     "no bottles or cones on the ground, no text"},
    {"id": "bg_06", "scene": "indoor_gym", "prompt":
     "Empty school gymnasium with beige floor, red boundary lines and a folded volleyball net in "
     "the far background, mixed daylight from side windows and ceiling lamps, wide fixed camera, "
     "no people, no balls, no cones, no signage text"},
)

SPEC_PROMPTS: tuple[dict[str, str], ...] = (
    {"id": "spec_ball", "prompt":
     "Close-up reference photo of a black and white football resting on a wooden indoor court floor, "
     "clear pentagon pattern, soft shadow, shot from about 2 meters height, clean composition"},
    {"id": "spec_cone", "prompt":
     "Close-up reference photo of a small blue plastic pyramid marker cone on a wooden sports floor, "
     "sharp edges, soft shadow, shot from about 2 meters height, clean composition"},
    {"id": "spec_flat_disc", "prompt":
     "Close-up reference photo of a flat round yellow plastic floor marker disc on a wooden sports "
     "floor, soft shadow, shot from about 2 meters height, clean composition"},
    {"id": "spec_hard", "prompt":
     "Two reference photos side by side on the same wooden sports floor: on the left a small blue "
     "marker cone, on the right a white disc marker, strong motion blur on the right one"},
)

STRESS_PROMPTS: tuple[dict[str, str], ...] = (
    {"id": "stress_night", "prompt":
     "Night-time amateur football match under uneven floodlights, many small distant players, "
     "one ball mid-air, strong noise and motion blur, shot from a high wide angle"},
    {"id": "stress_glare", "prompt":
     "Outdoor football scene at harsh noon with strong lens glare and blown highlights, "
     "players and a ball far away, wide angle, slight rolling-shutter smear"},
    {"id": "stress_shake", "prompt":
     "Handheld shaky phone video frame of an indoor futsal game, motion blur, tilted horizon, "
     "distant small players and a ball on the floor, dim lighting"},
    {"id": "stress_crowd", "prompt":
     "Football match seen from behind a fence with a crowd in the foreground, ball on the pitch "
     "far behind, heavy occlusion and depth of field blur"},
)


def _write_prompts(entries: Sequence[dict[str, str]], out_dir: Path, kind: str,
                   json_path: Path | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"kind": kind, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "target_dir": str(out_dir), "entries": []}
    for entry in entries:
        name = "%s_%s.jpg" % (kind, entry["id"])
        payload["entries"].append({**entry, "target_file": str(out_dir / name)})
    json_path = json_path or PROMPTS_JSON
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "PROMPTS.md").write_text(
        "# %s 提示词包\n\n把下面每条提示词喂给图像生成（skill: 多模态内容生成），"
        "图片存到本目录，文件名必须与 `target_file` 一致，然后跑 "
        "`python s2b_data_gen_backgrounds.py --validate`。\n\n" % kind
        + "\n".join("- **%s** → `%s`\n  - %s" % (e["id"], Path(e["target_file"]).name, e["prompt"])
                    for e in payload["entries"]) + "\n", encoding="utf-8")
    return json_path


def cmd_prompts(args: argparse.Namespace) -> int:
    path = _write_prompts(BACKGROUND_PROMPTS, base_layout.AI_GEN_BG_DIR, "bg")
    print("背景池提示词 %d 条 → %s" % (len(BACKGROUND_PROMPTS), path))
    print("  出图目录：%s（文件名见 %s）" % (base_layout.AI_GEN_BG_DIR, base_layout.AI_GEN_BG_DIR / "PROMPTS.md"))
    return 0


def cmd_spec_and_stress(args: argparse.Namespace) -> int:
    spec = _write_prompts(SPEC_PROMPTS, base_layout.AI_GEN_SPEC_DIR, "spec", SPEC_PROMPTS_JSON)
    stress = _write_prompts(STRESS_PROMPTS, base_layout.AI_GEN_STRESS_DIR, "stress", STRESS_PROMPTS_JSON)
    print("规范示例提示词 → %s（%d 条）" % (spec, len(SPEC_PROMPTS)))
    print("压力集提示词   → %s（%d 条，**不进训练集**）" % (stress, len(STRESS_PROMPTS)))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """校验背景池并登记 manifest/ledger。"""
    import torch
    from rfdetr import RFDETRSmall

    weights = Path(__file__).resolve().parents[2] / "models/rfdetr/rf-detr-small.pth"
    model = RFDETRSmall(pretrain_weights=str(weights), device="cuda")
    torch.set_grad_enabled(False)

    files = sorted(p for p in Path(args.dir).glob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print("待校验 %d 张：%s" % (len(files), args.dir))
    passed: list[dict[str, Any]] = []
    rejected: list[tuple[str, str]] = []
    seen_hashes: list[int] = []
    existing = {r["image"] for r in csv.DictReader(base_layout.MANIFEST_CSV.open(encoding="utf-8"))}
    for path in files:
        image = cv2.imread(str(path))
        rel = str(path.relative_to(base_layout.DATASET_ROOT))
        if image is None:
            rejected.append((path.name, "读图失败"))
            continue
        h, w = image.shape[:2]
        if min(h, w) < args.min_side:
            rejected.append((path.name, "过小 %dx%d" % (w, h)))
            continue
        # 宽场背景常见 1280x636（16:7.9）→ 闸门放宽到 0.35~2.8，只挡极端全景/竖条
        if not (0.35 <= h / w <= 2.8):
            rejected.append((path.name, "长宽比异常 %.2f" % (h / w)))
            continue
        sharp = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if sharp < args.min_sharpness:
            rejected.append((path.name, "糊图 tex%.0f" % sharp))
            continue
        if args.dedupe and any(hamming(dhash(image), prev) < args.dedupe for prev in seen_hashes):
            rejected.append((path.name, "与已有图近似重复"))
            continue
        det = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), threshold=args.detect_threshold)
        ids = [int(c) for c in np.asarray(det.class_id)]
        # COCO: 1=person, 37=sports ball；锥在这套权重里通常落到 ball
        hits = [c for c in ids if c in (1, 37)]
        if hits:
            rejected.append((path.name, "背景自带待标物体（person/球类 %d 个）" % len(hits)))
            continue
        # 裁掉底部水印带：生成图常在角落烧"AI生成"水印，留着会让模型学到"水印↔锥"的伪相关。
        # 三条硬约束：① 只裁**一次**（已在 manifest 里登记过的图不再裁，否则每跑一次就缩一圈：
        # 实测 720 → 677 → 636）；② 裁前把原图备份到 _orig/（可回退）；③ 记录 crop_bottom。
        note = "sharpness=%.1f" % sharp
        if args.crop_bottom > 0 and rel not in existing:
            cut = int(round(h * args.crop_bottom))
            if cut > 0:
                backup = path.parent / "_orig" / path.name
                if not backup.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(path.read_bytes())
                image = image[:h - cut]
                h = image.shape[0]
                if not cv2.imwrite(str(path), image):
                    rejected.append((path.name, "裁边后写回失败"))
                    continue
                note += " crop_bottom=%.3f（原图在 _orig/）" % args.crop_bottom
        elif args.crop_bottom > 0:
            note += " 已登记跳过重复裁边"
        seen_hashes.append(dhash(image))
        passed.append({"image": rel, "width": image.shape[1], "height": h,
                       "sharpness": round(sharp, 1), "note": note})

    rows = []
    for item in passed:
        if item["image"] in existing:
            continue
        rows.append({
            "image": item["image"], "source": "ai_gen_bg", "source_video": "",
            "frame_index": "", "time_sec": "", "width": item["width"], "height": item["height"],
            "scene": "ai_gen_bg", "view": "generated", "split": "train",
            "is_synthetic": "no", "ai_background": "yes", "is_pseudolabel": "no",
            "license": "自有生成（AI）", "note": item["note"],
        })
    written = append_manifest(rows)
    print("通过 %d / 剔除 %d；manifest 新增 %d 行" % (len(passed), len(rejected), written))
    if rejected:
        from collections import Counter
        print("  剔除原因:", dict(Counter(r.split("（")[0].split("%")[0] for _, r in rejected)))
        for name, why in rejected[:8]:
            print("    %-28s %s" % (name, why))
    # 台账：**幂等**——没有新登记的图就不追加（早先每次 --validate 都追加一行，实测重复了 6 条）
    rows_ledger = []
    with base_layout.LEDGER_CSV.open(encoding="utf-8") as fh:
        rows_ledger = list(csv.DictReader(fh))
    if written:
        rows_ledger.append({"name": "ai_background_pool", "url": "generated via image_gen skill",
                            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "license": "自有生成（AI）", "redistributable": "internal-only",
                            "n_images": written, "note": "新增 %d 张；提示词见 %s"
                                                        % (written, PROMPTS_JSON.name)})
    else:
        print("  （没有新图，台账不动）")
    with base_layout.LEDGER_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(base_layout.LEDGER_FIELDS))
        writer.writeheader()
        for row in rows_ledger:
            writer.writerow({k: row.get(k, "") for k in base_layout.LEDGER_FIELDS})
    print("台账 → %s" % base_layout.LEDGER_CSV)
    return 0 if passed else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI 背景池：提示词包 / 回收校验 / 台账")
    parser.add_argument("--prompts", action="store_true")
    parser.add_argument("--spec-and-stress", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--dir", default=str(base_layout.AI_GEN_BG_DIR))
    parser.add_argument("--min-side", type=int, default=480,
                        help="背景短边下限（宽场 1280x636 属合法，故不设 640）")
    parser.add_argument("--min-sharpness", type=float, default=60.0)
    parser.add_argument("--dedupe", type=int, default=4, help="dHash 汉明距离小于此值判重复（0=不去重）")
    parser.add_argument("--crop-bottom", type=float, default=0.06,
                        help="裁掉底部比例（去生成器水印；0=不裁）")
    parser.add_argument("--detect-threshold", type=float, default=0.3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.prompts:
        return cmd_prompts(args)
    if args.spec_and_stress:
        return cmd_spec_and_stress(args)
    if args.validate:
        return cmd_validate(args)
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
