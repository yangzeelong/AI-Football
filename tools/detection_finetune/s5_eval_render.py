#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把检测结果渲染成可看的产物：mp4（抽帧）+ GIF 片段（浏览器可播）+ 关键帧拼图 + 逐帧 CSV。

颜色约定（与标注规范一致）：**person=绿 · ball=蓝 · cone=红**；被"球∩锥抑制规则"压掉的
球框画成**灰色**，方便一眼看出这条规则救回了多少误检。

**只读视频**，产物全部写到 `--out-dir`（默认仓库 runs/render/），不碰 data/。

    $PY s5_eval_render.py --video "$AI_FOOTBALL_DATA_DIR/移动传球.mov" \\
        --model v1=runs/v1/checkpoint_best_ema.pth \\
        --start 0 --end 30 --stride 4 --gif-range 0,30 --contact 8 --tag mp_v1
    $PY s5_eval_render.py --video ... --model base=... --model v1=...   # 并排对照
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

import s5_eval_detector as ed

COLORS = {1: (80, 200, 80), 2: (230, 160, 60), 3: (60, 60, 235)}      # BGR: 绿/蓝/红
SUPPRESSED = (160, 160, 160)


def load_model(path: Path, resolution: int) -> Any:
    from rfdetr import RFDETRSmall

    return RFDETRSmall(pretrain_weights=str(path), resolution=resolution, device="cuda")


def dets_for(model: Any, frame: np.ndarray, thr: float) -> dict[str, Any]:
    """一帧的检测结果（本项目的 1/2/3 类）。"""
    from PIL import Image

    det = model.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), threshold=thr)
    out: dict[str, Any] = {"boxes": [], "scores": [], "labels": []}
    for box, score, label in zip(det.xyxy, det.confidence, det.class_id):
        our = ed.our_id_of(model.class_names).get(int(label))
        if our is None:
            continue
        out["boxes"].append([float(v) for v in box[:4]])
        out["scores"].append(float(score))
        out["labels"].append(int(our))
    return out


def draw(frame: np.ndarray, det: dict[str, Any], suppress_iou: float,
         stats: dict[str, int]) -> np.ndarray:
    """画框；被锥抑制的球画灰。返回画好的图。"""
    canvas = frame.copy()
    cones = [b for b, l in zip(det["boxes"], det["labels"]) if l == 3]
    for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
        color = COLORS[label]
        if label == 2 and suppress_iou > 0 and any(ed.iou(box, c) >= suppress_iou for c in cones):
            color = SUPPRESSED                      # 这条球框会被"球∩锥"规则压掉
            stats["suppressed"] += 1
        stats[{1: "person", 2: "ball", 3: "cone"}[label]] += 1
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, "%s %.2f" % (ed.NAME[label], score), (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def banner(img: np.ndarray, lines: list[str], tag: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22 + 20 * len(lines)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, tag, (8, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 2, cv2.LINE_AA)
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检测结果渲染（mp4 + GIF + 拼图）")
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="TAG=PATH")
    ap.add_argument("--resolution", type=int, default=960)
    ap.add_argument("--threshold", type=float, default=0.4, help="只画 ≥ 该置信度的框")
    ap.add_argument("--start", type=float, default=0.0, help="起始秒")
    ap.add_argument("--end", type=float, default=0.0, help="结束秒（0=到结尾）")
    ap.add_argument("--stride", type=int, default=6, help="每 N 帧取 1 帧（60fps 下 6 ≈ 10 帧/秒输出）")
    ap.add_argument("--suppress-iou", type=float, default=0.30,
                    help="球框与锥框 IoU ≥ 该值即视为'误检'画灰（0=关掉规则）")
    ap.add_argument("--gif-range", default="", help="额外出一个 GIF，如 0,30（秒区间，浏览器可播）")
    ap.add_argument("--gif-fps", type=float, default=8.0)
    ap.add_argument("--contact", type=int, default=0, help="出一张 N 格关键帧拼图")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "runs" / "render"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args(list(argv) if argv is not None else None)

    video = Path(a.video)
    if not video.exists():
        raise SystemExit("视频不存在：%s" % video)
    specs = []
    for item in a.model:
        tag, path = item.split("=", 1)
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).parent / path
        specs.append((tag, p))
    models = [(tag, load_model(p, a.resolution)) for tag, p in specs]

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_f = int(a.start * fps)
    end_f = int(a.end * fps) if a.end > 0 else total
    out_tag = a.tag or ("%s_%s" % (video.stem, "+".join(t for t, _ in specs)))
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fps = max(1.0, fps / a.stride)
    writer = cv2.VideoWriter(str(out_dir / (out_tag + ".mp4")),
                             cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
    if not writer.isOpened():
        raise SystemExit("VideoWriter 打不开（本机只有 mp4v）")
    print("视频 %s：%dx%d %.0ffps %d 帧（%.0f–%.0fs，stride=%d → 输出 %.1f fps）"
          % (video.name, width, height, fps, total, start_f / fps, end_f / fps, a.stride, out_fps))

    if start_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    gif_lo, gif_hi = None, None
    if a.gif_range:
        lo, hi = (float(v) for v in a.gif_range.split(","))
        gif_lo, gif_hi = lo * fps, hi * fps
    gif_frames: list[np.ndarray] = []
    contact_frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    index = start_f
    made = 0
    while index < end_f:
        ok = cap.grab()
        if not ok:
            break
        if (index - start_f) % a.stride:
            index += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        panels, notes = [], []
        for tag, model in models:
            stats = {"person": 0, "ball": 0, "cone": 0, "suppressed": 0}
            det = dets_for(model, frame, a.threshold)
            painted = draw(frame, det, a.suppress_iou, stats)
            notes.append("%s: person %d · ball %d（压制 %d） · cone %d"
                         % (tag, stats["person"], stats["ball"], stats["suppressed"],
                            stats["cone"]))
            panels.append(banner(painted, notes[-1:], tag))
            rows.append({"frame": index, "time": round(index / fps, 3), "model": tag,
                         **stats})
            if gif_lo is not None and gif_lo <= index <= gif_hi:
                gif_frames.append(cv2.resize(painted, (width // 2, height // 2)))
        strip = np.hstack(panels) if len(panels) > 1 else panels[0]
        writer.write(strip)
        if a.contact and made % max(1, (end_f - start_f) // (a.stride * max(1, a.contact))) == 0:
            contact_frames.append(cv2.resize(strip, (strip.shape[1] // 2, strip.shape[0] // 2)))
        made += 1
        index += 1
        if made % 200 == 0:
            print("  已渲染 %d 帧（frame %d / %.0fs）" % (made, index, index / fps), flush=True)
    cap.release()
    writer.release()
    print("视频 → %s（%d 帧）" % (out_dir / (out_tag + ".mp4"), made))

    with (out_dir / (out_tag + "_frames.csv")).open("w", newline="", encoding="utf-8") as fh:
        writer_csv = csv.DictWriter(fh, fieldnames=["frame", "time", "model", "person", "ball",
                                                    "cone", "suppressed"])
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    if a.contact and contact_frames:
        cols = min(4, len(contact_frames))
        rows_n = (len(contact_frames) + cols - 1) // cols
        tiles = [np.hstack(contact_frames[i * cols:(i + 1) * cols]) for i in range(rows_n)]
        wmin = min(t.shape[1] for t in tiles)
        sheet = np.vstack([t[:, :wmin] for t in tiles])
        cv2.imwrite(str(out_dir / (out_tag + "_contact.jpg")), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("拼图 → %s_contact.jpg（%d 格）" % (out_dir / out_tag, len(contact_frames)))

    if gif_frames:
        from PIL import Image

        step = max(1, int(round(out_fps / a.gif_fps)))
        gif_list = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                    .convert("P", palette=Image.ADAPTIVE, colors=128)
                    for f in gif_frames[::step]]
        gif_path = out_dir / (out_tag + ".gif")
        gif_list[0].save(str(gif_path), save_all=True, append_images=gif_list[1:],
                         duration=int(1000 / a.gif_fps), loop=0, optimize=True)
        print("GIF → %s（%d 帧，%.1f MB）" % (gif_path, len(gif_list),
                                            gif_path.stat().st_size / 1e6))

    tag0 = models[0][0]
    sub = [r for r in rows if r["model"] == tag0]
    if sub:
        n = len(sub)
        print("\n汇总（%s）：球 %d 框/%.1f 帧均值｜锥 %d｜被压制（球∩锥）%d｜人 %d"
              % (tag0, sum(r["ball"] for r in sub), sum(r["ball"] for r in sub) / n,
                 sum(r["cone"] for r in sub), sum(r["suppressed"] for r in sub),
                 sum(r["person"] for r in sub)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
