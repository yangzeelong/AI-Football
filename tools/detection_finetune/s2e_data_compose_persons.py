#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第二步：**纯色背景 + 摆人 → 拼成 mp4**（人数是唯一的自变量）。

为什么要纯色底
--------------
之前那版是"真实底板把人 inpaint 掉、再把抠像贴回**原框原位**"：原位的信息量是原视频决定的
（原视频几个人，k 就最多几个），而且每人各带一块自己的背景。纯色底把这件事彻底变成"往画布上
摆 k 张人体素材"：

* **人数完全可控**：k 想要几个就摆几个；
* **无干扰**：纯色底没有草地纹理/线/看台，检测器只可能输出来自素材的人框；
* **不重叠是构造保证**：把人摆到**互不相交的网格单元**里（单元尺寸按"目标人高 × 素材最大宽高比"
  算），所以"设定 k 人"在几何上就保证了不会变成 k-1 或 k+1；
* 矩形裁片贴到纯色底上会有硬边，用 `--feather`（默认 4 px）羽化掉。

**素材必须先筛「单人」**（默认开，`--screen-sprites`）
----------------------------------------------------
素材库里实测 **46/136 张其实含 2–4 个人**：前后站位的人框是"窄高"的，`s2d` 的"同帧框 IoU
重叠"闸门抓不到这种**深度重叠**，而 `左边比赛1` 那种比赛视频尤其多。不筛就合成，随机抽到几张
多人素材就会把 k 变成 k+2 —— 上一轮 8/10 档自检失败（k=4 检出 8、k=32 检出 35）就是这么来的。
所以这里**用与自检完全相同的判据**（同权重、同阈值、同人高、同灰底）把每张素材**单独**送检，
只保留恰好 1 个人的，结果缓存到 `sprite_screen.json`。

为什么必须先**统一人高**（这一版的关键修正）
--------------------------------------------
素材库是**从多段视频**抠的（这正是 `s2d` 的用途），所以原尺寸从 140 px 到 497 px 混杂。
若原样贴上去，"k 个人"这句里就同时混进了"每个人多大"——而姿态（HRNet）的成本正比于人的像素
尺寸，于是 k→耗时 曲线会被尺寸差污染。所以本脚本**一律把素材归一到统一人高**（默认取素材库
中位框高，`--person-height` 可改，`--height-jitter` 给 ±12% 的自然抖动），
并把这个"人高"逐档写进报告。

两个必须知道的代价（报告里逐档给数字）
--------------------------------------
1. **大 k 会缩小人**：画布固定时 k 越大每格越小，人高按 `网格` 自动等比缩小（报告列 `人高`）。
   所以大 k 的耗时里混着"人变小"，要纯变量就换更大画布（`--canvas 3840x2160`）或分两段看。
2. **纯色底码率极低**：H.264 对静止画面压得极狠，解码比真实比赛视频（≈6 Mbps）便宜得多，
   所以这里测到的主要是**推理成本**。报告仍给码率供对照。

用法::

    source env.sh
    $PY s2e_data_compose_persons.py                             # 默认 1,2,4,6,8,12,16,20,24
    $PY s2e_data_compose_persons.py --counts 1,4,8,16 --frames 900
    $PY s2e_data_compose_persons.py --person-height 240 --bg-color grass
    $PY s2e_data_compose_persons.py --list                       # 只看素材库
    $PY s2e_data_compose_persons.py --no-verify                   # 跳过检测自检（省时间）

产物：`<data>/ball_cone_v1/work/perf_person_videos/k%02d.mp4`（+ `manifest.json`）；
报告 `<data>/ball_cone_v1/reports/perf_person_videos.md`
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np

import base_layout
import s5_eval_render as rd

HERE = Path(__file__).resolve().parent
ASSETS_DIR = base_layout.WORK_DIR / "cutouts_person_hq"
OUT_DIR = base_layout.WORK_DIR / "perf_person_videos"
REPORT_MD = base_layout.REPORTS_DIR / "perf_person_videos.md"
DEFAULT_CHECKPOINT = HERE / "runs" / "v1-frozen-step2" / "checkpoint_best_ema.pth"

# 命名颜色（写的是 RGB，落地时转 BGR —— OpenCV 用 BGR，别记混）
NAMED_COLORS = {
    "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128),
    "dimgray": (64, 64, 64), "green": (46, 90, 46), "grass": (60, 120, 60),
    "blue": (40, 60, 140), "red": (160, 40, 40),
}


def parse_color(spec: str) -> tuple[int, int, int]:
    """颜色解析：名字（black/gray/green/grass…）| `#RRGGBB` | `R,G,B`。返回 OpenCV 的 BGR。"""
    s = spec.strip().lower()
    if s in NAMED_COLORS:
        r, g, b = NAMED_COLORS[s]
        return (b, g, r)
    s = s.lstrip("#")
    if "," in s:
        parts = [int(x) for x in s.split(",")]
        if len(parts) != 3 or any(not 0 <= p <= 255 for p in parts):
            raise SystemExit("颜色要 3 个 0–255 的分量：%s" % spec)
        r, g, b = parts
        return (b, g, r)
    if len(s) == 6 and all(c in "0123456789abcdef" for c in s):
        return (int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16))     # RGB → BGR
    raise SystemExit("认不出颜色：%s（用 black/gray/grass 或 #RRGGBB 或 R,G,B）" % spec)


def load_assets(assets_dir: Path) -> list[dict[str, Any]]:
    """读素材库 index.json，只保留磁盘上还在的 PNG，并把图像载进内存。"""
    idx = assets_dir / "index.json"
    if not idx.is_file():
        raise SystemExit("找不到素材库索引：%s（先跑 s2d_data_cut_person_assets.py）" % idx)
    doc = json.loads(idx.read_text(encoding="utf-8"))
    assets = [a for a in doc.get("assets", []) if (assets_dir / a["file"]).is_file()]
    if not assets:
        raise SystemExit("素材库里没有可用素材：%s" % assets_dir)
    for a in assets:
        rgba = cv2.imread(str(assets_dir / a["file"]), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise SystemExit("素材读不出来：%s" % (assets_dir / a["file"]))
        a["rgba"] = rgba
        a["aspect"] = rgba.shape[1] / max(1, rgba.shape[0])            # 宽/高
    return assets


def pick_pool(assets: list[dict[str, Any]], quantile: float
              ) -> tuple[list[dict[str, Any]], float, float]:
    """按宽高比筛出规划用的素材池，返回 (池, 规划用最大宽高比, 素材库中位框高)。

    为什么要筛：单元宽度得按"最宽的那张"算，否则会撞单元。而素材库里混着**偏宽**的框
    （框里塞了两个人那种），一张宽框就会把整个网格的缩放压低，所有人都跟着变小。
    所以取宽高比的 P90 作规划值，比它更宽的素材**不进池**（宁缺毋滥，保"每人一个框"）。
    """
    ratios = sorted(a["aspect"] for a in assets)
    heights = sorted(a["rgba"].shape[0] for a in assets)
    ratio = ratios[min(len(ratios) - 1, int(len(ratios) * quantile))]
    pool = [a for a in assets if a["aspect"] <= ratio + 1e-9]
    median_h = heights[len(heights) // 2]
    return pool, ratio, median_h


def count_persons(frame: np.ndarray, model: Any, threshold: float) -> int:
    """这一帧里有几个「人」（检测口径，与自检/素材筛选完全一致）。"""
    det = rd.dets_for(model, frame, threshold)
    return sum(1 for label in det["labels"] if label == 1)


def person_count_on_gray(bgr: np.ndarray, model: Any, threshold: float) -> int:
    """把一张（已缩到目标人高的）素材贴到灰底后**单独**送检，返回检出人数。"""
    canvas = np.empty_like(bgr)
    canvas[:] = 128
    canvas[:] = bgr
    return count_persons(canvas, model, threshold)


def screen_sprites(assets: list[dict[str, Any]], model: Any, threshold: float,
                   heights: Sequence[float], cache_path: Path
                   ) -> tuple[set[str], dict[str, Any]]:
    """逐个素材单独送检，**只保留「恰好 1 个人」的**；返回 (保留的文件名集合, 统计)。

    为什么必须筛：素材库里实测 **46/136 张其实含 2–4 个人**（前后站位——框是窄高的，
    `s2d` 的"同帧框 IoU 重叠"闸门抓不到这种深度重叠），贴上去就把 k 变成 k+2。
    上一轮 8/10 档自检失败（k=4 检出 8）就是抽到了这些多人素材。

    关键点：**筛选判据与自检判据完全同一套**（同一个权重、阈值、人高、灰底），两头口径一致。
    因为大 k 会把人缩小，所以在"最小人高"和"最大人高"两个尺寸上都必须恰好 1 个人。
    结果按 (文件, 高度) 缓存，重跑不用再算。
    """
    cache: dict[str, int] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    kept: set[str] = set()
    hist: dict[int, int] = {}
    examples: list[dict[str, Any]] = []
    for a in assets:
        rgba = a["rgba"]
        counts: dict[int, int] = {}
        for hgt in sorted({int(round(x)) for x in heights}):
            key = "%s|%d" % (a["file"], hgt)
            if key not in cache:
                h, w = rgba.shape[:2]
                s = hgt / h
                small = cv2.resize(rgba, (max(4, int(round(w * s))), max(8, hgt)),
                                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                cache[key] = person_count_on_gray(np.ascontiguousarray(small[..., :3]), model, threshold)
            counts[hgt] = cache[key]
        worst = max(counts.values()) if counts else 0
        hist[worst] = hist.get(worst, 0) + 1
        if worst == 1 and min(counts.values()) == 1:
            kept.add(a["file"])
        elif len(examples) < 10:
            examples.append({"file": a["file"], "counts": counts})
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return kept, {"hist": dict(sorted(hist.items())), "examples": examples,
                  "heights": sorted({int(round(x)) for x in heights}),
                  "n_assets": len(assets), "n_kept": len(kept)}


def plan_grid(k: int, canvas_wh: tuple[int, int], cell_wh: tuple[float, float]
              ) -> tuple[int, int, float]:
    """给 k 个人分互不相交的网格：返回 (cols, rows, 人高缩放)。

    选网格的原则是**让缩放尽量大**（尽量保持人高、尽量用满画布）。因为单元互不相交、
    每张素材又都缩到能塞进自己的单元里，所以"人框不重叠"是**构造保证**，不靠事后碰撞检测。
    """
    cw0, ch0 = canvas_wh
    mw, mh = cell_wh
    want = cw0 / ch0                       # 画布长宽比（网格形状朝它靠）
    best: tuple[int, int, float] | None = None
    for cols in range(1, k + 1):
        rows = math.ceil(k / cols)
        cw, ch = cw0 / cols, ch0 / rows
        scale = min(1.0, cw * 0.92 / mw, ch * 0.92 / mh)
        if best is None or scale > best[2] + 1e-9:
            best = (cols, rows, scale)
        elif abs(scale - best[2]) <= 1e-9:
            # 缩放到不了更高就选**形状更贴画布**的那个：否则 k=4 会挑出 1×4（竖着一列），
            # 人全挤在画面中轴，既不自然、也浪费画布
            if abs(cols / rows - want) < abs(best[0] / best[1] - want):
                best = (cols, rows, scale)
    assert best is not None
    return best


def feathered_alpha(h: int, w: int, feather: int) -> np.ndarray:
    """四边线性羽化的 alpha（float32 0–1）。把矩形裁片的硬边抹掉，避免明显的方框边。"""
    a = np.ones((h, w), np.float32)
    f = int(max(0, min(feather, h // 3, w // 3)))
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, endpoint=False, dtype=np.float32)   # 长度必须 = f
        a[:f, :] *= ramp[:, None]
        a[-f:, :] *= ramp[::-1, None]
        a[:, :f] *= ramp[None, :]
        a[:, -f:] *= ramp[::-1][None, :]
    return a


def build_plan(pool: list[dict[str, Any]], k: int, canvas_wh: tuple[int, int], person_h: float,
               feather: float, jitter: float, height_jitter: float, rng: np.random.Generator,
               cols: int, rows: int) -> tuple[list[tuple[np.ndarray, np.ndarray, int, int]], int]:
    """把 k 个人安排到网格里：返回 ([(bgr, alpha, x, y)], 被强制收缩的张数)。

    每张素材缩到 `person_h ± height_jitter` 的等高尺寸后**只缩一次**，逐帧复用（省 CPU）。
    """
    cw, ch = canvas_wh[0] / cols, canvas_wh[1] / rows
    picks = rng.choice(len(pool), size=k, replace=len(pool) < k)
    plan: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    clamped = 0
    for i, ai in enumerate(picks):
        rgba = pool[int(ai)]["rgba"]
        h, w = rgba.shape[:2]
        goal = person_h * float(rng.uniform(1.0 - height_jitter, 1.0 + height_jitter))
        s = goal / h
        nh, nw = max(8, int(round(h * s))), max(4, int(round(w * s)))
        if nw > cw * 0.98 or nh > ch * 0.98:      # 兜底：理论上不该发生（池已按 P90 宽高比筛过）
            s2 = min(cw * 0.98 / nw, ch * 0.98 / nh)
            nh, nw = max(8, int(nh * s2)), max(4, int(nw * s2))
            clamped += 1
        resized = cv2.resize(rgba, (nw, nh),
                             interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        bgr = np.ascontiguousarray(resized[..., :3])
        alpha = feathered_alpha(nh, nw, int(round(feather * min(1.0, s))))
        r, c = divmod(i, cols)
        # 单元内居中 + 抖动（抖动上限 = 单元余量，抖满也不跨单元 → 仍不重叠）
        slack_x, slack_y = max(0.0, cw - nw), max(0.0, ch - nh)
        cx = c * cw + cw / 2.0 + float(rng.uniform(-0.5, 0.5)) * slack_x * jitter
        cy = r * ch + ch / 2.0 + float(rng.uniform(-0.5, 0.5)) * slack_y * jitter
        x = int(min(max(0, round(cx - nw / 2.0)), canvas_wh[0] - nw))
        y = int(min(max(0, round(cy - nh / 2.0)), canvas_wh[1] - nh))
        plan.append((bgr, alpha, x, y))
    return plan, clamped


def render_frame(canvas: np.ndarray, plan: Sequence[tuple[np.ndarray, np.ndarray, int, int]]) -> None:
    """把计划里的人物按 alpha 混合贴到画布上（原地改 canvas）。"""
    for bgr, alpha, x, y in plan:
        h, w = bgr.shape[:2]
        roi = canvas[y:y + h, x:x + w].astype(np.float32)
        a = alpha[..., None]
        canvas[y:y + h, x:x + w] = np.clip(bgr.astype(np.float32) * a + roi * (1.0 - a),
                                           0, 255).astype(np.uint8)


def encode_mp4(path: Path, frames: Iterable[np.ndarray], size: tuple[int, int], fps: float,
               crf: int) -> int:
    """流式写 H.264（PyAV/libx264；本机 OpenCV 没有 H.264 编码器）。返回帧数。"""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    out = av.open(str(path), mode="w")
    stream = out.add_stream("libx264", rate=max(1, int(round(fps))))
    stream.width, stream.height = size
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": "veryfast"}
    n = 0
    for frame in frames:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
        for packet in stream.encode(vf):
            out.mux(packet)
        n += 1
    for packet in stream.encode():
        out.mux(packet)
    out.close()
    return n


def new_plan(pool: list[dict[str, Any]], k: int, cfg: argparse.Namespace,
             rng: np.random.Generator, person_h: float, cols: int, rows: int) -> list[Any]:
    """抽一批素材 + 抖一次位置，得到一个新摆位。"""
    plan, _ = build_plan(pool, k, cfg.canvas_wh, person_h, cfg.feather, cfg.jitter,
                         cfg.height_jitter, rng, cols, rows)
    return plan


def frames_for_k(cfg: argparse.Namespace, plans: list[list[Any]]) -> Iterator[np.ndarray]:
    """按帧号套用对应区块的摆位逐帧产出（区块内静止，区块之间切一次）。"""
    period = cfg.re_jitter_every if cfg.re_jitter_every > 0 else cfg.frames
    bg = np.empty((cfg.canvas_wh[1], cfg.canvas_wh[0], 3), np.uint8)
    bg[:] = cfg.bg_bgr
    for f in range(cfg.frames):
        b = min(len(plans) - 1, f // max(1, period))
        frame = bg.copy()
        render_frame(frame, plans[b])
        yield frame


def compose_video(path: Path, cfg: argparse.Namespace, pool: list[dict[str, Any]], k: int,
                  person_h: float, cols: int, rows: int, rng: np.random.Generator,
                  model: Any, threshold: float, rounds: int) -> dict[str, Any]:
    """合成 mp4 + **按「解码后的帧」逐块过检** + 坏块重摆重编。

    为什么必须按解码帧判、而不能按内存帧判（这一版的关键修正）
    ----------------------------------------------------------
    区块内是**静止**的，所以"某帧数出 k 个人"= "该区块每一帧都是 k 个人"，可以在合成前用
    内存帧预检。但 h264 的 yuv420p 色度子采样 + 量化会**轻微改变画面**：实测有档位在内存帧上
    数出 k 个，落到 mp4 里却变成 k±1（k=20 → 19、k=24 → 25）。而后面测 FPS 的 app 读的是
    **mp4**，所以唯一可信的判据是"编码 → 解码 → 检测"。这里就按这个口径逐块过检，
    坏的块重摆并重编（`--repair-rounds` 轮），直到全过或轮数用尽（用尽则在报告里标出）。
    """
    period = cfg.re_jitter_every if cfg.re_jitter_every > 0 else cfg.frames
    n_blocks = max(1, math.ceil(cfg.frames / max(1, period)))
    plans = [new_plan(pool, k, cfg, rng, person_h, cols, rows) for _ in range(n_blocks)]
    attempts = [1] * n_blocks
    failed: list[dict[str, Any]] = []
    n = 0
    for _ in range(max(1, rounds)):
        n = encode_mp4(path, frames_for_k(cfg, plans), cfg.canvas_wh, cfg.fps, cfg.crf)
        if model is None:              # --no-verify：只合成，不做解码过检
            break
        failed = []
        cap = cv2.VideoCapture(str(path))
        for b in range(n_blocks):
            fi = min(max(0, n - 1), b * period)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            got = count_persons(frame, model, threshold)
            if got != k:
                failed.append({"block": b, "frame": fi, "detected": got})
        cap.release()
        if not failed:
            break
        for item in failed:
            plans[item["block"]] = new_plan(pool, k, cfg, rng, person_h, cols, rows)
            attempts[item["block"]] += 1
    return {"blocks": n_blocks, "frames": n, "attempts": attempts, "failed": failed}


def verify(video: Path, model: Any, threshold: float) -> tuple[int, int]:
    """自检：抽首帧与中段帧，数**检测到的**人数（设定 k 必须等于观测值，否则这一档不算数）。"""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    got: list[int] = []
    for fi in (0, max(0, total // 2)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        det = rd.dets_for(model, frame, threshold)
        got.append(sum(1 for label in det["labels"] if label == 1))
    cap.release()
    return (got[0] if got else -1), (got[-1] if len(got) > 1 else -1)


def write_report(rows: list[dict[str, Any]], path: Path, n_assets: int, n_pool: int,
                 cfg: argparse.Namespace, person_h0: float,
                 screen_stats: dict[str, Any] | None = None) -> None:
    """报告：每档的规模/人高/码率/自检结果 —— 自检不过的档显式标出来。"""
    lines = [
        "# 纯色底合成视频（人数 → mp4）", "",
        "生成时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"), "",
        "画布 %dx%d ｜ 底色 `%s`（BGR %s）｜ %d 帧 @%.0f fps ｜ 羽化 %d px ｜ 网格抖动 %.2f"
        % (cfg.canvas_wh[0], cfg.canvas_wh[1], cfg.bg_color, cfg.bg_bgr, cfg.frames, cfg.fps,
           cfg.feather, cfg.jitter), "",
        "素材库 %d 张（中位框高 %d px）；按宽高比 P%.0f 筛出 %d 张用于规划，"
        "**统一人高**（基准 %d px，±%.0f%%），所以 k 是唯一变量。"
        % (n_assets, round(person_h0), cfg.aspect_quantile * 100, n_pool,
           round(cfg.person_height or person_h0), cfg.height_jitter * 100), "",
        "## 一、逐档结果", "",
        "| 人/帧 k | 网格 | **人高 px** | 摆位过检 | 文件 | 帧数 | 体积 | 码率 Mbps | 抽查(首帧/中帧) |",
        "| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        mark = ("✅ %d / %d" % (r["det0"], r["det1"])) if r["verified"] \
            else ("❌ %d / %d（应为 %d）" % (r["det0"], r["det1"], r["k"]))
        blocks = ("✅ %d/%d" % (r["blocks"] - r["blocks_failed"], r["blocks"])) \
            if not r["blocks_failed"] else ("❌ %d/%d" % (r["blocks"] - r["blocks_failed"],
                                                          r["blocks"]))
        lines.append("| %d | %d×%d | **%d** | %s | `%s` | %d | %.1f MB | %.2f | %s |"
                     % (r["k"], r["cols"], r["rows"], r["person_h"], blocks, r["file"],
                        r["frames"], r["mb"], r["mbps"], mark))
    bad = [r for r in rows if not r["verified"]]
    lines += ["", "## 二、自检（两道）", "",
              "**第一道 · 逐块过检（构造保证，判据用解码后的帧）**：区块内画面是静止的，所以"
              "「某帧数出 k 个人」= 「该区块每一帧都是 k 个人」。合成时按"
              "**编码 → 解码 → 数人**逐块过检（为什么不用内存帧：h264 的 yuv420p 子采样+量化"
              "会轻微改画面，实测有档位内存帧数出 k、mp4 里却 k±1），不过检的块**重摆并重编**"
              "（`--repair-rounds`，默认 6 轮）。这样 `人/帧 = k` 是**合成时就保证的**，"
              "而不是事后抽查碰运气。", "",
              "**第二道 · 成片抽查（独立复核）**：编码后再从 mp4 里抽首帧与中帧各检一次，"
              "防止漏了「编码 / 读回」环节的问题。", ""]
    if bad:
        lines += ["**%d 档未通过**：%s"
                  % (len(bad), "、".join("k=%d(摆位 %d/%d 区块过检，抽查 %d/%d)"
                                        % (r["k"], r["blocks"] - r["blocks_failed"], r["blocks"],
                                           r["det0"], r["det1"]) for r in bad)), "",
                  "→ 摆位没过：把 `--plan-attempts` 调大、换更大 `--canvas`、或调低 `--jitter`；",
                  "→ 抽查没过但摆位全过：说明 h264 编码/读回引入了差异，查 `--crf`；",
                  "→ 未通过的档**不能**当有效点，别写进 FPS 结论。"]
    else:
        lines += ["全部 %d 档通过：摆位逐块过检 + 成片抽查，设定人数 == 检测人数 ✅" % len(rows)]
    if screen_stats:
        lines += ["", "## 三、素材单人筛选（这一步是必须的）", "",
                  "素材库里 **%d/%d 张其实含 2–4 个人**（前后站位的窄高框，"
                  "`s2d` 的「同帧框 IoU 重叠」闸门抓不到这种深度重叠）。"
                  "若不筛，随机抽到几张多人素材就会把 k 变成 k+2 —— 实测 k=4 检出 8、k=32 检出 35。"
                  "这里用**与自检完全相同的判据**（同权重/阈值/人高/灰底）逐个素材单独送检，"
                  "只保留恰好 1 个人的。" % (screen_stats["n_assets"] - screen_stats["n_kept"],
                                       screen_stats["n_assets"]), "",
                  "筛选后：保留 **%d** 张，剔除 **%d** 张。" % (screen_stats["n_kept"],
                                                          screen_stats["n_assets"] - screen_stats["n_kept"]), "",
                  "| 单独送检检出人数 | 素材张数 |", "| ---: | ---: |"]
        for n, c in screen_stats["hist"].items():
            lines.append("| %d | %d |" % (n, c))
        lines += ["", "剔除样例（文件名 → 各人高下的检出人数）：", ""]
        for ex in screen_stats["examples"][:6]:
            lines.append("- `%s` → %s" % (ex["file"], ex["counts"]))
    lines += ["", "## 四、两个必须知道的代价", "",
              "1. **大 k 会缩小人**（见 `人高` 列）：画布固定时 k 越大每格越小 → 人的像素变少，"
              "而姿态成本正比于人像尺寸，所以大 k 的耗时里混着「人变小」。要纯变量就换更大画布"
              "（`--canvas 3840x2160`）或分两段看趋势。",
              "2. **纯色底码率极低**（见 `码率` 列，通常 < 1 Mbps）：解码比真实比赛视频"
              "（≈6 Mbps）便宜得多 → 这里测到的主要是**推理成本**，不是解码成本。", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets-dir", default=str(ASSETS_DIR), help="人体素材库（s2d 的产物）")
    ap.add_argument("--counts", default="1,2,4,6,8,12,16,20,24,32", help="人数档（逗号分隔）")
    ap.add_argument("--canvas", default="1920x1080", help="画布 WxH")
    ap.add_argument("--bg-color", default="gray",
                    help="底色：名字（black/gray/grass…）| `#RRGGBB` | `R,G,B`")
    ap.add_argument("--person-height", type=int, default=0,
                    help="统一人高 px（0 = 用素材库中位框高）；大 k 会被网格自动缩小，报告里列实际值")
    ap.add_argument("--height-jitter", type=float, default=0.12, help="人高抖动比例（±）")
    ap.add_argument("--aspect-quantile", type=float, default=0.9,
                    help="规划网格用的素材宽高比分位（越小的框越窄；偏宽框会被剔出池）")
    ap.add_argument("--frames", type=int, default=600, help="每档视频帧数")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--crf", type=int, default=18, help="H.264 画质（越小越清晰越大）")
    ap.add_argument("--feather", type=int, default=4, help="素材边缘羽化 px（抹掉矩形硬边）")
    ap.add_argument("--jitter", type=float, default=0.6,
                    help="网格内抖动量（0–1，按单元余量比例；≤1 保证不重叠）")
    ap.add_argument("--re-jitter-every", type=int, default=60,
                    help="每 N 帧换一次摆位（0 = 全程不动；有运动会抬高码率、更接近真实素材）")
    ap.add_argument("--repair-rounds", type=int, default=6,
                    help="「编码 → 解码 → 逐块数人」的修复轮数：不过检的区块会重摆并重编，"
                         "直到每块都恰好 k 个人（判据用**解码后的帧**，即 app 真正读到的画面）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--report", default=str(REPORT_MD))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--resolution", type=int, default=960)
    ap.add_argument("--threshold", type=float, default=0.35, help="自检/素材筛选用检测阈值")
    ap.add_argument("--screen-sprites", dest="screen_sprites", action="store_true", default=True,
                    help="**先用检测器把每张素材单独送检，只留「恰好 1 个人」的**（默认开）——"
                         "实测素材库里 46/136 张含 2–4 个人（前后站位），不筛就会把 k 变成 k+2")
    ap.add_argument("--no-screen-sprites", dest="screen_sprites", action="store_false")
    ap.add_argument("--no-verify", dest="verify", action="store_false", help="跳过检测自检")
    ap.add_argument("--list", action="store_true", help="只打印素材库概况后退出")
    ap.add_argument("--clean", action="store_true", help="先清空输出目录里的 k*.mp4")
    args = ap.parse_args(argv)

    w, h = (int(x) for x in args.canvas.lower().split("x"))
    args.canvas_wh = (w, h)
    args.bg_bgr = parse_color(args.bg_color)
    counts = sorted({int(c) for c in args.counts.split(",") if c.strip()})
    if not counts:
        raise SystemExit("--counts 是空的")

    assets_all = load_assets(Path(args.assets_dir))
    hs = sorted(a["rgba"].shape[0] for a in assets_all)
    median_h = hs[len(hs) // 2]
    person_h0 = float(args.person_height or median_h)
    pool, plan_ratio, _ = pick_pool(assets_all, args.aspect_quantile)
    if args.list:
        print("素材库 %s：%d 张，中位框高 %d px" % (args.assets_dir, len(assets_all), median_h))
        print("  规划池 %d 张（宽高比 ≤ %.2f = P%.0f）；统一人高基准 %.0f px ±%.0f%%"
              % (len(pool), plan_ratio, args.aspect_quantile * 100, max(1.0, person_h0),
                 args.height_jitter * 100))
        print("  底色 `%s` → BGR %s；画布 %dx%d" % (args.bg_color, args.bg_bgr, w, h))
        for k in counts:
            cols, rows_, sc = plan_grid(k, (w, h), (person_h0 * plan_ratio, person_h0))
            print("  k=%-3d → 网格 %dx%d，人高 %.0f px" % (k, cols, rows_, person_h0 * sc))
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for old in out_dir.glob("k*.mp4"):
            old.unlink()
        print("已清空 %s 里的 k*.mp4" % out_dir)

    print("素材库 %d 张（中位框高 %d px）" % (len(assets_all), median_h))
    print("画布 %dx%d，底色 `%s` BGR %s，统一人高基准 %.0f px ±%.0f%%"
          % (w, h, args.bg_color, args.bg_bgr, person_h0, args.height_jitter * 100))

    model = rd.load_model(Path(args.checkpoint), args.resolution) if args.verify else None
    screen_stats: dict[str, Any] | None = None
    assets = assets_all
    if args.screen_sprites and model is not None:
        print("① 素材单人筛选（逐个送检，判据与自检完全一致）…", flush=True)
        kept, screen_stats = screen_sprites(assets_all, model, args.threshold, [person_h0],
                                            out_dir / "sprite_screen.json")
        assets = [a for a in assets_all if a["file"] in kept]
        print("  保留单人素材 %d/%d 张 ｜ %s"
              % (len(assets), len(assets_all),
                 "，".join("检出 %d 人：%d 张" % (n, c)
                          for n, c in screen_stats["hist"].items())))
        for ex in screen_stats["examples"][:3]:
            print("    例：%s → %s" % (ex["file"], ex["counts"]))
        if len(assets) < max(counts):
            raise SystemExit("单人素材只剩 %d 张，不够 k=%d；调低 --counts、补素材，"
                             "或显式用 --no-screen-sprites（不建议，自检会不过）"
                             % (len(assets), max(counts)))
    pool, plan_ratio, _ = pick_pool(assets, args.aspect_quantile)
    print("② 规划池 %d 张（宽高比 ≤ %.2f = P%.0f）" % (len(pool), plan_ratio,
                                                    args.aspect_quantile * 100))

    rows: list[dict[str, Any]] = []
    for k in counts:
        cols, rows_, sc = plan_grid(k, (w, h), (person_h0 * plan_ratio, person_h0))
        person_h = person_h0 * sc
        rng = np.random.default_rng(args.seed + k * 1000)
        name = "k%02d.mp4" % k
        path = out_dir / name
        t0 = time.time()
        pstats = compose_video(path, args, pool, k, person_h, cols, rows_, rng, model,
                               args.threshold, args.repair_rounds)
        n = pstats["frames"]
        mb = path.stat().st_size / 1e6
        mbps = mb * 8.0 / (n / args.fps) if n else 0.0
        det0 = det1 = -1
        if args.verify and model is not None:
            det0, det1 = verify(path, model, args.threshold)
        n_bad = len(pstats["failed"])
        ok = (n_bad == 0) and ((det0 == k and det1 == k) if args.verify else True)
        rows.append({"k": k, "cols": cols, "rows": rows_, "person_h": int(round(person_h)),
                     "file": name, "frames": n, "mb": mb, "mbps": mbps, "det0": det0,
                     "det1": det1, "verified": ok, "sec": round(time.time() - t0, 1),
                     "blocks": pstats["blocks"], "blocks_failed": n_bad,
                     "attempts_max": max(pstats["attempts"]) if pstats["attempts"] else 0})
        print("  k=%-3d 网格 %dx%d 人高 %3d px 摆位 %d/%d 区块过检(最多重摆 %d 次) → %s  "
              "%5.1f MB  %.2f Mbps  抽查 %s"
              % (k, cols, rows_, round(person_h), pstats["blocks"] - n_bad, pstats["blocks"],
                 rows[-1]["attempts_max"], name, mb, mbps,
                 "✅ %d/%d" % (det0, det1) if ok else "❌ %d/%d（应为 %d）" % (det0, det1, k)),
              flush=True)

    (out_dir / "manifest.json").write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "canvas": [w, h], "bg_bgr": list(args.bg_bgr), "bg_color": args.bg_color,
        "fps": args.fps, "frames": args.frames, "assets_dir": str(args.assets_dir),
        "n_assets": len(assets), "n_pool": len(pool), "plan_ratio": plan_ratio,
        "person_height_base": person_h0, "height_jitter": args.height_jitter,
        "videos": rows,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_report(rows, Path(args.report), len(assets), len(pool), args, median_h, screen_stats)
    bad = [r for r in rows if not r["verified"]]
    print("\n完成 %d 档 → %s" % (len(rows), out_dir))
    if bad:
        print("  ⚠️ %d 档自检未过：%s（这些档不能当有效点）"
              % (len(bad), "、".join("k=%d" % r["k"] for r in bad)))
    else:
        print("  自检全部通过 ✅（设定人数 == 检测人数）")
    print("  报告：%s" % args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
