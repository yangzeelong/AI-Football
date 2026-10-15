#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人数 → FPS 受控曲线（合成法）：抠人 → 合成「人数已知」的帧序列 → 编码 mp4 → C++ app 直跑测 FPS。

为什么要合成，而不是直接拿现成视频跑
------------------------------------
真实视频里「人/帧」是检测与 ROI 的**结果**：换一段视频会同时换掉内容、分辨率、编码、机位
和镜头距离。于是"人数 vs FPS"里混进了一堆别的变量（本次实测撞过两个：ProRes 素材把 FPS
钉在 33、1080p 与 720p 差一档）。合成法把**人数做成自变量**：

1. **素材**（可单独跑：`--cutouts-only`）：从一段人多的高清视频里抽底板帧，用模型框人 →
   **直接裁检测框**（`--cutout-mode box`，默认）存到 `work/cutouts_person/`，并写 `index.json`
   + `contact_sheet.jpg`。够用的原因：素材只贴回**原框原位**，像素与原帧一致 → 无矩形接缝；
   像素级抠像（`--cutout-mode grabcut`，带 alpha）只在"要挪位置/换背景"时才需要，且对人像失败率高；
2. **背景**：同一帧把**所有人框** inpaint 掉（还是那块场地，只是没人）；
3. **合成**：对每个 k，把该帧最大的 k 个抠像**按原框中心贴回**（不缩放，原像素尺度）→
   "恰好 k 个人"的帧；
4. **编码**：循环补足到 `--frames-out` 帧，写 mp4（PyAV/libx264 —— 本机 OpenCV 没有
   H.264 编码器，只有 mp4v，见 rollout README）；
5. **测速**：每个 k 用**同一份 cpp 配置**直跑 app（不加 `--profile_timers`、不加 `--render`），
   读 wall-clock FPS；同时读 `observations.jsonl` 的 person 数**自检「设定人数 == 观测人数」**
   —— 对不上的档不能当有效点（说明合成或检测出了问题）。

用法::

    source env.sh
    $PY s5_eval_perf_persons.py                      # 默认 1/2/4/6/8/12/16
    $PY s5_eval_perf_persons.py --counts 1,3,6,10 --frames-out 900
    $PY s5_eval_perf_persons.py --compose-only       # 只抠图+合成 mp4，不跑 app

产物：抠像 `<data>/ball_cone_v1/work/cutouts_person/*.png`；
`tools/rollout/runs/perf_synth/<k>人/{video.mp4,run.log,observations.jsonl}`；
报告 `<data>/ball_cone_v1/reports/perf_persons_synth.md`
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import base_layout
import s2a_data_compose_cones as compose
import s5_eval_render as rd

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = base_layout.DATA_ROOT / "20260726" / "室外" / "左边比赛1-1080p60.mov"
DEFAULT_CONFIG = HERE.parent / "rollout" / "cpp_app" / "v1-frozen-step2" / "fastest.yaml"
DEFAULT_APP = Path("/home/hx1/yzl/Work/Nexusflow/build/bin/aifootball_app")
DEFAULT_OUT = HERE.parent / "rollout" / "runs" / "perf_synth"
CUTOUT_DIR = base_layout.WORK_DIR / "cutouts_person"
FPS = 60.0
CRF = 18


def probe_persons(model: Any, frame: np.ndarray, thr: float) -> list[list[float]]:
    """这一帧的人框（xyxy），按面积降序。"""
    det = rd.dets_for(model, frame, thr)
    boxes = [list(map(float, b)) for b, label in zip(det["boxes"], det["labels"]) if label == 1]
    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    return boxes


def pick_backgrounds(source: Path, model: Any, want: int, need: int, thr: float,
                     stride: int, scan_max: int) -> list[tuple[int, np.ndarray, list[list[float]]]]:
    """扫源视频，取「人最多」的 `want` 帧当底板；返回 (帧号, 帧, 人框)。"""
    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = total if scan_max <= 0 else min(total, scan_max)
    found: list[tuple[int, np.ndarray, list[list[float]]]] = []
    for fi in range(0, limit, max(1, stride)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        boxes = probe_persons(model, frame, thr)
        if len(boxes) >= need:
            found.append((fi, frame.copy(), boxes))
            print("  候选底板 f%-7d 人=%d" % (fi, len(boxes)), flush=True)
            if len(found) >= want:
                break
    cap.release()
    return found


PERSON_MIN_SIDE = 24          # grabcut 模式下：原框短边小于这个 px 就不抠（远景小人抠不出干净掩码）
PERSON_MIN_SIDE_BOX = 8       # box 模式（默认）：只挡退化的极小框
ALPHA_RATIO_RANGE = (0.12, 0.90)


def extract_person_box(image: np.ndarray, box: Sequence[float], pad: float = 0.10
                       ) -> tuple[np.ndarray | None, str, str]:
    """**默认模式**：直接裁检测框（带小 pad）当素材，不做像素级抠图。

    为什么够用：这些素材只服务于"合成受控人数的测试帧"，而合成时是把它贴回**原框原位**，
    所以贴回去的像素与原帧逐点一致 —— **不存在矩形接缝**，检测自然认得出。
    抠像（GrabCut）不但没收益，还会引入失败/降级/闸门丢弃：实测 86 个里 53 个掉到 Otsu，
    拼图一看全是碎片与白块（contact_sheet 里一目了然）。

    代价（必须知道）：这种素材**只能原位贴**。要挪位置 / 换背景做数据增强，
    得显式用 `--cutout-mode grabcut` 拿带 alpha 的抠像。
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
    if min(bw, bh) < PERSON_MIN_SIDE_BOX:
        return None, "-", "too-small(%dx%d)" % (int(bw), int(bh))
    px, py = bw * pad, bh * pad
    cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    cx2, cy2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return None, "-", "crop-degenerate"
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = 255            # 不透明：贴回去就是原像素
    return rgba, "box", "ok"


def extract_person_rgba(image: np.ndarray, box: Sequence[float], pad: float = 0.20
                        ) -> tuple[np.ndarray | None, str, str]:
    """人像专用抠图：GrabCut 用「外圈=背景 / 框内核心=前景」的**掩码初始化**，并过质量闸门。

    为什么不直接用 `s2a_data_compose_cones.extract_rgba`：那套是给**锥/碟**写的
    （高对比小目标，GrabCut 失败 → Otsu → 硬框，三级降级）。人像用它会出问题 ——
    实测 86 个抠像里 **53 个掉到 Otsu**，拼图一看全是碎片/白块/条纹。人的难点是
    "颜色杂、背景是草地"，所以这里：
      * 掩码初始化（外圈 3px 判背景、框内收缩后的核心区判前景）比整框初始化稳得多；
      * GrabCut 失败**直接丢弃**，不做 Otsu/硬框降级 —— 碎片进素材库只会污染合成帧；
      * 三道质量闸门：alpha 占比、最大连通域占比、原框被掩码覆盖的比例。
    返回 `(rgba|None, method, reason)`；`reason="ok"` 才算成功。
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
    if min(bw, bh) < PERSON_MIN_SIDE:
        return None, "-", "too-small(%dx%d)" % (int(bw), int(bh))
    px, py = bw * pad, bh * pad
    cx1, cy1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    cx2, cy2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None, "-", "crop-too-small"
    ch, cw = crop.shape[:2]
    rx, ry = int(x1 - cx1), int(y1 - cy1)
    rw, rh = int(max(2, bw)), int(max(2, bh))
    mask = np.full((ch, cw), cv2.GC_PR_BGD, np.uint8)
    ring = 3
    mask[:ring, :] = cv2.GC_BGD
    mask[-ring:, :] = cv2.GC_BGD
    mask[:, :ring] = cv2.GC_BGD
    mask[:, -ring:] = cv2.GC_BGD
    core = np.zeros((ch, cw), np.uint8)
    core[max(0, ry + rh // 6):min(ch, ry + rh - rh // 6),
         max(0, rx + rw // 6):min(cw, rx + rw - rw // 6)] = 1
    mask[core > 0] = cv2.GC_PR_FGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        alpha = compose._tidy_alpha(alpha)
    except cv2.error:
        alpha = None
    if alpha is None:
        return None, "-", "grabcut-empty"
    keep = alpha > 8
    ratio = float(keep.mean())
    if not (ALPHA_RATIO_RANGE[0] <= ratio <= ALPHA_RATIO_RANGE[1]):
        return None, "-", "alpha-ratio=%.2f" % ratio
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(alpha, 8)
    if count > 1 and stats[1:, cv2.CC_STAT_AREA].max() / max(1.0, float(keep.sum())) < 0.5:
        return None, "-", "fragmented"
    rect_mask = np.zeros((ch, cw), np.uint8)
    rect_mask[max(0, ry):min(ch, ry + rh), max(0, rx):min(cw, rx + rw)] = 1
    covered = float((keep & (rect_mask > 0)).sum()) / max(1.0, float(rect_mask.sum()))
    if covered < 0.45:
        return None, "-", "box-lost=%.2f" % covered
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = cv2.GaussianBlur(alpha, (3, 3), 0)
    ys, xs = np.where(keep)
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1], "grabcut-mask", "ok"


def cut_out(frame: np.ndarray, boxes: list[list[float]], stem: str, frame_index: int,
            source_name: str, mode: str = "box"
            ) -> tuple[list[tuple[np.ndarray, tuple[float, float]]], list[dict[str, Any]],
                       list[dict[str, Any]]]:
    """把每个人存成 RGBA 素材（`box`=直接裁框 / `grabcut`=像素级抠像），并返回索引条目。

    返回 `(sprites, entries, rejected)`：`sprites` = [(rgba, (原框中心 x, y))] 供合成用；
    `entries`/`rejected` 落成 `index.json`，记**出处**（源视频/帧号/原框/模式/尺寸）——
    素材质量直接决定合成帧还能不能被检测认出人，出问题必须能回查。
    """
    CUTOUT_DIR.mkdir(parents=True, exist_ok=True)
    extract = extract_person_box if mode == "box" else extract_person_rgba
    pad = 0.10 if mode == "box" else 0.20
    sprites: list[tuple[np.ndarray, tuple[float, float]]] = []
    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for i, box in enumerate(boxes, 1):
        rgba, method, reason = extract(frame, box, pad=pad)
        bw, bh = float(box[2]) - float(box[0]), float(box[3]) - float(box[1])
        if rgba is None:
            # 抠不干净就丢：碎片进素材库只会污染合成帧（闸门理由记进索引，便于调参）
            rejected.append({"source": source_name, "frame_index": frame_index, "index": i,
                             "box_xyxy": [round(float(v), 1) for v in box],
                             "box_px": int(bw * bh), "reason": reason})
            continue
        name = "%s_p%02d_q%.0f_%s.png" % (stem, i, bw * bh, method)
        path = CUTOUT_DIR / name
        if not cv2.imwrite(str(path), rgba):
            raise RuntimeError("抠图写失败：%s" % path)
        cx, cy = (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0
        sprites.append((rgba, (cx, cy)))
        entries.append({"file": name, "source": source_name, "frame_index": frame_index,
                        "box_xyxy": [round(float(v), 1) for v in box],
                        "box_px": int(bw * bh),
                        "rgba_wh": [int(rgba.shape[1]), int(rgba.shape[0])],
                        "alpha_ratio": round(float((rgba[..., 3] > 8).mean()), 3),
                        "method": method})
    return sprites, entries, rejected


def write_cutout_index(entries: list[dict[str, Any]], source: Path, frames_used: int,
                       rejected: list[dict[str, Any]] | None = None) -> Path:
    """把抠图台账写成 `cutouts_person/index.json`（谁、哪一帧、怎么抠的、多大、为什么丢）。"""
    CUTOUT_DIR.mkdir(parents=True, exist_ok=True)
    by_method: dict[str, int] = {}
    for entry in entries:
        by_method[entry["method"]] = by_method.get(entry["method"], 0) + 1
    by_reason: dict[str, int] = {}
    for item in rejected or []:
        key = str(item["reason"]).split("(")[0].split("=")[0]
        by_reason[key] = by_reason.get(key, 0) + 1
    path = CUTOUT_DIR / "index.json"
    path.write_text(json.dumps({
        "source": str(source), "frames_used": frames_used,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_cutouts": len(entries), "n_rejected": len(rejected or []),
        "by_method": by_method, "reject_reasons": by_reason,
        "cutouts": entries, "rejected": rejected or [],
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def write_cutout_sheet(entries: list[dict[str, Any]], path: Path,
                       cols: int = 12, cell: int = 96) -> None:
    """把抠像排成一张拼图（灰底、按比例缩略），一眼看抠图质量。"""
    if not entries:
        return
    rows = (len(entries) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 40, np.uint8)
    for i, entry in enumerate(entries):
        img = cv2.imread(str(CUTOUT_DIR / entry["file"]), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            continue
        h, w = img.shape[:2]
        scale = min(cell / max(1, h), cell / max(1, w))
        th, tw = max(1, int(h * scale)), max(1, int(w * scale))
        small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y0 = r * cell + (cell - th) // 2
        x0 = c * cell + (cell - tw) // 2
        tile = sheet[y0:y0 + th, x0:x0 + tw].astype(np.float32)
        alpha = small[..., 3:4].astype(np.float32) / 255.0
        sheet[y0:y0 + th, x0:x0 + tw] = np.clip(
            small[..., :3].astype(np.float32) * alpha + tile * (1.0 - alpha), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def erase_persons(frame: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    """把所有人框 inpaint 掉，得到"没人的同一块场地"。"""
    mask = np.zeros(frame.shape[:2], np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1) - 2), max(0, int(y1) - 2)
        x2, y2 = min(frame.shape[1], int(x2) + 2), min(frame.shape[0], int(y2) + 2)
        mask[y1:y2, x1:x2] = 255
    if mask.max() == 0:
        return frame.copy()
    # 略膨胀：检测框常常贴着人身边缘，不扩一点会留下"半个人"的残影被再检出来
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    return cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)


def paste(canvas: np.ndarray, rgba: np.ndarray, center: tuple[float, float]) -> None:
    """把抠像按原尺度贴到 center（原框中心），alpha 混合，越界自动裁剪。"""
    sh, sw = rgba.shape[:2]
    x0 = int(round(center[0] - sw / 2.0))
    y0 = int(round(center[1] - sh / 2.0))
    h, w = canvas.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x0 + sw - sx0), min(h, y0 + sh - sy0)
    if x1 <= x0 or y1 <= y0:
        return
    sprite = rgba[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    alpha = sprite[..., 3:4].astype(np.float32) / 255.0
    canvas[y0:y1, x0:x1] = np.clip(sprite[..., :3].astype(np.float32) * alpha
                                   + roi * (1.0 - alpha), 0, 255).astype(np.uint8)


def compose_k(frame: np.ndarray, boxes: list[list[float]],
              sprites: list[tuple[np.ndarray, tuple[float, float]]], k: int) -> np.ndarray:
    """恰好 k 个人：抹掉没被选中的人，选中的保留原像素，再把它的抠像贴上。

    两个坑都是实测踩出来的（自检当场抓到，别省这一步）：
    * 不能"全抹掉再贴回 k 个" —— 抠像经过 alpha 裁剪/羽化，4 个贴回去只被检出 3 个；
    * 抹人的掩码必须用**原检测框**，不能用抠像尺寸反推的框（抠像被裁到 alpha 紧边界、
      更小，抹不干净 → 残留像素被当成额外的"人"，1 个人被测成 10 个）。
    """
    canvas = erase_persons(frame, boxes[k:])
    for rgba, center in sprites[:k]:
        paste(canvas, rgba, center)
    return canvas


def write_mp4(path: Path, frames: list[np.ndarray], fps: float = FPS, crf: int = CRF) -> None:
    """PyAV/libx264 写 H.264（本机 OpenCV 无 H.264 编码器）。"""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    height, width = frames[0].shape[:2]
    out = av.open(str(path), mode="w")
    stream = out.add_stream("libx264", rate=max(1, int(round(fps))))
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": "veryfast"}
    for frame in frames:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
        for packet in stream.encode(vf):
            out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()


def invoke_app(app: Path, cfg: Path, video: Path, out_dir: Path, frames: int) -> None:
    """app 直跑（不加 `--profile_timers`、不加 `--render`），日志落 out_dir/run.log。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(app), str(cfg), "--video_path", str(video), "--output_dir", str(out_dir),
           "--stride", "1", "--max_frames", str(frames)]
    log = out_dir / "run.log"
    with log.open("w", encoding="utf-8") as fh:
        fh.write("$ cd %s\n$ %s\n\n" % (app.parent.parent, " ".join(cmd)))
        fh.flush()
        subprocess.run(cmd, cwd=str(app.parent.parent), stdout=fh, stderr=subprocess.STDOUT, check=False)


def parse_run(out_dir: Path) -> dict[str, Any]:
    """从 run.log + observations.jsonl 读 FPS、观测人数、各模块项。

    ⚠️ `max_module` 只作**参考**，别拿它当瓶颈判据：管线打满时 `Worker.Process.*` 里最大的那项
    总是 ≈ 每帧墙钟（含排队等待）。实测 1 人档 `SdkInput` = 8.99 ms / 每帧 9.94 ms、
    12 人档 `SdkInput` = 35.08 / 每帧 35.20 —— 都"等于"每帧耗时，所以它指不出真瓶颈
    （我第一版拿它当"解码瓶颈"探测器，是错的）。要判解码是否参与瓶颈，看**码率**（`video_mbps`）。
    """
    text = (out_dir / "run.log").read_text(encoding="utf-8", errors="ignore")
    fps = re.search(r"effective rate\s*:\s*([0-9.]+)", text)
    ms = re.search(r"per frame\s*:\s*([0-9.]+)", text)
    module, module_ms = "—", 0.0
    for hit in re.finditer(r"Execute 'Worker\.Process\.([A-Za-z]+)' for \d+ items cost: "
                           r"[0-9.]+ ms, qps: [0-9.]+ items/s, avg: ([0-9.]+) ms/item", text):
        if float(hit.group(2)) > module_ms:
            module, module_ms = hit.group(1), float(hit.group(2))
    n = persons = 0
    obs = out_dir / "observations.jsonl"
    if obs.is_file():
        with obs.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if '"type"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") != "frame":
                    continue
                n += 1
                persons += len(rec.get("persons") or [])
    return {"fps": float(fps.group(1)) if fps else None,
            "ms_per_frame": float(ms.group(1)) if ms else None,
            "frames": n, "persons_per_frame": (persons / n) if n else 0.0,
            "max_module": module, "max_module_ms": module_ms}


def video_mbps(path: Path, frames: int, fps: float = FPS) -> float:
    """mp4 码率（Mbps）。真实素材 ≈ 6 Mbps（射门1-1080p60 = 92 MB / 7201 帧）；
    合成视频明显高于这个量级时，解码会参与瓶颈、测到的就不是管线成本了。"""
    if frames <= 0:
        return 0.0
    return path.stat().st_size * 8.0 / (frames / fps) / 1e6


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="抽人的源视频（挑人多的那段）")
    ap.add_argument("--counts", default="1,2,4,6,8,12,16", help="要合成的人数档（逗号分隔）")
    ap.add_argument("--bg-frames", type=int, default=1,
                    help="用几张不同底板（循环铺满视频）。**默认 1**：多张互不相关的底板会把码率顶到"
                         "~50 Mbps、把瓶颈从管线挪到解码（实测 1 人档被拖到 27.8 FPS）。"
                         "要画面多样性就用 2–3，但必须看报告里的「瓶颈」列。")
    ap.add_argument("--crf", type=int, default=20, help="H.264 画质（越大越小、越小越清晰）")
    ap.add_argument("--scan-stride", type=int, default=30, help="找底板时的扫描步长（帧）")
    ap.add_argument("--scan-max", type=int, default=0, help="最多扫多少帧（0=全片）")
    ap.add_argument("--frames-out", type=int, default=600, help="每个 k 的 mp4 帧数（10 s @60fps）")
    ap.add_argument("--resolution", type=int, default=960, help="抽人用的检测分辨率")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--checkpoint", default=str(HERE / "runs" / "v1-frozen-step2" / "checkpoint_best_ema.pth"),
                    help="抽人用的权重（默认=部署版三分类）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="跑 app 用的 cpp 配置")
    ap.add_argument("--app", default=str(DEFAULT_APP))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--report", default=str(base_layout.REPORTS_DIR / "perf_persons_synth.md"))
    ap.add_argument("--compose-only", action="store_true", help="只抠图 + 合成 mp4，不跑 app")
    ap.add_argument("--cutouts-only", action="store_true",
                    help="**只做第一步**：抠人像存盘到 work/cutouts_person（+ index.json + contact_sheet.jpg），"
                         "不合成、不测速")
    ap.add_argument("--cutout-mode", choices=("box", "grabcut"), default="box",
                    help="素材模式。**box（默认）= 直接裁检测框**：够用且不会失败——素材只贴回"
                         "原框原位，像素与原帧一致、无矩形接缝；grabcut = 像素级抠像（带 alpha），"
                         "才可挪位置/换背景，但对人像失败率高（实测 86 个里 53 个掉到 Otsu）")
    ap.add_argument("--cutout-frames", type=int, default=3, help="抠图用几帧（取人最多的那几帧）")
    ap.add_argument("--cutout-min-persons", type=int, default=0,
                    help="候选帧至少多少人（0 = 取 --counts 的最大值）")
    ap.add_argument("--clean-cutouts", action="store_true",
                    help="抠图前先清空 work/cutouts_person，保证目录只含本次产物")
    ap.add_argument("--no-reuse", dest="reuse", action="store_false",
                    help="不复用已有 mp4/run.log，全部重算（默认复用，便于只重出报告）")
    args = ap.parse_args(argv)

    counts = sorted({int(c) for c in args.counts.split(",") if c.strip()})
    source = Path(args.source)
    if not source.is_file():
        raise SystemExit("找不到源视频：%s" % source)
    cfg, app = Path(args.config), Path(args.app)
    if not args.compose_only and not (cfg.is_file() and app.is_file()):
        raise SystemExit("缺 cpp 配置或 app：%s / %s" % (cfg, app))
    out_root = Path(args.out_dir)

    if args.cutouts_only:
        need = args.cutout_min_persons or (max(counts) if counts else 8)
        model = rd.load_model(Path(args.checkpoint), args.resolution)
        if args.clean_cutouts and CUTOUT_DIR.is_dir():
            for old in CUTOUT_DIR.iterdir():
                if old.is_file():
                    old.unlink()
            print("① 已清空 %s" % CUTOUT_DIR)
        print("① 找底板（每帧至少 %d 人，取人最多的 %d 帧）…" % (need, args.cutout_frames))
        bgs = pick_backgrounds(source, model, args.cutout_frames, need, args.threshold,
                               args.scan_stride, args.scan_max)
        if not bgs:
            raise SystemExit("源视频里找不到至少 %d 人的帧：换个源，或把 --cutout-min-persons 调小" % need)
        print("  底板：%s" % "，".join("f%d(%d 人)" % (fi, len(bx)) for fi, _, bx in bgs))
        print("② 抠图…")
        cut_entries: list[dict[str, Any]] = []
        cut_rejected: list[dict[str, Any]] = []
        for fi, frame, boxes in bgs:
            _sprites, got, bad = cut_out(frame, boxes, "%s_f%06d" % (source.stem, fi), fi,
                                         source.name, args.cutout_mode)
            cut_entries.extend(got)
            cut_rejected.extend(bad)
            print("  f%-7d 抠出 %d 个（丢弃 %d）" % (fi, len(got), len(bad)), flush=True)
        index = write_cutout_index(cut_entries, source, len(bgs), cut_rejected)
        sheet = CUTOUT_DIR / "contact_sheet.jpg"
        write_cutout_sheet(cut_entries, sheet)
        by_reason: dict[str, int] = {}
        for item in cut_rejected:
            key = str(item["reason"]).split("(")[0].split("=")[0]
            by_reason[key] = by_reason.get(key, 0) + 1
        sizes = sorted(e["box_px"] for e in cut_entries)
        print("\n  抠图完成：**保留 %d 个**，丢弃 %d 个 → %s" % (len(cut_entries), len(cut_rejected), CUTOUT_DIR))
        print("    丢弃理由：%s" % ("，".join("%s=%d" % (k, v) for k, v in sorted(by_reason.items())) or "无"))
        print("    原框尺寸：中位 %d px²，最小 %d，最大 %d"
              % (sizes[len(sizes) // 2] if sizes else 0, sizes[0] if sizes else 0,
                 sizes[-1] if sizes else 0))
        print("    索引：%s" % index)
        print("    拼图：%s" % sheet)
        return 0

    def k_dir_of(k: int) -> Path:
        return out_root / ("%02d人" % k)

    reuse_all = bool(args.reuse) and all(
        (k_dir_of(k) / "video.mp4").is_file()
        and (args.compose_only or (k_dir_of(k) / "run.log").is_file())
        for k in counts)

    plates: list[tuple[int, Any, list[list[float]], list[Any]]] = []
    n_cut = 0
    if reuse_all:
        print("① 复用已有产物（--reuse）：跳过抠图 / 合成 / 推理，只重算报告")
    else:
        model = rd.load_model(Path(args.checkpoint), args.resolution)
        print("① 找底板（每帧至少 %d 人）…" % max(counts))
        bgs = pick_backgrounds(source, model, args.bg_frames, max(counts), args.threshold,
                               args.scan_stride, args.scan_max)
        if not bgs:
            raise SystemExit("源视频里找不到至少 %d 人的帧：换个源，或把 --counts 的上限调小" % max(counts))
        if len(bgs) < args.bg_frames:
            print("  !! 只找到 %d 张合格底板（想 %d 张），按现有的用" % (len(bgs), args.bg_frames))
        print("  底板：%s" % "，".join("f%d(%d 人)" % (fi, len(bx)) for fi, _, bx in bgs))
        print("② 抠图…")
        entries: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for fi, frame, boxes in bgs:
            sprites, got, bad = cut_out(frame, boxes, "%s_f%06d" % (source.stem, fi), fi,
                                        source.name, args.cutout_mode)
            entries.extend(got)
            rejected.extend(bad)
            plates.append((fi, frame, boxes, sprites))   # 原帧 + 原检测框：合成时只抹掉没被选中的人
            print("  f%-7d 抠出 %d 个抠像（丢弃 %d）" % (fi, len(sprites), len(bad)), flush=True)
        n_cut = sum(len(item[3]) for item in plates)
        write_cutout_index(entries, source, len(bgs), rejected)
        write_cutout_sheet(entries, CUTOUT_DIR / "contact_sheet.jpg")
        print("  合计 %d 个抠像（丢弃 %d）→ %s（含 index.json + contact_sheet.jpg）"
              % (n_cut, len(rejected), CUTOUT_DIR))

    print("③ 合成 + 编码 + 测速…")
    rows = []
    for k in counts:
        out_dir_k = k_dir_of(k)
        video = out_dir_k / "video.mp4"
        row: dict[str, Any] = {"k": k, "video": str(video)}
        if args.reuse and video.is_file() and (args.compose_only or (out_dir_k / "run.log").is_file()):
            pass                       # 复用：不重算，下面直接出汇总行
        else:
            usable = [item for item in plates if len(item[3]) >= k]
            if not usable:
                print("  跳过 %d 人：没有这么多人的底板" % k)
                continue
            frames = [compose_k(fr, bx, sp, k) for _, fr, bx, sp in usable]
            seq = [frames[i % len(frames)] for i in range(max(len(frames), args.frames_out))]
            write_mp4(video, seq, crf=args.crf)
            if not args.compose_only:
                invoke_app(app, cfg, video, out_dir_k, len(seq))
        if not video.is_file():
            print("  跳过 %d 人：没有 mp4" % k)
            continue
        row["mb"] = video.stat().st_size / 1048576
        if args.compose_only:
            print("  %2d 人：%.1f MB" % (k, row["mb"]), flush=True)
        else:
            row.update(parse_run(out_dir_k))
            row["mbps"] = video_mbps(video, row.get("frames") or args.frames_out)
            print("  %2d 人：%.1f MB（%.1f Mbps）   FPS=%.1f   观测人/帧=%.2f   最大模块=%s"
                  % (k, row["mb"], row["mbps"], row["fps"] or 0, row["persons_per_frame"],
                     row.get("max_module", "—")), flush=True)
        rows.append(row)

    lines = ["# 人数 → FPS 受控曲线（合成法：抠人 + 粘贴）", "",
             "生成时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"), "",
             "源视频 `%s`%s｜每档 %d 帧 mp4｜配置 `%s`（**app 直跑**，无渲染、无 `--profile_timers`）"
             % (source.name,
                ("（本次复用已有产物，未重跑抠图/合成）" if reuse_all else
                 "（抠图 %d 个，存 `%s`）｜底板 %d 张"
                 % (n_cut, CUTOUT_DIR.relative_to(base_layout.DATASET_ROOT), len(plates))),
                args.frames_out, cfg.name), "",
             "做法：模型框人 → GrabCut 抠 RGBA → 底板把**没被选中的人** inpaint 掉（选中的保留原像素）"
             "→ 再把选中的 k 个抠像按原框中心贴上 → 编码 mp4。", "",
             "**自检**：`观测人/帧` 应 ≈ 设定人数；对不上的档不算有效点（合成或检测出了问题）。", "",
             "| 设定人数 | 观测人/帧 | FPS | ms/帧 | 码率 | 最大 Worker.Process 项（**参考**，非瓶颈判据） |",
             "| ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append("| %d | %s | %s | %s | %s | %s |" % (
            r["k"],
            "%.2f" % r["persons_per_frame"] if not args.compose_only else "—",
            "%.1f" % r["fps"] if r.get("fps") else "—",
            "%.2f" % r["ms_per_frame"] if r.get("ms_per_frame") else "—",
            "%.1f Mbps" % r["mbps"] if r.get("mbps") else "—",
            "%s = %.2f" % (r["max_module"], r["max_module_ms"]) if r.get("max_module") else "—"))
    heavy = [r for r in rows if r.get("mbps") and r["mbps"] > 12.0]
    if heavy:
        lines += ["", "> ⚠️ **这些档码率偏高**（%s）：%s —— 真实素材 ≈6 Mbps，超过 2× 就要怀疑解码"
                      "参与了瓶颈，把 `--bg-frames` 降到 1、或把 `--crf` 调大重跑。"
                  % ("、".join("%.1f Mbps" % r["mbps"] for r in heavy),
                     "、".join("%d 人" % r["k"] for r in heavy))]
    lines += ["", "> ℹ️ `Worker.Process.*` 的最大项**不是**瓶颈判据：管线打满时它总是 ≈ 每帧墙钟"
                  "（含排队等待）。实测 1 人档 `SdkInput` = 8.99 ms / 每帧 9.94 ms，12 人档 "
                  "35.08 / 35.20 —— 都「等于」每帧耗时，所以指不出真瓶颈。",
              "", "> ℹ️ FPS 对人数**不是纯线性**：实测 4 人 81.2 → 6 人 46.5 有一个明显的台阶，"
                  "与姿态分批（`batchFrameCount: 4`、`maxBatch: 8`）有关，因此 R² 会低于纯线性场景。"]
    valid = [r for r in rows if r.get("fps") and abs(r["persons_per_frame"] - r["k"]) <= 0.6]
    if len(valid) >= 2:
        xs = [r["persons_per_frame"] for r in valid]
        ys = [r["ms_per_frame"] for r in valid]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        inter = my - slope * mx
        r2 = 1 - sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys)) / sum((y - my) ** 2 for y in ys)
        lines += ["", "**拟合（仅取自检通过的 %d 档）**：" % len(valid), "", "```",
                  "ms/帧 ≈ %.2f + %.3f × 人/帧        R² = %.4f" % (inter, slope, r2), "```"]
    lines += ["", "产物：`%s/<人数>人/{video.mp4,run.log,observations.jsonl}`" % out_root, ""]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n→ %s" % report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
