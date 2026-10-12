#!/usr/bin/env python3
"""对比多个版本的逐视频观测（C++ 管线产物 observations.jsonl）。

用途：把 `runs/<版本>/<视频>/observations.jsonl` 横向比一遍，回答
"换成三分类模型后，每条视频上球/人的检测行为变了多少"。

口径与已知限制（写在报告里，避免误读）：
  * `observations.jsonl` 每帧只有 `persons` / `balls`（**没有 cone 框**）——
    锥的检测框在 C++ 里于跟踪阶段被丢弃，只有检测器的原始计数存在内存里，
    而序列化层目前只写 `{total, person, ball}`，所以**本表没有锥的逐帧指标**。
    锥维度请用固定评测集上的检测级 A/B（`tools/detection_finetune/s5_eval_detector.py`，
    那里有 GT，能给出"锥被判成球"的硬数字）。
  * 本表的 `球/帧`、`球/有球帧`、`人/帧` 是**行为代理**：把锥误检成球会让球框数偏高，
    所以同一段视频上"球框数下降"是锥误检减少的间接证据，不是直接测量。

用法：
  $PY compare_versions.py --version v1-frozen-step2 --version v1 \
      --output ../detection_finetune/runs/eval_ab/videos_compare.md
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

RUNS = pathlib.Path(__file__).resolve().parent / "runs"


def scan_one(args: tuple[str, str, pathlib.Path]) -> tuple[str, str, dict[str, Any] | None]:
    """扫一份 observations.jsonl，返回 (版本, 视频 tag, 统计)。"""
    version, tag, path = args
    if not path.is_file() or path.stat().st_size == 0:
        return version, tag, None
    frames = persons = balls = 0
    frames_with_ball = 0
    raw_person = raw_ball = raw_total = 0
    det_cone = det_ball = det_total = 0          # 检测器级（过滤前）计数，含锥
    frames_with_cone = 0
    cones_where_ball = 0                         # 有锥且同时出球框的帧数（锥误检成球的代理）
    balls_where_cone = 0
    bad = 0
    try:
        with path.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if '"type"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                if r.get("type") != "frame":
                    continue
                frames += 1
                np_ = len(r.get("persons") or [])
                nb = len(r.get("balls") or [])
                persons += np_
                balls += nb
                if nb:
                    frames_with_ball += 1
                rc = r.get("raw_detection_counts") or {}
                raw_person += int(rc.get("person") or 0)
                raw_ball += int(rc.get("ball") or 0)
                raw_total += int(rc.get("total") or 0)
                dc = r.get("detector_counts") or {}
                dc_cone = int(dc.get("cone") or 0)
                det_cone += dc_cone
                det_ball += int(dc.get("ball") or 0)
                det_total += int(dc.get("total") or 0)
                if dc_cone:
                    frames_with_cone += 1
                    if nb:
                        cones_where_ball += 1
                        balls_where_cone += nb
    except OSError as exc:                                    # 可能正被写入（批任务在跑）
        print("  !! 读取失败 %s: %s" % (path, exc), file=sys.stderr)
        return version, tag, None
    if not frames:
        return version, tag, None
    return version, tag, {
        "frames": frames,
        "persons": persons,
        "balls": balls,
        "persons_per_frame": persons / frames,
        "balls_per_frame": balls / frames,
        "balls_per_person": balls / persons if persons else 0.0,
        "ball_frames_ratio": frames_with_ball / frames,
        "raw_person_per_frame": raw_person / frames,
        "raw_ball_per_frame": raw_ball / frames,
        "raw_total_per_frame": raw_total / frames,
        "detector_cone_per_frame": det_cone / frames,
        "detector_ball_per_frame": det_ball / frames,
        "frames_with_cone_ratio": frames_with_cone / frames,
        "ball_in_cone_frame_ratio": cones_where_ball / frames_with_cone if frames_with_cone else 0.0,
        "balls_per_cone_frame": balls_where_cone / frames_with_cone if frames_with_cone else 0.0,
        "bad_lines": bad,
    }


def sort_key(tag: str) -> tuple[int, str]:
    return (0 if tag.startswith("20260726") else 1, tag)


def main() -> int:
    ap = argparse.ArgumentParser(description="对比多个版本的逐视频观测")
    ap.add_argument("--version", action="append", required=True, help="版本名（可重复，按这个顺序排表）")
    ap.add_argument("--output", required=True, help="输出 markdown 路径")
    ap.add_argument("--csv", default="", help="可选：同时输出 CSV")
    ap.add_argument("--jobs", type=int, default=2, help="并行进程数（批任务在跑时别开太大）")
    a = ap.parse_args()

    tasks: list[tuple[str, str, pathlib.Path]] = []
    for v in a.version:
        d = RUNS / v
        if not d.is_dir():
            print("!! 版本目录不存在：%s" % d, file=sys.stderr)
            return 2
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                tasks.append((v, sub.name, sub / "observations.jsonl"))

    stats: dict[tuple[str, str], dict[str, Any] | None] = {}
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for version, tag, st in ex.map(scan_one, tasks, chunksize=1):
            stats[(version, tag)] = st
            print("  扫完 %-18s %-46s %s" % (version, tag, "缺/空" if st is None else "%d 帧" % st["frames"]))

    tags = sorted({t for _, t in stats}, key=sort_key)
    L: list[str] = []
    L.append("# 逐视频观测对比（C++ 管线 %s）" % " vs ".join(a.version))
    L.append("")
    L.append("生成命令：`$PY tools/rollout/compare_versions.py %s`"
             % " ".join("--version %s" % v for v in a.version))
    L.append("")
    L.append("口径与限制（**务必先读**）：")
    L.append("")
    L.append("* 每帧只有 `persons`/`balls`，**没有锥框**（锥在跟踪阶段被丢、序列化层只写 ball/person/total）")
    L.append("  → 锥维度请看固定评测集上的 `s5_eval_detector.py`（有 GT 的硬数字）")
    L.append("* 本表的 `球/帧` 等是**行为代理**：锥被误检成球会让球框数偏高，所以同段视频球数下降是间接证据")
    L.append("* `raw_*` 是检测器**过滤前**的原始计数（含被置信度/尺寸过滤掉的框）")
    L.append("")

    def tbl(title: str, keys: list[tuple[str, str]], fmt: str) -> None:
        L.append("## %s" % title)
        L.append("")
        L.append("| 视频 | " + " | ".join(a.version) + " |")
        L.append("| --- |" + " --- |" * len(a.version))
        for tag in tags:
            cells = []
            for v in a.version:
                st = stats.get((v, tag))
                cells.append("—（未跑）" if st is None else fmt % st[keys[0][1]] if len(keys) == 1
                             else " / ".join(fmt % st[k] for _, k in keys))
            L.append("| `%s` | %s |" % (tag, " | ".join(cells)))
        L.append("")

    tbl("每帧球框数（锥误检的间接指标：越低越好）", [("", "balls_per_frame")], "%.2f")
    tbl("每帧人数（跟踪后）", [("", "persons_per_frame")], "%.2f")
    tbl("每帧球框数 ÷ 每帧人数（\"球\"相对人的密度，越低说明画面里的球越少）",
        [("", "balls_per_person")], "%.3f")
    if any(st and st.get("detector_cone_per_frame") for st in stats.values()):
        tbl("检测器级：锥框数 / 帧（本次新增字段 detector_counts.cone）", [("", "detector_cone_per_frame")], "%.2f")
        tbl("有锥的帧里同时出球框的比例（锥被判成球的代理，越低越好）",
            [("", "ball_in_cone_frame_ratio")], "%.3f")
        tbl("有锥帧里的球框数 / 帧（越低越好）", [("", "balls_per_cone_frame")], "%.2f")
    else:
        L.append("> ⚠️ 本批 observations **没有** `detector_counts`（旧二进制产物）→ 锥维度无法统计；")
        L.append("> 重新跑（当前二进制已在 JsonlWriter 里输出该字段）后本节会自动出现。")
        L.append("")

    tbl("原始检测（过滤前）球、人、总框数 / 帧", [("", "raw_ball_per_frame"), ("", "raw_person_per_frame"), ("", "raw_total_per_frame")], "%.1f")

    # 汇总
    L.append("## 汇总")
    L.append("")
    L.append("| 版本 | 已跑段数 | 总帧 | 球框合计 | 球/帧 | 人/帧 | 有球帧占比 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    csv_rows: list[list[Any]] = [["version", "tag", "frames", "persons", "balls",
                                 "balls_per_frame", "persons_per_frame", "ball_frames_ratio",
                                 "raw_person_per_frame", "raw_ball_per_frame", "raw_total_per_frame"]]
    for v in a.version:
        ss = [stats[(vv, t)] for (vv, t) in stats if vv == v and stats[(vv, t)]]
        if not ss:
            L.append("| `%s` | 0 | — | — | — | — | — |" % v)
            continue
        fr = sum(s["frames"] for s in ss)
        L.append("| `%s` | %d | %d | %d | %.2f | %.2f | %.1f%% |" % (
            v, len(ss), fr, sum(s["balls"] for s in ss),
            sum(s["balls"] for s in ss) / fr, sum(s["persons"] for s in ss) / fr,
            100.0 * sum(s["ball_frames_ratio"] * s["frames"] for s in ss) / fr))
    L.append("")
    for tag in tags:
        for v in a.version:
            st = stats.get((v, tag))
            if st:
                csv_rows.append([v, tag] + [st[k] for k in
                                ("frames", "persons", "balls", "balls_per_frame", "persons_per_frame",
                                 "ball_frames_ratio", "raw_person_per_frame", "raw_ball_per_frame",
                                 "raw_total_per_frame")])

    out = pathlib.Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n  已写：%s" % out)
    if a.csv:
        cp = pathlib.Path(a.csv)
        cp.parent.mkdir(parents=True, exist_ok=True)
        with cp.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(csv_rows)
        print("  已写：%s" % cp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
