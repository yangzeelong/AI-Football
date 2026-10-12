#!/usr/bin/env python
"""把两个版本在同一段视频上的渲染结果拼成左右对照视频（N 段输入 → N 段输出）。

每段视频 = 一个场景，输出一个 `<场景tag>.mp4`：
    左半 = --left 版本，右半 = --right 版本，中间竖线分隔，两侧标版本名。

同时汇总每个场景的检测计数到 `compare.md`（球框数是"是否有误报"最直接的信号）。

用法
----
    $PY stitch_compare.py                          # 默认读 rollout.yaml 里前两个 enabled 版本
    $PY stitch_compare.py --left v1 --right v1-frozen-step2
    $PY stitch_compare.py --root tools/rollout/runs_review
    $PY stitch_compare.py --overwrite              # 覆盖已拼好的
    $PY stitch_compare.py --list                   # 只列场景，不拼

前置：先跑 `rollout.py --out-root <root> run --all --stride N --max-frames M --render`。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "rollout.yaml"
SEPARATOR_PX = 4
LABEL_COLOR = (255, 255, 255)
LABEL_BG = (0, 0, 0)


def enabled_versions(cfg: dict) -> list[str]:
    return [name for name, spec in (cfg.get("versions") or {}).items()
            if (spec or {}).get("enabled")]


def list_scenes(root: Path, left: str, right: str) -> list[str]:
    """两边都产出过 rendered.mp4 的场景，按 tag 排序。"""
    def tags(version: str) -> set[str]:
        base = root / version
        if not base.is_dir():
            return set()
        return {p.name for p in base.iterdir()
                if (p / "rendered.mp4").is_file()}
    both = tags(left) & tags(right)
    return sorted(both)


def draw_label(frame, text: str) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (8, 8), (16 + tw, 20 + th + base), LABEL_BG, -1)
    cv2.putText(frame, text, (12, 16 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                LABEL_COLOR, 2, cv2.LINE_AA)


def stitch_one(left_path: Path, right_path: Path, out_path: Path,
               left_label: str, right_label: str) -> tuple[int, float]:
    cap_l = cv2.VideoCapture(str(left_path))
    cap_r = cv2.VideoCapture(str(right_path))
    if not (cap_l.isOpened() and cap_r.isOpened()):
        raise RuntimeError("打不开输入视频")

    fps = cap_r.get(cv2.CAP_PROP_FPS) or cap_l.get(cv2.CAP_PROP_FPS) or 6.0
    writer = None
    written = 0
    try:
        while True:
            ok_l, frame_l = cap_l.read()
            ok_r, frame_r = cap_r.read()
            if not (ok_l and ok_r):
                break
            # 两版渲染分辨率应当一致；万一不一致就按左边对齐，避免写不出去。
            if frame_l.shape != frame_r.shape:
                frame_r = cv2.resize(frame_r,
                                     (frame_l.shape[1], frame_l.shape[0]))
            if writer is None:
                height, width = frame_l.shape[:2]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (width * 2 + SEPARATOR_PX, height))
                if not writer.isOpened():
                    raise RuntimeError("VideoWriter 打不开，检查 mp4v 编码器")
            left_tagged = frame_l.copy()
            right_tagged = frame_r.copy()
            draw_label(left_tagged, left_label)
            draw_label(right_tagged, right_label)
            separator = np.full((frame_l.shape[0], SEPARATOR_PX, 3), 60,
                                dtype=np.uint8)
            canvas = cv2.hconcat([left_tagged, separator, right_tagged])
            writer.write(canvas)
            written += 1
    finally:
        cap_l.release()
        cap_r.release()
        if writer is not None:
            writer.release()
    if written == 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError("一个可读帧都没有")
    return written, fps


def scene_stats(path: Path) -> dict:
    """从 observations.jsonl 数"误报信号"。

    只用**原始检测**（`raw_detection_counts`，tracker 之前）——`balls` 那一路每帧都会被
    tracker 外推补出一个球，恒等于帧数，没有区分度。
    """
    stats = {"frames": 0, "ball_boxes": 0, "ball_frames": 0, "cone_boxes": 0,
             "cone_frames": 0, "observed": 0, "predicted": 0, "conf_sum": 0.0,
             "persons": 0}
    if not path.is_file():
        return stats
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") != "frame":
                continue
            stats["frames"] += 1
            raw = record.get("raw_detection_counts") or {}
            balls = int(raw.get("ball") or 0)
            cones = int(raw.get("cone") or 0)
            stats["ball_boxes"] += balls
            stats["cone_boxes"] += cones
            stats["ball_frames"] += 1 if balls > 0 else 0
            stats["cone_frames"] += 1 if cones > 0 else 0
            stats["persons"] += int(
                (record.get("detector_counts") or {}).get("person") or 0)
            for ball in record.get("balls") or []:
                if ball.get("state") == "observed":
                    stats["observed"] += 1
                    stats["conf_sum"] += float(ball.get("confidence") or 0.0)
                else:
                    stats["predicted"] += 1
    return stats


def collect_stats(root: Path, versions: list[str], scenes: list[str]) -> dict:
    return {tag: {version: scene_stats(root / version / tag / "observations.jsonl")
                  for version in versions}
            for tag in scenes}


def pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.0f}%" if whole else "—"


def mean_conf(stats: dict) -> str:
    observed = stats.get("observed") or 0
    return f"{stats['conf_sum'] / observed:.3f}" if observed else "—"


def write_report(out_dir: Path, scenes: list[str], versions: list[str],
                 stats: dict) -> Path:
    left, right = versions[0], versions[1]
    lines = [
        "# 版本对照（渲染视频 + 误报信号）", "",
        f"- 左：`{left}`　右：`{right}`",
        f"- 场景数：{len(scenes)}",
        f"- 对照视频：`{out_dir}/<场景tag>.mp4`", "",
        "指标全部取自 **`raw_detection_counts`（原始检测，tracker 之前）**，",
        "`balls` 那一路每帧都会被 tracker 外推补球（恒等于帧数），不能用来判误报。", "",
        "| 场景 | 帧 | 左球框 | 右球框 | **Δ球框** | 左球检出帧 | 右球检出帧 | "
        "左锥框 | 右锥框 | 左球置信 | 右球置信 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    flag_rows = []
    for tag in scenes:
        row = stats.get(tag, {})
        l, r = row.get(left, {}), row.get(right, {})
        lb, rb = l.get("ball_boxes", 0), r.get("ball_boxes", 0)
        lc, rc = l.get("cone_boxes", 0), r.get("cone_boxes", 0)
        if rb > lb or rc > lc:
            flag_rows.append((tag, rb - lb, rc - lc))
        lines.append(
            f"| {tag} | {l.get('frames') or 0} | {lb} | {rb} | **{rb - lb:+d}** | "
            f"{pct(l.get('ball_frames', 0), l.get('frames', 0))} | "
            f"{pct(r.get('ball_frames', 0), r.get('frames', 0))} | "
            f"{lc} | {rc} | {mean_conf(l)} | {mean_conf(r)} |")
    lines += ["", "## 优先复核（右版多于左版的场景）", ""]
    if flag_rows:
        flag_rows.sort(key=lambda x: -(x[1] + x[2]))
        for tag, db, dc in flag_rows:
            lines.append(f"- `{tag}`：球框 {db:+d}，锥框 {dc:+d}")
    else:
        lines.append("- 无（所有场景右版的原始球/锥框都不多于左版）")
    lines += ["",
              "> 判读：**Δ球框 > 0 且球检出帧占比也更高**，基本就是右版多出来的误报；",
              "> 只是 Δ 小、检出帧持平，多半是同一颗球的位置抖动，不用管。",
              "> `锥框` 是历史上误检最敏感的一项（step1/step2 的差异主要就在这）。", ""]
    path = out_dir / "compare.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--root", default="", help="跑批根目录（默认配 paths.output_root）")
    ap.add_argument("--left", default="", help="左半版本（默认第一个 enabled）")
    ap.add_argument("--right", default="", help="右半版本（默认第二个 enabled）")
    ap.add_argument("--out", default="", help="输出目录（默认 <root>/compare）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--list", action="store_true", help="只列场景，不拼")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    repo = Path(cfg["paths"]["repo_root"]).resolve()

    def resolve(raw: str, fallback: str) -> Path:
        # 相对路径按 repo_root 解析（与 rollout.py --out-root 一致），不跟 CWD 走。
        if not raw:
            return repo / fallback
        given = Path(raw)
        return given.resolve() if given.is_absolute() else (repo / given).resolve()

    root = resolve(a.root, str(cfg["paths"]["output_root"]))
    versions = enabled_versions(cfg)
    left = a.left or (versions[0] if versions else "")
    right = a.right or (versions[1] if len(versions) > 1 else "")
    if not (left and right):
        raise SystemExit("需要两个版本：--left / --right，或配置里至少两个 enabled")
    if not root.is_dir():
        raise SystemExit(f"根目录不存在：{root}（先跑 rollout.py run）")
    out_dir = resolve(a.out, "") if a.out else root / "compare"

    scenes = list_scenes(root, left, right)
    if not scenes:
        raise SystemExit(f"{root} 下没有同时存在 {left}/{right} 的场景（rendered.mp4）")
    print(f"根目录 {root}\n左 {left} | 右 {right}\n场景 {len(scenes)} 段 → {out_dir}")
    if a.list:
        for tag in scenes:
            print(f"  {tag}")
        return 0

    done: list[tuple[str, int, float]] = []
    failed: list[str] = []
    for index, tag in enumerate(scenes, 1):
        dst = out_dir / f"{tag}.mp4"
        if dst.is_file() and not a.overwrite:
            print(f"[{index}/{len(scenes)}] 跳过（已存在）{dst.name}")
            continue
        try:
            frames, fps = stitch_one(root / left / tag / "rendered.mp4",
                                     root / right / tag / "rendered.mp4", dst,
                                     left, right)
        except Exception as exc:                                  # noqa: BLE001
            print(f"[{index}/{len(scenes)}] 失败 {tag}：{exc}", file=sys.stderr)
            failed.append(tag)
            continue
        mb = dst.stat().st_size / 1e6
        print(f"[{index}/{len(scenes)}] {dst.name}  {frames} 帧 / {fps:.2f} fps / {mb:.1f} MB",
              flush=True)
        done.append((tag, frames, fps))

    report = write_report(out_dir, scenes, [left, right],
                          collect_stats(root, [left, right], scenes))
    print(f"\n拼成 {len(done)} 段，失败 {len(failed)} 段\n计数对照表：{report}")
    if failed:
        print("失败场景：" + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
