#!/usr/bin/env python
"""出**带锥**的对比视频（走 Nexusflow 的离线渲染器）。

为什么不能用 `rollout.py run --render`：C++ app 的渲染器
（`Nexusflow/examples/aifootball_app/VideoRenderer.cpp`）的 `DrawOverlay` 只画
person / ball，**全文没有 cone**。这是设计决定——SDK 头文件写明锥"只为调试/渲染
画框保留……是为了让 `tools/render_jsonl.py` 能把锥画成红框"。所以要看锥必须走离线渲染。

脚本做两件事，缺一不可：

1. 调 `Nexusflow/tools/render_jsonl.py`（源视频解码一次；`compare` 模式传 `--compare`）。
2. **按 JSONL 里的 stride 抽帧**。离线渲染器是"逐源帧输出、只在有观测的帧上画框"，
   我们抽帧跑的观测只在 0/10/20… 上，直接看会变成框每 10 帧闪一下。抽帧后每个保留帧
   都带标注，输出 `fps = 源fps / stride`，**实时播放不慢放**。

抽帧会重编码一次，所以默认用 **PyAV/libx264**（与 render_jsonl 同档画质），
PyAV 不可用时退回 cv2 `mp4v`（画质会掉一档）。

用法
----
    $PY render_cone_compare.py --mode single --video 移动传球      # 每版本一个独立视频
    $PY render_cone_compare.py --mode compare --video 移动传球     # 左右对照，一个视频
    $PY render_cone_compare.py --mode both  --video 移动传球       # 两者都出
    $PY render_cone_compare.py                              # 默认 compare，全部场景
    $PY render_cone_compare.py --list
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "rollout.yaml"


def enabled_versions(cfg: dict) -> list[str]:
    return [name for name, spec in (cfg.get("versions") or {}).items()
            if (spec or {}).get("enabled")]


def read_meta(jsonl: Path) -> dict:
    with jsonl.open(encoding="utf-8") as fh:
        return json.loads(fh.readline())


class FrameWriter:
    """优先 PyAV/libx264（画质与离线渲染器一致），退回 cv2 mp4v。"""

    def __init__(self, dst: Path, fps: float, width: int, height: int,
                 crf: int) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.kind = "pyav"
        self.cv = None
        try:
            import av
            self.av = av
            self.container = av.open(str(dst), mode="w", options={"movflags": "+faststart"})
            # PyAV 18 的 rate 必须是 Fraction：传 float 会报
            # "'float' object has no attribute 'numerator'"。
            self.stream = self.container.add_stream(
                "libx264", rate=Fraction(round(fps * 1000), 1000))
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": str(crf)}
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [warn] PyAV 不可用（{exc}），退回 cv2 mp4v（画质会降一档）",
                  file=sys.stderr)
            self.kind = "cv2"
            self.container = None
            self.stream = None
            self.cv = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (width, height))
            if not self.cv.isOpened():
                raise RuntimeError("cv2 VideoWriter 打不开")

    def write(self, frame) -> None:
        if self.kind == "pyav":
            video_frame = self.av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in self.stream.encode(video_frame):
                self.container.mux(packet)
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.kind == "pyav":
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
        elif self.cv is not None:
            self.cv.release()


def subsample(src: Path, dst: Path, step: int, crf: int) -> tuple[int, float, str]:
    """每 step 帧取一帧。观测正好落在这些帧上，所以保留帧都带标注。"""
    cap = cv2.VideoCapture(str(src))
    fps = (cap.get(cv2.CAP_PROP_FPS) or 60.0) / step
    writer = None
    index = written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = FrameWriter(dst, fps, width, height, crf)
                writer.write(frame)
                written += 1
            index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.close()
    return written, fps, (writer.kind if writer else "—")


def concat_videos(sources: list[Path], dst: Path, crf: int) -> tuple[int, float]:
    """顺序拼接多个视频（本机没有 ffmpeg CLI，用 PyAV 解码+重编码）。"""
    writer = None
    fps = 60.0
    total = 0
    size = None
    try:
        for src in sources:
            cap = cv2.VideoCapture(str(src))
            fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if size is None:
                    size = (frame.shape[1], frame.shape[0])
                elif (frame.shape[1], frame.shape[0]) != size:
                    raise RuntimeError(
                        f"{src.name} 分辨率 {frame.shape[1]}x{frame.shape[0]} "
                        f"与第一段 {size[0]}x{size[1]} 不一致，不能直接拼")
                if writer is None:
                    writer = FrameWriter(dst, fps, size[0], size[1], crf)
                writer.write(frame)
                total += 1
            cap.release()
    finally:
        if writer is not None:
            writer.close()
    return total, fps


def render(renderer: Path, observations: Path, compare: Path | None,
           label_a: str, label_b: str, output: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(renderer),
           "--observations", str(observations),
           "--label-a", label_a,
           "--show-counts", "--output", str(output)]
    if compare is not None:
        cmd += ["--compare", str(compare), "--label-b", label_b]
    return subprocess.run(cmd, capture_output=True, text=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--root", default="", help="跑批根目录（默认配 paths.output_root）")
    ap.add_argument("--left", default="")
    ap.add_argument("--right", default="")
    ap.add_argument("--out", default="", help="输出目录（默认 <root>/cone）")
    ap.add_argument("--mode", choices=["compare", "single", "both"],
                    default="compare",
                    help="compare=左右一个文件；single=每版本一个文件；both=都出")
    ap.add_argument("--crf", type=int, default=18,
                    help="抽帧重编码质量（越小越好，默认 18；离线渲染器本身是 23）")
    ap.add_argument("--concat", action="store_true",
                    help="single/both 模式下，额外把各版本按顺序拼成一个文件")
    ap.add_argument("--video", action="append", default=[], help="只跑这些场景 tag")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep-temp", action="store_true", help="保留抽帧前的全帧率中间文件")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    repo = Path(cfg["paths"]["repo_root"]).resolve()
    nexus = Path(cfg["paths"]["nexusflow_root"]).resolve()
    renderer = nexus / "tools" / "render_jsonl.py"
    if not renderer.is_file():
        raise SystemExit(f"找不到离线渲染器：{renderer}")

    def resolve(raw: str, fallback: str) -> Path:
        if not raw:
            return repo / fallback
        given = Path(raw)
        return given.resolve() if given.is_absolute() else (repo / given).resolve()

    root = resolve(a.root, str(cfg["paths"]["output_root"]))
    versions = enabled_versions(cfg)
    left = a.left or (versions[0] if versions else "")
    right = a.right or (versions[1] if len(versions) > 1 else "")
    if not (left and right):
        raise SystemExit("需要两个版本：--left / --right")
    out_dir = resolve(a.out, "") if a.out else root / "cone"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes: list[str] = []
    for entry in sorted((root / left).iterdir()) if (root / left).is_dir() else []:
        if not entry.is_dir():
            continue
        if not (root / right / entry.name / "observations.jsonl").is_file():
            continue
        if a.video and entry.name not in set(a.video):
            continue
        scenes.append(entry.name)
    if not scenes:
        raise SystemExit(f"{root} 下没有可比对的场景")
    print(f"根目录 {root}\n左 {left} | 右 {right}\nmode={a.mode} 场景 {len(scenes)} 段 → {out_dir}")
    if a.list:
        for tag in scenes:
            print(f"  {tag}")
        return 0

    done = failed = 0
    for index, tag in enumerate(scenes, 1):
        src_jsonl = root / left / tag / "observations.jsonl"
        right_jsonl = root / right / tag / "observations.jsonl"
        try:
            # stride 在 metadata 的嵌套字段里：{"type":"metadata","video":{...,"stride":10}}。
            meta = read_meta(src_jsonl)
            step = max(1, int((meta.get("video") or {}).get("stride") or 1))
        except Exception as exc:                                  # noqa: BLE001
            print(f"[{index}/{len(scenes)}] 跳过 {tag}：读 metadata 失败 {exc}",
                  file=sys.stderr)
            failed += 1
            continue

        jobs: list[tuple[Path, Path, Path | None]] = []
        if a.mode in ("compare", "both"):
            jobs.append((out_dir / f"{tag}.mp4", src_jsonl, right_jsonl))
        if a.mode in ("single", "both"):
            jobs.append((out_dir / f"{tag}.{left}.mp4", src_jsonl, None))
            jobs.append((out_dir / f"{tag}.{right}.mp4", right_jsonl, None))

        for dst, observations, compare in jobs:
            if dst.is_file() and not a.overwrite:
                print(f"[{index}/{len(scenes)}] 跳过（已存在）{dst.name}")
                done += 1
                continue
            label_a = left if compare is not None else (
                left if observations == src_jsonl else right)
            label_b = right
            with tempfile.TemporaryDirectory() as tmp:
                full = Path(tmp) / "full.mp4"
                proc = render(renderer, observations, compare, label_a, label_b, full)
                if proc.returncode != 0 or not full.is_file():
                    print(f"[{index}/{len(scenes)}] 渲染失败 {dst.name}："
                          f"{proc.stderr[-300:]}", file=sys.stderr)
                    failed += 1
                    continue
                try:
                    if step == 1:
                        # stride=1：抽帧是空操作，直接采用渲染器的输出，
                        # 省掉一次全量重编码（16037 帧约 7 分钟）。
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(full), str(dst))
                        probe = cv2.VideoCapture(str(dst))
                        frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = probe.get(cv2.CAP_PROP_FPS) or 60.0
                        probe.release()
                        codec = "libx264(直拷)"
                    else:
                        frames, fps, codec = subsample(full, dst, step, a.crf)
                except Exception as exc:                          # noqa: BLE001
                    print(f"[{index}/{len(scenes)}] 抽帧失败 {dst.name}：{exc}",
                          file=sys.stderr)
                    failed += 1
                    continue
                if a.keep_temp and full.is_file():
                    full.replace(out_dir / f"{dst.stem}.full.mp4")
            mb = dst.stat().st_size / 1e6
            print(f"[{index}/{len(scenes)}] {dst.name}  {frames} 帧 / {fps:.2f} fps "
                  f"(源 stride={step}, {codec}, crf={a.crf}) / {mb:.1f} MB", flush=True)
            done += 1

        if a.concat and a.mode in ("single", "both"):
            parts = [out_dir / f"{tag}.{left}.mp4", out_dir / f"{tag}.{right}.mp4"]
            joined = out_dir / f"{tag}.{left}+{right}.mp4"
            if not all(part.is_file() for part in parts):
                print(f"[{index}/{len(scenes)}] 跳过拼接 {joined.name}：单版本文件不全",
                      file=sys.stderr)
                failed += 1
            elif joined.is_file() and not a.overwrite:
                print(f"[{index}/{len(scenes)}] 跳过（已存在）{joined.name}")
                done += 1
            else:
                try:
                    frames, jfps = concat_videos(parts, joined, a.crf)
                except Exception as exc:                          # noqa: BLE001
                    print(f"[{index}/{len(scenes)}] 拼接失败 {joined.name}：{exc}",
                          file=sys.stderr)
                    failed += 1
                    continue
                print(f"[{index}/{len(scenes)}] {joined.name}  {frames} 帧 / "
                      f"{jfps:.2f} fps / {joined.stat().st_size/1e6:.1f} MB "
                      f"(= {parts[0].name} + {parts[1].name})", flush=True)
                done += 1

    print(f"\n完成 {done} 个文件，失败 {failed} 个 → {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
