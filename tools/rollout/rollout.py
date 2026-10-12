#!/usr/bin/env python3
"""rollout —— 从 checkpoint 到 C++ 批量跑视频的一条流水线（编排，不重写推理）。

    $PY rollout.py status                     # 版本/engine/权重软链/任务进度 一览
    $PY rollout.py videos                     # 列出扫描到的视频（tag/帧数/时长）
    $PY rollout.py export --version <版本>     # checkpoint → ONNX → fp16 → TRT engine → 模型卡
    $PY rollout.py cpp-config --all           # 生成每个版本专用的 C++ config（三分类 + engine 路径）
    $PY rollout.py run --all                  # 逐帧批量跑（可续跑：manifest 里 rc==0 的自动跳过）
    $PY rollout.py summary                    # 汇总表（帧数/耗时/FPS/人球数）

设计取舍（为什么这么写）
----------------------
* **只编排**：导出复用 `tools/detection_finetune/s6_deploy_export.py` + `s6_deploy_pin_model.py`；
  推理用 Nexusflow 的 `aifootball_app`（TRT engine）。这里不实现任何推理，避免口径分叉。
* **每次运行一个独立目录**：`runs/<版本>/<视频tag>/{app.yaml,run.log,observations.jsonl}`，
  生效配置也留一份 → 几个月后还能复现"当时到底跑了什么"。
* **续跑靠 manifest**：完成 = manifest 里该任务 `returncode == 0`；
  只查文件会把中断的半截产物当成完成（实测踩过）。
* **配置是生成物**：C++ config 里 engine 路径 / numClasses / targetClasses / classId /
  tracker ballClassId 全部由 `cpp-config` 按版本注入并断言，防止"配置写 step2、加载 v1"。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "rollout.yaml"
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}


# --------------------------------------------------------------------------- 基础
def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit("配置文件不存在：%s" % path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("配置顶层必须是 mapping：%s" % path)
    for key in ("paths", "versions"):
        if key not in data:
            raise SystemExit("配置缺少 `%s` 段：%s" % (key, path))
    return data


def _derived(raw: Any, default: Path) -> Path:
    """配置里留空就用推导出的默认值（空字符串/None 都算未设置）。"""
    text = str(raw or "").strip()
    return Path(text).expanduser().resolve() if text else default.resolve()


def ctx(cfg: dict[str, Any]) -> dict[str, Any]:
    """把配置里的路径解析成绝对路径，返回一个小的上下文对象。

    根路径（repo_root / nexusflow_root / cpp_config_dir / video_roots）**可以留空**，
    此时按「本文件位置 + 兄弟目录」推导，所以换机器/换目录不必改这个 yaml：

        repo_root       = <本文件>/../..              （…/AI-Football）
        nexusflow_root  = <WORKSPACE>/Nexusflow
        cpp_config_dir  = <repo>/tools/rollout/cpp_app
        video_roots     = [<WORKSPACE>/data]

    要拆开布局就用环境变量覆盖（AI_FOOTBALL_WORKSPACE / AI_FOOTBALL_NEXUSFLOW /
    AI_FOOTBALL_DATA_DIR），与 detection_finetune/base_layout.py 是同一套约定。
    """
    p = cfg["paths"]
    here = Path(__file__).resolve().parent
    repo_default = here.parents[1]
    workspace = Path(os.environ.get("AI_FOOTBALL_WORKSPACE")
                     or repo_default.parent).resolve()

    repo = _derived(p.get("repo_root"), repo_default)
    nx = _derived(p.get("nexusflow_root"),
                  Path(os.environ.get("AI_FOOTBALL_NEXUSFLOW")
                       or workspace / "Nexusflow"))
    cpp_dir = _derived(p.get("cpp_config_dir"),
                       repo / "tools" / "rollout" / "cpp_app")

    raw_roots: list[Any] = list(p.get("video_roots") or [])
    if not raw_roots:
        raw_roots = [os.environ.get("AI_FOOTBALL_DATA_DIR") or workspace / "data"]
    video_roots: list[Path] = []
    for raw in raw_roots:
        given = Path(str(raw)).expanduser()
        video_roots.append((given if given.is_absolute() else repo / given).resolve())

    return {
        "repo": repo, "nx": nx, "workspace": workspace,
        "app": nx / str(p["app"]),
        "cpp_ref": nx / str(p["cpp_reference_config"]),
        "cpp_dir": cpp_dir,
        "models_root": repo / str(p["models_root"]),
        "out_root": repo / str(p["output_root"]),
        "pose_engine": repo / str(p["pose_engine"]),
        "video_roots": video_roots,
        "classes": cfg["classes"],
        "run": cfg.get("run") or {},
        "pose_profiles": cfg.get("pose_profiles") or {},
    }


def version_paths(c: dict[str, Any], version: str, spec: dict[str, Any]) -> dict[str, Any]:
    """版本相关路径。C++ 配置按"版本 × 姿态档位"存成矩阵文件。

    `cpp_config` = 当前 `run.pose_profile` 那一份（run/bench 用它）；
    `cpp_config_for(档位)` 取任意档位（做对照/回归用）。
    """
    model_dir = c["models_root"] / version
    cfg_dir = c["cpp_dir"] / version
    profile = str(c["run"].get("pose_profile") or "accurate")
    return {
        "version": version,
        "note": spec.get("note", ""),
        "checkpoint": (c["repo"] / str(spec["checkpoint"])).resolve(),
        "model_dir": model_dir,
        "engine": model_dir / ("rf-detr-small-%s-960-fp16-b8-fp32io.engine" % version),
        "onnx": model_dir / ("rf-detr-small-%s-960-dynamic.onnx" % version),
        "onnx_fp16": model_dir / ("rf-detr-small-%s-960-dynamic-fp16-fp32io.onnx" % version),
        "link": model_dir / "rf-detr-small.pth",
        "card": model_dir / "meta.json",
        "cpp_dir": cfg_dir,
        "cpp_config": cfg_dir / ("%s.yaml" % profile),
        "cpp_config_for": lambda p: cfg_dir / ("%s.yaml" % p),
    }


def scan_videos(cfg: dict[str, Any], c: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[tuple[Path, Path]] = []
    for base in c["video_roots"]:            # ctx() 已解析成绝对路径（含默认推导）
        if base.is_dir():
            found += [(p, base) for p in sorted(base.rglob("*")) if p.suffix.lower() in VIDEO_EXTS]
    seen: set[Path] = set()
    out: list[dict[str, Any]] = []
    for path, base in found:
        real = path.resolve()
        if real in seen:
            continue
        seen.add(real)
        tag = str(path.relative_to(base).with_suffix("")).replace("/", "_").replace(" ", "_")
        out.append({"tag": tag, "path": path})
    return out


def manifest_path(c: dict[str, Any]) -> Path:
    return c["out_root"] / "manifest.json"


def load_manifest(c: dict[str, Any]) -> dict[str, Any]:
    path = manifest_path(c)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"jobs": []}


def save_manifest(c: dict[str, Any], data: dict[str, Any]) -> None:
    path = manifest_path(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def job_done(c: dict[str, Any], manifest: dict[str, Any], version: str, tag: str) -> bool:
    """任务是否已完成：manifest 里 rc==0，**或** 文件系统证据（observations.jsonl + 日志收尾行）。

    文件系统那一条是为了从"manifest 被并发写坏/覆盖"里恢复：真跑完过的任务不该重跑。
    """
    if any(j.get("version") == version and j.get("tag") == tag and j.get("returncode") == 0
           for j in manifest.get("jobs", [])):
        return True
    run_dir = c["out_root"] / version / tag
    log = run_dir / "run.log"
    if not log.is_file() or not (run_dir / "observations.jsonl").is_file():
        return False
    tail = log.read_text(encoding="utf-8", errors="ignore")[-4000:]
    return "processed " in tail and "observations=" in tail


def acquire_run_lock() -> Path:
    """真运行锁：/tmp/rollout.pid 里的 pid 还活着就拒绝启动（避免并发跑批互相覆盖 manifest）。"""
    lock = Path("/tmp/rollout.pid")
    if lock.is_file():
        raw = lock.read_text(encoding="utf-8").strip()
        try:
            old = int(raw)
            os.kill(old, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            print("（发现过期的 pid 文件 %s：%s，覆盖之）" % (lock, raw))
        else:
            raise SystemExit("已有 rollout 在跑（pid=%d）：先 kill -- -%d，或删掉 %s 再启动"
                             % (old, old, lock))
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def count_observations(path: Path) -> dict[str, int]:
    """从 observations.jsonl 数帧数/人数/球数（流式，别把大文件读进内存）。"""
    stats = {"frames": 0, "persons": 0, "balls": 0, "bytes": 0}
    if not path.is_file():
        return stats
    stats["bytes"] = path.stat().st_size
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or '"type"' not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "frame":
                continue
            stats["frames"] += 1
            stats["persons"] += len(record.get("persons") or [])
            stats["balls"] += len(record.get("balls") or [])
    return stats


def pick_versions(cfg: dict[str, Any], a: argparse.Namespace) -> list[str]:
    """要操作的版本：显式 `--version` 优先（即使 `enabled: false` 也能单独操作），否则取 enabled 的。"""
    if a.version:
        if a.version not in cfg["versions"]:
            raise SystemExit("未知版本：%s（配置里有 %s）" % (a.version, list(cfg["versions"])))
        return [a.version]
    out = [v for v, spec in cfg["versions"].items() if spec.get("enabled", True)]
    if not out:
        raise SystemExit("配置里没有 enabled: true 的版本（用 --version 指定要跑哪个）")
    return out


def module_timings(c: dict[str, Any], version: str) -> dict[str, float]:
    """该版本最近一次跑批的"各模块 ms/帧"（C++ 每帧都会写 module_timings_ms）。"""
    manifest = load_manifest(c)
    done = [j for j in manifest.get("jobs", [])
            if j.get("version") == version and j.get("returncode") == 0]
    if not done:
        return {}
    path = Path(done[-1]["run_dir"]) / "observations.jsonl"
    if not path.is_file():
        return {}
    acc: dict[str, float] = {}
    frames = 0
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if '"module_timings_ms"' not in line:
                continue
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if record.get("type") != "frame":
                continue
            frames += 1
            for key, value in (record.get("module_timings_ms") or {}).items():
                acc[key] = acc.get(key, 0.0) + float(value)
    if not frames:
        return {}
    return {k: v / frames for k, v in acc.items()}


# --------------------------------------------------------------------------- status / videos
def cmd_status(cfg: dict[str, Any], c: dict[str, Any], _: argparse.Namespace) -> int:
    manifest = load_manifest(c)
    videos = scan_videos(cfg, c)
    print("视频：%d 段（roots=%s）" % (len(videos), c["video_roots"]))
    print("app：%s %s" % (c["app"], "✓" if c["app"].is_file() else "✗ 未编译"))
    print("姿态engine：%s %s" % (c["pose_engine"].name, "✓" if c["pose_engine"].is_file() else "✗ 缺"))
    print("")
    print("  %-20s %-8s %-8s %-8s %-8s %s" % ("版本", "权重", "ONNX", "engine", "权重软链", "已完成任务"))
    for version, spec in cfg["versions"].items():
        vp = version_paths(c, version, spec)
        done = sum(1 for j in manifest.get("jobs", [])
                   if j.get("version") == version and j.get("returncode") == 0)
        print("  %-20s %-8s %-8s %-8s %-8s %d/%d" % (
            version, "✓" if vp["checkpoint"].is_file() else "✗",
            "✓" if list(vp["model_dir"].glob("rf-detr-small-*-960-dynamic.onnx")) else "✗",
            "✓" if vp["engine"].is_file() else "✗",
            "✓" if vp["link"].exists() else "✗", done, len(videos)))
    return 0


def cmd_videos(cfg: dict[str, Any], c: dict[str, Any], _: argparse.Namespace) -> int:
    videos = scan_videos(cfg, c)
    print("共 %d 段：" % len(videos))
    for v in videos:
        print("  %-40s %s" % (v["tag"], v["path"]))
    return 0


# --------------------------------------------------------------------------- export
def cmd_export(cfg: dict[str, Any], c: dict[str, Any], a: argparse.Namespace) -> int:
    targets = pick_versions(cfg, a)
    c["out_root"].mkdir(parents=True, exist_ok=True)
    lock = c["out_root"] / ".exporting"
    ft = c["repo"] / "tools" / "detection_finetune"
    lock.write_text(",".join(targets), encoding="utf-8")
    try:
        return _export_targets(cfg, c, a, targets, ft)
    finally:
        lock.unlink(missing_ok=True)


def _export_targets(cfg: dict[str, Any], c: dict[str, Any], a: argparse.Namespace,
                    targets: list[str], ft: Path) -> int:
    # 解释器在这里自己取：调用方 cmd_export 的作用域变量传不进来（曾因此 NameError）。
    py = os.environ.get("PY") or sys.executable
    for version in targets:
        vp = version_paths(c, version, cfg["versions"][version])
        if not vp["checkpoint"].is_file():
            print("!! %s：权重不存在，跳过（%s）" % (version, vp["checkpoint"]), file=sys.stderr)
            continue
        print("\n=== %s：导出 engine（ONNX → fp16 → TRT，含与 PyTorch 对拍）===" % version)
        cmd = [py, "-u", "s6_deploy_export.py", "--checkpoint", str(vp["checkpoint"])]
        if a.skip_engine:
            cmd.append("--skip-engine")
        if getattr(a, "force", False):
            cmd.append("--force")
        code = subprocess.run(cmd, cwd=str(ft), check=False).returncode
        if code != 0:
            print("!! 导出失败（rc=%d），不继续 pin" % code, file=sys.stderr)
            continue
        subprocess.run([py, "s6_deploy_pin_model.py", "--version", version,
                        "--description", cfg["versions"][version].get("note", "")],
                       cwd=str(ft), check=False)
    return 0


# --------------------------------------------------------------------------- cpp-config
def apply_pose_profile(text: str, c: dict[str, Any], profile: str) -> tuple[str, str]:
    """把姿态档位写进 C++ 配置（只动 HRNetPoseEstimator 块的 enginePath / flipTest / useDark）。

    返回 (新文本, 摘要)。找不到姿态块或 engine 文件缺失就报错——宁可失败也别静默用错姿态模型。
    """
    profiles = c.get("pose_profiles") or {}
    if profile not in profiles:
        raise SystemExit("未知姿态档位：%s（可选 %s）" % (profile, sorted(profiles)))
    spec = profiles[profile]
    engine = (c["repo"] / str(spec["engine"])).resolve()
    if not engine.is_file():
        raise SystemExit("档位 %s 的姿态 engine 不存在：%s" % (profile, engine))
    lines = text.splitlines(keepends=True)
    pose_idx = next((i for i, ln in enumerate(lines) if "class: HRNetPoseEstimator" in ln), None)
    if pose_idx is None:
        raise SystemExit("配置里找不到 HRNetPoseEstimator 模块")
    fields = {
        "enginePath": str(engine),
        "inputWidth": str(int(spec["input_width"])),
        "inputHeight": str(int(spec["input_height"])),
        "heatmapWidth": str(int(spec["heatmap_width"])),
        "heatmapHeight": str(int(spec["heatmap_height"])),
        "flipTest": "true" if spec.get("flip_test", True) else "false",
        "useDark": "true" if spec.get("use_dark", True) else "false",
    }
    # 一致性校验：engine 名字里带 "256x192" 就不能配 288x384 的输入（反之亦然）
    name_dims = re.findall(r"(\d{3})x(\d{3})", engine.name)
    if name_dims:
        expect_w, expect_h = int(name_dims[0][0]), int(name_dims[0][1])
        # engine 名字里的尺寸是"高x宽"（如 256x192 = H256×W192），配置里是 W/H，两种顺序都接受
        got = (int(spec["input_width"]), int(spec["input_height"]))
        if got not in ((expect_w, expect_h), (expect_h, expect_w)):
            raise SystemExit("档位 %s 的输入尺寸 %sx%s 与 engine 名 %s 不符"
                             % (profile, spec["input_width"], spec["input_height"], engine.name))
    seen: set[str] = set()
    for i in range(pose_idx, min(pose_idx + 40, len(lines))):
        ln = lines[i]
        key = ln.strip().split(":", 1)[0]
        if key in fields and not ln.lstrip().startswith("#"):
            indent = ln[: len(ln) - len(ln.lstrip())]
            comment = ""
            marker = ln.find("#")                            # 保留行尾注释，便于人工 review
            if marker > 0:
                comment = "  " + ln[marker:].rstrip("\n")
            lines[i] = "%s%s: %s%s\n" % (indent, key, fields[key], comment)
            seen.add(key)
    missing = sorted(set(fields) - seen)
    if missing:
        raise SystemExit("姿态块里没找到这些字段，拒绝生成（免得半套配置）：%s" % missing)
    summary = "%s（%s，输入 %sx%s，flipTest=%s，useDark=%s）" % (
        profile, engine.name, spec["input_width"], spec["input_height"],
        spec.get("flip_test"), spec.get("use_dark"))
    return "".join(lines), summary


def cmd_cpp_config(cfg: dict[str, Any], c: dict[str, Any], a: argparse.Namespace) -> int:
    targets = pick_versions(cfg, a)
    ref = c["cpp_ref"]
    if not ref.is_file():
        raise SystemExit("C++ 底板配置不存在：%s" % ref)
    base = ref.read_text(encoding="utf-8")
    c["cpp_dir"].mkdir(parents=True, exist_ok=True)
    cls = c["classes"]
    built: list[str] = []
    profiles = [a.pose_profile] if a.pose_profile else sorted(c["pose_profiles"])
    for version in targets:
        vp = version_paths(c, version, cfg["versions"][version])
        if not vp["engine"].is_file():
            print("!! %s：engine 不存在，先跑 `rollout.py export --version %s`（%s）"
                  % (version, version, vp["engine"]), file=sys.stderr)
            continue
        vp["cpp_dir"].mkdir(parents=True, exist_ok=True)
        text = base
        # ① 检测 engine
        text = re.sub(r"(?m)^(\s*)enginePath:.*$",
                      lambda m: "%senginePath: %s" % (m.group(1), vp["engine"]),
                      text, count=1)
        # ② 类别契约
        text = re.sub(r"(?m)^(\s*)numClasses:.*$",
                      lambda m: "%snumClasses: %d" % (m.group(1), cls["num_classes"]),
                      text, count=1)
        text = re.sub(r"(?m)^(\s*)targetClasses:.*$",
                      lambda m: ("%stargetClasses: [%s]   # classId = logits 列下标（不加 1）；列 0 = no-object\n"
                                 "%spersonClassId: %d\n"
                                 "%sballClassId: %d\n"
                                 "%sconeClassId: %d")
                                % (m.group(1), ", ".join(str(i) for i in cls["target_classes"]),
                                   m.group(1), cls["person_class_id"],
                                   m.group(1), cls["ball_class_id"],
                                   m.group(1), cls["cone_class_id"]),
                      text, count=1)
        # ②b 断言：检测器自己那段必须有类别号。`RFDetrDetector.cpp:35-37` 是用**它自己**的
        #     config 读 personClassId/ballClassId/coneClassId 来做统计计数的；缺省 1/37/-1 时
        #     三分类模型会把 ball/cone 计数记成 0（检测结果不受影响，但日志与统计会误导排查）。
        #     `coneClassId` 只有检测器段有，所以用它做判据。
        if ("coneClassId: %d" % cls["cone_class_id"]) not in text:
            raise SystemExit("生成结果里检测器段缺 coneClassId（统计计数会错），请检查 targetClasses 注入")
        # ③ tracker 的 ballClassId（COCO 是 37，三分类是 2）
        text, n_ball = re.subn(r"(?m)^(\s*)ballClassId:\s*37\s*$",
                               lambda m: "%sballClassId: %d" % (m.group(1), cls["ball_class_id"]),
                               text)
        # ③b ByteTracker 段补 `coneClassId`：锥检测本身不参与跟踪，这个键只控制
        #     "要不要把锥框带进 observations（供调试渲染画红框）"。默认 -1 = 不带，
        #     与 COCO 版行为一致。注意检测器段也有个同名键，那是**统计计数**用的，两码事。
        _m_bt = re.search(r"(?m)^    - name: ByteTracker\s*$", text)
        if not _m_bt:
            raise SystemExit("生成结果里找不到 ByteTracker 段（无法注入 coneClassId）")
        _tail = text[_m_bt.end():]
        _m_ball = re.search(r"(?m)^(\s*)ballClassId:\s*\d+\s*$", _tail)
        if not _m_ball:
            raise SystemExit("ByteTracker 段里找不到 ballClassId（无法注入 coneClassId）")
        _at = _m_bt.end() + _m_ball.end()
        text = (text[:_at] + "\n" + _m_ball.group(1)
                + "coneClassId: %d" % cls["cone_class_id"] + text[_at:])
        if ("coneClassId: %d" % cls["cone_class_id"]) not in text:
            raise SystemExit("coneClassId 注入失败")
        # ④ ROI：跨视角批量默认关（单多边形只对某机位有效）。
        #    ⚠️ 只准动 `algorithm:` 下那个 `roi:` 块里的 enabled ——
        #    早先用"第一个 enabled: true"改成 false，结果误关了 **KeypointSmoother**
        #    （关键点平滑，正是逐帧跑要保住的东西），而真正的 ROI 还开着。
        lines = text.splitlines(keepends=True)
        roi_idx = next((i for i, ln in enumerate(lines)
                        if re.match(r"^  roi:\s*$", ln)), None)
        n_roi = 0
        if roi_idx is None:
            raise SystemExit("底板配置里找不到 `algorithm.roi:`（结构与预期不符，先人工确认）")
        if not (a.keep_roi or c["run"].get("keep_roi")):
            for i in range(roi_idx + 1, min(roi_idx + 8, len(lines))):
                if re.match(r"^\s+enabled:\s*true\s*$", lines[i]):
                    lines[i] = re.sub(r"enabled:\s*true", "enabled: false", lines[i])
                    n_roi = 1
                    break
            text = "".join(lines)
        # 断言：KeypointSmoother 必须与底板一致（默认 true），不许被顺手关掉
        smoother = re.search(r"(?ms)KeypointSmoother\s*\n\s*class:.*?config:\s*\n\s*enabled:\s*(\w+)",
                             text)
        smoother_state = smoother.group(1) if smoother else "?"
        if smoother_state != "true":
            raise SystemExit("生成结果里 KeypointSmoother.enabled=%s（应为 true）：关键点平滑不能关"
                             % smoother_state)
        # 底板配置自带注释头（口径与 FPS 是"底板那份文件"的实测值），套到别的版本会误导读者 →
        # 生成前把底板开头的整块注释剥掉，只保留正文与行内注释，下面再写本版本自己的头。
        keep = text.splitlines(keepends=True)
        while keep and (keep[0].lstrip().startswith("#") or not keep[0].strip()):
            keep.pop(0)
        text = "".join(keep)
        # 实测吞吐（口径：`射门1-1080p60`、无渲染、16 核空闲、机器空闲）——
        # 人的数量是主因（每多 1 人 ≈ +1.6 ms/帧），所以 ROI 开关会让同一档位差 2–3 倍：
        #   本目录默认 ROI 关 → 人/帧 ≈10.7；同机位 `--keep-roi` → 人/帧 ≈3.6。
        PROFILE_FPS = {
            "accurate": "14.1 FPS / 71.0 ms",   # ROI 关、人/帧 10.7（1200 帧实测）
            "fast":     "24.6 FPS / 40.6 ms",   # 同上
            "fastest":  "46.5 FPS / 21.5 ms",   # 同上；开 ROI 实测 108.0 FPS / 9.26 ms
        }
        for profile in profiles:
            per_profile, profile_desc = apply_pose_profile(text, c, profile)
            out_path = vp["cpp_config_for"](profile)
            # header 依赖姿态档（每档有自己的实测 FPS），所以必须在循环里逐档格式化
            header = ("# 由 tools/rollout/rollout.py cpp-config 生成，勿手改（改 rollout.yaml 后重跑）。\n"
                      "#   version   = %s（%s）\n"
                      "#   engine    = %s\n"
                      "#   类别      = numClasses:%d targetClasses:%s person/ball/cone=%d/%d/%d\n"
                      "#   ROI       = %s\n"
                      "#   实测 FPS  = %s（姿态档 %s；口径 = 射门1-1080p60、无渲染、16 核空闲）\n"
                      "#               开渲染约 -35%%；人/帧 是主因（每多 1 人 ≈ +1.6 ms/帧），\n"
                      "#               同机位开 ROI（人/帧 3.6）比关 ROI（10.7）快 2–3 倍\n"
                      "# 生成时间：%s\n" % (
                          version, cfg["versions"][version].get("note", ""), vp["engine"],
                          cls["num_classes"], cls["target_classes"], cls["person_class_id"],
                          cls["ball_class_id"], cls["cone_class_id"],
                          "关（--keep-roi 可开）" if n_roi else "保持底板设置",
                          PROFILE_FPS.get(profile, "（未登记该档实测值）"), profile,
                          time.strftime("%F %T")))
            out_path.write_text(header + per_profile, encoding="utf-8")
            built.append(str(out_path))
            print("  已生成 %s（ballClassId 改动 %d 处，ROI %s，姿态档 %s）"
                  % (out_path, n_ball, "关闭" if n_roi else "保持", profile_desc))
    if not built:
        raise SystemExit("没有生成任何配置")
    return 0


# --------------------------------------------------------------------------- run
def cmd_run(cfg: dict[str, Any], c: dict[str, Any], a: argparse.Namespace) -> int:
    if a.pose_profile:
        c["run"]["pose_profile"] = a.pose_profile      # 临时选档，不改 yaml
    run_cfg = dict(c["run"])
    if a.stride is not None:
        run_cfg["stride"] = a.stride
    if a.max_frames is not None:
        run_cfg["max_frames"] = a.max_frames
    if a.render:
        run_cfg["render"] = True
    versions = pick_versions(cfg, a)
    videos = scan_videos(cfg, c)
    if a.video:
        want = set(a.video)
        videos = [v for v in videos if v["tag"] in want]
        if not videos:
            raise SystemExit("--video 没匹配到：%s" % sorted(want))
    if not c["app"].is_file():
        raise SystemExit("app 未编译：%s（cd %s && cmake --build build --target aifootball_app）"
                         % (c["app"], c["nx"]))

    jobs: list[tuple[str, dict[str, Any]]] = []
    for version in versions:
        vp = version_paths(c, version, cfg["versions"][version])
        if not vp["cpp_config"].is_file():
            print("   跳过 %s：还没有 C++ 配置（先 `rollout.py export` + `cpp-config`）" % version,
                  file=sys.stderr)
            continue
        for video in videos:
            jobs.append((version, {"version": version, "tag": video["tag"],
                                   "video": video["path"], "vp": vp}))
    if not jobs:
        raise SystemExit("没有可跑的版本（先 export + cpp-config）")
    manifest = load_manifest(c)
    todo = [j for j in jobs if a.force or not job_done(c, manifest, j[0], j[1]["tag"])]
    lock = c["out_root"] / ".exporting"
    if lock.is_file():
        raise SystemExit("有导出在进行（%s 里写着 %s）——等它结束再跑批，否则会加载到半截 engine"
                         % (lock, lock.read_text(encoding="utf-8")))
    print("任务：%d 个（已完成 %d，本次要跑 %d）" % (len(jobs), len(jobs) - len(todo), len(todo)))
    if a.dry_run:
        for version, job in todo:
            print("  $ cd %s && %s %s --video_path %s --output_dir %s --stride %s%s"
                  % (c["nx"], c["app"].name, job["vp"]["cpp_config"], job["video"],
                     c["out_root"] / version / job["tag"], run_cfg.get("stride", 1),
                     " --render" if run_cfg.get("render") else ""))
        return 0
    if not todo:
        print("没有要跑的任务（都完成了；--force 可重跑）")
        return 0

    pid_file = acquire_run_lock()
    print("停止：kill -- -$(cat %s)\n" % pid_file)
    for index, (version, job) in enumerate(todo, 1):
        run_dir = c["out_root"] / version / job["tag"]
        run_dir.mkdir(parents=True, exist_ok=True)
        app_cfg = run_dir / "app.yaml"                  # 每次运行留一份生效配置
        shutil.copyfile(job["vp"]["cpp_config"], app_cfg)
        log = run_dir / "run.log"
        cmd = [str(c["app"]), str(app_cfg), "--video_path", str(job["video"]),
               "--output_dir", str(run_dir), "--stride", str(run_cfg.get("stride", 1)),
               "--device", str(run_cfg.get("device", 0))]
        if int(run_cfg.get("max_frames") or 0) > 0:
            cmd += ["--max_frames", str(int(run_cfg["max_frames"]))]
        if run_cfg.get("render"):
            cmd.append("--render")
        cmd += [str(x) for x in (run_cfg.get("extra_args") or [])]
        print("[%d/%d] %s × %s → %s" % (index, len(todo), version, job["tag"], run_dir), flush=True)
        started = time.time()
        with log.open("w", encoding="utf-8") as fh:
            fh.write("$ cd %s\n$ %s\n\n" % (c["nx"], " ".join(cmd)))
            fh.flush()
            proc = subprocess.run(cmd, cwd=str(c["nx"]), stdout=fh, stderr=subprocess.STDOUT,
                                  check=False)
        wall = time.time() - started
        obs = count_observations(run_dir / "observations.jsonl")
        entry = {"version": version, "tag": job["tag"], "video": str(job["video"]),
                 "run_dir": str(run_dir), "cmd": cmd, "returncode": proc.returncode,
                 "wall_sec": round(wall, 1), "frames": obs["frames"],
                 "fps": round(obs["frames"] / wall, 2) if wall > 0 and obs["frames"] else None,
                 "persons": obs["persons"], "balls": obs["balls"],
                 "observations_mb": round(obs["bytes"] / 1e6, 2),
                 "stride": run_cfg.get("stride", 1), "render": bool(run_cfg.get("render"))}
        manifest.setdefault("jobs", []).append(entry)
        save_manifest(c, manifest)
        print("  → rc=%s  帧=%s  wall=%.1fs  FPS=%s  人=%s 球=%s"
              % (proc.returncode, entry["frames"], wall, entry["fps"],
                 entry["persons"], entry["balls"]), flush=True)
    ok = sum(1 for j in manifest["jobs"] if j.get("returncode") == 0)
    print("\n完成：manifest 里 rc==0 的任务共 %d 个 → %s" % (ok, manifest_path(c)))
    return 0


# --------------------------------------------------------------------------- summary
def cmd_summary(cfg: dict[str, Any], c: dict[str, Any], _: argparse.Namespace) -> int:
    manifest = load_manifest(c)
    jobs = manifest.get("jobs", [])
    if not jobs:
        raise SystemExit("还没有跑批记录（先跑 rollout.py run）")
    versions = [v for v in cfg["versions"] if any(j["version"] == v for j in jobs)]
    lines = ["# rollout 跑批汇总", "",
             "| 视频 | " + " | ".join("%s 帧/FPS/球" % v for v in versions) + " |",
             "| --- |" + " --- |" * len(versions)]
    last: dict[tuple[str, str], dict[str, Any]] = {}
    for j in jobs:                                    # 同一 (版本,视频) 只看最后一次
        last[(j["version"], j["tag"])] = j
    failed = sorted(key for key, j in last.items() if j.get("returncode") != 0)
    tags = sorted({j["tag"] for j in jobs if j.get("returncode") == 0})
    totals: dict[str, dict[str, float]] = {v: {"frames": 0, "wall": 0.0, "balls": 0} for v in versions}
    for tag in tags:
        cells = []
        for version in versions:
            found = [j for j in jobs if j["version"] == version and j["tag"] == tag]
            if not found:
                cells.append("—")
                continue
            j = found[-1]
            totals[version]["frames"] += j.get("frames") or 0
            totals[version]["wall"] += j.get("wall_sec") or 0
            totals[version]["balls"] += j.get("balls") or 0
            cells.append("%s / %s / %s" % (j.get("frames"), j.get("fps"), j.get("balls")))
        lines.append("| %s | %s |" % (tag, " | ".join(cells)))
    lines += ["", "## 按版本合计", "", "| 版本 | 帧数 | 总耗时(s) | 平均FPS | 球框总数 |",
              "| --- | --- | --- | --- | --- |"]
    for version in versions:
        t = totals[version]
        lines.append("| %s | %d | %.0f | %.2f | %d |"
                     % (version, t["frames"], t["wall"],
                        (t["frames"] / t["wall"]) if t["wall"] else 0, t["balls"]))
    mod = {v: module_timings(c, v) for v in versions}
    mod_names = sorted({k for m in mod.values() for k in m})
    if mod_names and any(mod.values()):
        lines += ["", "## 各模块耗时（ms/帧，取该版本最近一次跑批）",
                  "| 版本 | " + " | ".join(mod_names) + " |",
                  "| --- |" + " --- |" * len(mod_names)]
        for version in versions:
            cells = ["%.2f" % mod[version][k] if k in mod[version] else "—" for k in mod_names]
            lines.append("| %s | %s |" % (version, " | ".join(cells)))
    if failed:
        lines += ["", "> 另有 %d 个 (版本, 视频) 组合最近一次是失败（rc≠0），未计入上表；"
                      "详情看 `manifest.json` 或对应 run.log。" % len(failed)]
    out = c["out_root"] / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n→ %s" % out)
    return 0


def cmd_doctor(cfg: dict[str, Any], c: dict[str, Any], a: argparse.Namespace) -> int:
    """跑批前预检：app/engine/姿态 engine/类别契约/生成配置 是否就绪。任何一条不过就退出非 0。"""
    problems: list[str] = []
    print("== 预检 ==")
    checks = [
        ("app 已编译", c["app"].is_file(), str(c["app"])),
        ("姿态 engine", c["pose_engine"].is_file(), c["pose_engine"].name),
        ("C++ 底板配置", c["cpp_ref"].is_file(), str(c["cpp_ref"])),
    ]
    for name, ok, detail in checks:
        print("  [%s] %-14s %s" % ("✓" if ok else "✗", name, detail))
        if not ok:
            problems.append("%s 缺失：%s" % (name, detail))
    free_gb = shutil.disk_usage(str(c["out_root"] if c["out_root"].exists() else c["repo"])).free / 1e9
    print("  [%s] %-14s %.0f GB 可用（observations 约 1–2 MB/百帧，注意留量）"
          % ("✓" if free_gb > 20 else "✗", "磁盘余量", free_gb))
    if free_gb <= 20:
        problems.append("磁盘余量不足 20 GB：%.0f GB" % free_gb)

    # 类别契约：classes.json（训练/导出侧的真相源）必须与 rollout.yaml 一致
    classes_json = c["repo"] / "tools" / "detection_finetune" / "classes.json"
    if classes_json.is_file():
        declared = json.loads(classes_json.read_text(encoding="utf-8"))["categories"]
        ids = {str(x["name"]): x["id"] for x in declared}
        want = {"person": c["classes"]["person_class_id"], "ball": c["classes"]["ball_class_id"],
                "cone": c["classes"]["cone_class_id"]}
        same = all(ids.get(k) == v for k, v in want.items()) and len(declared) == c["classes"]["num_classes"]
        print("  [%s] %-14s classes.json=%s / rollout.yaml=%s"
              % ("✓" if same else "✗", "类别契约", ids, want))
        if not same:
            problems.append("类别契约不一致：classes.json=%s，rollout.yaml=%s" % (ids, want))
    else:
        print("  [ ] 类别契约    跳过（没找到 %s）" % classes_json)

    for version in pick_versions(cfg, a):
        vp = version_paths(c, version, cfg["versions"][version])
        state = {"权重": vp["checkpoint"].is_file(), "engine": vp["engine"].is_file(),
                 "权重软链": vp["link"].exists(), "模型卡": vp["card"].is_file(),
                 "C++配置": vp["cpp_config"].is_file()}
        ok = state["engine"] and state["C++配置"]
        print("  [%s] %-14s %s" % ("✓" if ok else "✗", "版本 " + version,
                                  "  ".join("%s=%s" % (k, "有" if v else "无") for k, v in state.items())))
        if not ok:
            problems.append("%s 未就绪（先 `rollout.py export --version %s` → `cpp-config`）"
                            % (version, version))
    if problems:
        print("\n预检未通过 %d 项：" % len(problems))
        for item in problems:
            print("  - %s" % item)
        return 1
    print("\n预检通过 ✓")
    return 0


def cmd_all(cfg: dict[str, Any], c: dict[str, Any], a: argparse.Namespace) -> int:
    """一条命令走完流程：预检 → （缺 engine 的版本）导出 → 生成 C++ 配置 → 跑批 → 汇总。"""
    if cmd_doctor(cfg, c, a) != 0 and not a.force:
        raise SystemExit("预检没过；确认要跳过就加 --force")
    versions = pick_versions(cfg, a)
    missing = [v for v in versions
               if not version_paths(c, v, cfg["versions"][v])["engine"].is_file()]
    if missing and not a.no_export:
        print("\n== 导出缺失 engine：%s ==" % ", ".join(missing))
        cmd_export(cfg, c, a)
    cmd_cpp_config(cfg, c, a)
    code = cmd_run(cfg, c, a)
    if not a.dry_run:
        cmd_summary(cfg, c, a)
    return code


def parse_app_rate(log: Path) -> dict[str, float]:
    """从 app 日志里读它自己算的速率（不含引擎加载时间）。"""
    out: dict[str, float] = {}
    if not log.is_file():
        return out
    text = log.read_text(encoding="utf-8", errors="ignore")
    rate = re.search(r"effective rate\s*:\s*([0-9.]+)\s*FPS", text)
    per = re.search(r"per frame\s*:\s*([0-9.]+)\s*ms", text)
    if rate:
        out["fps"] = float(rate.group(1))
    if per:
        out["ms_per_frame"] = float(per.group(1))
    return out


def cmd_bench(cfg: dict[str, Any], c: dict[str, Any], a: argparse.Namespace) -> int:
    """单视频 FPS 基准：重复跑 N 次，报告 app 自报 FPS/每帧耗时 + 各模块 ms/帧。

    跑批在跑时会拒绝（共用 GPU/CPU，同时跑会互相污染数字）。
    """
    if not a.video:
        raise SystemExit("bench 需要 --video <视频 tag>（用 `rollout.py videos` 看有哪些）")
    lock = Path("/tmp/rollout.pid")
    if lock.is_file():
        try:
            os.kill(int(lock.read_text(encoding="utf-8").strip()), 0)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        else:
            raise SystemExit("有 run 在跑（pid 见 %s），先等它结束再 bench，否则数字不准" % lock)
    if a.pose_profile:
        c["run"]["pose_profile"] = a.pose_profile
    version = a.version or pick_versions(cfg, a)[0]
    if version not in cfg["versions"]:
        raise SystemExit("未知版本：%s" % version)
    vp = version_paths(c, version, cfg["versions"][version])
    if not (vp["cpp_config"].is_file() and c["app"].is_file()):
        raise SystemExit("缺 C++ 配置或 app：先 `rollout.py cpp-config --version %s`" % version)
    tag = a.video[0]
    videos = [v for v in scan_videos(cfg, c) if v["tag"] == tag]
    if not videos:
        raise SystemExit("没有这个视频 tag：%s" % tag)
    video = videos[0]
    run_cfg = dict(c["run"])
    if a.stride is not None:
        run_cfg["stride"] = a.stride
    if a.max_frames is not None:
        run_cfg["max_frames"] = a.max_frames
    if a.render:
        run_cfg["render"] = True
    repeats = max(1, int(a.repeats))
    base_dir = c["out_root"] / "bench" / version / tag
    rows: list[dict[str, Any]] = []
    print("基准：%s × %s，重复 %d 次（stride=%s，max_frames=%s，render=%s）"
          % (version, tag, repeats, run_cfg.get("stride"), run_cfg.get("max_frames") or "全部",
             bool(run_cfg.get("render"))))
    for i in range(1, repeats + 1):
        run_dir = base_dir / ("run%d" % i)
        run_dir.mkdir(parents=True, exist_ok=True)
        app_cfg = run_dir / "app.yaml"
        shutil.copyfile(vp["cpp_config"], app_cfg)
        log = run_dir / "run.log"
        cmd = [str(c["app"]), str(app_cfg), "--video_path", str(video["path"]),
               "--output_dir", str(run_dir), "--stride", str(run_cfg.get("stride", 1)),
               "--device", str(run_cfg.get("device", 0)), "--profile_timers"]
        if int(run_cfg.get("max_frames") or 0) > 0:
            cmd += ["--max_frames", str(int(run_cfg["max_frames"]))]
        if run_cfg.get("render"):
            cmd.append("--render")
        started = time.time()
        with log.open("w", encoding="utf-8") as fh:
            fh.write("$ cd %s\n$ %s\n\n" % (c["nx"], " ".join(cmd)))
            fh.flush()
            proc = subprocess.run(cmd, cwd=str(c["nx"]), stdout=fh, stderr=subprocess.STDOUT,
                                  check=False)
        wall = time.time() - started
        obs = count_observations(run_dir / "observations.jsonl")
        rate = parse_app_rate(log)
        row = {"run": i, "returncode": proc.returncode, "wall_sec": round(wall, 1),
               "frames": obs["frames"], "persons": obs["persons"],
               "fps": rate.get("fps"), "ms_per_frame": rate.get("ms_per_frame")}
        rows.append(row)
        print("  第 %d 次：rc=%s 帧=%s  墙钟 %.1fs  app自报 %s FPS / %s ms·帧-1"
              % (i, proc.returncode, obs["frames"], wall,
                 row["fps"] if row["fps"] is not None else "—",
                 row["ms_per_frame"] if row["ms_per_frame"] is not None else "—"), flush=True)
    mod = module_timings(c, version) if not a.video else {}
    bench_mods: dict[str, float] = {}
    last_obs = base_dir / ("run%d" % repeats) / "observations.jsonl"
    if last_obs.is_file():
        acc: dict[str, float] = {}
        frames = 0
        with last_obs.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if '"module_timings_ms"' not in line:
                    continue
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if record.get("type") != "frame":
                    continue
                frames += 1
                for key, value in (record.get("module_timings_ms") or {}).items():
                    acc[key] = acc.get(key, 0.0) + float(value)
        bench_mods = {k: v / frames for k, v in acc.items()} if frames else {}
    lines = ["# FPS 基准：%s × %s" % (version, tag), "",
             "| 次数 | 退出码 | 帧数 | 墙钟(s) | app 自报 FPS | app 自报 ms/帧 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append("| %d | %s | %s | %.1f | %s | %s |" % (
            row["run"], row["returncode"], row["frames"], row["wall_sec"],
            row["fps"] if row["fps"] is not None else "—",
            row["ms_per_frame"] if row["ms_per_frame"] is not None else "—"))
    if bench_mods:
        lines += ["", "## 各模块 ms/帧（最后一次）", "",
                  "| 模块 | ms/帧 |", "| --- | --- |"]
        for key, value in sorted(bench_mods.items(), key=lambda kv: -kv[1]):
            lines.append("| %s | %.2f |" % (key, value))
    out = base_dir / "bench.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n→ %s" % out)
    return 0


COMMANDS = {"status": cmd_status, "videos": cmd_videos, "doctor": cmd_doctor,
            "export": cmd_export, "cpp-config": cmd_cpp_config, "run": cmd_run,
            "bench": cmd_bench, "summary": cmd_summary, "all": cmd_all}


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out-root", default="",
                    help="覆盖配置里的 paths.output_root（相对路径按 repo_root 解析）。"
                         "跑批产物落到别处，不覆盖已有结果；manifest/summary 也跟着换根。")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in COMMANDS:
        p = sub.add_parser(name, help=COMMANDS[name].__doc__ or name)
        p.add_argument("--version", default="", help="版本名（默认全部）")
        p.add_argument("--all", action="store_true", help="所有版本")
        p.add_argument("--video", action="append", default=[], help="视频 tag（可重复）")
        p.add_argument("--stride", type=int, default=None)
        p.add_argument("--max-frames", type=int, default=None)
        p.add_argument("--render", action="store_true", help="出渲染视频（更慢）")
        p.add_argument("--keep-roi", action="store_true", help="生成配置时保留 ROI（同机位才用）")
        p.add_argument("--pose-profile", default="", help="姿态档位：accurate/fast/fastest（默认取配置 run.pose_profile）")
        p.add_argument("--skip-engine", action="store_true", help="导出时跳过 engine 编译")
        p.add_argument("--force", action="store_true", help="已完成也重跑")
        p.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
        p.add_argument("--repeats", type=int, default=1, help="bench：重复次数")
        p.add_argument("--no-export", action="store_true", help="all：缺 engine 也不现场导出")
    a = ap.parse_args(list(argv) if argv is not None else None)
    cfg = load_config(Path(a.config))
    c = ctx(cfg)
    if a.out_root:
        raw = Path(a.out_root)
        c["out_root"] = (raw if raw.is_absolute() else c["repo"] / raw).resolve()
        print("输出根目录已覆盖：%s（配置里是 %s）"
              % (c["out_root"], cfg["paths"]["output_root"]))
    return COMMANDS[a.cmd](cfg, c, a)


if __name__ == "__main__":
    sys.exit(main())
