#!/usr/bin/env python3
"""配置驱动的批量跑视频：读 YAML → 逐条拼 `code/app.py` 命令 → 执行 → 记账。

分工（重要）
------------
* **干活的是 `code/app.py`**（完整管线：检测 + 跟踪 + 姿态 + 渲染 + `observations.jsonl`），
  本脚本只做编排：校验、拼命令、串行执行、跳过已完成、写 manifest。这样出口径和线上一致，
  且每条命令都原样打印 —— 出问题能直接复制重跑（`--dry-run` 只看不跑）。
* 只想"快速体检检测结果（含锥统计、耗时）"用 `s7_run_ab_videos.py`：
  `app.py` 落的 `observations.jsonl` **只有 person/ball，没有 cone**（锥只在渲染视频里画）。

校验前置（防静默错配）
----------------------
`models.<名>.checkpoint` 必须与 `app_config` 里 `models.detector.rfdetr_model_dir/rf-detr-small.pth`
**指向同一个文件**，否则直接报错退出 —— 否则很容易出现"配置写着 step2、实际加载 v1"这种
不报错但结论全错的情况。

用法
----
    source env.sh
    $PY s7_run_batch_videos.py --dry-run                  # 打印命令，不执行
    $PY s7_run_batch_videos.py                            # 跑全部（已有产物的自动跳过）
    $PY s7_run_batch_videos.py --force --only-model v1    # 强制重跑某版本
    $PY s7_run_batch_videos.py --config my_batch.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "s7_batch_videos.yaml"


VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}
NL = chr(10)


def sync_videos(cfg_path: Path, cfg: dict[str, Any], repo_root: Path) -> int:
    """按 `paths.video_roots` 递归扫描视频，重写配置的 `videos:` 段（保序、去重、tag 由相对路径生成）。

    只替换 `videos:` 及其之后的内容：前面的注释与其它段一字不动，所以"数据目录加了视频"
    这件事可以反复跑，不会把注释冲掉。
    """
    roots = [Path(str(r)) for r in (cfg["paths"].get("video_roots") or [])]
    if not roots:
        raise SystemExit("配置里没有 paths.video_roots，无法扫描（先加上要扫的目录）")
    found: list[tuple[Path, Path]] = []
    for root in roots:
        base = root if root.is_absolute() else repo_root / root
        if not base.is_dir():
            print("  !! 目录不存在，跳过：%s" % base, file=sys.stderr)
            continue
        found += [(p, base) for p in sorted(base.rglob("*")) if p.suffix.lower() in VIDEO_EXTS]
    seen: set[Path] = set()
    entries: list[tuple[str, Path]] = []
    for path, base in found:
        real = path.resolve()
        if real in seen:                                   # 多个 root 覆盖同一文件时只留一次
            continue
        seen.add(real)
        rel = path.relative_to(base)
        tag = str(rel.with_suffix("")).replace("/", "_").replace(" ", "_")
        entries.append((tag, rel))

    head = ("videos:            # 由 `s7_run_batch_videos.py --sync-videos` 自动生成"
            "（勿手改；改 paths.video_roots）")
    lines = [head, "  # path 一律相对 paths.video_roots[0] 书写 → 换机器/换目录不用改这里"]
    for tag, rel in entries:
        lines += ['  - path: "%s"' % rel, "    tag: %s" % tag]
    text = cfg_path.read_text(encoding="utf-8")
    marker = text.index(NL + "videos:")
    cfg_path.write_text(text[:marker] + NL + NL.join(lines) + NL, encoding="utf-8")
    print("已写入 %d 个视频（%d 个 root）→ %s" % (len(entries), len(roots), cfg_path.name))
    return len(entries)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit("配置文件不存在：%s" % path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("配置文件顶层必须是 mapping：%s" % path)
    for key in ("paths", "models", "videos"):
        if key not in data:
            raise SystemExit("配置缺少 `%s` 段：%s" % (key, path))
    return data


def model_dir_of_app_config(app_config: Path, repo_root: Path) -> tuple[Path, set[str]]:
    """从管线 YAML 里抠出模型目录与 class_names（轻量读 YAML，不 import 管线，避免拖依赖）。"""
    raw = yaml.safe_load(app_config.read_text(encoding="utf-8"))
    det = (raw or {}).get("models", {}).get("detector", {}) or {}
    model_dir = Path(str(det.get("rfdetr_model_dir", "models/rfdetr")))
    if not model_dir.is_absolute():
        model_dir = repo_root / model_dir
    names = det.get("class_names") or []
    if isinstance(names, str):
        names = [names]
    return model_dir.resolve(), {str(n) for n in names}


def default_video_root(repo_root: Path) -> Path:
    """`paths.video_roots` 留空时的默认值：`<WORKSPACE>/data`。

    与 base_layout.py / rollout.py 同一套约定，env 可覆盖。
    """
    workspace = Path(os.environ.get("AI_FOOTBALL_WORKSPACE")
                     or repo_root.parent).resolve()
    return Path(os.environ.get("AI_FOOTBALL_DATA_DIR")
                or workspace / "data").expanduser().resolve()


def video_base(paths: dict[str, Any], repo_root: Path) -> Path:
    """`videos[].path` 里相对路径的基准目录 = `paths.video_roots` 的第一个。

    这样 yaml 只写 `20260726/室内/带球-720p60.mov` 这类相对路径，换机器/换目录不用改文件。
    留空时**自己在函数内推导默认值**——别依赖调用方先注入，否则直接调用会解析到错目录。
    """
    roots = paths.get("video_roots") or []
    if not roots:
        return default_video_root(repo_root)
    first = Path(str(roots[0])).expanduser()
    return first if first.is_absolute() else (repo_root / first)


def resolve_video(raw: Any, paths: dict[str, Any], repo_root: Path) -> Path:
    given = Path(str(raw)).expanduser()
    return given if given.is_absolute() else video_base(paths, repo_root) / given


def build_jobs(cfg: dict[str, Any], repo_root: Path, only_models: set[str],
               only_videos: set[str]) -> list[dict[str, Any]]:
    paths = cfg["paths"]
    defaults = dict(cfg.get("defaults") or {})
    out_root = Path(str(paths["output_root"]))
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    template = str(paths.get("out_template", "{output_root}/{model}/{tag}"))

    jobs: list[dict[str, Any]] = []
    for video in cfg["videos"]:
        tag = str(video.get("tag") or Path(str(video["path"])).stem)
        if only_videos and tag not in only_videos:
            continue
        merged = {**defaults, **{k: v for k, v in video.items() if k not in ("path", "tag", "models")}}
        for model_name in video.get("models") or list(cfg["models"]):
            if only_models and model_name not in only_models:
                continue
            spec = cfg["models"][model_name]
            out_dir = Path(str(video.get("out_dir") or template.format(
                output_root=out_root, model=model_name, tag=tag)))
            if not out_dir.is_absolute():
                out_dir = repo_root / out_dir
            jobs.append({"model": model_name, "video_tag": tag,
                         "video": resolve_video(video["path"], paths, repo_root),
                         "spec": spec, "opts": merged, "out_dir": out_dir,
                         "checkpoint": spec["checkpoint"], "app_config": spec["app_config"]})
    return jobs


def validate_job(job: dict[str, Any], paths: dict[str, Any], repo_root: Path) -> list[str]:
    """返回错误列表（空 = 通过）。"""
    errs: list[str] = []
    app = repo_root / str(paths["app"])
    if not app.is_file():
        errs.append("找不到 app.py：%s" % app)
    video = Path(job["video"])
    if not video.is_file():
        errs.append("找不到视频：%s" % video)
    ckpt = Path(job["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = repo_root / ckpt
    if not ckpt.is_file():
        errs.append("找不到权重：%s" % ckpt)
    app_cfg = Path(job["app_config"])
    if not app_cfg.is_absolute():
        app_cfg = repo_root / app_cfg
    if not app_cfg.is_file():
        errs.append("找不到管线配置：%s" % app_cfg)
        return errs
    model_dir, names = model_dir_of_app_config(app_cfg, repo_root)
    link = model_dir / "rf-detr-small.pth"
    if not link.exists():
        errs.append("管线配置里的模型目录下没有 rf-detr-small.pth（先跑 s6_deploy_pin_model.py）：%s" % link)
    else:
        try:
            resolved = link.resolve()
        except OSError as exc:                                # noqa: PERF203
            errs.append("软链无法解析：%s（%s）" % (link, exc))
        else:
            if ckpt.is_file() and resolved.samefile(ckpt):
                pass
            else:
                errs.append("权重错配：管线配置实际会加载 %s，但配置声明的 checkpoint 是 %s"
                            % (resolved, ckpt))
    if names and "cone" not in names:
        errs.append("管线配置 class_names 里没有 cone（%s）—— 三分类模型会丢锥类" % sorted(names))
    return errs


def job_done(out_dir: Path) -> bool:
    """任务是否真跑完：observations.jsonl 存在 **且** 日志里有 app.py 的收尾行。

    只查文件会踩坑：被中断的任务也会留下半截 observations.jsonl，下一次运行就把它"跳过"了，
    于是静默缺数据（实测踩过）。
    """
    obs = out_dir / "observations.jsonl"
    log = out_dir / "run.log"
    if not (obs.is_file() and log.is_file()):
        return False
    tail = log.read_text(encoding="utf-8", errors="ignore")[-4000:]
    return ("processed " in tail and "frames from" in tail) or "rendered" in tail


def build_command(job: dict[str, Any], paths: dict[str, Any], repo_root: Path, python: str) -> list[str]:
    opts = job["opts"]
    cmd = [python, "-u", str(paths["app"]),
           "--video", str(job["video"]),
           "--output-dir", str(job["out_dir"]),
           "--config", str(job["app_config"]),
           "--device", str(opts.get("device", "cuda:0")),
           "--camera-id", str(opts.get("camera_id", "C1"))]
    if opts.get("roi_config"):
        cmd += ["--roi-config", str(opts["roi_config"])]
    if opts.get("use_roi"):
        cmd.append("--use-roi")
    # app.py 的 --stride 与 --target-fps 互斥：配了 target_fps 就走后者（"每秒采 N 帧"更直观）
    if int(opts.get("target_fps") or 0) > 0:
        cmd += ["--target-fps", str(opts["target_fps"])]
    elif opts.get("stride"):
        cmd += ["--stride", str(opts["stride"])]
    if int(opts.get("max_frames") or 0) > 0:
        cmd += ["--max-frames", str(int(opts["max_frames"]))]
    if not opts.get("render", True):
        cmd.append("--no-render")
    cmd += [str(a) for a in (opts.get("extra_args") or [])]
    return cmd


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="批量配置 YAML")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--list", action="store_true", help="只列出任务（含输出目录与是否已完成）")
    ap.add_argument("--only-model", action="append", default=[], help="只跑这些版本（可重复）")
    ap.add_argument("--only-video", action="append", default=[], help="只跑这些视频 tag（可重复）")
    ap.add_argument("--force", action="store_true", help="已有产物也重跑")
    ap.add_argument("--sync-videos", action="store_true",
                    help="按 paths.video_roots 扫描视频并重写配置的 videos 段后退出")
    ap.add_argument("--python", default="", help="默认取 $PY（env.sh），其次 sys.executable")
    # 冒烟/预览用的一次性覆盖（配置为主；命令行给的值优先于配置文件）
    ap.add_argument("--max-frames", type=int, default=None, help="覆盖所有任务的 max_frames（0=全部）")
    ap.add_argument("--stride", type=int, default=None, help="覆盖所有任务的 stride")
    ap.add_argument("--target-fps", type=float, default=None,
                    help="覆盖所有任务的 target_fps（>0 时优先于 stride）")
    ap.add_argument("--no-render", action="store_true", help="覆盖 render=false（只出 observations.jsonl）")
    ap.add_argument("--render", action="store_true", help="覆盖 render=true（配置里关了渲染时用）")
    a = ap.parse_args(list(argv) if argv is not None else None)

    cfg_path = Path(a.config)
    if not cfg_path.is_absolute():
        cfg_path = (HERE / cfg_path) if not cfg_path.exists() else cfg_path.resolve()
    cfg = load_config(cfg_path)
    paths = cfg["paths"]
    # repo_root / video_roots 留空即推导（与 base_layout.py 同一套约定），换机器/换目录不用改 yaml：
    #   repo_root  = <本文件>/../..            （…/AI-Football）
    #   video_roots= [<WORKSPACE>/data]；可用 AI_FOOTBALL_WORKSPACE / AI_FOOTBALL_DATA_DIR 覆盖
    repo_raw = str(paths.get("repo_root") or "").strip()
    repo_root = (Path(repo_raw).expanduser().resolve() if repo_raw
                 else HERE.parents[1].resolve())
    if not (paths.get("video_roots") or []):
        paths["video_roots"] = [default_video_root(repo_root)]
    if a.sync_videos:
        return 0 if sync_videos(cfg_path, cfg, repo_root) else 1
    python = a.python or __import__("os").environ.get("PY") or sys.executable

    jobs = build_jobs(cfg, repo_root, set(a.only_model), set(a.only_video))
    for job in jobs:                                  # 命令行覆盖优先
        if a.max_frames is not None:
            job["opts"]["max_frames"] = a.max_frames
        if a.stride is not None:
            job["opts"]["stride"] = a.stride
        if a.target_fps is not None:
            job["opts"]["target_fps"] = a.target_fps
        if a.no_render:
            job["opts"]["render"] = False
        if a.render:
            job["opts"]["render"] = True
    if not jobs:
        raise SystemExit("没有匹配的任务（检查 --only-model/--only-video 与配置里的 tag）")

    # 校验：任何一条不过就整体退出（不要跑一半才发现错配）
    bad = 0
    for job in jobs:
        errs = validate_job(job, paths, repo_root)
        job["errors"] = errs
        if errs:
            bad += 1
            print("!! %s × %s" % (job["model"], job["video_tag"]))
            for e in errs:
                print("     %s" % e)
    if bad:
        raise SystemExit("有 %d 个任务校验不通过，已中止（先修配置/先跑 s6_deploy_pin_model.py）" % bad)

    plan: list[dict[str, Any]] = []
    for job in jobs:
        cmd = build_command(job, paths, repo_root, python)
        done = job_done(job["out_dir"])
        plan.append({**job, "cmd": cmd, "done": done})

    if a.list or a.dry_run:
        print("=== 任务清单（%d 个）===" % len(plan))
        for job in plan:
            state = "已完成" if job["done"] else "待跑"
            print("  [%s] %-18s × %-16s → %s" % (state, job["model"], job["video_tag"], job["out_dir"]))
        if a.dry_run:
            for job in plan:
                print("\n$ cd %s\n$ %s" % (repo_root, " ".join(job["cmd"])))
        return 0

    manifest_path = Path(paths["output_root"])
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest_path.mkdir(parents=True, exist_ok=True)
    # pid 文件（与训练脚本同约定）：停止时用负号杀整个进程组，避免 pkill 误杀自己的 shell
    pid_file = Path("/tmp/batch_videos.pid")
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    print("停止：kill -- -$(cat %s)\n" % pid_file)

    results: list[dict[str, Any]] = []
    for index, job in enumerate(plan, 1):
        head = "[%d/%d] %s × %s" % (index, len(plan), job["model"], job["video_tag"])
        if job["done"] and not a.force:
            print("%s → 跳过（已有 observations.jsonl；--force 可重跑）" % head)
            results.append({**{k: job[k] for k in ("model", "video_tag")},
                            "out_dir": str(job["out_dir"]), "skipped": True, "cmd": job["cmd"]})
            continue
        job["out_dir"].mkdir(parents=True, exist_ok=True)
        log_path = job["out_dir"] / "run.log"
        print("%s\n  → %s\n  → 日志 %s" % (head, " ".join(job["cmd"]), log_path), flush=True)
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ cd %s\n$ %s\n\n" % (repo_root, " ".join(job["cmd"])))
            log.flush()
            proc = subprocess.run(job["cmd"], cwd=str(repo_root), stdout=log,
                                  stderr=subprocess.STDOUT, check=False)
        elapsed = time.time() - started
        obs = job["out_dir"] / "observations.jsonl"
        rec = {"model": job["model"], "video_tag": job["video_tag"], "video": str(job["video"]),
               "out_dir": str(job["out_dir"]), "cmd": job["cmd"], "returncode": proc.returncode,
               "seconds": round(elapsed, 1), "log": str(log_path),
               "observations": obs.is_file(),
               "observations_bytes": obs.stat().st_size if obs.is_file() else 0,
               "rendered": (job["out_dir"] / "rendered.mp4").is_file()}
        results.append(rec)
        print("  → 退出码 %s，用时 %.1fs，observations %s"
              % (proc.returncode, elapsed, "有" if rec["observations"] else "无"), flush=True)

    (manifest_path / "batch_manifest.json").write_text(
        json.dumps({"config": str(cfg_path), "repo_root": str(repo_root), "jobs": results},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for r in results if r.get("returncode") == 0)
    lines = ["# 批量跑视频结果（配置 %s）" % cfg_path.name, "",
             "| 任务 | 版本 | 视频 | 退出码 | 用时(s) | observations | rendered.mp4 | 输出目录 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, rec in enumerate(results, 1):
        if rec.get("skipped"):
            lines.append("| %d | %s | %s | 跳过 | — | 已有 | — | %s |"
                         % (i, rec["model"], rec["video_tag"], rec["out_dir"]))
            continue
        lines.append("| %d | %s | %s | %s | %.1f | %s | %s | %s |"
                     % (i, rec["model"], rec["video_tag"], rec["returncode"], rec["seconds"],
                        "有" if rec["observations"] else "无",
                        "有" if rec["rendered"] else "无", rec["out_dir"]))
    (manifest_path / "batch_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n完成 %d/%d 个任务（成功 %d）\n  清单 → %s\n  汇总 → %s"
          % (len(results), len(plan), ok, manifest_path / "batch_manifest.json",
             manifest_path / "batch_summary.md"))
    return 0 if ok == len([r for r in results if not r.get("skipped")]) else 1


if __name__ == "__main__":
    sys.exit(main())
