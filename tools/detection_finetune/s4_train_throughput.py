#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""吞吐调优探针：实测不同 `batch_size` × `gradient_checkpointing` 组合的**显存峰值**与**每步耗时**。

为什么需要：`v1.yaml` 是按"显存吃紧"写的（batch 4 + grad_accum 4 + checkpointing 开），
实测显存只用了一半（7.3G/16G）、GPU 利用率很低。与其猜，不如量：

* 每个组合跑 `--subset N --epochs 1`（N 张图，够跑满几个 micro-batch 稳定下来）；
* 全程用 `nvidia-smi` 采样显存峰值；
* 从训练日志里解析 `time:`（每个 micro-batch 秒数）、`data:`（取数时间）、`max mem:`（框架自报）；
* 输出对比表 + 推荐（在"显存不炸"前提下选 img/s 最高的）。

用法::

    $PY s4_train_throughput.py --configs 4:on,8:on,8:off,12:off --subset 64 --steps 8
    $PY s4_train_throughput.py --configs 8:off --base train_configs/v1.yaml   # 只测一个
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

import yaml

import base_layout

HERE = Path(__file__).parent
TIME_RE = re.compile(r"\btime: ([0-9.]+)\s+data: ([0-9.]+)\s+max mem: ([0-9]+)")
SAFE_FRACTION = 0.92          # 显存用到 92% 以上就算"危险"，不推荐


def gpu_peak_mb(stop: threading.Event, out: dict[str, int], interval: float = 0.4) -> None:
    """后台采样显存峰值（MiB）。"""
    while not stop.is_set():
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10, check=False)
            value = int(res.stdout.strip().splitlines()[0])
            out["peak"] = max(out.get("peak", 0), value)
        except (ValueError, IndexError, subprocess.SubprocessError, OSError):
            pass
        stop.wait(interval)


def total_memory_mb() -> int:
    res = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=False)
    try:
        return int(res.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0


def write_config(base: Path, target: Path, batch: int, checkpointing: bool,
                 epochs: int, grad_accum: int | None, output_dir: Path,
                 workers: int | None = None, expanded: bool | None = None) -> dict:
    data = yaml.safe_load(base.read_text(encoding="utf-8"))
    data.setdefault("model", {})["gradient_checkpointing"] = bool(checkpointing)
    train = data.setdefault("train", {})
    train["batch_size"] = batch
    if grad_accum:
        train["grad_accum_steps"] = grad_accum
    train["epochs"] = epochs
    train["output_dir"] = str(output_dir)
    if workers:
        train["num_workers"] = workers
        train["persistent_workers"] = True
    if expanded is not None:
        train["expanded_scales"] = expanded
    train["checkpoint_interval"] = 1
    train["early_stopping"] = False          # 探针不需要早停
    train["tensorboard"] = False
    target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return data


def run_one(py: str, base: Path, spec: str, subset: int, steps: int,
            alloc_conf: str = "", no_expanded: bool = False) -> dict:
    parts = spec.split(":")
    batch_s, ckpt_s = parts[0], parts[1]
    workers = int(parts[2]) if len(parts) > 2 and parts[2] else None
    tag = "%s_%s%s" % (batch_s, ckpt_s.lower(), "_w%d" % workers if workers else "")
    batch, ckpt = int(batch_s), ckpt_s.lower() in ("on", "true", "1", "yes")
    workdir = Path("/tmp/tune_" + tag)
    config = Path("/tmp/tune_%s.yaml" % tag)
    log = workdir / "probe.log"
    workdir.mkdir(parents=True, exist_ok=True)
    grad_accum = max(1, round(16 / batch))          # 保持等效 batch ≈16（与正式训练一致）
    write_config(base, config, batch, ckpt, epochs=1, grad_accum=grad_accum,
                 output_dir=workdir, workers=workers,
                 expanded=False if no_expanded else None)

    cmd = [py, "-u", str(HERE / "s4_train_finetune.py"), "--config", str(config), "--subset", str(subset)]
    stop = threading.Event()
    peak: dict[str, int] = {}
    sampler = threading.Thread(target=gpu_peak_mb, args=(stop, peak), daemon=True)
    sampler.start()
    started = time.time()
    env = os.environ.copy()
    if alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf      # 例：expandable_segments:True
    with log.open("w") as handle:
        subprocess.run(cmd, cwd=str(HERE), stdout=handle, stderr=subprocess.STDOUT, check=False,
                       timeout=1800, env=env)
    stop.set()
    sampler.join(timeout=2)

    text = log.read_text(encoding="utf-8", errors="ignore")
    samples = [tuple(map(float, m.groups())) for m in TIME_RE.finditer(text)]
    train_samples = [s for s in samples]
    if not train_samples:
        return {"spec": spec, "batch": batch, "ckpt": ckpt, "workers": workers, "ok": False,
                "why": "没解析到 time: 行",
                "log": str(log)}
    # 跳过首个 micro-batch（含 warmup/编译），用中位数更能代表稳态
    times = sorted(s[0] for s in train_samples[1:]) or [train_samples[0][0]]
    mid = times[len(times) // 2]
    data_mid = sorted(s[1] for s in train_samples[1:] or train_samples)[len(times) // 2]
    max_mem = max(s[2] for s in train_samples)
    return {"spec": spec, "batch": batch, "ckpt": ckpt, "workers": workers, "ok": True, "step_s": mid,
            "data_s": data_mid, "imgs_s": batch / mid if mid else 0.0,
            "peak_mb": peak.get("peak", 0), "fw_mem_mb": int(max_mem),
            "elapsed": time.time() - started, "log": str(log)}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="吞吐调优探针（batch × gradient_checkpointing）")
    parser.add_argument("--base", default=str(HERE / "train_configs" / "v1.yaml"))
    parser.add_argument("--configs", default="4:on,8:on,8:off,12:off",
                        help="逗号分隔的 batch:checkpointing 组合")
    parser.add_argument("--subset", type=int, default=64, help="每 split 用前 N 张（够跑满几个 micro-batch）")
    parser.add_argument("--alloc", default="", help="PYTORCH_CUDA_ALLOC_CONF（如 expandable_segments:True）")
    parser.add_argument("--no-expanded", action="store_true", help="关掉 expanded_scales（减少输入尺寸抖动）")
    args = parser.parse_args(list(argv) if argv is not None else None)
    base_layout.ensure_layout()

    py = sys.executable
    base = (HERE / args.base) if not Path(args.base).is_absolute() else Path(args.base)
    specs = [s.strip() for s in args.configs.split(",") if s.strip()]
    total_mb = total_memory_mb()
    print("GPU 总显存 %d MiB；逐个试 %s（每个跑 1 epoch / subset %d）\n"
          % (total_mb, "、".join(specs), args.subset))

    results = []
    for spec in specs:
        print("== 试 batch:ckpt = %s ==" % spec, flush=True)
        res = run_one(py, base, spec, args.subset, 1, args.alloc, args.no_expanded)
        results.append(res)
        if res["ok"]:
            print("   每 micro-batch %.3fs（取数 %.3fs）→ %.1f img/s；显存峰值 %d MiB（框架自报 %d）"
                  
                  % (res["step_s"], res["data_s"], res["imgs_s"], res["peak_mb"], res["fw_mem_mb"]))
        else:
            print("   失败：%s（详见 %s）" % (res["why"], res["log"]))

    good = [r for r in results if r["ok"]]
    if not good:
        print("\n全部失败，先看日志再决定")
        return 1
    print("\n%-14s %8s %8s %10s %10s %s" % ("batch:ckpt", "步(s)", "img/s", "显存(MiB)", "占比", "结论"))
    best = None
    for res in good:
        ratio = (res["peak_mb"] / total_mb) if total_mb else 0
        risky = ratio >= SAFE_FRACTION
        verdict = "风险高（显存接近上限）" if risky else "可用"
        print("%-14s %8.3f %8.1f %10d %9.0f%% %s"
              % (res["spec"], res["step_s"], res["imgs_s"], res["peak_mb"], 100 * ratio, verdict))
        if not risky and (best is None or res["imgs_s"] > best["imgs_s"]):
            best = res
    if best is None:
        best = min(good, key=lambda r: r["peak_mb"])
        print("\n所有组合都接近显存上限 → 建议取最省显存的那个")
    print("\n推荐：`batch_size: %d` + `gradient_checkpointing: %s`"
          "（%.1f img/s，显存峰值 %d MiB）" % (best["batch"], "true" if best["ckpt"] else "false",
                                             best["imgs_s"], best["peak_mb"]))
    print("把这两项写进 %s 即可；grad_accum_steps 记得按 等效batch/新batch 调整"
          % base.relative_to(HERE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
