#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长跑训练监视器：解析训练日志，打印紧凑进度（epoch / loss / mAP / lr / 显存 / ETA）。

长跑（几十小时）时不想开 tensorboard 翻图、又不想被刷屏日志淹没，就用它：

    $PY s4_train_watch.py                       # 每 20 秒刷新一次
    $PY s4_train_watch.py --once                # 只看一眼当前状态
    $PY s4_train_watch.py --log runs/v1/train.log --interval 60

信息取自日志里已有的行（不额外占 GPU）：

* `Epoch: [n] [i/N] ... loss: a (b) ... lr: x`
* `Test: [...]`（每个 epoch 后的评测）
* `mAP` 行与 `Early stopping: ... Best: ...`
* `max mem: n`（框架自报显存）
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

EPOCH_RE = re.compile(r"Epoch: \[(\d+)\]\s+\[\s*(\d+)/(\d+)\].*?loss: ([0-9.]+) \(([0-9.]+)\)"
                      r"(?:.*?lr: ([0-9.eE+-]+))?")
TEST_RE = re.compile(r"Test:\s+\[\s*(\d+)/(\d+)\]")
MEM_RE = re.compile(r"max mem: (\d+)")
MAP_RE = re.compile(r"mAP \(EMA\): ([0-9.]+)|Best: ([0-9.]+)")
EARLY_RE = re.compile(r"Early stopping: (Current mAP \(EMA\): [0-9.]+, Best: [0-9.]+)")


def parse(log: Path) -> dict:
    """扫一遍日志，取最后一次出现的各类指标。"""
    out: dict[str, object] = {"epoch": 0, "total_epochs": 0, "step": 0, "steps": 0, "loss": None,
                              "avg_loss": None, "lr": None, "mAP": None, "best": None,
                              "max_mem": None, "phase": "train", "lines": 0}
    if not log.exists():
        return out
    text = log.read_text(encoding="utf-8", errors="ignore")
    out["lines"] = text.count("\n")
    for match in EPOCH_RE.finditer(text):
        out.update(epoch=int(match.group(1)) + 1, step=int(match.group(2)),
                   steps=int(match.group(3)), loss=float(match.group(4)),
                   avg_loss=float(match.group(5)))
        if match.group(6):
            out["lr"] = match.group(6)
    for match in TEST_RE.finditer(text):
        out.update(phase="eval", step=int(match.group(1)), steps=int(match.group(2)))
    for match in MEM_RE.finditer(text):
        out["max_mem"] = int(match.group(1))
    for match in MAP_RE.finditer(text):
        value = float(match.group(1) or match.group(2))
        if match.group(1):
            out["mAP"] = value
        else:
            out["best"] = value
    for match in EARLY_RE.finditer(text):
        cur = re.search(r"Current mAP \(EMA\): ([0-9.]+)", match.group(1))
        best = re.search(r"Best: ([0-9.]+)", match.group(1))
        if cur:
            out["mAP"] = float(cur.group(1))
        if best:
            out["best"] = float(best.group(1))
    # 只看"本次启动"之后的日志：续训/改名后日志会混进上一次（甚至被停掉）的进度，
    # 实测把上一轮 run 的 epoch/mAP 读了出来。启动器每次都会写 `==== <时间> 启动 ====`。
    marker = text.rfind("启动 ====")
    if marker >= 0:
        text = text[max(0, marker - 40):]
    out["total_epochs"] = int(re.findall(r"Epoch: \[(\d+)\]", text)[-1]) + 1 if "Epoch" in text else 0
    return out


def planned_epochs(log: Path) -> int:
    """计划 epoch 数：从 `train_configs/<版本>.yaml` 读（版本 = 日志所在目录名）。

    早先这里把 100 写死在渲染里 → 换成 step2（epochs 20）后显示"（配置 100）"，
    剩余时间估算也跟着错。
    """
    cfg = Path(__file__).resolve().parent / "train_configs" / (log.parent.name + ".yaml")
    if not cfg.is_file():
        return 0
    found = re.search(r"^\s*epochs:\s*(\d+)", cfg.read_text(encoding="utf-8"), re.M)
    return int(found.group(1)) if found else 0


def gpu_line() -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    res = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                          "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return "  GPU " + res.stdout.strip() if res.stdout.strip() else ""


def alive() -> str:
    res = subprocess.run(["pgrep", "-fc", "s4_train_finetune.py --config"], capture_output=True, text=True,
                         check=False)
    try:
        count = int(res.stdout.strip() or 0)
    except ValueError:
        count = 0
    return "训练进程在跑" if count else "!! 没有训练进程（可能已结束/被停）"


def render(state: dict, log: Path, started: float) -> str:
    phase = "评测/测试" if state["phase"] == "eval" else "训练"
    epochs = state["total_epochs"] or 0
    planned = planned_epochs(log)
    lines = [
        "=== %s ===" % log.name,
        "  阶段：%s   已到 epoch %s%s   步 %s/%s"
        % (phase, epochs, ("／共 %d" % planned) if planned else "", state["step"], state["steps"] or "?"),
        "  loss：%s（平均 %s）  lr：%s" % (state["loss"], state["avg_loss"], state["lr"]),
        "  mAP(EMA)：%s   历史最佳：%s" % (state["mAP"], state["best"]),
        "  框架自报显存峰值：%s MiB%s" % (state["max_mem"], gpu_line()),
        "  日志：%s（%d 行）  已监视 %.0f 分钟" % (log, state["lines"], (time.time() - started) / 60),
        "  %s" % alive(),
    ]
    if epochs and planned:
        lines.append("  粗估剩余：还需 %d epoch（每 epoch 约 1–2 分钟，视负载波动）"
                      % max(0, planned - epochs))
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练进度监视器（只读日志，不占 GPU）")
    parser.add_argument("--log", default="runs/v1/train.log")
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--once", action="store_true", help="只打印一次")
    args = parser.parse_args(list(argv) if argv is not None else None)

    log = Path(args.log)
    started = time.time()
    if not log.is_absolute():
        log = Path(__file__).parent / log
    while True:
        state = parse(log)
        text = render(state, log, started)
        if args.once:
            print(text)
            return 0
        sys.stdout.write("\033[2J\033[H" + text + "\n")   # 清屏重画，避免刷屏
        sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
