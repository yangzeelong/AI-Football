#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把散落的模型名统一成规范名（唯一口径见 README「模型管理（唯一命名：v1）」）。

为什么需要：早期评测命令行里的 tag 用过旧名（`finetune-v1` / `finetune-v1-frozen`），
后来只把**文件名**改成了规范名，文件内部的 `tag` 字段没动 → 同一份权重出现两个名字
（台账里会变成两行，人也会看糊涂）。本脚本一次改干净：

1. 指标文件内部 `tag` → 规范名，旧名保留在 `tag_legacy`（留档，不丢信息）；
2. `checkpoint` 路径若已失效 → 指向 `runs/<规范名>/<同名文件>`（**该文件存在才改**），
   旧路径存 `checkpoint_legacy`；
3. 文件名里的 tag 也改成规范名；若目标名已存在（同一权重评了两遍）→ 把多余的那份移到
   `runs/eval_ab/_legacy/`（保留证据，且 `_legacy` 不参与台账扫描）。

映射表的唯一真源是 `metrics_aliases.json`。默认 **dry-run**，只打印将做的改动；加 `--apply` 才写。

    $PY util_unify_model_names.py            # 看要改什么
    $PY util_unify_model_names.py --apply    # 真的改
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import s5_eval_ledger as ml

TOOL_DIR = Path(__file__).resolve().parent
RUNS_DIR = TOOL_DIR / "runs"
LEGACY_DIRNAME = "_legacy"


def plan_file(path: Path, aliases: dict[str, str], runs_dir: Path) -> dict[str, Any] | None:
    """算出单个指标文件要做哪些改动；无需改动返回 None。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                       # noqa: BLE001
        print("  !! 跳过（读不了）：%s（%s）" % (path, exc), file=sys.stderr)
        return None
    if not isinstance(data.get("coco"), dict):
        return None                                # 不是评测产物（例如台账自己）

    tag_raw = str(data.get("tag", path.stem.replace("metrics_", "")))
    # 只认映射表：`tag_legacy` 是留档字段，**不能**拿它当映射来源（否则会把规范名改回旧名）
    tag = aliases.get(tag_raw, tag_raw)
    changes: dict[str, Any] = {"path": path, "tag_raw": tag_raw, "tag": tag,
                               "set_tag": tag != tag_raw, "set_ckpt": None, "move_to": None}

    # ② checkpoint 失效 → 指到规范名目录下的同名文件
    ckpt = str(data.get("checkpoint", ""))
    if ckpt and not Path(ckpt).is_file():
        candidate = runs_dir / tag / Path(ckpt).name
        if candidate.is_file():
            changes["set_ckpt"] = str(candidate)
        else:
            changes["ckpt_missing"] = ckpt         # 找不到就不猜，留给人工

    # ③ 文件名里的 tag 也统一；撞名说明同一权重评了两遍 → 多余那份归档
    want = path.with_name("metrics_%s.json" % tag)
    if want != path:
        if want.exists():
            changes["move_to"] = path.parent.parent / LEGACY_DIRNAME / path.name
        else:
            changes["rename_to"] = want
    return changes


def apply_changes(ch: dict[str, Any]) -> None:
    path: Path = ch["path"]
    if ch["set_tag"] or ch["set_ckpt"]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if ch["set_tag"]:
            if "tag_legacy" not in data:
                data["tag_legacy"] = ch["tag_raw"]
            data["tag"] = ch["tag"]
        if ch["set_ckpt"]:
            if "checkpoint_legacy" not in data:
                data["checkpoint_legacy"] = data.get("checkpoint")
            data["checkpoint"] = ch["set_ckpt"]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if ch.get("move_to"):
        ch["move_to"].parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(ch["move_to"]))
    elif ch.get("rename_to"):
        path.rename(ch["rename_to"])


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="统一模型名（dry-run 默认）")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR), help="扫描根（默认 tools/detection_finetune/runs）")
    ap.add_argument("--aliases", default=str(ml.DEFAULT_ALIASES), help="映射表（历史 tag → 规范 tag）")
    ap.add_argument("--apply", action="store_true", help="真的写盘（默认只打印）")
    args = ap.parse_args(list(argv) if argv is not None else None)

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    aliases = ml.load_aliases(Path(args.aliases).expanduser().resolve())
    if not aliases:
        raise SystemExit("映射表为空或缺，先确认 %s" % args.aliases)

    todo: list[dict[str, Any]] = []
    for path in sorted(runs_dir.rglob("metrics_*.json")):
        if path.stem.startswith("metrics_ledger") or LEGACY_DIRNAME in path.parts:
            continue
        ch = plan_file(path, aliases, runs_dir)
        if ch and (ch["set_tag"] or ch["set_ckpt"] or ch.get("move_to") or ch.get("rename_to")):
            todo.append(ch)

    print("  扫描 %s：需要统一 %d 个文件%s" % (runs_dir, len(todo),
                                             "" if args.apply else "（dry-run，未写盘）"))
    for ch in todo:
        rel = ch["path"].relative_to(runs_dir)
        acts = []
        if ch["set_tag"]:
            acts.append("tag: %s → %s（旧名存 tag_legacy）" % (ch["tag_raw"], ch["tag"]))
        if ch["set_ckpt"]:
            acts.append("checkpoint: 失效路径 → %s" % ch["set_ckpt"])
        if ch.get("ckpt_missing"):
            acts.append("!! checkpoint 失效且找不到替代（%s）" % ch["ckpt_missing"])
        if ch.get("move_to"):
            acts.append("归档（同名目标已存在）→ %s" % ch["move_to"].relative_to(runs_dir))
        elif ch.get("rename_to"):
            acts.append("文件名 → %s" % ch["rename_to"].name)
        print("    %-52s %s" % (rel, "；".join(acts)))
        if args.apply:
            apply_changes(ch)
    if todo and not args.apply:
        print("\n  加 --apply 执行上述改动。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
