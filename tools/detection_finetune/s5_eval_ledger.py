#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""指标台账：把每次评测的 mAP / 主指标**累积**成一份可长期对比的历史记录。

为什么需要它：`s5_eval_detector.py` 写的 `compare.md` / `compare.json` 是**覆盖式**的
（只反映最近一次 A/B，重跑即冲掉），跨次对比只能手翻文件、还容易把不同评测集/分辨率的
数字混着比。本脚本扫描 `runs/**/metrics_*.json`，按 `(tag, split, resolution)` 追加：

    runs/metrics_ledger.json    机器可读全量历史（含来源文件 sha 与时间，保留每条 revision）
    runs/metrics_ledger.csv     扁平表（一行一条记录，便于 diff / 画曲线）
    <data>/ball_cone_v1/reports/metrics_ledger.md   人读表（各 split 最新一份 + 相对 base 的差值 + 决策门）

幂等：来源文件 sha 与**评测集指纹**都没变 → 不追加；变了 → 追加新 revision 并记 `prev_sha256`，
历史不丢。评测集指纹取 `dataset/<split>/_annotations.coco.json` 的 sha256[:12] —— 指纹不同
说明 GT 动过，两次指标不能直接比（表里会标出来）。

    $PY s5_eval_ledger.py                  # 扫描 + 更新台账 + 打印最新表
    $PY s5_eval_ledger.py --quiet          # 只更新不打印
    $PY s5_eval_ledger.py --runs-dir runs  # 换扫描根（默认 tools/detection_finetune/runs）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import base_layout

TOOL_DIR = Path(__file__).resolve().parent
RUNS_DIR = TOOL_DIR / "runs"
DEFAULT_LEDGER = RUNS_DIR / "metrics_ledger.json"
DEFAULT_CSV = RUNS_DIR / "metrics_ledger.csv"
DEFAULT_MD = base_layout.REPORTS_DIR / "metrics_ledger.md"
DEFAULT_ALIASES = TOOL_DIR / "metrics_aliases.json"   # 历史 tag → 规范 tag 的归并表
LEGACY_DIRNAME = "_legacy"                            # 归档目录（util_unify_model_names.py 用），不参与扫描

CLASSES = ("person", "ball", "cone")
OVERALL_KEYS = ("mAP@.5", "mAP@.5:.95", "mAP@.75", "AR@100")
MAIN_KEYS = ("cone_as_ball_frames", "cone_as_ball_boxes", "ball_frame_precision",
             "ball_frame_recall", "empty_ball_fp_ratio", "frames")
# person mAP50 允许掉点（百分点）—— 方案里的决策门
PERSON_DROP_GATE_PT = 1.0
CHECKPOINT_HASH_LIMIT = 1024 * 1024 * 1024        # >1 GB 的权重不整文件哈希（记 "-"）


def sha12(path: Path, limit: int | None = None) -> str:
    """文件 sha256 的前 12 位；缺失/超限返回 "-"。"""
    try:
        size = path.stat().st_size
    except OSError:
        return "-"
    if limit is not None and size > limit:
        return "-"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def file_sha12_checked(path: Path | None) -> tuple[str, str]:
    """返回 (sha12, 状态)；状态用于区分"没有路径 / 文件没了 / 太大不哈希 / 正常"。"""
    if path is None:
        return "-", "none"
    try:
        size = path.stat().st_size
    except OSError:
        return "-", "missing"
    if size > CHECKPOINT_HASH_LIMIT:
        return "-", "too-large"
    return sha12(path), "ok"


def gt_fingerprint(split: str) -> str:
    """评测集指纹：GT 的 sha256[:12]（换 GT 就该看出来）。"""
    return sha12(base_layout.DATASET_DIR / split / "_annotations.coco.json")


def load_aliases(path: Path) -> dict[str, str]:
    """读标签归并表（历史 tag → 规范 tag）；忽略 `_` 开头的说明键。

    为什么需要：早期评测命令行的 tag 用了旧名（`finetune-v1` / `finetune-v1-frozen`），
    后来只把指标**文件名**改成了规范名，文件内的 `tag` 字段按 README:43「留档不回溯改」
    保持原样 → 台账若不归并，同一个模型会出现两行。归并只在读取时发生，不动历史文件。
    """
    if not path.is_file():
        print("  !! 别名表不存在（%s）：台账保留原始 tag，不做归并。" % path, file=sys.stderr)
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                       # noqa: BLE001
        print("  !! 别名表读不了（%s）：%s" % (path, exc), file=sys.stderr)
        return {}
    return {str(k): str(v) for k, v in data.items() if not str(k).startswith("_")}


def extract(path: Path, aliases: dict[str, str] | None = None) -> dict[str, Any] | None:
    """把一个 metrics_*.json 压成台账记录；结构不符则返回 None（并提示）。

    tag 取文件内 `tag` 字段并**按别名表归并**；同时记 `tag_raw`（原始）与 `tag_file`
    （文件名推断），便于暴露"文件名与内部 tag 不一致"这类历史遗留。
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                       # noqa: BLE001 - 坏文件不该中断全量扫描
        print("  !! 跳过（读不了）：%s（%s）" % (path, exc), file=sys.stderr)
        return None
    coco = data.get("coco") or {}
    if not isinstance(coco, dict) or "overall" not in coco:
        print("  !! 跳过（不是 evaluate_detector 的产物）：%s" % path, file=sys.stderr)
        return None

    tag_file = path.stem.replace("metrics_", "")
    tag_raw = str(data.get("tag", tag_file))
    aliases = aliases or {}
    tag = aliases.get(tag_raw, tag_raw)
    checkpoint = str(data.get("checkpoint", ""))
    ckpt_path = Path(checkpoint) if checkpoint else None
    ckpt_sha, ckpt_state = file_sha12_checked(ckpt_path)
    behavior = data.get("behavior") or {}
    return {
        "tag": tag,
        "tag_raw": tag_raw,
        "tag_file": tag_file,
        "split": str(data.get("split", "?")),
        "resolution": int(data.get("resolution", 0)),
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(path),
        "source_sha256": sha12(path),
        "prev_sha256": None,
        "gt_sha256": gt_fingerprint(str(data.get("split", "?"))),
        "checkpoint": checkpoint,
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_state": ckpt_state,
        "threshold": data.get("threshold"),
        "n_predictions": data.get("n_predictions"),
        "overall": {k: coco.get("overall", {}).get(k) for k in OVERALL_KEYS},
        "per_class": {c: {k: (coco.get("per_class", {}).get(c) or {}).get(k)
                          for k in ("mAP@.5", "mAP@.5:.95")} for c in CLASSES},
        "main": {k: behavior.get(k) for k in MAIN_KEYS if k in behavior},
        "note": ("COCO 80 类（无 cone 类 → cone 恒为 0，overall 被结构性拉低）"
                 if tag == "base" else ""),
    }


def key_of(rec: dict[str, Any]) -> tuple[Any, ...]:
    return (rec["tag"], rec["split"], rec["resolution"])


def load_ledger(path: Path) -> dict[str, Any]:
    if path.is_file():
        try:
            ledger = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(ledger, dict) and isinstance(ledger.get("records"), list):
                return ledger
        except Exception as exc:                   # noqa: BLE001
            print("  !! 台账损坏，将重建：%s（%s）" % (path, exc), file=sys.stderr)
    return {"version": 1, "records": []}


def scan(runs_dir: Path, skip: set[Path] | None = None,
         aliases: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """扫描 runs/ 下的 metrics_*.json；skip 用于排除本脚本自己的产物（台账会命中同名 glob）。"""
    skip = skip or set()
    found: list[dict[str, Any]] = []
    for path in sorted(runs_dir.rglob("metrics_*.json")):
        if path.resolve() in skip or path.stem.startswith("metrics_ledger") \
                or LEGACY_DIRNAME in path.parts:
            continue
        rec = extract(path, aliases)
        if rec:
            found.append(rec)
    return found


def metric_signature(rec: dict[str, Any]) -> tuple[Any, ...]:
    ov, pc = rec["overall"], rec["per_class"]
    return (ov.get("mAP@.5:.95"), ov.get("mAP@.5"),
            pc["person"]["mAP@.5"], pc["ball"]["mAP@.5"], pc["cone"]["mAP@.5"])


def tag_aliases(records: Iterable[dict[str, Any]]) -> tuple[list[list[str]], list[list[str]]]:
    """同一模型挂了不同标签的两种情况：权重 sha 相同（确定）/ 同一 split 内指标逐项相同（疑似）。

    后者是实测需要：早期几次评测把 checkpoint 路径写成了别的 cwd 下的位置（sha 记不到），
    但同一份权重在同一 split 上评两次，指标会逐项相同。
    """
    by_ckpt: dict[str, set[str]] = {}
    by_split_metric: dict[tuple[Any, ...], set[str]] = {}
    for rec in records:
        sha = rec.get("checkpoint_sha256")
        if sha and sha != "-":
            by_ckpt.setdefault(sha, set()).add(rec["tag"])
        by_split_metric.setdefault((rec.get("split"), metric_signature(rec)), set()).add(rec["tag"])
    certain = sorted(sorted(tags) for tags in by_ckpt.values() if len(tags) > 1)
    suspected = sorted(sorted(tags) for tags in by_split_metric.values() if len(tags) > 1)
    return certain, suspected


def merge(ledger: dict[str, Any], found: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    """追加/更新记录；返回（新增, 更新, 未变, 重复）。

    * 未变 = 同一来源文件、内容没动；
    * 重复 = 同 key 且指标/GT/预测数逐项相同，只是换了文件名（或换了 tag）再评一遍 →
      **不追加**，只在 `ledger["duplicates"]` 留一行溯源，免得台账里同一模型出现两条一样的行。
    """
    records: list[dict[str, Any]] = ledger["records"]
    duplicates: list[dict[str, Any]] = ledger.setdefault("duplicates", [])
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for rec in records:
        latest[key_of(rec)] = rec
    # 折叠记录本身也要幂等：同一份重复评测只留一行溯源
    seen_dup = {(d.get("tag"), d.get("split"), d.get("resolution"), d.get("source"))
                for d in duplicates}
    added = updated = unchanged = duplicate = 0
    for rec in found:
        prev = latest.get(key_of(rec))
        if prev is None:
            added += 1
        elif prev.get("source_sha256") == rec["source_sha256"] \
                and prev.get("gt_sha256") == rec["gt_sha256"]:
            unchanged += 1
            continue
        elif prev.get("source") != rec["source"] \
                and prev.get("gt_sha256") == rec["gt_sha256"] \
                and prev.get("n_predictions") == rec["n_predictions"] \
                and metric_signature(prev) == metric_signature(rec):
            duplicate += 1
            sig = (rec["tag"], rec["split"], rec["resolution"], rec["source"])
            if sig not in seen_dup:
                seen_dup.add(sig)
                duplicates.append({"tag": rec["tag"], "tag_raw": rec.get("tag_raw"),
                                   "tag_file": rec.get("tag_file"),
                                   "split": rec["split"], "resolution": rec["resolution"],
                                   "source": rec["source"],
                                   "source_sha256": rec["source_sha256"],
                                   "recorded_at": rec["recorded_at"],
                                   "same_as": prev.get("source")})
            continue
        else:
            rec["prev_sha256"] = prev.get("source_sha256")
            updated += 1
        records.append(rec)
        latest[key_of(rec)] = rec
    return added, updated, unchanged, duplicate


def latest_per_key(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for rec in records:
        latest[key_of(rec)] = rec          # 后写的覆盖先写的 = 最新
    return sorted(latest.values(), key=lambda r: (r["split"], r["resolution"], r["tag"]))


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return "—" if value < 0 else ("%.*f" % (digits, value))
    return "—" if value is None else str(value)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["recorded_at", "tag", "tag_raw", "split", "resolution", "threshold", "n_predictions",
              "mAP@.5", "mAP@.5:.95", "mAP@.75", "AR@100", "person_mAP@.5", "ball_mAP@.5",
              "cone_mAP@.5", "cone_as_ball_frames", "cone_as_ball_boxes",
              "ball_frame_precision", "ball_frame_recall", "empty_ball_fp_ratio",
              "gt_sha256", "source_sha256", "prev_sha256", "checkpoint_sha256",
              "checkpoint_state", "source"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "recorded_at": rec["recorded_at"], "tag": rec["tag"],
                "tag_raw": rec.get("tag_raw"), "split": rec["split"],
                "resolution": rec["resolution"], "threshold": rec.get("threshold"),
                "n_predictions": rec.get("n_predictions"),
                "mAP@.5": rec["overall"].get("mAP@.5"),
                "mAP@.5:.95": rec["overall"].get("mAP@.5:.95"),
                "mAP@.75": rec["overall"].get("mAP@.75"),
                "AR@100": rec["overall"].get("AR@100"),
                "person_mAP@.5": rec["per_class"]["person"]["mAP@.5"],
                "ball_mAP@.5": rec["per_class"]["ball"]["mAP@.5"],
                "cone_mAP@.5": rec["per_class"]["cone"]["mAP@.5"],
                "cone_as_ball_frames": rec["main"].get("cone_as_ball_frames"),
                "cone_as_ball_boxes": rec["main"].get("cone_as_ball_boxes"),
                "ball_frame_precision": rec["main"].get("ball_frame_precision"),
                "ball_frame_recall": rec["main"].get("ball_frame_recall"),
                "empty_ball_fp_ratio": rec["main"].get("empty_ball_fp_ratio"),
                "gt_sha256": rec["gt_sha256"], "source_sha256": rec["source_sha256"],
                "prev_sha256": rec.get("prev_sha256") or "",
                "checkpoint_sha256": rec.get("checkpoint_sha256"),
                "checkpoint_state": rec.get("checkpoint_state"),
                "source": rec["source"],
            })


def table_for(recs: list[dict[str, Any]], tags: list[str]) -> list[str]:
    by_tag = {r["tag"]: r for r in recs}
    lines = ["| 模型 | mAP50 | mAP50-95 | person | ball | cone | 锥→球帧 | 球帧P | 球帧R |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for tag in tags:
        rec = by_tag[tag]
        ov, pc, main = rec["overall"], rec["per_class"], rec["main"]
        lines.append("| %s%s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            tag, "（base）" if tag == "base" else "",
            fmt(ov.get("mAP@.5")), fmt(ov.get("mAP@.5:.95")),
            fmt(pc["person"]["mAP@.5"]), fmt(pc["ball"]["mAP@.5"]), fmt(pc["cone"]["mAP@.5"]),
            fmt(main.get("cone_as_ball_frames"), 0) if main.get("cone_as_ball_frames") is not None else "—",
            fmt(main.get("ball_frame_precision")), fmt(main.get("ball_frame_recall"))))
    return lines


def write_md(path: Path, ledger: dict[str, Any], latest: list[dict[str, Any]]) -> None:
    records: list[dict[str, Any]] = ledger["records"]
    duplicates: list[dict[str, Any]] = ledger.get("duplicates", [])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 指标台账（累积记录，供后续对比）", "",
             "生成时间：%s ｜ 记录 %d 条（含历史 revision）" % (
                 time.strftime("%Y-%m-%d %H:%M:%S"), len(records)),
             "", "来源：`tools/detection_finetune/runs/**/metrics_*.json`"
             "（由 `s5_eval_detector.py` 产出）；本文件由 `s5_eval_ledger.py` 重新生成，别手改。", ""]
    splits = sorted({r["split"] for r in latest})
    fingerprints = {r["split"]: r["gt_sha256"] for r in latest}
    lines += ["**评测集指纹（GT sha256[:12]）**：%s" % "，".join(
        "%s=%s" % (s, fingerprints[s]) for s in splits),
        "", "> 指纹不同 = GT 动过，两次指标不能直接比；同 split 内多次 revision 会在最后一节列出。",
        "", "**各 split 已评测标签**：%s" % "；".join(
            "%s：%s" % (s, ", ".join("`%s`" % t for t in sorted(
                r["tag"] for r in latest if r["split"] == s))) for s in splits),
        "", "**命名口径（内外一致，2026-10-12 定）**：四个名字都直说 —— `base` / `v1` / "
        "`v1-frozen-step1`（`--freeze encoder`：主干冻结、只训新类头）/ `v1-frozen-step2`"
        "（从 step1 warm start、解冻主干精修）。`v1-frozen` 只是这两个 step 的谱系简写，"
        "**不单独当名字用**（会和 step1/step2 混）。当前部署/交付的是 **`v1-frozen-step2`**"
        "（`config/app_v1-frozen-step2.yaml` + `models/rfdetr/fine-tune/v1-frozen-step2/`）。"]
    applied = sorted({(r["tag_raw"], r["tag"]) for r in records if r.get("tag_raw") != r["tag"]})
    if applied:
        lines += ["", "**标签归并（历史 tag → 规范 tag，来自 `metrics_aliases.json`，只影响台账不改历史文件）**：%s"
                  % "；".join("`%s` → `%s`" % (a, b) for a, b in applied)]
    # 被折叠的文件不在 records 里，但同样是"文件名≠内部 tag"，要一并统计
    pairs = {(r["split"], r["tag_file"]) for r in records
             if r.get("tag_raw") and r.get("tag_file") != r["tag_raw"]}
    pairs |= {(d["split"], d["tag_file"]) for d in duplicates
              if d.get("tag_file") and d.get("tag_raw") and d["tag_file"] != d["tag_raw"]}
    mismatched = sorted(pairs)
    if mismatched:
        lines += ["", "> 注：%d 个历史指标文件的**文件名与内部 `tag` 字段不一致**（当时只改了文件名，"
                  "内部字段按 README「留档不回溯改」保留）：%s → 已按上面的归并表处理。"
                  % (len(mismatched), "、".join("`%s`(%s)" % (f, s) for s, f in mismatched))]
    if duplicates:
        lines += ["", "**重复评测（指标逐项相同，已折叠，不计入上表）**：%s" % "；".join(
            "`%s` ← `%s`（同 `%s`）" % (d["tag"], Path(d["source"]).name, Path(d["same_as"]).name)
            for d in duplicates)]
    certain, suspected = tag_aliases(records)
    if certain:
        lines += ["", "> ⚠️ **同权重不同标签**（checkpoint sha256 相同）：%s —— 对比请按**权重**对齐，别按标签名对齐。"
                  % "；".join(" ↔ ".join(a) for a in certain)]
    if suspected:
        lines += ["", "> ⚠️ **疑似同权重不同标签**（同一 split 内指标逐项相同）：%s —— 后续重评请固定标签。"
                  % "；".join(" ↔ ".join(a) for a in suspected)]
    missing = sorted({r["checkpoint"] for r in latest if r.get("checkpoint_state") == "missing"})
    if missing:
        lines += ["", "> ⚠️ 以下 checkpoint 路径**当前不存在**（评测时 cwd 与现在不同）：%s —— 溯源以实存文件为准。"
                  % "；".join("`%s`" % m for m in missing)]
    lines += [""]
    for split in splits:
        group = [r for r in latest if r["split"] == split]
        res = sorted({r["resolution"] for r in group})
        for r in res:
            recs = sorted([g for g in group if g["resolution"] == r], key=lambda x: x["tag"])
            tags = [x["tag"] for x in recs]
            lines += ["## %s @ resolution %s（最新）" % (split, r), ""]
            lines += table_for(recs, tags)
            base = next((x for x in recs if x["tag"] == "base"), None)
            if base:
                lines += ["", "**相对 base（同一份 GT）的差值**："]
                for x in recs:
                    if x is base:
                        continue
                    item = []
                    for cls in CLASSES:
                        b, c = base["per_class"][cls]["mAP@.5"], x["per_class"][cls]["mAP@.5"]
                        if b is None or c is None:
                            continue
                        item.append("%s mAP50 %+.3f" % (cls, c - b))
                    for k, label in (("cone_as_ball_frames", "锥→球帧"),
                                     ("ball_frame_recall", "球帧R"),
                                     ("empty_ball_fp_ratio", "无球帧出球")):
                        b, c = base["main"].get(k), x["main"].get(k)
                        if b is None or c is None:
                            continue
                        item.append("%s %+.3f" % (label, c - b))
                    lines.append("  * `%s`：%s" % (x["tag"], "，".join(item)))
                drops = []
                for x in recs:
                    if x is base or x["per_class"]["person"]["mAP@.5"] is None:
                        continue
                    d = (base["per_class"]["person"]["mAP@.5"]
                         - x["per_class"]["person"]["mAP@.5"]) * 100.0
                    drops.append((x["tag"], d))
                if drops:
                    # 逐档给结论：只看"最差档"会误判（探索性权重也在表里，不代表部署档）
                    lines += ["", "**决策门：person mAP50 掉点 vs base（阈值 ≤ %.1f 个百分点）**：%s"
                              % (PERSON_DROP_GATE_PT,
                                 "；".join("`%s` %+.2f pt %s"
                                           % (tag, -d, "✓" if d <= PERSON_DROP_GATE_PT else "✗ 超阈值")
                                           for tag, d in sorted(drops, key=lambda x: x[1])))]
            lines += [""]
    lines += ["## 历史 revision（时间序）", "",
              "| 时间 | 模型 | split | res | 来源 sha | GT 指纹 | 说明 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for rec in records:
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            rec["recorded_at"], rec["tag"], rec["split"], rec["resolution"],
            rec["source_sha256"], rec["gt_sha256"],
            "新记录" if not rec.get("prev_sha256") else "更新（前 %s）" % rec["prev_sha256"]))
    lines += ["", "端到端（触球名额级 P/R/F1）不在本表，归 `examples/touch_eval` 的回归记录。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="指标台账：累积 runs/ 下的评测指标供后续对比")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR), help="扫描根（默认 tools/detection_finetune/runs）")
    ap.add_argument("--ledger", default=str(DEFAULT_LEDGER), help="台账 JSON 路径")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="扁平 CSV 路径")
    ap.add_argument("--md", default=str(DEFAULT_MD), help="人读 MD 路径")
    ap.add_argument("--aliases", default=str(DEFAULT_ALIASES),
                    help="标签归并表（JSON：历史 tag → 规范 tag）")
    ap.add_argument("--no-aliases", action="store_true", help="不做归并，保留文件内原始 tag")
    ap.add_argument("--quiet", action="store_true", help="不把表打到终端")
    args = ap.parse_args(list(argv) if argv is not None else None)

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    if not runs_dir.is_dir():
        raise SystemExit("扫描根不存在：%s" % runs_dir)
    ledger_path = Path(args.ledger).expanduser().resolve()

    ledger = load_ledger(ledger_path)
    skip = {ledger_path.resolve(), Path(args.csv).expanduser().resolve(),
            Path(args.md).expanduser().resolve()}
    aliases = {} if args.no_aliases else load_aliases(Path(args.aliases).expanduser().resolve())
    found = scan(runs_dir, skip, aliases)
    added, updated, unchanged, duplicate = merge(ledger, found)
    ledger["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    records = ledger["records"]
    latest = latest_per_key(records)
    write_csv(Path(args.csv).expanduser().resolve(), records)
    write_md(Path(args.md).expanduser().resolve(), ledger, latest)

    print("  扫描 %s：命中 %d 个指标文件 → 新增 %d，更新 %d，未变 %d，重复折叠 %d（台账共 %d 条）"
          % (runs_dir, len(found), added, updated, unchanged, duplicate, len(records)))
    print("  台账 JSON：%s" % ledger_path)
    print("  扁平 CSV ：%s" % Path(args.csv).expanduser().resolve())
    print("  人读报告 ：%s" % Path(args.md).expanduser().resolve())
    if not args.quiet:
        for split in sorted({r["split"] for r in latest}):
            for res in sorted({r["resolution"] for r in latest if r["split"] == split}):
                recs = sorted([r for r in latest if r["split"] == split and r["resolution"] == res],
                              key=lambda x: x["tag"])
                print("\n  === %s @ %s（最新；GT 指纹 %s）===" % (split, res, recs[0]["gt_sha256"]))
                print("    " + "\n    ".join(table_for(recs, [r["tag"] for r in recs])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
