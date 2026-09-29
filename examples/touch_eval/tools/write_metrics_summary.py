#!/usr/bin/env python3
"""把一次运行的指标写成 `<output-dir>/metrics.md` 与 `metrics.csv`。

数字全部从产物里读（`touch_report.json`、标注 CSV、插桩重放一次判定层），不手抄，
所以标注一改、跑一次评估、再跑一次这个脚本，文件夹里的指标就同步了。叙述版
（带结论与出处）在 `docs/touch_summary.md`。

    uv run --project examples/touch_eval/py python3 \
        examples/touch_eval/tools/write_metrics_summary.py \
        --output-dir output/touch_eval_moving_pass-update2
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from datetime import date
from pathlib import Path

from touch_eval import cli as cli_module
from touch_eval import detect as detect_module
from touch_eval import gt as gt_module
from touch_eval.params import DetectorParams


def cluster_stats(output_dir: Path, params) -> dict | None:
    """插桩重放一次判定层：簇的数量、被丢弃的簇、一个簇出几个事件。"""
    observations = output_dir / "observations.jsonl"
    if not observations.is_file():
        return None
    frames = cli_module.load_evidence(str(observations), params)
    clusters: list[tuple[int, int, int]] = []
    original = detect_module.TouchEventAggregator.close_cluster

    def close(self):
        active = self.m_active
        length = len(self.m_cluster)
        contact = sum(1 for entry in self.m_cluster if entry.in_contact)
        events = original(self)
        if active:
            clusters.append((length, contact, len(events)))
        return events

    detect_module.TouchEventAggregator.close_cluster = close
    try:
        detector = detect_module.TouchDetector(params)
        kept: list = []
        for frame in frames:
            kept.extend(detector.push_evidence(frame))
        kept.extend(detector.flush())
    finally:
        detect_module.TouchEventAggregator.close_cluster = original
    dropped = [item for item in clusters if item[1] < 2]
    return {
        "clusters": len(clusters),
        "dropped": len(dropped),
        "dropped_zero": sum(1 for item in dropped if item[1] == 0),
        "dropped_one": sum(1 for item in dropped if item[1] == 1),
        "kept_clusters": len(clusters) - len(dropped),
        "candidate_events": sum(item[2] for item in clusters),
        "double_clusters": sum(1 for item in clusters if item[2] > 1),
        "length_min": min(item[0] for item in clusters),
        "length_median": statistics.median(item[0] for item in clusters),
        "length_max": max(item[0] for item in clusters),
        "contact_median": statistics.median(item[1] for item in clusters),
        "contact_max": max(item[1] for item in clusters),
        "events": len(kept),
        "suppressed_foot": detector.suppressed_by_refractory,
        "suppressed_person": detector.suppressed_by_person_refractory,
        "rejected_ball_gate": detector.m_aggregator.m_rejected_by_ball_gate,
        "rejected_speed_gate": detector.m_aggregator.m_rejected_by_speed_gate,
    }


def slot_composition(output_dir: Path, report: dict, annotations_path: Path) -> dict:
    """FN 的名额级构成：整行未命中 vs 部分命中（需要按匹配算，不是按窗口内预测数）。"""
    rows = [row for row in gt_module.parse_annotation_file(str(annotations_path)).events]
    events = report["events"]
    matched: dict[int, int] = {}
    for index, event in enumerate(events):
        if event["gt_index"] >= 0:
            matched[rows[event["gt_index"]].index] = matched.get(
                rows[event["gt_index"]].index, 0) + 1
    full_miss = partial = full_slots = partial_slots = 0
    for row in rows:
        got = matched.get(row.index, 0)
        if got >= (row.count or 1):
            continue
        missing = (row.count or 1) - got
        if got == 0:
            full_miss += 1
            full_slots += missing
        else:
            partial += 1
            partial_slots += missing
    return {"full_miss_rows": full_miss, "full_miss_slots": full_slots,
            "partial_rows": partial, "partial_slots": partial_slots}


def score_row(tag: str, score: dict) -> list:
    return [tag, score.get("capacity", ""), score["true_positives"], score["false_positives"],
            score["false_negatives"], round(score["precision"], 4), round(score["recall"], 4),
            round(score["f1"], 4)]


def resolve_annotations(given: str, recorded: str) -> Path:
    """标注路径的三级回退：显式给的 → 报告里记的 → 换个基准目录再试。

    报告里记的是"跑评估时那条命令的工作目录"下的相对路径，所以从仓库根跑这个脚本时
    要补上 `examples/touch_eval/py/` 前缀。
    """
    for candidate in (given, recorded):
        if not candidate:
            continue
        path = Path(candidate)
        for base in (Path("."), Path("examples/touch_eval/py"), Path("data")):
            probe = base / path if base != Path("data") else base / path.name
            if probe.is_file():
                return probe
    raise SystemExit("找不到标注文件：%s" % (given or recorded))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="output/touch_eval_moving_pass")
    parser.add_argument("--annotations", default="",
                        help="标注文件；默认用报告里记录的那一份（ground_truth_source）")
    parser.add_argument("--no-clusters", action="store_true",
                        help="跳过插桩重放（不写簇统计那一节）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report = json.loads((output_dir / "touch_report.json").read_text(encoding="utf-8"))
    main_score = report["primary_count_aware"]
    clean = report["low_quality"]
    clean_score = clean["primary_count_aware"]
    official = main_score           # 对外那个数：名额级、含 low_quality
    stats = None if args.no_clusters else cluster_stats(
        output_dir, DetectorParams.parse(report["params"]))
    annotations_path = resolve_annotations(args.annotations, report["ground_truth_source"])
    if stats:   # 漏斗是独立重放出来的，必须和报告完全一致才写文件
        assert (stats["events"], stats["suppressed_foot"], stats["suppressed_person"],
                stats["rejected_ball_gate"], stats["rejected_speed_gate"]) == (
            report["predicted_events"], main_score["suppressed_by_refractory"],
            main_score["suppressed_by_person_refractory"],
            main_score["rejected_by_ball_gate"], main_score["rejected_by_speed_gate"]), \
            "插桩重放与 touch_report.json 不一致，先别写指标"
    slots = slot_composition(output_dir, report, annotations_path)

    # --- CSV：一行一个口径 ---------------------------------------------------
    row_capacity = report["counts"]["records"]      # 行级口径的分母是"行数"
    rows = [
        score_row("名额级 · 含 low_quality（对外）", official),
        score_row("名额级 · 不含 low_quality", clean_score),
        score_row("行级 ±0 帧 · 含 low_quality",
                  {**report["primary"], "capacity": row_capacity}),
        score_row("行级 ±0 帧 · 不含 low_quality",
                  {**clean["primary"], "capacity": row_capacity}),
    ]
    sweep = {point["tolerance_frames"]: point for point in report["tolerance_sweep"]}
    for frames in sorted(sweep):
        if frames == 0:
            continue
        rows.append(score_row("行级 ±%d 帧 · 含 low_quality" % frames,
                              {**sweep[frames], "capacity": row_capacity}))
    clean_sweep = {point["tolerance_frames"]: point for point in clean["tolerance_sweep"]}
    for frames in sorted(clean_sweep):
        if frames == 0:
            continue
        rows.append(score_row("行级 ±%d 帧 · 不含 low_quality" % frames,
                              {**clean_sweep[frames], "capacity": row_capacity}))
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["口径", "容量(名额/行)", "TP", "FP", "FN", "precision", "recall", "F1"])
        writer.writerows(rows)

    # --- Markdown ------------------------------------------------------------
    def table(header: list[str], body: list[list]) -> list[str]:
        return ["| " + " | ".join(header) + " |",
                "| " + " | ".join("---" for _ in header) + " |"] + \
               ["| " + " | ".join(str(cell) for cell in line) + " |" for line in body]

    ground_truth = report["ground_truth"]
    lines = [
        "# 触球评估指标（自动生成）",
        "",
        "数字全部从本目录的产物读出；叙述版（结论、机制、出处）见 "
        "[`docs/touch_summary.md`](../../docs/touch_summary.md)。生成日期 %s。" % date.today(),
        "",
        "## 运行身份",
        "",
        *table(["项", "值"], [
            ["帧数 / 时长 / fps", "%d / %.2f s / %.2f" % (report["frames"],
                                                          report["duration_sec"], report["fps"])],
            ["观测", report["observations"]],
            ["标注", "%s（%d 行 / %d 次触球 / %d 个 no_touch / %d 行低质量）"
             % (report["ground_truth_source"], ground_truth["events"], ground_truth["touches"],
                ground_truth["no_touch_spans"], clean["rows"])],
            ["窗口锚点 / 点窗口", "%s / %.2f s" % (report["point_window_anchor"],
                                                   main_score["point_window_sec"])],
            ["预测事件数", report["predicted_events"]],
            ["判定参数", "`%s`" % report["params"]],
        ]),
        "",
        "## 1. 关键指标",
        "",
        *table(["口径", "容量(名额/行)", "TP", "FP", "FN", "precision", "recall", "F1"], rows[:4]),
        "",
        "> 名额级 = 一行标注写 `count` 次就算 `count` 个名额（对外口径）；行级 = 每个标注最多配一个事件，"
        "多触球行里第 2 个及以后的事件在那一口径下会被算成 FP，所以行级的 P 偏低。",
        "",
        "## 2. 容差扫描（行级）",
        "",
        *table(["档位", "含 low_quality TP/FP/FN → P/R/F1",
                "不含 low_quality TP/FP/FN → P/R/F1"],
               [["±%d 帧 (%.3f s)" % (frames, sweep[frames]["tolerance_sec"]),
                 "%d/%d/%d → %.4f/%.4f/%.4f" % (sweep[frames]["true_positives"],
                                                sweep[frames]["false_positives"],
                                                sweep[frames]["false_negatives"],
                                                sweep[frames]["precision"],
                                                sweep[frames]["recall"], sweep[frames]["f1"]),
                 "%d/%d/%d → %.4f/%.4f/%.4f" % (clean_sweep[frames]["true_positives"],
                                                clean_sweep[frames]["false_positives"],
                                                clean_sweep[frames]["false_negatives"],
                                                clean_sweep[frames]["precision"],
                                                clean_sweep[frames]["recall"],
                                                clean_sweep[frames]["f1"])]
                for frames in sorted(sweep)]),
        "",
        "## 3. 误检 FP 分解（%d 条）" % main_score["false_positives"],
        "",
        *table(["类别", "条数"], [[key, value] for key, value in sorted(
            ((key, value) for key, value in report["false_positive_breakdown"].items()
             if key != "examples"), key=lambda item: -item[1])]),
        "",
        "## 4. 漏检 FN 构成（名额级，%d 个名额）" % main_score["false_negatives"],
        "",
        *table(["构成", "行数", "名额数"], [
            ["整行未命中", slots["full_miss_rows"], slots["full_miss_slots"]],
            ["部分命中（写了多次只填了一部分）", slots["partial_rows"], slots["partial_slots"]],
        ]),
        "",
        "报告给出的原因分组（只统计整行未命中的行）：",
        "",
        *table(["原因", "行数"], [[group["reason"], group["count"]]
                                 for group in report["failures"]
                                 if not group["reason"].startswith("false positive")]),
        "",
        "## 5. 计数表",
        "",
        *table(["项", "值"], [
            ["行数 / 标注触球数", "%d / %d" % (report["counts"]["records"],
                                               report["counts"]["gt_touches"])],
            ["计数完全正确", "%d 行" % report["counts"]["exact_records"]],
            ["偏多 / 偏少", "%d / %d 行" % (report["counts"]["over_records"],
                                            report["counts"]["under_records"])],
            ["平均计数误差", "%.3f" % report["counts"]["mean_abs_count_error"]],
        ]),
        "",
    ]
    if stats:
        lines += [
            "## 6. 判定漏斗与簇统计（插桩重放）",
            "",
            *table(["环节", "数量"], [
                ["帧", report["frames"]],
                ["簇（候选接触片段）", stats["clusters"]],
                ["丢弃：接触帧 < 2", "−%d（零接触 %d、单帧 %d）"
                 % (stats["dropped"], stats["dropped_zero"], stats["dropped_one"])],
                ["保留的簇", stats["kept_clusters"]],
                ["候选事件（%d 个簇各出 2 个）" % stats["double_clusters"],
                 stats["candidate_events"]],
                ["抑制：同脚不应期", "−%d" % stats["suppressed_foot"]],
                ["抑制：同人不应期", "−%d" % stats["suppressed_person"]],
                ["球门控 / 速度门控拒绝", "%d / %d" % (stats["rejected_ball_gate"],
                                                       stats["rejected_speed_gate"])],
                ["**最终事件**", "**%d**" % stats["events"]],
            ]),
            "",
            *table(["簇的形状", "最小 / 中位 / 最大"], [
                ["簇长（帧）", "%d / %s / %d" % (stats["length_min"], stats["length_median"],
                                                 stats["length_max"])],
                ["簇内接触帧数", "— / %s / %d" % (stats["contact_median"], stats["contact_max"])],
                ["一个簇出 2 个事件的簇", "%d 个" % stats["double_clusters"]],
            ]),
            "",
            "> 插桩重放得到的最终事件数、两个不应期计数与 `touch_report.json` 一致（本次 %d / %d / %d）。"
            % (stats["events"], main_score["suppressed_by_refractory"],
               main_score["suppressed_by_person_refractory"]),
            "",
        ]
    threshold_counts = collections.Counter(
        int(round(event["threshold_px"])) for event in report["events"])
    lines += [
        "## 7. 事件级阈值 τ 的分布",
        "",
        *table(["τ (px)"] + [str(key) for key in sorted(threshold_counts)],
               [["事件数"] + [threshold_counts[key] for key in sorted(threshold_counts)]]),
        "",
        "> 两个台阶分别对应「人框高项 / 下限（22 px）」与「画面短边 × 0.045（720p ≈ 32.4 px）」。",
        "",
        "## 8. 本目录产物",
        "",
        "```",
        "observations.jsonl   基模逐帧输出        touch_report.json   评估报告（本文件的来源）",
        "touch_events.jsonl   事件表              contact_signal.csv  逐帧接触信号",
        "rendered.mp4         基模渲染视频        metrics.md/metrics.csv  本指标汇总",
        "tp_fp_fn_detail.csv  逐条明细            timeline.csv        每行标注的核对表",
        "pr_review/           tp 79 / fp 18 / fn 15 / low_quality 10 张图 + timeline.md/csv",
        "                     + touch_playback.mp4（整段实时，底部 pred/gt 两条轨道）",
        "```",
        "",
        "## 9. 复现",
        "",
        "```bash",
        "# 1) 判定 + 评估（观测来自基模那一遍推理；路径是绝对路径，在哪跑都对）",
        "cd examples/touch_eval/py",
        "uv run touch-eval --observations %s \\" % (output_dir / "observations.jsonl").resolve(),
        "                  --touches_csv %s \\" % annotations_path.resolve(),
        "                  --output_dir %s" % output_dir.resolve(),
        "# 2) 指标汇总（本文件）",
        "uv run --project . python3 ../tools/write_metrics_summary.py --output-dir %s"
        % output_dir.resolve(),
        "# 3) 复核图与播放视频",
        "uv run --project . --extra tools python3 ../tools/render_review_folders.py \\",
        "       --output-dir %s --playback" % output_dir.resolve(),
        "```",
        "",
    ]
    (output_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[metrics] 写好了 %s 与 %s" % (output_dir / "metrics.md", output_dir / "metrics.csv"))
    print("[metrics] 对外口径：容量 %d｜TP %d FP %d FN %d｜P %.4f R %.4f F1 %.4f"
          % (official["capacity"], official["true_positives"], official["false_positives"],
             official["false_negatives"], official["precision"], official["recall"],
             official["f1"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
