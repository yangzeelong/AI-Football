#!/usr/bin/env python3
"""ball_cone_v1 三分类数据集的路径、类别与"三方对拍"校验。

唯一真相源是 `classes.json`；本模块把它翻译成训练/导出/部署三方都要对齐的东西：

* 数据集 roboflow 三目录（`train/valid/test` 各自的 `_annotations.coco.json`）
* 训练时的 `class_names`（`.train(class_names=...)`，否则 num_classes 不生效）
* 导出/部署时的类别号与列数（列数 = max(category_id)+1，第 0 列是背景）
* 两侧管线的 YAML 值（`numClasses/targetClasses/personClassId/ballClassId/coneClassId`）

被 s1_data_extract_frames.py / s2_data_preannotate.py / s3_label_convert_dataset.py / s2a_data_compose_cones.py /
s4a_train_person_pseudo.py / s4b_train_person_coco.py / s4_train_finetune.py 等复用，避免任何脚本
自己硬编码类别。

用法：
    python base_layout.py --check        # 检查目录骨架 + 打印三方对齐值
    python base_layout.py --init         # 建目录骨架与 manifest/ledger 表头（幂等）
    python base_layout.py --selftest     # 自检对拍函数能抓住错误
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TOOL_DIR = Path(__file__).resolve().parent
CLASSES_JSON = TOOL_DIR / "classes.json"

# --- 根路径：全部可推导，环境变量只作覆盖 ------------------------------------
# 约定布局（三个目录**相邻**即可，换机器/换目录不用改代码）：
#   <WORKSPACE>/AI-Football    ← 本仓库
#   <WORKSPACE>/Nexusflow      ← C++ / TensorRT 侧
#   <WORKSPACE>/data           ← 视频与数据集
# 默认值都从本文件位置推出；要拆开布局就用下面的环境变量覆盖。
REPO_ROOT = TOOL_DIR.parents[1]                      # …/AI-Football


def _root(env: str, default: Path) -> Path:
    """环境变量优先，否则用推导出的默认值（空字符串视为未设置）。"""
    raw = os.environ.get(env, "").strip()
    return Path(raw).expanduser().resolve() if raw else default.resolve()


WORKSPACE = _root("AI_FOOTBALL_WORKSPACE", REPO_ROOT.parent)
NEXUSFLOW_ROOT = _root("AI_FOOTBALL_NEXUSFLOW", WORKSPACE / "Nexusflow")
#: 视频与数据集所在的 `data/` 目录
DATA_ROOT = _root("AI_FOOTBALL_DATA_DIR", WORKSPACE / "data")

# --- 数据根（不进 git）--------------------------------------------------------
#: **命名规则（唯一口径，新素材一律照这个来）**
#:
#: | 前缀          | 含义                                  | 目录                     | manifest source | 生成脚本                | 进训练 |
#: | ---           | ---                                   | ---                      | ---             | ---                     | ---    |
#: | `raw_video`   | 现场视频抽帧（原始数据）              | `sources/raw_video/`     | raw_video       | `s1_data_extract_frames.py`  | ✅     |
#: | `coco`        | COCO 数据集里来的                     | `sources/coco_person/`   | coco_person     | `s4b_train_person_coco.py`    | ✅(兜底)|
#: | `ai_gen_bg`   | AI 生成的空场地背景（**素材池**）     | `sources/ai_gen_bg/`     | ai_gen_bg       | `s2b_data_gen_backgrounds.py`      | ❌     |
#: | `spec`        | 标注规范对照示例（画了框的示例图）    | `sources/spec_examples/` | spec_example    | `s2c_data_spec_stress.py`  | ❌     |
#: | `stress`      | 分布外压力集（难帧队列 + 入选原因）   | `sources/stress_set/`    | stress_set      | `s2c_data_spec_stress.py`  | ❌     |
#: | `pseudo`      | 模型伪标签产物                        | `sources/pseudo_person/` | pseudo_person   | `s4a_train_person_pseudo.py`| ✅(兜底)|
#: | `ai_cut`      | 脚本抠图**粘贴**到真实背景（GT 精确） | `work/ai_cut/`           | ai_cut          | `s2a_data_compose_cones.py`       | ✅     |
#: | `ai_bg_cut`   | 脚本抠图粘贴到 **AI 背景**（GT 精确） | `work/ai_bg_cut/`        | ai_bg_cut       | `s2a_data_compose_cones.py`       | ✅     |
#: | `web`         | 网络/公开渠道采集（需许可台账）       | `sources/web/`           | web             | `s1_data_collect_web.py`        | ✅(需许可)|
#:
#: **脚本命名规则**：产出素材的脚本按它产出的前缀命名（上表最后一列）；流程步骤脚本仍用动词
#: （`s2_data_preannotate.py` `s3_label_review_roundtrip.py` `s3_label_convert_dataset.py` `s3_label_dataset_stats.py` `s4_train_finetune.py`
#: `s6_deploy_export.py` `s5_eval_detector.py` `recycle_hard_examples.py`）。
#:
#: 三层结构：`dataset/`（① 最终数据，可直接训练）→ `sources/`（② 素材，原样保存）
#: → `work/`（③ 中间产物，可整目录删掉重建）。顶层另放 `manifest.csv`、`ledger.csv`、`reports/`。
#: 数据集本体（默认 `data/ball_cone_v1`）；指到别处就设 AI_FOOTBALL_DATASET_DIR
DATASET_ROOT = _root("AI_FOOTBALL_DATASET_DIR", DATA_ROOT / "ball_cone_v1")

# ① 最终数据（roboflow 三目录）
DATASET_DIR = DATASET_ROOT / "dataset"
# ② 素材来源（原样保存，不直接训练；命名前缀见上表）
SOURCES_DIR = DATASET_ROOT / "sources"
RAW_VIDEO_DIR = SOURCES_DIR / "raw_video"          # raw_video_* 现场视频抽帧
AI_GEN_BG_DIR = SOURCES_DIR / "ai_gen_bg"          # ai_gen_bg_* AI 空场地背景（粘贴用素材池）
SPEC_EXAMPLE_DIR = SOURCES_DIR / "spec_examples"  # spec_* 标注规范对照示例（只进文档）
STRESS_DIR = SOURCES_DIR / "stress_set"          # stress_* 压力集（难帧队列 + 原因，不进训练）
COCO_PERSON_DIR = SOURCES_DIR / "coco_person"      # coco_person_* COCO 人体子集
PSEUDO_PERSON_DIR = SOURCES_DIR / "pseudo_person"  # pseudo_person_* 伪标签
# ③ 中间工作区（可整目录删掉重生成）
WORK_DIR = DATASET_ROOT / "work"
ANNOTATION_DIR = WORK_DIR / "annotations"          # GT.json（人工 GT）/ preannot.json（机器基线）

#: **标注命名（唯一口径）**：
#: | 文件 | 含义 | 谁写 |
#: | --- | --- | --- |
#: | `GT.json` | **人工复核后的权威标注**（全量、可直接训练） | `s3_label_review_roundtrip.py --import` |
#: | `preannot.json` | 机器预标注基线（base 模型 + 锥候选清单） | `s2_data_preannotate.py` |
#: | `GT_reviewed.json` | 已复核帧台账（清单） | `s3_label_review_roundtrip.py --import` |
#: | `GT_excluded.json` | 人工判废清单（整图不进任何 split） | `s3_label_review_roundtrip.py --import` |
GT_JSON = ANNOTATION_DIR / "GT.json"
PREANNOT_JSON = ANNOTATION_DIR / "preannot.json"
GT_REVIEWED_JSON = ANNOTATION_DIR / "GT_reviewed.json"
GT_EXCLUDED_TXT = ANNOTATION_DIR / "GT_excluded.txt"
#: 旧名（本次改名之前）：读到就认，保证老目录/老流程不炸
LEGACY_GT_JSON = ANNOTATION_DIR / "review.json"
LEGACY_REVIEWED_JSON = ANNOTATION_DIR / "reviewed.json"
LEGACY_EXCLUDED_TXT = ANNOTATION_DIR / "excluded.txt"


def first_existing(*paths: Path) -> Path | None:
    """返回第一个存在的路径（None = 都不存在）。"""
    return next((p for p in paths if p.exists()), None)


def gt_annotation_path(fallback_preannot: bool = True) -> Path | None:
    """人工 GT 路径：`GT.json` → 旧名 `review.json` → `preannot.json`（bootstrap 基线）。"""
    candidates = [GT_JSON, LEGACY_GT_JSON] + ([PREANNOT_JSON] if fallback_preannot else [])
    return first_existing(*candidates)


def reviewed_ledger_path() -> Path:
    return first_existing(GT_REVIEWED_JSON, LEGACY_REVIEWED_JSON) or GT_REVIEWED_JSON


def excluded_path() -> Path:
    return first_existing(GT_EXCLUDED_TXT, LEGACY_EXCLUDED_TXT) or GT_EXCLUDED_TXT
CUTOUT_DIR = WORK_DIR / "cutouts"                  # 锥素材库（抠图 RGBA）
CUTOUT_META = CUTOUT_DIR / "cutouts.json"
AI_CUT_DIR = WORK_DIR / "ai_cut"                   # 真实背景 + 粘贴（GT 精确，进训练）
AI_CUT_IMAGE_DIR = AI_CUT_DIR / "images"
AI_CUT_ANNOTATIONS = AI_CUT_DIR / "annotations.json"
AI_BG_CUT_DIR = WORK_DIR / "ai_bg_cut"             # AI 背景 + 粘贴（GT 精确，进训练）
AI_BG_CUT_IMAGE_DIR = AI_BG_CUT_DIR / "images"
AI_BG_CUT_ANNOTATIONS = AI_BG_CUT_DIR / "annotations.json"
PROMPT_DIR = WORK_DIR / "prompts"                  # 三类 AI 提示词包
# 报告与台账
REPORTS_DIR = DATASET_ROOT / "reports"
REVIEW_BATCH_DIR = REPORTS_DIR / "review_batches"  # 待标包 zip
MANIFEST_CSV = DATASET_ROOT / "manifest.csv"
LEDGER_CSV = DATASET_ROOT / "ledger.csv"

#: 进训练集的来源（用于"合成占比"等统计）
TRAINABLE_SOURCES: tuple[str, ...] = ("raw_video", "ai_cut", "ai_bg_cut", "coco_person",
                                     "pseudo_person", "web")
#: 合成样本（用于 ≤30% 上限统计）
COMPOSITE_SOURCES: tuple[str, ...] = ("ai_cut", "ai_bg_cut")

#: roboflow 布局必需的三个 split（`run_test` 默认开启，缺 test/ 会构建失败）
SPLITS: tuple[str, ...] = ("train", "valid", "test")

#: 逐帧来源登记：谁、哪段视频、第几帧、什么场景/视角、落在哪个 split、是否合成/伪标签
MANIFEST_FIELDS: tuple[str, ...] = (
    "image",           # 相对 ball_cone_v1 的图片路径
    "source",          # raw_video | ai_cut | ai_bg_cut | ai_gen_bg | spec_example |
                       # stress_set | web | coco_person | pseudo_person（见文件顶部命名规则表）
    "source_video",    # 原始视频相对 /home/hx1/yzl/Work/data 的路径（合成图为背景图来源）
    "frame_index",     # 源视频帧号（合成图为背景帧号）
    "time_sec",        # 源视频时间戳
    "width", "height", # 原始分辨率（分辨率不统一缩放，训练侧统一 letterbox 到 960）
    "scene",           # indoor_gym | outdoor_pitch | ...
    "view",            # 视角/机位（wide | sideline | handheld | ...）
    "split",           # train | valid | test
    "is_synthetic",    # 合成锥桶样本
    "ai_background",   # 合成图所用背景是否 AI 生成（yes/no；单列统计用）
    "is_pseudolabel",  # person 伪标签
    "license",         # 许可（自有素材 | 自有合成 | AI 生成背景 | CC-BY-4.0 | ...）
    "note",
)

#: 采集台账（网络/公开数据）
LEDGER_FIELDS: tuple[str, ...] = (
    "name", "url", "fetched_at", "license", "redistributable", "n_images", "note",
)


# --- 类别真相源 ---------------------------------------------------------------


def load_classes(path: Path | None = None) -> list[dict[str, Any]]:
    """读 classes.json 的 categories（顺序即类别顺序）。"""
    raw = json.loads((path or CLASSES_JSON).read_text(encoding="utf-8"))
    cats = raw["categories"]
    if not cats:
        raise ValueError(f"{path or CLASSES_JSON} 的 categories 为空")
    return cats


def categories(path: Path | None = None) -> list[dict[str, str | int]]:
    """COCO 用的 categories：只保留 id/name/supercategory。

    两条硬约束：
    * `id` 必须 **0-based 连续**（RF-DETR 用 `len(categories)` 当 num_classes，
      logits 最后一列是 no-object，写成 1..N 会在训练时 target 越界触发
      CUDA device-side assert）；
    * `supercategory` 不能写 "none"（`detr.py:133` 会把 supercategory=="none" 的类
      从 class_names 里滤掉，导致类别名比 num_classes 少）。
    """
    out = []
    for cat in load_classes(path):
        name = cat["name"]
        out.append({"id": int(cat["id"]), "name": name,
                    "supercategory": str(cat.get("supercategory") or name)})
    return out


def class_names(path: Path | None = None) -> list[str]:
    """训练时必须显式传给 `.train(class_names=...)` 的名字列表。

    与 `rfdetr/detr.py:133` 的过滤规则一致：排除 `supercategory == "none"` 的占位类。
    """
    return [str(cat["name"]) for cat in categories(path)
            if str(cat.get("supercategory")) != "none"]


def name_to_id(path: Path | None = None) -> dict[str, int]:
    return {str(cat["name"]): int(cat["id"]) for cat in categories(path)}


def id_to_name(path: Path | None = None) -> dict[int, str]:
    return {int(cat["id"]): str(cat["name"]) for cat in categories(path)}


def num_columns(path: Path | None = None) -> int:
    """模型输出 logits 的列数 = `len(categories)`（含 id=0 的占位背景类）。

    含 3 个真实类别时是 4 列；部署侧 `RFDetrDetectorInfer.cpp:134-137` 直接从
    logits 维度读 numClasses，所以 YAML 里那个值只是书面一致。
    """
    return len(categories(path))


def deploy_class_id(name: str, path: Path | None = None) -> int:
    """部署侧 classId：**就是 category_id**。

    `RFDetrDetectorInfer.cpp:751-752` 是 `Candidate{Sigmoid(logit[c]), q, c}` ——
    直接用列下标当 classId，不加 1；Python 侧 `detr.py:216` 的 class_names 映射
    才是 `{i+1: name}`（这正是我们把 id 排成 1..3 的原因）。
    """
    return name_to_id(path)[name]


def pipeline_expectations(path: Path | None = None) -> dict[str, Any]:
    """两侧管线该写的值（Nexusflow YAML / 文档对拍用）。"""
    return {
        "numClasses": num_columns(path),
        "targetClasses": [deploy_class_id(n, path) for n in ("person", "ball", "cone")],
        "personClassId": deploy_class_id("person", path),
        "ballClassId": deploy_class_id("ball", path),
        "coneClassId": deploy_class_id("cone", path),
        "class_names": class_names(path),
        "dataset_category_ids": [c["id"] for c in categories(path)],
    }


# --- 路径 ---------------------------------------------------------------------


def split_dir(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split 必须是 {SPLITS} 之一，收到 {split!r}")
    return DATASET_DIR / split


def split_images(split: str) -> Path:
    return split_dir(split) / "images"


def split_annotations(split: str) -> Path:
    return split_dir(split) / "_annotations.coco.json"


def ensure_layout() -> list[Path]:
    """建目录骨架与 manifest/ledger 表头（幂等）。返回新建/确认过的目录。"""
    dirs = [DATASET_ROOT, DATASET_DIR, SOURCES_DIR, RAW_VIDEO_DIR, AI_GEN_BG_DIR,
            SPEC_EXAMPLE_DIR, STRESS_DIR, COCO_PERSON_DIR, PSEUDO_PERSON_DIR,
            WORK_DIR, ANNOTATION_DIR, CUTOUT_DIR, AI_CUT_DIR, AI_CUT_IMAGE_DIR,
            AI_BG_CUT_DIR, AI_BG_CUT_IMAGE_DIR, PROMPT_DIR, REPORTS_DIR,
            REVIEW_BATCH_DIR] + [split_images(s) for s in SPLITS]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
    for csv_path, fields in ((MANIFEST_CSV, MANIFEST_FIELDS), (LEDGER_CSV, LEDGER_FIELDS)):
        if not csv_path.exists():
            csv_path.write_text(",".join(fields) + "\n", encoding="utf-8")
    ensure_manifest_schema()
    return dirs


def ensure_manifest_schema() -> int:
    """给已存在的 manifest.csv 补上新增列（保留行序与内容）。返回补了几行。"""
    import csv as _csv

    if not MANIFEST_CSV.exists() or MANIFEST_CSV.stat().st_size == 0:
        return 0
    with MANIFEST_CSV.open(encoding="utf-8") as handle:
        rows = list(_csv.DictReader(handle))
        header = list(rows[0].keys()) if rows else list(MANIFEST_FIELDS)
    if header == list(MANIFEST_FIELDS):
        return 0
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
    return len(rows)


def naming_table_md() -> str:
    """命名规则 markdown 表（README / 目录树 / 文档共用同一份，别各写各的）。"""
    rows = [
        ("`raw_video_*`", "现场视频抽帧（**原始数据**）", "`sources/raw_video/`", "raw_video",
         "`s1_data_extract_frames.py`", "✅"),
        ("`coco_*`", "COCO 数据集里来的", "`sources/coco_person/`", "coco_person",
         "`s4b_train_person_coco.py`", "✅（兜底③）"),
        ("`ai_gen_bg_*`", "AI 生成的空场地背景（**素材池**）", "`sources/ai_gen_bg/`", "ai_gen_bg",
         "`s2b_data_gen_backgrounds.py`", "❌（只作粘贴背景）"),
        ("`spec_*`", "标注规范对照示例（画框示例图）", "`sources/spec_examples/`", "spec_example",
         "`s2c_data_spec_stress.py`", "❌"),
        ("`stress_*`", "分布外压力集（难帧队列 + 原因）", "`sources/stress_set/`", "stress_set",
         "`s2c_data_spec_stress.py`", "❌"),
        ("`pseudo_person_*`", "模型伪标签", "`sources/pseudo_person/`", "pseudo_person",
         "`s4a_train_person_pseudo.py`", "✅（兜底①）"),
        ("**`ai_cut_*`**", "抠图**粘贴到真实背景**（GT 精确）", "`work/ai_cut/`", "ai_cut",
         "`s2a_data_compose_cones.py`", "✅"),
        ("**`ai_bg_cut_*`**", "**AI 背景** + 抠图粘贴（GT 精确）", "`work/ai_bg_cut/`", "ai_bg_cut",
         "`s2a_data_compose_cones.py`", "✅"),
        ("`web_*`", "网络/公开渠道采集（需许可台账）", "`sources/web/`", "web",
         "`s1_data_collect_web.py`", "✅（需许可）"),
    ]
    head = ("| 前缀 | 含义 | 目录 | manifest `source` | 生成脚本 | 进训练 |\n"
            "| --- | --- | --- | --- | --- | --- |")
    return "\n".join([head] + ["| %s | %s | %s | `%s` | %s | %s |" % r for r in rows])


def _dataset_counts() -> list[str]:
    """从 dataset/ 与 manifest 现算规模，写进 README（别写死数字）。"""
    lines = []
    for split in SPLITS:
        coco = split_annotations(split)
        if not coco.exists():
            lines.append("- `%s`：尚未生成" % split)
            continue
        data = json.loads(coco.read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in data.get("categories", [])}
        counts = collections.Counter(names.get(a["category_id"], "?")
                                     for a in data.get("annotations", []))
        lines.append("- `dataset/%s`：**%d 图 / %d 框**（%s）"
                     % (split, len(data.get("images", [])), len(data.get("annotations", [])),
                        " · ".join("%s %d" % (k, v) for k, v in sorted(counts.items()))))
    if MANIFEST_CSV.exists():
        with MANIFEST_CSV.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        by_source = collections.Counter(str(r.get("source", "")) for r in rows)
        lines.append("- manifest：**%d 行**（%s）" % (
            len(rows), " · ".join("%s %d" % (k, v) for k, v in sorted(by_source.items()))))
        composite = sum(by_source.get(s, 0) for s in COMPOSITE_SOURCES)
        if rows:
            lines.append("- 合成占比：**%.1f%%**（%d / %d；上限 30%%）"
                         % (100.0 * composite / len(rows), composite, len(rows)))
    return lines


def write_dataset_readme() -> Path:
    """写数据根 README.md（自解释：命名规则 / 三层结构 / 当前规模 / 重建命令）。"""
    tree = "\n".join("  %s%s  %s" % ("✓ " if (DATASET_ROOT / name).exists() else "· ", name + "/",
                                    text.split("。")[0][:70])
                     for name, text in (("dataset", "① 最终数据（可直接训练）"),
                                        ("sources", "② 素材（原样保存，不直接训练）"),
                                        ("work", "③ 中间产物（可整目录删掉重建）"),
                                        ("reports", "报告与待标包")))
    text = """# ball_cone_v1 —— 人 / 足球 / 锥形桶 三分类数据集

为"锥形桶被误检成足球"这类误检做的自建数据集；工具集在
`AI-Football/tools/detection_finetune/`，完整方案见 `AI-Football/doc/detection_finetune_plan.md`。
**这个目录不进 git**（`AI-Football/.gitignore` 已忽略 `data`）。

## 命名规则（唯一口径）

{_naming_table}
**脚本命名规则**：产出来源的脚本按它产出的前缀命名（上表"生成脚本"列）；流程步骤脚本仍用动词
（`s2_data_preannotate.py` `s3_label_review_roundtrip.py` `s3_label_convert_dataset.py` `s3_label_dataset_stats.py` `s4_train_finetune.py`
`s6_deploy_export.py` `s5_eval_detector.py` `recycle_hard_examples.py`）。

## 三层结构

```
{_tree}
```

* **`dataset/`** 是唯一"能直接喂训练/评测"的地方，由 `s3_label_convert_dataset.py` 生成，可整目录重建。
* **`sources/`** 原样保存素材，只读；素材变了要重建 `dataset/`。
* **`work/`** 是中间产物：标注、抠图素材、粘贴合成、提示词。删了能重建（合成是确定性的：同 `--seed` 出同样样本）。
* 每个目录里都有 `_README.md`（由 `base_layout.py --readmes` 写）。

## 当前规模

{_counts}

## 常用命令

```bash
cd AI-Football/tools/detection_finetune && source env.sh   # 之后用 $PY

$PY base_layout.py --tree                  # 看目录结构 + 每目录一句话说明
$PY base_layout.py --check                 # 四类对拍（classes.json vs 数据集 vs 两侧配置）
$PY s3_label_dataset_stats.py                  # 数据体检（含来源分层、目标尺寸分布）
$PY s2a_data_compose_cones.py --qa --n 12       # 合成样本抽检拼图（人工只抽 10%，查粘贴质量）
$PY s3_label_review_roundtrip.py --status      # S3 人工校验进度
$PY s3_label_review_ui.py                      # S3 复核界面（零依赖，一次载入全部批次）
$PY s3_label_convert_dataset.py --link         # 重建 dataset/（默认优先用 work/annotations/GT.json）
```

## 不要手改的东西

* `dataset/**`：重建会覆盖；要改标注请改 `work/annotations/*.json` 或走 S3 往返流程。
* `manifest.csv`：由脚本登记（抽帧 / 合成 / 校验时写），手改会丢追踪关系。
* `ledger.csv`：许可台账，新增外部素材（网络/公开数据集/AI 生成）必须登记。

## 判废与例外

人工校验时判"这张图不要了"的（合成图悬空/越界/压人、真实帧标不出来）→ 记在
`work/annotations/GT_excluded.txt`（`s3_label_review_roundtrip.py --import` 自动写），`s3_label_convert_dataset.py` 会跳过这些图。
""".format(_naming_table=naming_table_md(), _tree=tree, _counts="\n".join(_dataset_counts()))
    path = DATASET_ROOT / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def missing_layout() -> list[Path]:
    return [p for p in [DATASET_DIR, SOURCES_DIR, RAW_VIDEO_DIR, AI_GEN_BG_DIR,
                        SPEC_EXAMPLE_DIR, STRESS_DIR, COCO_PERSON_DIR, WORK_DIR,
                        ANNOTATION_DIR, CUTOUT_DIR,
                        AI_CUT_DIR, AI_BG_CUT_DIR, PROMPT_DIR, REPORTS_DIR,
                        MANIFEST_CSV, LEDGER_CSV]
            + [split_images(s) for s in SPLITS] if not p.exists()]


# --- 目录命名：迁移与自解释 -----------------------------------------------------

#: 旧 → 新 目录映射；`python base_layout.py --migrate` 会搬目录并改写文件里的路径前缀
MIGRATION_MAP: tuple[tuple[str, str], ...] = (
    ("raw", "sources/raw_video"),                 # raw_video_* 现场视频抽帧
    ("sources/video", "sources/raw_video"),
    ("synthetic/cutouts", "work/cutouts"),         # 锥素材库
    ("synthetic/images", "work/ai_cut/images"),    # 真实背景粘贴（AI 背景那批由 SPLIT_COMPOSITES 分出去）
    ("synthetic/ai_backgrounds", "sources/ai_gen_bg"),
    ("synthetic/ai_spec_examples", "sources/spec_examples"),
    ("sources/ai_background", "sources/ai_gen_bg"),
    ("sources/ai_gen_spec", "sources/spec_examples"),
    ("sources/ai_spec", "sources/spec_examples"),
    ("ai_stress", "sources/stress_set"),
    ("sources/ai_gen_stress", "sources/stress_set"),
    ("sources/ai_stress", "sources/stress_set"),
    ("coco_person", "sources/coco_person"),
    ("sources/public_person", "sources/coco_person"),
    ("pseudolabel", "sources/pseudo_person"),
    ("sources/pseudolabel", "sources/pseudo_person"),
    ("annotations", "work/annotations"),
    ("synthetic", "work/ai_cut"),                  # 兜底：剩下的散件（json）跟着走
    ("work/synthetic", "work/ai_cut"),
    ("train", "dataset/train"),
    ("valid", "dataset/valid"),
    ("test", "dataset/test"),
)

#: 文件内路径前缀改写（顺序敏感：长前缀在前，避免被短前缀先吃掉）
PATH_REWRITES: tuple[tuple[str, str], ...] = (
    ("synthetic/cutouts/", "work/cutouts/"),
    ("synthetic/images/", "work/ai_cut/images/"),
    ("synthetic/ai_backgrounds/", "sources/ai_gen_bg/"),
    ("synthetic/ai_spec_examples/", "sources/spec_examples/"),
    ("work/synthetic/images/", "work/ai_cut/images/"),
    ("sources/video/", "sources/raw_video/"),
    ("raw/", "sources/raw_video/"),
    ("sources/ai_background/", "sources/ai_gen_bg/"),
    ("sources/ai_gen_spec/", "sources/spec_examples/"),
    ("sources/ai_spec/", "sources/spec_examples/"),
    ("sources/ai_gen_stress/", "sources/stress_set/"),
    ("sources/ai_stress/", "sources/stress_set/"),
    ("ai_stress/", "sources/stress_set/"),
    ("sources/public_person/", "sources/coco_person/"),
    ("coco_person/", "sources/coco_person/"),
    ("sources/pseudolabel/", "sources/pseudo_person/"),
    ("pseudolabel/", "sources/pseudo_person/"),
)

#: manifest `source` 取值改名（与目录前缀一一对应）
SOURCE_RENAMES: dict[str, str] = {
    "video": "raw_video",
    "synthetic": "ai_cut",          # 具体是 ai_cut 还是 ai_bg_cut 由 ai_background 列决定
    "ai_background": "ai_gen_bg",
    "ai_stress": "stress_set",
    "ai_gen_stress": "stress_set",
    "ai_spec": "spec_example",
    "ai_gen_spec": "spec_example",
    "pseudolabel": "pseudo_person",
    "crawl": "raw_video",           # 旧的网络采集占位值
}


def rewrite_path_prefixes(text: str) -> tuple[str, int]:
    """把文本里的旧目录前缀换成新前缀；已迁移过的（前面已是 sources/、work/、dataset/）不动。"""
    import re as _re

    changed = 0
    for old, new in PATH_REWRITES:
        pattern = _re.compile(r"(?<!sources/)(?<!work/)(?<!dataset/)" + _re.escape(old))
        text, n = pattern.subn(new, text)
        changed += n
    return text, changed


def _merge_move(src: Path, dst: Path) -> int:
    """把 src 搬到 dst；dst 已存在时合并（同名不覆盖）。返回搬动条目数。"""
    import shutil as _shutil

    if not src.exists():
        return 0
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        _shutil.move(str(src), str(dst))
        return 1
    moved = 0
    for child in list(src.iterdir()):
        target = dst / child.name
        if target.exists():
            # 同名目录必须**递归合并**（早先直接跳过，导致 work/synthetic/images/* 整个没搬过去）
            if child.is_dir() and target.is_dir():
                moved += _merge_move(child, target)
            continue
        if child.is_dir():
            moved += _merge_move(child, target)
        else:
            _shutil.move(str(child), str(target))
            moved += 1
    if not any(src.iterdir()):
        src.rmdir()
    return moved


def _read_manifest() -> list[dict[str, str]]:
    import csv as _csv

    if not MANIFEST_CSV.exists():
        return []
    with MANIFEST_CSV.open(encoding="utf-8") as handle:
        return list(_csv.DictReader(handle))


def _write_manifest(rows: Sequence[dict[str, Any]]) -> None:
    import csv as _csv

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def _split_composites(actions: list[tuple[str, str, str]]) -> None:
    """合成样本按背景来源拆两处：`ai_cut`（真实背景）/ `ai_bg_cut`（AI 背景）。"""
    import json as _json
    import shutil as _shutil

    rows = _read_manifest()
    if not rows:
        return
    moved = 0
    for row in rows:
        if not str(row.get("source", "")).startswith(("synthetic", "ai_cut", "ai_bg_cut")):
            continue
        is_ai_bg = str(row.get("ai_background", "")).lower() in ("yes", "1", "true")
        target_dir = AI_BG_CUT_DIR if is_ai_bg else AI_CUT_DIR
        basename = Path(row["image"]).name
        new_rel = str((target_dir / "images" / basename).relative_to(DATASET_ROOT))
        dst = DATASET_ROOT / new_rel
        if not dst.exists():
            # 实体可能还在另一个桶里（早先只改写文本、没搬文件）→ 两个桶都找一遍
            found = None
            for cand in (AI_CUT_IMAGE_DIR, AI_BG_CUT_IMAGE_DIR):
                if (cand / basename).exists():
                    found = cand / basename
                    break
            if found is not None:
                dst.parent.mkdir(parents=True, exist_ok=True)
                _shutil.move(str(found), str(dst))
                moved += 1
            else:
                actions.append(("missing", basename, new_rel))
        row["image"] = new_rel
        row["source"] = "ai_bg_cut" if is_ai_bg else "ai_cut"
    _write_manifest(rows)
    actions.append(("split %d 张合成图" % moved, "work/ai_cut", "work/ai_bg_cut"))

    # 标注文件也拆成两份：按 **manifest 的 ai_background 列**判定（比猜 file_name 可靠，且可重复执行）
    candidates = [AI_CUT_DIR / "annotations_synth.json", AI_CUT_ANNOTATIONS, AI_BG_CUT_ANNOTATIONS]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return
    is_ai_bg_of = {
        Path(row["image"]).name: str(row.get("ai_background", "")).lower() in ("yes", "1", "true")
        for row in rows
        if str(row.get("source", "")).startswith(("synthetic", "ai_cut", "ai_bg_cut"))
    }
    # 多个输入文件（旧单文件 + 已拆过的两桶）的 id 空间是**互相冲突**的（都从 1 开始），
    # 所以逐文件加命名空间偏移后再合并，否则同一 image_id 会把别的桶的标注也吸过来（实测 cone 从 433 涨到 1889）。
    merged_images: dict[str, dict[str, Any]] = {}
    categories: list[dict[str, Any]] = []
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for file_index, path in enumerate(existing):
        data = _json.loads(path.read_text(encoding="utf-8"))
        categories = data.get("categories") or categories
        offset = file_index * 10 ** 7
        for img in data.get("images", []):
            merged_images[Path(img["file_name"]).name] = {**img, "id": img["id"] + offset}
        for ann in data.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"] + offset, []).append(
                {**ann, "image_id": ann["image_id"] + offset})
    buckets: dict[str, dict[str, Any]] = {
        key: {"images": [], "annotations": [], "categories": categories}
        for key in ("ai_cut", "ai_bg_cut")
    }
    for basename, img in sorted(merged_images.items()):
        key = "ai_bg_cut" if is_ai_bg_of.get(basename) else "ai_cut"
        bucket = buckets[key]
        prefix = str((AI_BG_CUT_IMAGE_DIR if key == "ai_bg_cut" else AI_CUT_IMAGE_DIR)
                     .relative_to(DATASET_ROOT))
        new_id = len(bucket["images"]) + 1
        bucket["images"].append({**img, "id": new_id, "file_name": "%s/%s" % (prefix, basename)})
        for ann in anns_by_image.get(img["id"], []):
            bucket["annotations"].append({**ann, "id": len(bucket["annotations"]) + 1, "image_id": new_id})
    for key, path in (("ai_cut", AI_CUT_ANNOTATIONS), ("ai_bg_cut", AI_BG_CUT_ANNOTATIONS)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(buckets[key], ensure_ascii=False), encoding="utf-8")
        actions.append(("annotations %d 图" % len(buckets[key]["images"]), key, str(path.name)))
    for stale in (AI_CUT_DIR / "annotations_synth.json",):
        stale.unlink(missing_ok=True)


def migrate_layout(apply: bool = True) -> list[tuple[str, str, str]]:
    """按 `MIGRATION_MAP` 搬目录，改写路径前缀与 `source` 取值，并拆分合成样本。

    返回 [(动作, 旧, 新)] 明细；`apply=False` 时只做演练（dry-run）。
    """
    ensure_layout()
    actions: list[tuple[str, str, str]] = []
    for old, new in MIGRATION_MAP:
        src, dst = DATASET_ROOT / old, DATASET_ROOT / new
        if not src.exists() or old == new:
            continue
        if apply:
            n = _merge_move(src, dst)
            actions.append(("moved %d 项" % n, old, new))
        else:
            actions.append(("would move", old, new))
    if not apply:
        return actions

    # 文件内路径改写
    targets = [MANIFEST_CSV]
    for folder in (ANNOTATION_DIR, CUTOUT_DIR, AI_CUT_DIR, AI_BG_CUT_DIR, PROMPT_DIR):
        if folder.exists():
            targets += [p for p in folder.rglob("*.json")]
    for path in targets:
        if not path.exists() or path.suffix not in (".csv", ".json"):
            continue
        text = path.read_text(encoding="utf-8")
        new_text, n = rewrite_path_prefixes(text)
        if n:
            path.write_text(new_text, encoding="utf-8")
            actions.append(("rewrote %d 处路径" % n, str(path.relative_to(DATASET_ROOT)), ""))

    # source 取值改名（与目录前缀一一对应），再拆分合成样本
    rows = _read_manifest()
    renamed = 0
    for row in rows:
        old = str(row.get("source", ""))
        if old in SOURCE_RENAMES:
            row["source"] = SOURCE_RENAMES[old]
            renamed += 1
    if renamed:
        _write_manifest(rows)
        actions.append(("renamed %d 行 source" % renamed, "manifest.csv", ""))
    _split_composites(actions)
    return actions


#: 每个目录的自解释说明（`--readmes` 写成 `_README.md`，解决"这目录干什么用的"）
DIR_HELP: tuple[tuple[str, str], ...] = (
    ("dataset", "① **最终数据**（可直接训练）：`train/ valid/ test/`。"
                "由 `s3_label_convert_dataset.py` 生成，可整目录重建。"),
    ("dataset/train", "训练集（`raw_video_*` 真实抽帧 + `ai_cut_*`/`ai_bg_cut_*` 合成）。"),
    ("dataset/valid", "验证集（纯真实，不掺合成）。"),
    ("dataset/test", "测试集/固定评测集（纯真实：移动传球、颠球、室外/右边比赛1）。"),
    ("sources", "② **素材来源**（原样保存，不直接训练）。命名前缀："
                "`raw_video_*` 现场视频抽帧 · `coco_*` COCO 数据集 · `ai_gen_*` AI 生成 · "
                "`ai_cut_*`/`ai_bg_cut_*` 脚本抠图粘贴。"),
    ("sources/raw_video", "`raw_video_*`：现场视频抽帧（源分辨率原样保存）。由 `s1_data_extract_frames.py` 生成。"),
    ("sources/ai_gen_bg", "`ai_gen_bg_*`：**AI 生成**的空场地背景（**素材池**，无球/无人/无器材），"
                         "只作粘贴背景，不直接训练。由 `s2b_data_gen_backgrounds.py --validate` 校验登记"
                         "（裁水印只做一次，原图备份在 `_orig/`）。"),
    ("sources/spec_examples", "`spec_*`：**标注规范对照示例图**（真实帧上画好框 + 说明），只进文档、不进训练。"
                             "由 `s2c_data_spec_stress.py --spec` 生成。"),
    ("sources/stress_set", "`stress_*`：**分布外压力集**（难帧队列 + 入选原因，单独评测、不进训练）。"
                          "由 `s2c_data_spec_stress.py --stress` 生成（AI 生成服务停用 → 用真实难帧）。"),
    ("sources/coco_person", "`coco_person_*`：COCO 数据集的人体子集（兜底第三级，需登记台账）。"),
    ("sources/pseudo_person", "`pseudo_person_*`：模型伪标签产物（兜底第一级，单列统计）。"),
    ("work", "③ **中间工作区**（可整目录删掉重生成）：标注、抠图素材、粘贴合成、提示词。"),
    ("work/annotations", "标注工作区：**`GT.json` = 人工复核后的权威标注**；"
                        "`preannot.json` = 机器预标注基线；`GT_reviewed.json` = 复核台账；"
                        "`GT_excluded.txt` = 判废清单。"),
    ("work/cutouts", "锥素材库（GrabCut/Otsu 抠出的 RGBA PNG）+ `cutouts.json` 元数据。"),
    ("work/ai_cut", "`ai_cut_*`：**抠图粘贴到真实背景**的合成样本 + `annotations.json`"
                    "（GT 由粘贴几何精确算出）。"),
    ("work/ai_bg_cut", "`ai_bg_cut_*`：**抠图粘贴到 AI 背景**的合成样本 + `annotations.json`"
                       "（GT 同样精确）。"),
    ("work/prompts", "三类 AI 提示词包（背景池 / 规范示例 / 压力集）。"),
    ("reports", "报告与抽检图：三色拼图、锥候选 CSV、数据体检、合成抽检、待标包。"),
)


def write_dir_readmes() -> list[Path]:
    """给每个目录写 `_README.md`（自解释：这是什么、谁生成、是否进训练）。"""
    written = []
    for rel, text in DIR_HELP:
        target = DATASET_ROOT / rel
        target.mkdir(parents=True, exist_ok=True)
        path = target / "_README.md"
        path.write_text("# %s\n\n%s\n" % (rel, text), encoding="utf-8")
        written.append(path)
    return written


# --- 对拍校验 -----------------------------------------------------------------


def categories_from_coco(path: Path | str) -> list[dict[str, Any]]:
    """从 `_annotations.coco.json` 读出 categories（只取 id/name/supercategory）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "categories" not in data:
        raise ValueError(f"{path} 缺少 'categories' 字段")
    return [{"id": int(c["id"]), "name": c.get("name"),
             "supercategory": c.get("supercategory")} for c in data["categories"]]


def assert_categories(actual: Sequence[Mapping[str, Any]], source: str = "?",
                      expected: Sequence[Mapping[str, Any]] | None = None) -> None:
    """数据集 categories 必须与 classes.json 逐项一致（顺序、数量、id、name）。

    不一致直接抛 ValueError —— 训练侧的列数与部署侧的类别号都由它推导，
    错一位就是"球消失"或"锥桶又被算进球"。
    """
    want = list(expected if expected is not None else categories())
    got = list(actual)
    if len(got) != len(want):
        raise ValueError(f"{source}: categories 数量 {len(got)} != 期望 {len(want)}（{want}）")
    for i, (g, w) in enumerate(zip(got, want)):
        for key in ("id", "name", "supercategory"):
            if g.get(key) != w.get(key):
                raise ValueError(
                    f"{source}: categories[{i}].{key} = {g.get(key)!r}，期望 {w.get(key)!r}"
                    f"（顺序即类别顺序，不可调换）")
    if {int(c["id"]) for c in got} != set(range(len(want))):
        raise ValueError(f"{source}: category_id 必须是连续的 0..{len(want) - 1}（id 0 为"
                         f" supercategory=none 的占位背景类），实际 "
                         f"{sorted(int(c['id']) for c in got)}")
    placeholders = [c for c in got if str(c.get("supercategory")) == "none"]
    if len(placeholders) != 1 or int(placeholders[0]["id"]) != 0:
        raise ValueError(f"{source}: 必须有且仅有 id=0 的占位背景类（supercategory=none），"
                         f"实际 {placeholders}")
    real = [c for c in want if str(c.get("supercategory")) != "none"]
    if len(got) != len(real) + 1:
        raise ValueError(f"{source}: 应为 {len(real)} 个真实类别 + 1 个占位类，实际 {len(got)} 项")


def assert_split_annotations() -> dict[str, int]:
    """三目录的 categories 全部对拍，返回每个 split 的图片数。"""
    counts: dict[str, int] = {}
    for split in SPLITS:
        ann = split_annotations(split)
        if not ann.exists():
            raise FileNotFoundError(f"缺少 {ann}（roboflow 布局必需）")
        assert_categories(categories_from_coco(ann), source=str(ann))
        data = json.loads(ann.read_text(encoding="utf-8"))
        counts[split] = len(data.get("images", []))
    return counts


def assert_manifest_licenses(rows: Iterable[Mapping[str, str]], allowed_redistributable: bool = False
                             ) -> None:
    """合成/伪标签/采集样本必须登记来源；不可再分发的不能进 test 集。"""
    for i, row in enumerate(rows, 2):
        for field in ("source", "license"):
            if not str(row.get(field, "")).strip():
                raise ValueError(f"manifest 第 {i} 行缺少 {field}")
        if row.get("split") == "test" and not allowed_redistributable \
                and str(row.get("source")) in {"crawl", "ai_stress"}:
            raise ValueError(f"manifest 第 {i} 行：来源 {row.get('source')} 不应进 test 集")


# --- CLI ----------------------------------------------------------------------


def _print_alignment() -> None:
    exp = pipeline_expectations()
    print("类别真源:", CLASSES_JSON)
    print("  categories      :", json.dumps(categories(), ensure_ascii=False))
    print("  class_names     :", exp["class_names"], "（必须显式传给 .train(class_names=...)）")
    print("  数据集 category_id:", exp["dataset_category_ids"], "（0-based 连续；列数 =", exp["numClasses"], "）")
    print("  Nexusflow YAML  : numClasses=%d targetClasses=%s personClassId=%d ballClassId=%d coneClassId=%d"
          % (exp["numClasses"], exp["targetClasses"], exp["personClassId"],
             exp["ballClassId"], exp["coneClassId"]))
    print("  数据根          :", DATASET_ROOT)


def _cmd_check() -> int:
    _print_alignment()
    missing = missing_layout()
    print("骨架缺失项:", "无" if not missing else "")
    for path in missing:
        print("   !", path)
    for split in SPLITS:
        ann = split_annotations(split)
        n_img = len(list(split_images(split).glob("*"))) if split_images(split).exists() else 0
        print("  %-5s: %s（已就位 %d 张图）" % (split, "有标注" if ann.exists() else "缺标注", n_img))
    return 1 if missing else 0


def _selftest() -> int:
    ok = [c for c in categories()]
    assert_categories(ok, source="selftest")  # 正确的不该抛
    bad_cases = {
        "数量少一个（缺占位类）": [c for c in ok if c["id"] != 0],
        "顺序调换": [ok[0], ok[2], ok[1], ok[3]],
        "名字改了": [{**c, "name": "sport ball"} if c["id"] == 2 else c for c in ok],
        "id 不连续": [{**c, "id": c["id"] + 10} for c in ok],
        "真实类被写成 supercategory=none": [{**c, "supercategory": "none"} for c in ok],
        "占位类不属于 id=0": [{**c, "id": (c["id"] - 1) % 4} for c in ok],
    }
    for label, case in bad_cases.items():
        try:
            assert_categories(case, source="selftest")
        except ValueError:
            print("  对拍能抓住：", label)
        else:
            raise AssertionError(f"对拍漏掉了异常：{label}")
    print("layout 自检通过：列数=%d，class_names=%s" % (num_columns(), class_names()))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ball_cone_v1 类别真源与目录对拍")
    parser.add_argument("--init", action="store_true", help="建目录骨架与 manifest/ledger 表头")
    parser.add_argument("--check", action="store_true", help="检查骨架并打印三方对齐值")
    parser.add_argument("--check-annotations", action="store_true",
                        help="对拍三个 split 的 _annotations.coco.json 类别")
    parser.add_argument("--selftest", action="store_true", help="自检对拍函数")
    parser.add_argument("--migrate", action="store_true",
                        help="按新命名整理目录（搬目录 + 改写文件内路径前缀）")
    parser.add_argument("--migrate-dry-run", action="store_true", help="只演练迁移，不动文件")
    parser.add_argument("--readmes", action="store_true", help="给每个目录写 _README.md（自解释）")
    parser.add_argument("--tree", action="store_true", help="打印目录树 + 每个目录一行说明")
    args = parser.parse_args(argv)

    if args.init:
        for path in ensure_layout():
            print("ok:", path)
    if args.migrate_dry_run or args.migrate:
        for action, old, new in migrate_layout(apply=args.migrate):
            print("  %-16s %s → %s" % (action, old, new))
    if args.readmes:
        print("ok:", write_dataset_readme())
        for path in write_dir_readmes():
            print("readme:", path.relative_to(DATASET_ROOT))
    if args.selftest:
        return _selftest()
    if args.check_annotations:
        print("每 split 图片数:", assert_split_annotations())
    if args.tree:
        print("\n%s/" % DATASET_ROOT)
        for rel, text in DIR_HELP:
            exists = "✓" if (DATASET_ROOT / rel).exists() else "·"
            print("  %s %-24s %s" % (exists, rel + "/", text.split("\n")[0][:80]))
    if args.check:
        return _cmd_check()
    if not any((args.init, args.check, args.check_annotations, args.selftest,
                args.migrate, args.migrate_dry_run, args.readmes, args.tree)):
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
