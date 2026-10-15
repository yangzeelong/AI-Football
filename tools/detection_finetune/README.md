# detection_finetune —— 人/足球/锥形桶 三分类微调工具集

完整方案与逐步骤手册见 **`../../doc/detection_finetune_plan.md`**（含每步验收判据与人工工作量估计）。
数据侧的说明在 **`$AI_FOOTBALL_DATASET_DIR/README.md`**（由 `base_layout.py --readmes` 生成，含命名规则与当前规模）。

## 一句话流程

```
抽帧 → 半自动预标注 → 合成扩充(真实背景 + AI 背景) / AI 示例图与压力集
     → 人工校验(GUI 往返：Label Studio / CVAT，合成图只抽检 10%) → 回收+体检
     → 微调(512 冒烟 → 960 正式) → person 保障 → 检测级 A/B
     → 导出 ONNX+fp16+engine → 两侧接入(类别号 + cone 抑制) → 端到端触球回归 → 误检回收回灌
```

> 合成与 AI 步骤排在人工校验**之前**：一次校验过完"真实 + 合成"的最终候选集，避免二次开标注任务。

## 模型管理（唯一命名：v1）

> **训练出来后怎么上线**：`tools/rollout/`（独立流水线目录）——`rollout.py export` 出 engine、
> `cpp-config` 生成三分类 C++ 配置、`run` 用 C++ + TensorRT 批量跑视频、`summary` 出汇总。
> 实测同一张卡：TRT fp16 engine 检测 3.9 ms/帧、姿态 0.92 ms/人，比 Python PyTorch 路线快 30–270×。


**规则：一个版本一个目录、一个名字；别处只准引用这个名字，不允许多名指同一物。**

| 东西 | 唯一名字 | 位置 |
| --- | --- | --- |
| 训练产物（权重） | `checkpoint_best_ema.pth` / `checkpoint.pth` | `runs/v1/` |
| 管线加载的权重 | `rf-detr-small.pth`（**软链**到上面那个；文件名是 `RfdetrDetector` 的硬契约，不能改） | `models/rfdetr/fine-tune/v1/` |
| 导出 ONNX（fp32 / fp16） | `rf-detr-small-v1-960-dynamic{,-fp16-fp32io}.onnx`（+`.data`） | 同上目录 |
| 模型卡 | `meta.json`（类别 id、部署字段、对拍结论） | 同上目录 |
| 训练配置 | `train_configs/v1.yaml`（`output_dir: runs/v1`） | 本目录 |
| 长跑启动器 | **`s4_train_launch.sh <版本>`**（如 `./s4_train_launch.sh v1-frozen-step2`；日志 `runs/<版本>/train.log`，pid `/tmp/<版本>.pid`）——一个启动器跑所有版本 | 本目录 |
| 管线配置（Python/C++） | `config/app_v1.yaml`（`rfdetr_model_dir: models/rfdetr/fine-tune/v1`） | `AI-Football/config/` |
| 第一步（只学分类层） | **`v1-frozen-step1`**（`--freeze encoder`：主干冻结、只训新类头；**内外同名，直接用**） | `runs/v1-frozen-step1/` |
| 第二步（优化 encoder/卷积层） | **`v1-frozen-step2`**（解冻主干精修，warm start 自 step1 的 `checkpoint_best_ema.pth`；**内外同名，当前部署/交付的就是这一份**） | `runs/v1-frozen-step2/` |
| 对照基准 | **base** = COCO 权重 `models/rfdetr/rf-detr-small.pth`（名字保持不动，勿覆盖） | `models/rfdetr/` |

改名/新增版本时**照抄这套结构**（版本名 = 血缘，别处只引用它）：
`runs/v1-frozen-step1/`、`train_configs/v1-frozen-step1.yaml`、
`runs/v1-frozen-step2/`、`train_configs/v1-frozen-step2.yaml`、
`config/app_v2.yaml`（**`v2` 预留给"小目标优化"**：更小尺度的合成样本/更高分辨率）。
**命名已于 2026-10-12 统一**（此前评测命令行用过旧 tag `finetune-v1` / `finetune-v1-frozen`，
只改了文件名没改内部字段 → 一度两名指一物）：
历史指标文件的内部 `tag` 已改成规范名，旧名留在 `tag_legacy`；失效的 `checkpoint` 路径指向
规范名目录，旧值留在 `checkpoint_legacy`；同一权重评两遍的多余文件归档在 `runs/eval_ab/_legacy/`
（不参与台账）。工具：`util_unify_model_names.py`（默认 dry-run）；台账侧另有 `metrics_aliases.json`
兜底，遇到旧名自动归并。**新增版本时照抄规范名，别再起别名。**

## 目录与命名规则（唯一口径）

三层结构：**`dataset/` ① 最终数据 → `sources/` ② 素材 → `work/` ③ 中间产物**（可整目录删掉重建）。

| 前缀 | 含义 | 目录 | manifest `source` | 生成脚本 | 进训练 |
| --- | --- | --- | --- | --- | --- |
| `raw_video_*` | 现场视频抽帧（**原始数据**） | `sources/raw_video/` | `raw_video` | `s1_data_extract_frames.py` | ✅ |
| `coco_*` | COCO 数据集里来的 | `sources/coco_person/` | `coco_person` | `s4b_train_person_coco.py` | ✅（兜底③） |
| `ai_gen_bg_*` | AI 生成的空场地背景（**素材池**） | `sources/ai_gen_bg/` | `ai_gen_bg` | `s2b_data_gen_backgrounds.py` | ❌（只作粘贴背景） |
| `spec_*` / `stress_*` | 标注规范示例（画框） / 分布外压力集 | `sources/spec_examples/`、`sources/stress_set/` | `spec_example` / `stress_set` | `s2c_data_spec_stress.py` | ❌ |
| `pseudo_person_*` | 伪标签产物 | `sources/pseudo_person/` | `pseudo_person` | `s4a_train_person_pseudo.py` | ✅（兜底①） |
| **`ai_cut_*`** | 抠图**粘贴到真实背景**（GT 精确） | `work/ai_cut/` | `ai_cut` | `s2a_data_compose_cones.py` | ✅ |
| **`ai_bg_cut_*`** | **AI 背景** + 抠图粘贴（GT 精确） | `work/ai_bg_cut/` | `ai_bg_cut` | `s2a_data_compose_cones.py` | ✅ |
| `web_*` | 网络/公开渠道采集（需许可台账） | `sources/web/` | `web` | `s1_data_collect_web.py` | ✅（需许可） |

**脚本命名规则**：`<阶段>_<域>_<动作>.py`，`ls` 天然按流程排序。

| 段 | 取值 | 说明 |
| --- | --- | --- |
| 阶段 | `s1` → `s7` | `s1` 抽帧采集 · `s2` 预标注与合成 · `s3` 人工校验与回收 · `s4` 训练与 person 保障 · `s5` 评估 · `s6` 部署 · `s7` 端到端跑视频 |
| 子步骤 | 阶段后加字母 | `s2a_` `s2b_` `s2c_`（同阶段的第 2/3/4 步，按执行顺序） |
| 域 | `data`/`label`/`train`/`eval`/`deploy`/`run` | 该脚本在哪一段流程里干活 |
| 基础设施 | `base_` | `base_layout.py`（类别真源与目录对拍）、`env.sh` |
| 一次性工具 | `util_` | `util_unify_model_names.py` |

例：`s4a_train_person_pseudo.py` = 训练阶段第 a 步、person 保障、伪标签。

> 注意区分两层命名：上表里的 `raw_video_*` / `coco_*` / `ai_cut_*` 是**数据**的命名规则
> （目录名与 manifest 的 `source` 字段），与**脚本文件名**无关。
> 脚本改名时必须同步 `import`（脚本间互相 import）；`base_layout.py` 被 18 个脚本引用。

```bash
$PY base_layout.py --tree        # 打印目录树 + 每个目录一行说明
$PY base_layout.py --readmes     # 写数据根 README.md + 每个目录 _README.md（自解释）
$PY base_layout.py --migrate     # 按命名规则搬目录 + 改写文件内路径/source（幂等）
```

## 路径约定（换机器 / 换目录不用改代码）

三个目录**相邻**即可，所有路径都能从仓库位置推导出来：

```
<WORKSPACE>/                      ← 工作区根
├── AI-Football/                  ← 本仓库
├── Nexusflow/                    ← C++ / TensorRT 侧
└── data/                         ← 视频与数据集（ball_cone_v1 在这下面）
```

`source env.sh` 之后下面这些变量就绪；**脚本与文档统一用它们，不再出现绝对路径**：

| 变量 | 默认值（推导） | 说明 |
| --- | --- | --- |
| `AI_FOOTBALL_REPO_ROOT` | `<WORKSPACE>/AI-Football` | 本仓库根 |
| `AI_FOOTBALL_WORKSPACE` | 仓库的父目录 | 三个目录的共同父级 |
| `AI_FOOTBALL_NEXUSFLOW` | `<WORKSPACE>/Nexusflow` | C++ / TensorRT 仓库 |
| `AI_FOOTBALL_DATA_DIR` | `<WORKSPACE>/data` | 视频与数据集所在目录 |
| `AI_FOOTBALL_DATASET_DIR` | `<AI_FOOTBALL_DATA_DIR>/ball_cone_v1` | 数据集本体 |
| `AI_FOOTBALL_ENV` | `/root/miniconda3/envs/ai-football` | conda 环境目录（`$PY` 的来源） |

Python 侧读的是同一批变量（`base_layout.py`），所以 shell 与脚本口径一致。

**要拆开布局**（例如数据放另一块盘）就先设变量再 source——`env.sh` 只补缺省、**不覆盖**已有值：

```bash
export AI_FOOTBALL_DATA_DIR=/mnt/bigdisk/data
export AI_FOOTBALL_ENV=/opt/conda/envs/ai-football
source env.sh
```

配置文件（`tools/rollout/rollout.yaml`、`s7_batch_videos.yaml`）里的根路径**留空即自动推导**；
`s7_batch_videos.yaml` 的 `videos[].path` 一律写相对 `video_roots[0]` 的相对路径。
所以换机器只需要保证三个目录相邻，**不必改任何 yaml**。

## 数据集目录要求（训练输入的唯一口径）

训练走 rfdetr 的 **roboflow 布局**。结构不对会**启动即失败**；类别写错会**静默错指标、甚至 CUDA assert**。

```
<dataset_dir>/                        # 默认 = $AI_FOOTBALL_DATASET_DIR/dataset
├── train/
│   ├── _annotations.coco.json
│   └── images/*.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── images/*.jpg
└── test/                             # ⚠️ 必须有：run_test 默认开启
    ├── _annotations.coco.json
    └── images/*.jpg
```

| 要求 | 说明 |
| --- | --- |
| split 目录名 | 固定 `train` / `valid` / `test`，三个都得有 |
| 每个目录的内容 | `_annotations.coco.json` + `images/`，缺一不可 |
| `categories` | **必须 4 条**：`id=0` 是 `supercategory:"none"` 的占位背景类，真实类 `1/2/3 = person/ball/cone`。原因与后果见「两条不能忘的契约」 |
| `images[].file_name` | **必须**写成 `images/<文件名>`（roboflow 加载器按这个前缀拼路径） |
| `annotations[].bbox` | `[x, y, w, h]`（左上角 + 宽高），`id` 唯一；越界会被裁回图内，面积 ≤1px 的框被丢弃 |
| 类别名 | 逐字 `person` / `ball` / `cone`（`classes.json` 是唯一真源，按**名字**对拍） |
| 同源泄漏 | 同一段视频**不得**跨 split |

三道校验，违规直接报错：

```bash
$PY base_layout.py --check-annotations   # 对拍三个 split 的 categories 与 classes.json
$PY s3_label_convert_dataset.py --link   # 转换时对拍 + 泄漏检查 + 文件名冲突
$PY s3_label_dataset_stats.py            # 类别/尺度/来源/空标注/过小框哨兵
```

训练启动时也会自检，并把结果写进 `finetune_meta.json`：

```
类别: ['person','ball','cone']（应为 ['person','ball','cone']）| 输出列数应为 4
```

**别手搓 COCO JSON。** 用生成器产出这一整套结构（它会把 `file_name` 强制写成 `images/xxx`、
做类别对拍、并按 `split_policy.json` 保证同源不跨集）：

```bash
$PY s3_label_convert_dataset.py --link
```

* `--dataset-dir` 可以指到别处（或设 `AI_FOOTBALL_DATASET_DIR`），但**结构必须照上表**；
* 冒烟用的 `--subset N` 会把每个 split 前 N 张复制到 `/tmp/ds_subset`（结构同上，并**保证 `test/` 存在**），不动原数据。

## 环境准备（每个新终端一次）

**必须用 `ai-football` 环境跑**；本机默认激活的是 conda **base**，裸 `python` 里没有 `rfdetr`。
下文所有命令里的 `$PY` 都指这个环境的 python：

```bash
cd AI-Football/tools/detection_finetune
source env.sh                # 最省事：设好 $PY 与 PATH（推荐）
PY=$AI_FOOTBALL_ENV/bin/python   # 等价写法（任何 shell 都稳）
```

自检：`$PY -c "import torch, rfdetr, cv2; print(torch.__version__, torch.cuda.is_available())"`。
装包用 `$PY -m pip install <pkg>`——**裸 `pip` 会装到 conda base**。
标注 GUI（Label Studio）装在独立 venv `.venv-labelstudio/`，**不污染训练环境**。

## 命令索引

工作目录为本目录；数据根：`$AI_FOOTBALL_DATASET_DIR`。

按流程顺序排列（`ls` 的顺序就是执行顺序）。

| 脚本 | 作用 | 命令 | 状态 |
| --- | --- | --- | --- |
| `base_layout.py` | 类别真源 + 目录骨架 + 三方对拍（`classes.json` 的唯一解释器） | `$PY base_layout.py --selftest` / `--check` / `--init` / `--tree` / `--readmes` / `--migrate` | ✅ |
| `s1_data_extract_frames.py` | 现场视频抽帧，登记 manifest（保留原分辨率与帧号） | `$PY s1_data_extract_frames.py --priority high,normal --stride-sec 3 --per-video-max 30` | ✅（529 帧） |
| `s1_data_collect_web.py` | 网络/公开素材入库（`web_*` + 许可台账） | `$PY s1_data_collect_web.py --ping <url>` / `--from-dir ... --license CC-BY-4.0` / `--stats` | ✅ |
| `s2_data_preannotate.py` | base 模型半自动预标注 + 锥桶候选清单 | `$PY s2_data_preannotate.py --resolution 960 --threshold 0.25 --montage-splits train,valid,test` | ✅（529 图 / 7479 框 + 69 锥候选） |
| `s2a_data_compose_cones.py` | 抠素材贴到背景合成锥桶，自动精确 GT | `--extract --min-score 0.75` → `--compose --n 400 --backgrounds mixed --ai-share 0.25 --per-image-max 4` → `--qa` → `--ablate` | ✅（12 素材 → 128 + 43 张） |
| `s2b_data_gen_backgrounds.py` | AI 生成空场地背景池（只作粘贴背景） | `--prompts` → 用 `[skill:多模态内容生成]` 出图 → `--validate`（裁水印只做一次，原图备份 `_orig/`） | ✅ 3 张背景池（**云服务已停用**，见坑表） |
| `s2c_data_spec_stress.py` | 标注规范示例图 + 分布外压力集 | `$PY s2c_data_spec_stress.py --spec --count 6` / `--stress --per-bucket 8` / `--list` | ✅（真实帧派生） |
| `s2d_data_cut_person_assets.py` | **人体素材库**：多视频扫描 + 七道质量闸门（框高/面积/形态/截断/遮挡/清晰度/去重）→ PNG + index + 拼图 | `$PY s2d_data_cut_person_assets.py`（默认自动挑 6 段）｜`--min-height 160` ｜`--videos 移动传球.mov,颠球.mov` ｜`--list-videos` | ✅（136 张 / 6 段，框高中位 170 px） |
| `s3_label_review_ui.py` | 人工复核界面（零依赖，标准库实现） | `$PY s3_label_review_ui.py`（一次载入全部批次，界面内下拉切换；改前端后跑 `node tests/ui_smoke.js` 回归） | ✅ |
| `s3_label_start_ui.sh` | 复核界面启动/重启（pidfile 管理；**别用 pkill**，见坑表） | `./s3_label_start_ui.sh`（`PORT=8090` 可改） | ✅ |
| `s3_label_review_roundtrip.py` | 校验往返：导出待标包 / 回收 + 差异报告 | `--export` / `--ls-tasks <batch>` / `--ls-setup` / `--import <导出.json> --batch <batch>` / `--status` | ✅（S3） |
| `s3_label_convert_dataset.py` | 标注回收 → COCO 三分类 → roboflow 三目录（带泄漏检查） | `$PY s3_label_convert_dataset.py --link`（默认优先 `work/annotations/GT.json`（人工 GT）；`--cone-min-score 0.6` 仅 bootstrap 用） | ✅ |
| `s3_label_dataset_stats.py` | 数据体检：类别 / 尺度 / 分辨率 / 来源 / 空标注 | `$PY s3_label_dataset_stats.py` | ✅ |
| `s4_train_finetune.py` | 三分类微调入口（把 `train_configs/*.yaml` 喂给 rfdetr） | `$PY s4_train_finetune.py --config train_configs/smoke_512.yaml --subset 24 --subset-require-cone` | ✅（冒烟通过） |
| `s4_train_launch.sh` | 长跑启动器（一个脚本跑所有版本，自动续训并落日志/pid） | `./s4_train_launch.sh <版本> [--fresh] [--epochs N]` | ✅ |
| `s4_train_watch.py` | 训练进度监视（只读日志，不占 GPU） | `$PY s4_train_watch.py --log runs/<版本>/train.log [--once]` | ✅ |
| `s4_train_throughput.py` | 吞吐探针（batch × gradient checkpointing） | `$PY s4_train_throughput.py --subset 24` | ✅ |
| `s4a_train_person_pseudo.py` | person 保障①：base 模型伪标签扩量 | `$PY s4a_train_person_pseudo.py --count 200 --stride-sec 2 --conf 0.6` / `--stats` | ✅（已跑 10 帧冒烟） |
| `s4b_train_person_coco.py` | person 保障③：COCO val2017 人体子集（按需） | `$PY s4b_train_person_coco.py --ping` → `--download-annotations` → `--mix --max-images 400` / `--stats` | ✅（按需，已验证资源可达） |
| `s5_eval_detector.py` | 检测级 A/B（base vs 微调，有 GT 的硬数字） | `$PY s5_eval_detector.py --split test --model base=models/rfdetr/rf-detr-small.pth --model v1=runs/v1/checkpoint_best_ema.pth` | ✅（产物 `runs/eval_ab/<split>_<res>/`） |
| `s5_eval_diag_ab.py` | person/ball 回归诊断 + 三方对照图 | `$PY s5_eval_diag_ab.py --split test --model base=... --model v1=... --overlays` | ✅ |
| `s5_eval_render.py` | 检测结果渲染（mp4 + GIF + 拼图） | `$PY s5_eval_render.py --video <视频> --model v1=... --contact` | ✅ |
| `s5_eval_ledger.py` | 指标台账（幂等累积，跨次对比的唯一入口） | `$PY s5_eval_ledger.py`（扫描 `runs/**/metrics_*.json`；`--no-aliases` 保留原始 tag） | ✅ 台账 `runs/metrics_ledger.{json,csv}` + `data/ball_cone_v1/reports/metrics_ledger.md` |
| `s6_deploy_export.py` | 导出 ONNX/fp16/engine，并与 PyTorch 对拍 | `$PY s6_deploy_export.py --checkpoint <new.pth> --resolution 960` | ✅（v1 已产出 ONNX/fp16/engine + 对拍） |
| `s6_deploy_pin_model.py` | 模型上架：建权重软链 + 写模型卡 `meta.json` | `$PY s6_deploy_pin_model.py --version v1` | ✅ |
| `s7_run_batch_videos.py` | 批量跑 `code/app.py`（读 YAML → 拼命令 → 执行 → 记账） | `$PY s7_run_batch_videos.py [--dry-run] [--only-model v1] [--only-video <tag>]` | ✅ |
| `s7_run_ab_videos.py` | 多视频检测级 A/B（同一批视频上比多个版本，源视频只解码一遍） | `$PY s7_run_ab_videos.py --videos-dir <目录> --version v1 --version v1-frozen-step2` | ✅ |
| `util_unify_model_names.py` | 把历史指标文件的 tag/文件名/checkpoint 统一成规范名 | `$PY util_unify_model_names.py`（**dry-run**）→ `--apply` 才写 | ✅（2026-10-12 已对 3 个历史文件执行一次） |
| `s7_recycle_hard_examples.py` | **误检回收闭环**：跑视频 → 挑「锥被当球」的帧 → 出图 + COCO 预标注 + 登 manifest → 交回 S3/S4 | `$PY s7_recycle_hard_examples.py --video <视频> [--dry-run]` | ✅（判据 `cross` A 判球/B 判锥 + `within` 球∩锥；**默认拒绝从 valid/test 源回收**，防污染评测集） |

数据/配置文件（非脚本）：

| 文件 | 作用 |
| --- | --- |
| `classes.json` | 类别唯一真相源（训练 / 导出 / 部署三方对拍） |
| `split_policy.json` | 视频 → split/场景/视角/采样优先级（split 按整段视频划分，防同源泄漏） |
| `metrics_aliases.json` | 历史 tag → 规范 tag 归并表（`finetune-v1` → `v1`）；命中就说明还有旧名没清 |
| `s7_batch_videos.yaml` | `s7_run_batch_videos.py` 的任务清单（由 `--sync-videos` 生成） |
| `env.sh` | `source` 即设好 `$PY` 与 PATH |

## S3 人工校验（自带界面为主，Label Studio / CVAT 备选）

只复核**真实帧**的框 + 锥候选改类；**合成帧只抽检 10%**（查粘贴质量：悬空 / 越界 / 压人）。

```bash
# 1) 出批次（默认跳过已进过批次的图；100 图/批）
$PY s3_label_review_roundtrip.py --export --split train --source raw_video --limit 100
$PY s3_label_review_roundtrip.py --export --split valid --source raw_video --limit 100
$PY s3_label_review_roundtrip.py --export --split test  --source raw_video --limit 100
$PY s3_label_review_roundtrip.py --export --split all --source ai_cut,ai_bg_cut --synth-sample 0.1
$PY s3_label_review_roundtrip.py --status

# 2) 复核（零依赖界面：标准库 http.server + Canvas）
$PY s3_label_review_ui.py                     # 默认载入**全部批次**（也可 --batch a,b 只载指定批）
#    本地另开终端：ssh -L 8090:localhost:8090 <user>@<host> → 浏览器 http://localhost:8090
#    界面：左上角下拉切换批次、点批次表跳转。操作：
#      拖拽 = **框选多个框**（Shift+拖拽 = 加选）→ 1 person / 2 ball / 3 cone **批量改类**，Del **批量删除**
#      N = 切换**新建模式**（拖拽画新框；类别取"新建框类别"下拉，默认 cone；画完自动选中可 1/2/3 改）
#      点框 = 选中单个，Shift+点 = 加选；单击空白/Esc = 清空选择；Ctrl+Z = 撤销
#      **C = 叠加"上一张新增/改过的框"到本图**（本图原有框保留；同位置已有框→只同步类别；
#        不同位置→新增）。**Shift+C = 用上一张的全部框整体覆盖本图**（两帧几乎一致时更快）。
#      勾"叠加后自动下一张"后 C 会直接跳下一张
#      ←/→ 翻图；Enter = 本图已核对并跳到下一张未核对；S 保存（各批写各自 <batch>_edited.json）

# 3) 回收（按类别名映射，产出 GT.json + 差异报告 + 判废清单）
$PY s3_label_review_roundtrip.py --import <batch>_edited.json --batch <batch>
$PY s3_label_convert_dataset.py --link     # 自动优先用 GT.json，并跳过 GT_excluded.txt
$PY s3_label_dataset_stats.py
```

**Label Studio / CVAT 备选**：脚本已内置（`--ls-tasks` 打印导入 curl；`--ls-setup` 一条命令建项目+导入；
CVAT 用 `<batch>.zip` 的 `images/` + `preannot.json` 走 COCO 1.0）。**但本机装不上 Label Studio**：
其依赖 `python-json-logger` 安装时创建 `pythonjsonlogger` 目录被拒（EACCES，root 也一样；
换 conda / 换目录 / `/tmp` 同样被拦，已实测），
所以本机一律用上面的自带界面；换到不受限的机器再用 LS/CVAT。

## 长跑训练（S4）

```bash
$PY s4_train_throughput.py --configs 4:on,6:on,8:on --alloc expandable_segments:True  # 换机器/换分辨率先测
./s4_train_launch.sh v1        # 后台长跑（自动续训；日志 runs/v1/train.log）
$PY s4_train_watch.py                 # 监视：epoch / loss / mAP / 显存 / 进度（20s 刷新）
kill -- -"$(cat /tmp/v1.pid)"   # 停止（负号=杀进程组；**别用 pkill**，会误杀自己的 shell）
./s4_train_launch.sh v1 --fresh   # 忽略 checkpoint 从头跑
./s4_train_launch.sh v1-frozen-step2        # 第二步：解冻 encoder 精修（warm start 自 v1-frozen-step1）
./s4_train_launch.sh v1 --freeze encoder   # stage-1：冻结主干，只学新类
```

**本机吞吐实测（16G 卡，960 输入）**——结论是**不要加大 batch**：

| 配置 | 每 micro-batch | img/s | 显存峰值 | 结论 |
| --- | --- | --- | --- | --- |
| batch 4 + checkpointing on | **0.197s**（长跑实测）/ 0.44s（探针） | 20 / 9.2 | 6.9–8.7 G（53%） | ✅ 最快 |
| batch 4，num_workers 8 + persistent | — | — | — | ❌ **卡死在第一步**（CUDA fork 死锁） |
| batch 6 + on | 2.25s | 2.7 | 12.4 G（76%） | ❌ 更慢 |
| batch 8 + on | 1.01–2.66s | 3.0–7.9 | 15.7 G（96%） | ❌ 顶上限→分配器抖动 |
| batch 8 + off（关 ckpt） | 1.16s | 6.9 | 15.8 G | ❌ 关掉 checkpointing 也不会更快 |
| batch 12 + on | 1.26s | 9.5 | 15.8 G | ❌ 风险高、波动大 |

* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：显存峰值 8657 → 7873 MiB（已作为默认写进启动器）。
* 加 worker 无收益（`data:` 只占 0.05–0.1s，不是瓶颈）；**显存空着 ≠ 能提速**。
* 长跑实测：约 **60 秒/epoch**（batch 4，36 步），比最初的 113 秒/epoch 快 1.9x。

## 两条不能忘的契约（写错就白烧几小时）

1. **类别**：COCO 里放 4 个 category —— `id=0` 是 `supercategory:"none"` 的占位背景类，真实类 **1=person / 2=ball / 3=cone**。
   直接写 1/2/3 三类会在训练时 target 越界触发 **CUDA device-side assert**；写 0/1/2 会让 Python 侧的
   `{i+1: name}` 类别名映射错位、person 被丢。（目录结构与其余字段要求见「数据集目录要求」）
2. **分辨率**：只传 `resolution=960`，**不要**传 `positional_encoding_size=60`（base 权重位置编码是 32²，
   传 60² 会在 `load_state_dict(strict=False)` 上 size mismatch 报错）。

部署侧 classId = logits 列下标（**不加 1**），所以 Nexusflow 侧写
`targetClasses:[1,2,3]`、`personClassId:1 / ballClassId:2 / coneClassId:3`、`numClasses:4`。

## 踩过的坑（都在这里，别再踩）

| 坑 | 现象 | 现在的做法 |
| --- | --- | --- |
| 抽帧去重用 8×8 dHash | 把"人走了几步"的真实变化当重复，30 帧只剩 2 帧 | 用 16×16（256 位），默认阈值 6 |
| `cv2.imwrite` 不建父目录且失败静默 | manifest 记了 529 行，`raw/` 里一张图都没有 | 写图后断言"返回 True + 文件非空" |
| `argparse` help 里的 `%` | `--help` 直接崩（`unsupported format character`） | 百分号写 `%%` |
| 水印裁剪原地覆盖 | 每跑一次 `--validate` 就缩一圈（720→677→636） | 只裁一次 + 原图备份 `_orig/`；短边闸门 480、长宽比 0.35–2.8 |
| 目录合并跳过同名目录 | `work/synthetic/images/*` 171 张全没搬过去 | `_merge_move` 递归合并 |
| 多标注文件按 id 合并 | 两桶 id 都从 1 开始 → cone 从 433 涨到 1889 | 合并前给每个文件加命名空间偏移 |
| 训练脚本默认数据目录 | 迁移后指向 `ball_cone_v1/train`（不存在） | 默认 `base_layout.DATASET_DIR` |
| 合成文件名用 `syn_` 前缀 | 与命名规则不符，找不着来源 | 文件名前缀 = 来源键（`ai_cut_*` / `ai_bg_cut_*`） |
| GUI 导出重排 category id | 按 id 映射会把 ball 当 person | 只按**名字**映射（`classes.json` 为真源） |
| 素材文件名按"框高+清晰度"命名 | 同帧两个人框高与清晰度都相同时**撞名**：后者覆盖前者 PNG，前者被去重删掉 → 索引指向不存在的文件 | 文件名必须带框坐标（`_x%04d_y%04d_`），并在收尾做一次**索引 vs 磁盘一致性自检** |
| Label Studio 依赖装不上 | 装 LS 时创建 `pythonjsonlogger` 目录被拒（EACCES，root 也不行） | 用零依赖的 `s3_label_review_ui.py`；换机器再上 LS |
| **云端图片生成已停用** | 文生图服务返回"图片和视频功能已停止服务" | 规范示例/压力集改**真实帧派生**（`s2c_data_spec_stress.py`）；背景池沿用已有的 3 张 |
| 预标注里的"小到离谱的球框" | train 有 **516 个 <5px 的框（ball 512）**，全是静止锥/碟被当球 | `s3_label_dataset_stats.py` 的"过小框哨兵"盯着这个数；S3 复核后应显著下降 |
| 大文件下载（pip / urllib） | pyarrow 48MB、label-studio 73MB 都会卡住不动 | 一律用 `curl -C - --retry`（`s4b_train_person_coco.py` 已内置） |
| 台账重复行 | `s2b_data_gen_backgrounds.py --validate` 每次跑都追加一行（实测 6 条重复） | 只有真登记了新图才写台账（幂等） |
| 图片路径带空格/中文 | Label Studio 打不开图 | local-files URL 做 `quote()` 编码 |

## 目录约定

| 路径 | 用途 |
| --- | --- |
| `classes.json` | 类别唯一真相源（训练 / 导出 / 部署三方对拍） |
| `split_policy.json` | 视频 → split/场景/视角/采样优先级（split 按整段视频划分，防同源泄漏） |
| `train_configs/*.yaml` | 训练参数（外置，含 `gradient_checkpointing`、EMA、早停） |
| `runs/` | 训练产物（`finetune_meta.json` 记录冻结模式与可训练参数量） |
| `docs/` | 标注规范（含对照示例图）与操作手册 |
| `env.sh` | `source` 即设好 `$PY` 与 PATH |
