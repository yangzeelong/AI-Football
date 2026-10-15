# 标注规范（人 / 足球 / 锥形桶）

本文件规定**标什么、怎么标**。改标注的操作方式见下一节（本项目自带复核界面为主，
Label Studio / CVAT 只在换到不受限的机器时作备选）。

配套材料：三色抽查拼图（`$AI_FOOTBALL_DATASET_DIR/reports/montage_*.jpg`）、
锥桶候选清单（`reports/preannot_cone_candidates.csv`）。

> 路径约定：下文 `$PY` 与 `$AI_FOOTBALL_*` 由 `tools/detection_finetune/env.sh` 设置
> （`source env.sh` 一次即可）。数据根默认是 `<WORKSPACE>/data/ball_cone_v1`，
> 换机器不用改文档里的命令，见 README「目录与命名规则」。

## 类别定义

| 类别 | id | 部署 classId | 定义 | 边界样本 |
| --- | --- | --- | --- | --- |
| `person` | 1 | 1 | 场上球员（含背影、半身、被球门网遮挡） | 观众席人群、海报/围栏上的人像、球门柱 → **不标** |
| `ball` | 2 | 2 | 本场使用的那颗足球（滚动 / 空中 / 被拿在手里 / 贴脚） | 器材筐里的旧球、明显不属于本场的球 → **不标**（当背景） |
| `cone` | 3 | 3 | 场地上的训练器材：小锥、标志碟、标志桶（黄/蓝/白/粉等，7–35 px） | 地板上印的线/圈/点、白线交点 → **不标**（当背景，重要！） |
| 背景 | 0 | — | 地板印花、场地线、球门网、器材筐等一切"看起来像球但不是球"的东西 | **绝对不标任何类** |

> 「地板印花」这一类是本任务的**核心负样本**：base 模型会把它们判成 `sports ball`，
> 训练集里必须保留它们并**不给标签**，模型才会学会不看它们。

## 三种典型难例（对照拼图看）

1. **静止的小锥/标志碟**（室内最常见）：7–35 px，颜色饱和、纹理低，会在固定像素位置反复被检成球。
   → 标 `cone`。同一物体在相邻帧重复出现时**每帧都要标**（跟踪器需要连续性）。
2. **真球停在地上一动不动**：与上一条现象相同，但看黑白花纹/圆形高光 → 标 `ball`。
   分辨不了时：**标 ball 并在 note 里写"疑似静止球，待复核"**（宁可错在 ball，别把真球吞进 cone——吞掉真球会直接压低 recall）。
3. **严重运动模糊/被遮挡 > 50% 的锥**：能确认位置就标 `cone`（框可略大）；完全不可辨认 → **删框**（当负样本）。

## 标注粒度

* 框贴紧目标外沿，允许 ±2 px；
* 同一帧里同一物体只标一个框；
* `person` 只需保证"人在框内"即可（下游是关键点/跟踪，不是检测精度竞赛）；
* 遮挡严重的人：可见部分 > 30% 就标，框住可见部分。

## 改标注的操作方式（自带复核界面，推荐）

完整流程说明见主 README 的「S3 人工校验」；这里只列与标注规范相关的要点。

```bash
cd tools/detection_finetune && source env.sh        # 设好 $PY 与路径变量

# ① 出待标批次（默认跳过已进过批次的图；100 图/批）
$PY s3_label_review_roundtrip.py --export --split train --source raw_video --limit 100
$PY s3_label_review_roundtrip.py --export --split valid --source raw_video --limit 100
$PY s3_label_review_roundtrip.py --export --split test  --source raw_video --limit 100
$PY s3_label_review_roundtrip.py --export --split all   --source ai_cut,ai_bg_cut --synth-sample 0.1
$PY s3_label_review_roundtrip.py --status

# ② 复核（零依赖界面：标准库 http.server + Canvas，一次载入全部批次）
./s3_label_start_ui.sh
#    本地另开终端：ssh -L 8090:localhost:8090 <user>@<host> → 浏览器 http://localhost:8090
#    与标注规范相关的操作：
#      拖拽框选（Shift+拖拽加选）→ 1 person / 2 ball / 3 cone 批量改类，Del 批量删除
#      N 切换新建模式（拖拽画框，类别取"新建框类别"下拉）→ 补漏标的目标
#      C 叠加"上一张新增/改过的框"到本图（相邻帧几乎一致时最快）；Shift+C 整图覆盖
#      S 保存 → 各批写各自的 reports/review_batches/<批>/<批>_edited.json

# ③ 回收（按类别名映射回 classes.json，产出 GT.json + 差异报告 + 判废清单）
$PY s3_label_review_roundtrip.py --import \
    $AI_FOOTBALL_DATASET_DIR/reports/review_batches/<批>/<批>_edited.json --batch <批>
$PY s3_label_convert_dataset.py --link       # 默认优先用 GT.json，并跳过判废清单
$PY s3_label_dataset_stats.py                # 类别 / 尺度 / 来源 分布
```

**为什么以自带界面为主**：它直接读写本项目的 COCO 结构，类别名与 `classes.json` 天生一致，
不会像第三方工具那样**重排 category id**；改完的差异会进 `review_diff.md`，可逐条追溯。
Label Studio / CVAT 仍可用（脚本内置 `--ls-tasks` / `--ls-setup`，CVAT 走 `<批>.zip` 的
`images/` + `preannot.json`，COCO 1.0），但换到那些工具时唯一要注意的是——
**标签名必须逐字为 `person` / `ball` / `cone`**，差一个字符就会在 `--import` 时报「未知标签」。

### 备选：直接改 JSON（无界面环境）

```bash
DATA=$AI_FOOTBALL_DATASET_DIR/work/annotations
cp "$DATA/preannot.json" "$DATA/GT.json"
```

* 改类：把该条的 `"category_id": 2`（ball）改成 `3`（cone）；
* 删框：删掉整条 `annotations` 记录；
* 补框：复制相邻记录，改 `image_id` 与 `bbox`（`[x, y, w, h]`，左上角 + 宽高），`id` 保持唯一。

## 合成帧怎么审（只抽检 10%）

合成帧（`source=ai_cut` / `ai_bg_cut`，含 AI 背景合成）的 GT 是**粘贴时几何算出来的**，本来就精确，
所以**不要逐条对框**，只抽检 10% 查粘贴质量：

| 抽检项 | 处置 |
| --- | --- |
| 锥悬空（阴影/接触点不在同一平面） | 删掉该图并写 note（合成图是"可弃样本"，不是必须修） |
| 锥越出画面 / 尺寸明显不合理（比人还大等） | 删图 |
| 锥与球员框重叠（本该被脚本拦下） | 删图 + 报告脚本 bug（说明避让断言失效） |
| 背景里自带球/人/锥（AI 背景没筛干净） | 删图 + 该背景图移出 `ai_bg_cut` 的素材池 |

删图的动作在复核界面里就是"把这张图的框全删掉 + 在 note 里写 `drop: 原因`"，
`s3_label_review_roundtrip.py --import` 会把它记进 `review_diff.md`，之后该图不进任何 split。

## 提交前自检（脚本会强制）

```bash
cd tools/detection_finetune && source env.sh
$PY s3_label_convert_dataset.py --link      # 类别对拍 / 泄漏检查 / 文件名冲突
$PY s3_label_dataset_stats.py               # 类别分布 / 尺度分布 / 空标注统计
```

* `categories` 顺序或数量与 `classes.json` 不一致 → **直接报错**（这是故意的，错一位就是"球消失"或"锥桶又算进球"）；
* 同一段视频不得跨 split（同源泄漏）；
* bbox 越界会被裁回图内，面积 ≤ 1 px 的框被丢弃（丢弃数会打印出来，异常偏大要回头看标注）。

## 对照示例图（真实帧派生）

示例图由自有视频帧派生（`$PY s2c_data_spec_stress.py --spec`），框色：人=蓝、球=绿、锥=橙。
全部在 `$AI_FOOTBALL_DATASET_DIR/sources/spec_examples/`，逐图说明见同目录 `SPEC_INDEX.md`：

| 文件 | 看什么 |
| --- | --- |
| `spec_01_all_three_classes.jpg` | 同一帧里人/球/锥三类齐全，框该多大、贴多紧 |
| `spec_02_cone_and_ball.jpg` | 锥与球同帧：**别把锥标成球**（这正是本项目的核心误检） |
| `spec_03_tiny_targets.jpg` | 远景小目标（球在 960 下 ~12px）：要标，但别把<5px 的噪声框留下 |
| `spec_04_synth_sample.jpg` | 合成样本（`ai_cut_*`）：GT 是粘贴几何算的，抽检只看"贴得像不像" |

> 示例图若仍来自 bootstrap 预标注，人工复核完成后应重跑一次
> `$PY s2c_data_spec_stress.py --spec`（它会自动优先用复核后的 `GT.json`）。
