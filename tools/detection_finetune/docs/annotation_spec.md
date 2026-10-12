# 标注规范（人 / 足球 / 锥形桶）

配套：三色抽查拼图（`ball_cone_v1/reports/montage_*.jpg`）、锥桶候选清单（`reports/preannot_cone_candidates.csv`）。

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

## 改标注的操作方式：GUI 往返（推荐）

完整 UI 步骤见 `../../doc/detection_finetune_plan.md` 的 S3；这里只列要点。

```bash
cd /home/hx1/yzl/Work/AI-Football/tools/detection_finetune
# $PY = /root/miniconda3/envs/ai-football/bin/python（必须用该环境，见 plan 文档 §0.5；裸 python 是 conda base）
$PY s3_label_review_roundtrip.py --export --split train        # ① 出待标包 reports/review_batches/train.zip
#  ② 浏览器里改（Label Studio 或 CVAT）
$PY s3_label_review_roundtrip.py --import ~/Downloads/project-*.json --batch reports/review_batches/train.zip
#  → work/annotations/GT.json + reports/review_diff.md（改类 N / 新增 M / 删除 K / 移动 L）
```

**Label Studio**（本机唯一可行：pypi 可达、docker registry 不可达、机器 headless）：
`pip install label-studio` → `label-studio start --port 8080 --no-browser` → 本机 `ssh -L 8080:localhost:8080`
→ 建项目（Labels 必须叫 `person`/`ball`/`cone`）→ Import 选 **COCO** 传 `preannot.json`
→ 改完 Export **COCO**。

**CVAT**（需 docker registry 可达）：`docker compose up -d` → 建 project（Labels 同样三个名字）
→ 建 task 上传 `images/` → **Upload annotations → COCO 1.0**（传 `preannot.json`）
→ 改完 **Export → COCO 1.0** → 取 zip 里的 `annotations/instances_default.json` 交给 `--import`。

> 两个工具都会**重排 category id**，`s3_label_review_roundtrip.py` **按名字**映射回 `classes.json`；
> 标签名差一个字符就会在导入时报"未知标签"。图片路径带 `images/` 前缀或 URL 会被自动归一。

### 备选：直接改 JSON（无 GUI 环境）

```bash
cp /home/hx1/yzl/Work/data/ball_cone_v1/work/annotations/preannot.json \
   /home/hx1/yzl/Work/data/ball_cone_v1/work/annotations/GT.json
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
| 背景里自带球/人/锥（AI 背景没筛干净） | 删图 + 该背景图移出 `ai_backgrounds/` |

删图的动作在 GUI 里就是"把这张图的框全删掉 + 在 note 里写 `drop: 原因`"，
`s3_label_review_roundtrip.py --import` 会把它记进 `review_diff.md`，S4 之后该图不进任何 split。

## 提交前自检（脚本会强制）

```bash
cd /home/hx1/yzl/Work/AI-Football/tools/detection_finetune
$PY s3_label_convert_dataset.py --annotations work/annotations/review.json --link   # 类别对拍 / 泄漏检查 / 文件名冲突
$PY s3_label_dataset_stats.py                                                  # 类别分布 / 尺度分布 / 空标注统计
```

* `categories` 顺序或数量与 `classes.json` 不一致 → **直接报错**（这是故意的，错一位就是"球消失"或"锥桶又算进球"）；
* 同一段视频不得跨 split（同源泄漏）；
* bbox 越界会被裁回图内，面积 ≤ 1 px 的框被丢弃（丢弃数会打印出来，异常偏大要回头看标注）。

## 对照示例图（真实帧派生）

云端图片生成服务停用后，示例图改为从自有视频帧派生（`s2c_data_spec_stress.py --spec`），框色：人=蓝、球=绿、锥=橙。全部在 `/home/hx1/yzl/Work/data/ball_cone_v1/sources/spec_examples/`，逐图说明见同目录 `SPEC_INDEX.md`：

| 文件 | 看什么 |
| --- | --- |
| `spec_01_all_three_classes.jpg` | 同一帧里人/球/锥三类齐全，框该多大、贴多紧 |
| `spec_02_cone_and_ball.jpg` | 锥与球同帧：**别把锥标成球**（这正是本项目的核心误检） |
| `spec_03_tiny_targets.jpg` | 远景小目标（球在 960 下 ~12px）：要标，但别把<5px 的噪声框留下 |
| `spec_04_synth_sample.jpg` | 合成样本（`ai_cut_*`）：GT 是粘贴几何算的，抽检只看"贴得像不像" |

> 注意：示例图当前仍来自 bootstrap 预标注，S3 人工复核完应重跑 `s2c_data_spec_stress.py --spec`（它会自动优先用 `work/annotations/review.json`）。
