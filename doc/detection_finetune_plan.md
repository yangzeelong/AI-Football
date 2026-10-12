# 三分类微调（人 / 足球 / 锥形桶）方案与 Step-by-Step 操作手册

> 目标：消除"锥形桶（场地小锥、标志碟）被误检成足球"这一整类误检。
> 做法：自建三分类数据集 → 微调 RF-DETR small(960) → 导出 fp16/TRT → 两侧管线按类别过滤 → 交付 base vs 微调的前后对比。
> 本文档 = 方案 + 可照做的分步命令 + 每步的验收判据 + 人工工作量估计。

---

## 0. 一页速览

| 阶段 | 谁做 | 主要命令/产物 | 人工量（估计） |
| --- | --- | --- | --- |
| S0 环境自检 | 自动 | `base_layout.py --check --selftest` | 0 |
| S1 抽帧 | 自动 | `s1_data_extract_frames.py` → `data/ball_cone_v1/raw/**` + `manifest.csv` | 0 |
| S2 半自动预标注 | 自动（GPU） | `s2_data_preannotate.py` → `work/annotations/preannot.json`、`reports/*.csv`、三色拼图 46 张 | 0 |
| S2.5 合成扩充（真实背景 + **AI 背景**） | 自动 | `s2a_data_compose_cones.py`（抠素材 → 粘贴 → 自动精确 GT → 避让人体断言） | 0 |
| S2.6 AI 生成图（规范示例 + 独立压力集） | 自动（skill） | `s2b_data_gen_backgrounds.py` + `[skill:多模态内容生成]` | 0 |
| **S3 人工校验（最重）** | **人工** | **Label Studio / CVAT 往返**（`s3_label_review_roundtrip.py`）；合成图只**抽检 10%** | **1.5–3 h / 529 帧** |
| S4 回收 + 体检 | 自动 | `s3_label_convert_dataset.py` → `train/valid/test`；`s3_label_dataset_stats.py` → 体检报告 | 0（对不上会报错） |
| S5 冒烟 + 960 训练 | 自动（GPU） | `s4_train_finetune.py --config train_configs/*.yaml` | 0（只等） |
| S7 person 保障 | 自动 | `s4a_train_person_pseudo.py` / `s4b_train_person_coco.py` | 0（决策门触发才用） |
| S8 检测级 A/B + 导出 | 自动 | `s5_eval_detector.py` / `s6_deploy_export.py` | 0.5 h 看报告 |
| S9 两侧接入 + 回归 | 人工（改配置） | Nexusflow YAML / `RFDetrDetector.cpp` | 0.5–1 h |
| S10 闭环 + 汇报 | 自动 + 人工润色 | `s7_recycle_hard_examples.py`、`docx` 报告 | 0.5 h |

> **顺序要点**：S2.5 / S2.6 都排在 **S3 人工校验之前**——这样一次校验就把"真实帧 + 合成帧"的最终候选集过完，不需要二次开标注任务。
> 但**合成/AI 背景帧的 GT 是几何精确的**，所以它们在批次里只做**抽检 10%**（查粘贴质量），不逐条对框。
> 原"S6 合成锥桶扩充"已并入 S2.5；要做"有/无合成"消融时用 `s2a_data_compose_cones.py --ablate` 出两版数据集。

**一句话回答"最麻烦的是不是准备数据+人工校验"：是，而且几乎全部人工成本都压在 S3。**
其余环节都是一条命令的事；S3 之所以重，是因为"球 vs 锥 vs 地板花纹"这种判定只能靠人眼。
本文档 S3 给了**把人工量压到最小**的具体做法（只改类、只补漏、分层抽查），并给出时间预算。

---

## 0.5 环境准备（每个新终端做一次）

**所有脚本都必须用 `ai-football` 这个 conda 环境跑。**
⚠️ 本机默认激活的是 conda **base**（`/root/miniconda3`），**裸 `python` 是错的环境**——base 里没有 `rfdetr`。
四种写法任选一种（前两种最省事），后文一律用 `$PY` 代表"这个环境的 python"：

```bash
# 【方式 1 · 零配置，任何 shell / 脚本里都稳】（推荐写进脚本）
PY=/root/miniconda3/envs/ai-football/bin/python
$PY base_layout.py --check

# 【方式 2 · 本机默认 shell 是 zsh，~/.zshrc:129 已 conda init，可直接激活】
conda activate ai-football
python base_layout.py --check
# 激活后也才能直接用该环境的可执行文件：label-studio、pip、uv …

# 【方式 3 · bash / 非交互 shell：先 source hook 再激活】
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ai-football

# 【方式 4 · 一条命令、不改当前环境（适合一次性调用 / CI）】
conda run -n ai-football python base_layout.py --check
```

可选简写（写进 `~/.zshrc`）：

```bash
alias py3c='/root/miniconda3/envs/ai-football/bin/python'            # 少打字
export PATH=/root/miniconda3/envs/ai-football/bin:$PATH              # 让 python/pip/label-studio 都指向该环境
```

环境自检（三条都应通过）：

```bash
$PY -c "import torch, rfdetr, cv2; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
$PY -c "import rfdetr, os; print('rfdetr:', os.path.dirname(rfdetr.__file__))"   # 应指向 envs/ai-football/…
$PY --version                                                                    # 期望 Python 3.11.x
```

三个容易踩的点：

* **装包必须装进这个环境**：用 `$PY -m pip install <pkg>`（或先 `conda activate ai-football` 再 `pip install`）；
  裸 `pip install` 会装到 conda base，之后 `$PY -c "import ..."` 依然找不到。
* **该环境的可执行文件在 `/root/miniconda3/envs/ai-football/bin/`**：未激活时要写全路径，
  例如 `/root/miniconda3/envs/ai-football/bin/label-studio start --port 8080`。
* Nexusflow 侧的触球评估是另一套（`cd examples/touch_eval/py && uv run touch-eval …`），与 `$PY` 无关。

---

## 1. 已实测的关键契约（**不要照抄网上的老写法**）

这三条都踩过坑，写错任何一个都会白烧几小时：

### 1.1 数据集类别必须写成 `1/2/3` + 一个 `id=0` 的占位背景类

RF-DETR 用 `len(categories)` 当 `num_classes`（`rfdetr/detr.py:130-133`），而 logits 的最后一列是 no-object。
于是：

| 写法 | `len(categories)` | 头行数 | 结果 |
| --- | --- | --- | --- |
| 只写 `1/2/3` 三类 | 3 | 3 | 训练时 target 索引越界 → **CUDA device-side assert**（1 图冒烟"通过"只是因为那张图没有 cone，属假绿） |
| 写 `0/1/2` 三类 | 3 | 3 | 能训，但部署侧 classId 变 0/1/2；Python 侧 `detr.py:216` 的类别名映射是 `{i+1: name}`，person（列 0）查不到名字 → 被 `football_vision.py:543` 丢掉 |
| **`0=background(supercategory:"none")` + `1/2/3`（本方案）** | 4 | 4 | ✅ 预测 classId = 1/2/3，两侧默认值全对上 |

`supercategory:"none"` 的占位类会被 `detr.py:133` 从 `class_names` 里滤掉，所以类别名仍是 3 个、`num_classes` 是 4。

### 1.2 部署侧 `classId` **不加 1**，就是 logits 列下标

`RFDetrDetectorInfer.cpp:751-752` 是 `Candidate{Sigmoid(logit[c]), q, c}` —— 直接用列下标当 classId。
所以 Nexusflow 侧该写：

```yaml
RFDetrDetector.config.numClasses: 4          # 会被 engine 的 logits 宽度覆盖，只求书面一致
RFDetrDetector.config.targetClasses: [1, 2, 3]
ByteTracker.config.personClassId: 1
ByteTracker.config.ballClassId: 2
ByteTracker.config.coneClassId: 3
FootballTracker.config.ballClassId: 2
```

`RFDetrDetector.cpp:178-179` 的 `classId==1 → person / ==37 → ball` **只影响统计计数**，真正过滤在
`RFDetrDetectorInfer.cpp:725/767` 读 `targetClasses`。

### 1.3 分辨率只改 `resolution`，**不要**动 `positional_encoding_size`

base 权重的位置编码是 32² = 1025；传 960 对应的 60² = 3601 会在 `main.py:130` 的
`load_state_dict(strict=False)` 上直接 size mismatch（`strict=False` 只忽略缺/多键，**形状不符照样抛**）。
960 输入靠窗口注意力 + 插值，现有 `rf-detr-small-960-*` engine 就是这么建出来的。

### 1.4 另外两条小契约

* `.train(class_names=[...])` **必须显式传**，否则 `main.py:165-167` 不会采用它的 `num_classes`；
* roboflow 布局必须有 `train/ valid/ test/` 三个目录（`run_test` 默认开启），且 `file_name` 要写成
  `images/<文件名>`（加载器按 `dataset_dir/<split>/file_name` 找图）。

---

## 2. 人工工作量：真相与压缩办法

### 2.1 为什么 S3 是瓶颈

判定"这个绿框是足球还是场地小锥/标志碟"没有自动化捷径：

* 小锥/碟只有 **7–35 px**（720p 源 → 960 输入后 ~7–26 px），比足球还小；
* 它和足球在 base 模型眼里都是 `sports ball`，置信度 0.30–0.65，与真球分布重叠；
* 真球也会停在地上不动（这时"静止"判据失效），必须逐个人眼看。

### 2.2 现成工具已经把人工量压到了什么程度

* base 模型自动给出 **人（5355 框）/ 球（2077 框）** → 这两类**只抽查不改**；
* **锥候选 69 个**（score≥0.6 共 47 个）已经排在 CSV 里，从高分往下看即可；
* 三色拼图 46 张（每张 12 帧，球=绿、锥候选=红、人=灰）→ **看"哪些绿框其实是锥"比看原图快 5–10 倍**；
* 一个锥/碟物体在同一位置会被反复检出（实测：`(886,189)` 那一处触发了 5 次 FP）→ **改一处等于修一串**。

### 2.3 具体压缩策略（建议照这个顺序做）

| 优先 | 做什么 | 覆盖量 | 估计耗时 |
| --- | --- | --- | --- |
| ① | 只看 69 个候选（CSV 从高分到低分）：是锥就改成 cone；不是就留 ball | 69 处 | 30–50 min |
| ② | 翻 test 的 7 张拼图 + train 里室内那几段（移动传球/颠球/绕障碍/1V2/a/未命名/折返跑）：把**候选漏掉**的小锥补上 | 约 150 帧 | 60–90 min |
| ③ | 抽查 10% 的人/球框（尤其室外 1080p）确认没有大错；明显漏标的人框补几个 | 约 60 帧 | 20–30 min |
| ④ | **合成帧抽检 10%**（S2.5 产出，只查粘贴质量：悬空/越界/与人物重叠 → 删图记 note） | 约 30–50 张 | 10–20 min |
| ⑤ | 合成锥桶把 cone 实例补到 300–500 个（S2.5，全自动，GT 精确） | 无需人工 | 0 |

> 也就是说：**第一版数据集的人工成本 = 约 1.5–3 小时**；之后每轮回灌（S10）通常 0.5 h 以内。

### 2.4 人工怎么改：GUI 往返（推荐）

用支持 COCO 的 Web 工具改（本机 headless，用浏览器从你自己的电脑连过来），往返由脚本保证：

```bash
# ① 导出待标包（zip：images/ + preannot.json + annotation_spec.md + batch.json）
$PY s3_label_review_roundtrip.py --export --split train

# ② 浏览器里改（Label Studio 或 CVAT，步骤见 S3）
#    只做三件事：把"其实是小锥/标志碟"的 ball 改成 cone、补漏、删掉认不出的框

# ③ 导入导出结果：按【类别名】映射回 classes.json，并出差异报告
$PY s3_label_review_roundtrip.py --import ~/Downloads/project-*.json --batch reports/review_batches/train.zip
#    → work/work/annotations/GT.json + reports/review_diff.md（改类 N / 新增 M / 删除 K / 移动 L）
```

**备选（无 GUI 环境）**：直接改 JSON —— `cp work/annotations/preannot.json work/work/annotations/GT.json`，
改 `annotations[].category_id`（`2`=ball → `3`=cone）、删条目、或复制邻近条目补框（`bbox` 是 `[x, y, w, h]`）。

无论哪种方式，最后都必须过 S4 的对拍：`categories` 顺序/数量对不上会**直接报错**（这是故意的）。

---

## 3. Step-by-Step

所有命令的工作目录：`/home/hx1/yzl/Work/AI-Football/tools/detection_finetune/`
Python 解释器：**一律用 ai-football 环境**（下称 `$PY`，设置方式见 §0.5）——
⚠️ 裸 `python` 默认是 conda **base**，里面没有 `rfdetr`，跑脚本会 `ModuleNotFoundError`。

约定：`DS=/home/hx1/yzl/Work/data/ball_cone_v1`

---

### S0 环境与类别真源自检（0 分钟）

```bash
cd /home/hx1/yzl/Work/AI-Football/tools/detection_finetune
$PY base_layout.py --selftest     # 对拍函数自检：6 类故意写错的 categories 都必须被抓出来
$PY base_layout.py --check        # 打印三方对齐值 + 目录骨架状态
```

**通过判据**：`列数=4，class_names=['person','ball','cone']`；`Nexusflow YAML : numClasses=4 targetClasses=[1,2,3] personClassId=1 ballClassId=2 coneClassId=3`；骨架缺失项为空。

---

### S1 抽帧（约 1.5 分钟 GPU 空闲，纯 CPU）

```bash
$PY s1_data_extract_frames.py --priority high,normal --stride-sec 3 --per-video-max 30 --limit-total 600
```

* `split_policy.json` 决定每段视频的 `split/scene/view/priority`；**split 按整段视频划分**（同源不跨集，避免泄漏），
  `test` 同时是 S8 的固定评测集（`移动传球.mov`、`颠球.mov`、`室外/右边比赛1`）。
* 相邻帧用 16×16 dHash 去重（阈值 6/256 ≈ 2.3%）；**源分辨率原样保存不缩放**（部署侧 letterbox 到 960，
  源分辨率不改变模型看到的目标像素数，决定目标大小的是取景/距离）。

**产物**：`$DS/raw/<视频>/<视频>__f000540__t0009.00.jpg`（529 张，243 MB）+ `$DS/manifest.csv`（529 行）
**通过判据**：`du -sh $DS/raw` ≈ 243 MB；`manifest.csv` 行数 = 图片数；分辨率分布 `{1920x1080: 172, 1280x720: 357}`

> 想加新素材：往 `/home/hx1/yzl/Work/data` 丢视频 → 在 `split_policy.json` 里登记 split/场景 → 重跑本步（已登记的图会自动跳过）。

---

### S2 半自动预标注（约 1 分钟 GPU）

```bash
$PY s2_data_preannotate.py --resolution 960 --threshold 0.25 --montage-splits train,valid,test
```

做四件事：① base 模型出 person/ball 框（逐类 NMS，避免重复框）；② 跨帧聚类找**静止球状物**当锥候选；
③ 写 COCO 预标注；④ 出三色拼图。

**产物**

| 文件 | 内容 |
| --- | --- |
| `$DS/annotations/preannot.json` | 529 图 / 7479 框（person 5355 + ball 2077，锥仍是 ball 待人工改） |
| `$DS/reports/preannot_cone_candidates.csv` | **69 个锥候选**（score≥0.6 共 47），带 `score/persistence/spread_px/sat/texture/inside_person/xyxy` |
| `$DS/reports/montage_{train,valid,test}_*.jpg` | 46 张三色拼图（球=绿、锥候选=红、人=灰） |

**通过判据**：候选 `persistence ≤ 1.0`、`spread_px` 通常 < 1 px（静止物应几乎不动）；拼图里能肉眼看清框。

---

### S2.5 合成扩充：真实背景 + **AI 生成背景**（自动，无需人工）

```bash
# ① 抠素材：从 S2 的高分候选里抠出锥/碟 RGBA（几何已知 → 以后的 GT 精确）
$PY s2a_data_compose_cones.py --extract --min-score 0.75 --min-side 8

# ② 背景池：真实帧（默认）+ AI 生成背景（S2.6 产出的目录）
$PY s2a_data_compose_cones.py --compose --n 400 --backgrounds mixed \
                   --bg-dir $DS/sources/ai_gen_bg --max-ratio 0.30

# ③ 只做"有/无合成"的消融版本（不覆盖主数据集，另开一份）
$PY s2a_data_compose_cones.py --ablate        # → /tmp/ds_no_synth 与 /tmp/ds_with_synth
```

**AI 生成图在这里的唯一用途 = 当背景**：AI 出"没有待标物体的球场/草地/看台"，
我们把**真实锥素材**贴上去 —— 因此 **GT 仍然是几何精确的**，同时借 AI 换来场景多样性
（黄昏、强光、湿滑、陌生场地），缓解"合成图背景全是自家球馆 → 过拟合"。

硬约束（脚本会断言，违反直接失败）：

* 合成样本 ≤ 训练集 **30%**（`--max-ratio`，超出即报错）；
* 粘贴位置**必须避让 person 框**（重叠 → 断言失败；错误 GT 比没有 GT 更糟）；
* 越界/悬空/尺度过大过小的粘贴一律拒绝；
* 每张合成图在 `manifest.csv` 记 `source=synthetic` + `is_synthetic=yes` +
  `ai_background=yes/no`（用于**单列统计**与消融）；
* AI 背景图进 `ledger.csv` 台账（生成工具、时间、提示词、许可=自有生成）。

### S2.6 AI 生成图：规范示例 + 独立压力集（自动 + skill）

```bash
$PY s2b_data_gen_backgrounds.py --prompts      # 生成提示词包（背景池 / 规范示例 / 压力集三类）
# 用 [skill:多模态内容生成] 按提示词包出图，落到：
#   $DS/sources/ai_gen_bg/          ← 背景池（进 S2.5，会参与训练）
#   $DS/sources/spec_examples/      ← 标注规范对照示例（真实帧画框；AI 服务停用后改派生）
$PY s2b_data_gen_backgrounds.py --validate --dir $DS/sources/ai_gen_bg
```

`--validate` 会做四件事：尺寸/长宽比合法、无明显水印与文字、拉普拉斯方差不过低（不是糊图）、
**用 base 模型确认背景里没有球/人/锥**（有就剔除——背景不能自带待标物体），
并把通过的图登记进 `manifest.csv`（`source=ai_gen_bg`）与 `ledger.csv`。

**压力集与训练集严格隔离**：不抽帧、不进 S4、只用于 S8 报告里的"分布外稳健性"一栏。

### S3 人工校验（GUI 往返）

**只复核真实帧的框 + 锥候选改类；合成帧只抽检 10%**（查粘贴质量：悬空 / 越界 / 压人）。

```bash
cd AI-Football/tools/detection_finetune && source env.sh

# ① 出批次（默认跳过已进过批次的图；100 图/批；train 400 张 = 4 批）
$PY s3_label_review_roundtrip.py --export --split train --source raw_video --limit 100
$PY s3_label_review_roundtrip.py --export --split valid --source raw_video --limit 100
$PY s3_label_review_roundtrip.py --export --split test  --source raw_video --limit 100
$PY s3_label_review_roundtrip.py --export --split all --source ai_cut,ai_bg_cut --synth-sample 0.1   # 合成抽检
$PY s3_label_review_roundtrip.py --status          # 看批次与复核进度
```
每批产出：`<batch>.zip`（`images/` + `preannot.json` + 标注规范 + `batch.json`，给 CVAT/离线）、
`ls_tasks.json`（Label Studio 任务，含预标注预测）、`preannot.json`（COCO）。

**复核界面（主用：`s3_label_review_ui.py`，零依赖，标准库 http.server + Canvas）**

```bash
$PY s3_label_review_ui.py                     # 默认载入全部批次，界面内下拉切换（--batch a,b 可只载指定批）
# 本地另开终端：ssh -L 8090:localhost:8090 <user>@<host> → 浏览器 http://localhost:8090
# 拖拽=框选多个框（Shift 加选）→ 1/2/3 批量改类、Del 批量删除；点框=选中，单击空白/Esc=清空
# N=新建模式（拖拽画新框，类别取"新建框类别"下拉，默认 cone）；Ctrl+Z=撤销
# ←/→ 翻图；Enter=本图已核对并跳下一张未核对；S 保存
# **C=叠加"上一张新增/改过的框"（本图原有框保留；Shift+C=整体覆盖）**；
# 勾"叠加后自动下一张"可一路 C 过连帧
# 保存产物：reports/review_batches/<batch>/<batch>_edited.json（与 preannot.json 同构，仍按批分）
```

**Label Studio（备选，本机装不上）**

```bash
cd AI-Football && LABEL_STUDIO_LOCAL_FILES_SOURCES=/home/hx1/yzl/Work/data/ball_cone_v1 \
  LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/hx1/yzl/Work/data/ball_cone_v1 \
  .venv-labelstudio/bin/label-studio start --port 8080 --no-browser
# 本地另开终端：ssh -L 8080:localhost:8080 <user>@<host> → 浏览器 http://localhost:8080
$PY s3_label_review_roundtrip.py --ls-tasks <batch>      # 打印导入用的 curl（Token 在 Settings → Access Token）
```
页面里：`Create Project` → Labeling Setup 加 3 个 **RectangleLabels：`person` / `ball` / `cone`**
（**名字必须与 `classes.json` 完全一致，大小写敏感**）→ 用上面的 curl 导入（带预标注）→ 改 → **Export → COCO**。

**CVAT（备选，换到 registry 可达的机器）**：`docker compose up -d` → project（Labels 同名）
→ task 上传 `<batch>.zip` 里的 `images/` → **Upload annotations → COCO 1.0**（`preannot.json`）
→ 改 → **Export → COCO 1.0**。

**回收（脚本按类别名映射，绝不按 id）**

```bash
$PY s3_label_review_roundtrip.py --import <导出的 coco.json> --batch <batch>
#   → work/work/annotations/GT.json（全量可训练标注：复核帧用人工结果，其余保留基线）
#   → work/annotations/reviewed.json（复核台账）、excluded.txt（判废图，0 框的自动进）
#   → reports/review_diff.md / review_diff.csv（改类 N · 新增 M · 删除 K · 移动 L，逐条明细）
$PY s3_label_convert_dataset.py --link     # 默认优先用 review.json，并跳过 excluded.txt 里的图
$PY s3_label_dataset_stats.py              # 复查类别分布与合成占比
```

**判据**：每批复核完后差异报告里 `kept` 占绝大多数、`added/deleted` 是人工补漏/删误标；
合成抽检批次里凡"粘贴悬空/越界/压人"的图删成 0 框 → 自动进 `excluded.txt`，不进任何 split。

### S4 标注回收 + 数据体检（约 10 秒）

```bash
# 第一版（还没人工校验完）可以先 bootstrap：把候选自动改类，够跑通流程
$PY s3_label_convert_dataset.py --cone-min-score 0.6 --link

# 人工校验完（正事）：
$PY s3_label_convert_dataset.py --annotations work/work/annotations/GT.json --link
$PY s3_label_dataset_stats.py
```

**做的事**：统一 COCO → 写 `train/valid/test` 三目录 → **类别对拍 classes.json**（错一位直接抛错）→
同源视频跨 split 检查 → 文件名冲突检查 → bbox 越界裁剪（实测裁回 22 个）→ 体检报告。

**产物**：`$DS/{train,valid,test}/{images,_annotations.coco.json}`、`$DS/reports/dataset_stats.{md,json}`
**通过判据**：体检表的"三目录类别对拍"= ✅；"合成样本占比 ≤ 30%" = ✅；`valid` 非空。

---

### S5 冒烟 → 960 正式训练

```bash
# ① 冒烟（1 分钟）：必须用"含 cone"的子集，否则会假绿
$PY s4_train_finetune.py --config train_configs/smoke_512.yaml --subset 24 --subset-require-cone --epochs 6

# ② 正式（960，保守超参 + EMA + 早停）
$PY s4_train_finetune.py --config train_configs/v1.yaml --freeze encoder    # 掉点超阈值时才加 --freeze
```

**冒烟通过判据**：`class_embed.bias 列数 = 4`；COCO 评估的 `class_map` 里**出现 `cone`**；
预测 classId 只出现 1/2/3（实测样例：`{1: 174, 2: 15, 3: 1}`）。

**`person` 决策门**：960 训练后与 base 在固定评测集上的 `person mAP50` 比，**掉点 ≤ 1 个百分点**算通过；
超阈值按四级递进升级（S7 ①②③④），每一步的指标变化写进报告。

---

### S6 已并入 S2.5

原"合成锥桶扩充"现在排在人工校验之前（S2.5），并支持 AI 生成背景。
要做"有/无合成"的消融，用 `s2a_data_compose_cones.py --ablate` 生成两版数据集，在 S8 里对比。

---

### S7 person 保障四级递进（按决策门结果启用）

```bash
$PY s4a_train_person_pseudo.py --confidence 0.6          # ① 伪标签扩量（零人工）
$PY s4_train_finetune.py --config train_configs/v1.yaml --freeze encoder   # ② 保守微调/选择性冻结
python s4b_train_person_coco.py --target-ratio 0.5        # ③ 混入 COCO val2017 person 子集（先测外网可达性）
# ④ 双模型兜底：base 出 person + 微调出 ball/cone（管线最复杂，最后手段）
```

---

### S8 检测级 A/B + 导出

```bash
$PY s5_eval_detector.py --split test --resolution 960 \
    --model base=models/rfdetr/rf-detr-small.pth \
    --model v1-frozen-step2=runs/v1-frozen-step2/checkpoint_best_ema.pth
$PY s5_eval_ledger.py            # 幂等累积指标台账（跨次对比用）
$PY s6_deploy_export.py --checkpoint runs/v1-frozen-step2/checkpoint_best_ema.pth --resolution 960
```

**主指标**：固定评测集上"锥形桶被判成球的帧数"下降比例（base vs 微调）；
次级：ball 按帧 P/R、`cone` mAP50、`person` mAP50 不掉点。输出逐帧差异 CSV + 按来源（真实/合成）消融。

**指标台账（后续对比的唯一入口）**：`s5_eval_ledger.py` 扫描 `runs/**/metrics_*.json`，按
`(tag, split, resolution)` 幂等累积——`runs/metrics_ledger.{json,csv}` 是机器可读全量历史（含来源 sha、
**评测集 GT 指纹**、checkpoint sha），`<data>/ball_cone_v1/reports/metrics_ledger.md` 是人读表
（各 split 最新一份 + 相对 base 的差值 + **person 决策门逐档判定** + 历史 revision）。
`compare.md/compare.json` 是**覆盖式**的（只反映最近一次 A/B），别拿它做跨次对比。

> ⚠️ 导出脚本的产物名只由 `--resolution` 决定，**会覆盖现有 COCO ONNX**：必须换 `--output-dir` 或先备份
> `models/rfdetr/onnx/rf-detr-small-960-dynamic.onnx`。fp16 与 engine 编译用
> `AI-Football/tools/model_conversion/{convert_fp16_onnx.py,build_engines.sh}`（装完 ONNX 工具链记得 `pip install "numpy<2"`）。

---

### S9 两侧接入 + 端到端回归

1. **Python 侧**：只需在 YAML 里把 `models.detector.class_names` 改成 `["person","ball","cone"]`
   （`football_vision.py:495` 用 checkpoint 自带的 `class_names`；`labels` 段仍是 `{person, ball}`，
   `app_config.py:197` 的严格键校验**不用改代码**）。
2. **C++ 侧**：新增 `examples/aifootball_app/config/finetuned_3c.yaml`（§1.2 的五个键）；
   `RFDetrDetector.cpp` 的统计计数改为读 `personClassId/ballClassId/coneClassId`（默认 1/37/-1 保持向后兼容）。
3. **cone 抑制规则**：球候选与 cone 框高度重叠/邻近 → 抑制该球（阈值可配、默认保守，避免误杀真球）。
4. **回归**：`examples/touch_eval/py` 跑 `移动传球`，对比 base 基线
   **名额级含 low_quality P/R/F1 = 0.8144**，允许 1–2 个百分点内的波动。

---

### S10 闭环回收 + 汇报

```bash
$PY s7_recycle_hard_examples.py --video /home/hx1/yzl/Work/data/移动传球.mov
```

一条命令：跑视频 → 挑出误检帧（cone 被当球）→ 导出待标注图 + 预标注 → 回到 S3/S4 → 重训。
最后用 `docx` skill 把 A/B 报告导出成可直接发送的 Word 版（内容与本文档一致）。

---

## 4. 验收阈值（默认值，可按你意见调整）

| 指标 | 阈值 |
| --- | --- |
| 锥形桶被判成球的帧数 | 下降 ≥ 90% |
| `ball` 按帧召回 | 相对 base 掉 ≤ 2 个百分点 |
| `person` mAP50 | 绝对下降 ≤ 1 个百分点 |
| `cone` mAP50 | ≥ 0.5（合成+真实混合后） |
| 端到端触球（名额级含 low_quality P/R/F1） | ≥ 0.8044（基线 0.8144 − 1 个点） |

---

## 5. 已存在的产物清单（当前进度）

### 5.1 目录命名规则（唯一口径）

| 前缀 | 含义 | 目录 | manifest `source` | 生成脚本 | 进训练 |
| --- | --- | --- | --- | --- | --- |
| `raw_video_*` | **原始数据**：现场视频抽帧 | `sources/raw_video/` | `raw_video` | `s1_data_extract_frames.py` | ✅ |
| `coco_*` | **COCO 数据集**里来的 | `sources/coco_person/` | `coco_person` | `s4b_train_person_coco.py` | ✅（兜底三级） |
| `ai_gen_bg_*` | AI 生成的**空场地背景**（**素材池**，不直接训练；裁水印只做一次，原图备份 `_orig/`） | `sources/ai_gen_bg/` | `ai_gen_bg` | `s2b_data_gen_backgrounds.py` | ❌（只作粘贴背景） |
| `spec_*` | 标注规范对照示例（真实帧画框） | `sources/spec_examples/` | `spec_example` | `s2c_data_spec_stress.py` | ❌ |
| `stress_*` | 分布外压力集（难帧队列 + 原因） | `sources/stress_set/` | `stress_set` | `s2c_data_spec_stress.py` | ❌ |
| `pseudo_person_*` | 模型伪标签产物 | `sources/pseudo_person/` | `pseudo_person` | `s4a_train_person_pseudo.py` | ✅（兜底一级） |
| **`ai_cut_*`** | **AI（脚本）抠图粘贴到真实背景**，GT 几何精确 | `work/ai_cut/` | `ai_cut` | `s2a_data_compose_cones.py` | ✅ |
| **`ai_bg_cut_*`** | **AI 背景 + 抠图粘贴**，GT 几何精确 | `work/ai_bg_cut/` | `ai_bg_cut` | `s2a_data_compose_cones.py` | ✅ |
| `web_*` | 网络/公开渠道采集（需许可台账） | `sources/web/` | `web` | `s1_data_collect_web.py` | ✅（需许可） |

**脚本命名规则**：**产出来源的脚本按它产出的前缀命名**（上表"生成脚本"列）；流程步骤脚本仍用动词：
`s2_data_preannotate.py`（半自动预标注）· `s3_label_review_roundtrip.py`（GUI 往返）· `s3_label_convert_dataset.py`（回收成三目录）·
`s3_label_dataset_stats.py`（体检）· `s4_train_finetune.py`（微调）· `s6_deploy_export.py`（导出对拍）·
`s5_eval_detector.py`（A/B）· `s7_recycle_hard_examples.py`（闭环回灌）。

三层结构：**`dataset/`（① 最终数据，可直接训练）→ `sources/`（② 素材，原样保存）→ `work/`（③ 中间产物，可整目录删掉重建）**。
`base_layout.py --tree` 打印目录树 + 每目录一行说明；`base_layout.py --readmes` 会在每个目录写 `_README.md`（自解释"这目录干什么用的"）；
`base_layout.py --migrate` 负责按新命名搬目录并改写文件内路径与 `source` 取值（幂等）。

### 5.2 当前产物

```
/home/hx1/yzl/Work/data/ball_cone_v1/
├── dataset/                     # ① 最终数据（s3_label_convert_dataset.py 生成，可整目录重建）
│   ├── train/images  572 张（_annotations.coco.json：6265 框；person 4192 / ball 1640 / cone 433）
│   ├── valid/images   53 张（纯真实）
│   └── test/images    75 张（纯真实 = 固定评测集：移动传球 / 颠球 / 室外右边比赛1）
├── sources/                     # ② 素材（原样保存，不直接训练）
│   ├── raw_video/              529 张抽帧（243 MB；720p 357 + 1080p 172；24 个视频目录）
│   ├── ai_gen_bg/              ✅ 3 张 AI 空场地背景（已裁水印 1280x677，base 模型确认无球/人/锥）
│   ├── spec_examples/          ✅ 6 张规范对照示例图 + SPEC_INDEX.md（不进训练）
│   ├── stress_set/             ✅ 30 帧压力集 cohort.json + stress_sheet.jpg（原地引用）
│   ├── coco_person/  pseudo_person/   ⏳ 兜底数据（目录已就位）
├── work/                        # ③ 中间产物（可删可重建）
│   ├── annotations/preannot.json    base 预标注 529 图 / 7479 框
│   ├── cutouts/                ✅ 12 个锥素材（6 条质量闸门筛过）+ cutouts.json
│   ├── ai_cut/images           ✅ 128 张合成（339 锥框，真实背景）
│   ├── ai_bg_cut/images        ✅ 43 张合成（94 锥框，AI 背景）
│   └── prompts/                三类提示词包（背景池 / 规范示例 / 压力集）
├── manifest.csv                703 行（raw_video 529 + ai_gen_bg 3 + ai_cut 128 + ai_bg_cut 43）
├── ledger.csv                  许可台账（自有素材 / AI 生成 / 公开数据）
└── reports/                    54 项：46 张三色拼图 · 锥候选 CSV(69) · synth_qa.jpg · dataset_stats.md/json
                                 ⏳ review_batches/*.zip（待标包）· review_diff.md（往返差异）

AI-Football/tools/detection_finetune/
├── classes.json · base_layout.py（含 --tree/--readmes/--migrate）· split_policy.json
├── s1_data_extract_frames.py · s2_data_preannotate.py · s3_label_convert_dataset.py · s3_label_dataset_stats.py      ← ✅
├── s4_train_finetune.py · train_configs/{smoke_512,full_960}.yaml                           ← ✅（冒烟通过）
├── s2a_data_compose_cones.py（S2.5 抠图+合成+抽检+消融）· s2b_data_gen_backgrounds.py（S2.6 提示词+校验+台账） ← ✅
├── s3_label_review_roundtrip.py（S3 导出待标包 / 导入 LS·CVAT 结果 + 差异报告）             ← ⏳ 待实现
├── s1_data_collect_web.py · s4a_train_person_pseudo.py · s4b_train_person_coco.py
├── s5_eval_detector.py · s6_deploy_export.py · s7_recycle_hard_examples.py
├── env.sh（source 即设好 $PY 与 PATH）· README.md · docs/annotation_spec.md
└── runs/smoke_cone/            冒烟产物（6 epoch：head=4 列，cone 进 class_map）
```

---

## 6. 坑清单（含本轮实测新增）

* **类别**：`1/2/3` + `id=0` 占位（§1.1）；`supercategory` 不能给真实类写 `"none"`。
* **分辨率**：只改 `resolution`（§1.3）。
* **部署 classId**：不加 1（§1.2）。
* 训练必须显式传 `class_names`；三目录必须都有 `_annotations.coco.json`；`file_name` 必须 `images/...`。
* 装 ONNX/fp16 工具链后压回 `numpy<2`；`onnx.save` 外部数据会追加写，保存前删旧文件；换 GPU/TRT 版本必须重建 engine。
* 合成图粘贴与 person 框重叠会造出错误 GT → **必须避让 + 断言**。
* COCO 里 `iscrowd=1` 的实例会被数据加载器丢弃，统计实例数要扣除。
* `imwrite` 不会建父目录，写失败还是静默的 → 本套脚本已加"写失败即报错 + 空文件检查"。
* 抽帧去重别用 8×8 dHash（会把真实变化当重复，实测 30 帧只剩 2 帧），用 16×16。
* **GUI 往返**：Label Studio 与 CVAT 都会**重排 category id**（常 0-based、按标签顺序）→ 只按**名字**映射
  （`s3_label_review_roundtrip.py` 已内置）；标签名与大小写必须与 `classes.json` 完全一致；
  导出里图片路径可能带 `images/` 前缀或 URL → 按 basename 归一；`bbox` 统一 `[x,y,w,h]`。
* **AI 背景**：背景图必须先用 base 模型确认**没有球/人/锥**（背景自带待标物体 = 造出错标签），
  并过尺寸/长宽比、模糊度（拉普拉斯方差）、水印与文字筛查；AI 背景占比**单列统计**，不许混进"真实"那栏。
  生成器常在角落烧"AI生成"水印 → 必须裁掉（否则模型会学到"水印↔锥"的伪相关）。
* **目录迁移**：`_merge_move` 遇到"目标同名目录已存在"必须**递归合并**，直接跳过会漏搬整个子目录
  （实测 `work/synthetic/images/*` 171 张全没搬过去）；合并多个标注文件时要给各自的 id 加**命名空间偏移**，
  否则同一 `image_id` 会把别的桶的标注一起吸过来（实测 cone 从 433 涨到 1889）。
* **合成帧**：粘贴必须避让 person 框（重叠即断言失败）；人工只**抽检 10%**，发现悬空/越界/重叠就**删图**并记 note。

---

## 7. 数据合规

* 自有素材：`license=自有素材`，可全量用于训练与评测；
* 网络/公开数据：入库前登记 `ledger.csv`（URL/时间/许可/可否再分发），只入库许可明确的；
* **AI 生成图分两种用途**：① **当合成背景**（图上本来没有待标物体，粘贴真实锥/碟后 GT 仍精确）→
  可进训练集，按 `source=ai_gen_bg` 登记，并把生成工具/时间/提示词写进 `ledger.csv`；
  ② **AI 原图（自带球/人/锥）** → 只做标注规范示例与分布外压力集，**不进训练集**，与训练集严格隔离；
* COCO val2017（若启用第③级）：标注 CC-BY 4.0、图片为 Flickr 各自许可 → 仅内部训练用途并登记台账。

---

## 8. 不自动提交

所有改动留在工作区；需要提交时按你指示执行。汇报一律带命令原文与数字。
