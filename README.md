# AI-Football

用于足球视频单视角感知与观测输出的实验项目。

## 目录结构

```markdown
├── code                    # 单视角推理管线（检测 / 跟踪 / 关键点 → observations.jsonl）
├── config                  # ROI、模型路径、app 过滤配置
├── doc                     # 方案与路线图文档
├── models                  # 本地权重（.pth / .pt / .engine，不进 git）
├── reports                 # 质量评估报告输出
├── runs                    # 跑批产物
├── scripts                 # 可直接参考的示例脚本
├── tools                   # 微调与部署流水线（各有自己的 README，见下节）
│   ├── detection_finetune  # 抽帧 → 标注 → 复核 → 三分类微调
│   ├── model_conversion    # ONNX / fp16 / TensorRT engine 转换
│   └── rollout             # 版本 → 导出 → C++ 配置 → 批量跑视频 → 汇总
├── tmp                     # 中间输出，如 JSONL 观测文件
├── requirements.txt
└── README.md
```

> **数据不在本仓库内**：约定放在工作区的兄弟目录 `<WORKSPACE>/data`（视频 + `ball_cone_v1` 数据集）。
> 详见下节「路径约定」。

## 目录说明

| 目录 | 说明 |
| --- | --- |
| `code` | 单视角推理核心代码（检测、跟踪、关键点、JSONL 输出） |
| `config` | ROI、模型路径和 app 过滤配置 |
| `doc` | 任务说明、路线图和方案文档 |
| `models` | 本地权重（不含 git） |
| `reports` | 质量评估报告输出 |
| `runs` | 跑批产物 |
| `scripts` | 可直接参考的示例脚本 |
| `tools` | 微调 / 模型转换 / 部署跑批三条流水线 |
| `tmp` | 中间输出，如 JSONL 观测文件 |

## 路径约定（换机器不用改代码）

三个目录**相邻**即可，所有路径都能从仓库位置推导出来：

```
<WORKSPACE>/                      ← 工作区根
├── AI-Football/                  ← 本仓库
├── Nexusflow/                    ← C++ / TensorRT 侧
└── data/                         ← 视频与数据集（ball_cone_v1 在这下面）
```

`source tools/detection_finetune/env.sh` 之后，`AI_FOOTBALL_WORKSPACE / DATA_DIR /
DATASET_DIR / NEXUSFLOW / ENV` 这批变量就绪（脚本与文档统一用它们，仓库里不再出现绝对路径）；
要拆开布局就先设这些变量，`env.sh` 只补缺省、不覆盖。完整说明见
`tools/detection_finetune/README.md` 的「路径约定」。

## 训练 / 部署流水线（`tools/`）

单视角推理之外的「微调 → 导出 → 部署跑批」在 `tools/` 下，每条链都有自己的 README：

| 目录 | 做什么 | 入口文档 |
| --- | --- | --- |
| `tools/detection_finetune/` | 抽帧 → 预标注 → 合成扩充 → 人工复核 → 三分类微调 | 该目录 `README.md`（含**数据集目录要求**、路径约定、命令索引） |
| `tools/model_conversion/` | 权重 → ONNX → fp16 → TensorRT engine（含与 PyTorch 数值对拍） | 该目录 `README.md` |
| `tools/rollout/` | 版本 → 导出 engine → 生成 C++ 配置 → 批量跑视频 → 汇总对比 | 该目录 `README.md` |

跑之前务必先看两条硬约束：

* **数据集目录**：训练输入必须是 roboflow 三目录（`train/ valid/ test` 各带 `_annotations.coco.json`
  与 `images/`），且 `categories` **必须 4 条**（`id=0` 为占位背景类，真实类 `1/2/3 = person/ball/cone`）
  —— 写错会**静默错指标甚至 CUDA assert**。完整要求见
  `tools/detection_finetune/README.md` 的「数据集目录要求（训练输入的唯一口径）」一节。
* **类别名**：逐字 `person` / `ball` / `cone`，唯一真源是 `tools/detection_finetune/classes.json`。

## 核心脚本

- `code/app.py`：主入口，完成检测、跟踪、关键点和 JSONL 输出
- `code/replay_observations.py`：根据 JSONL 回放并生成可视化结果
- `code/evaluate_observations.py`：根据 JSONL 生成质量评估报告
- `scripts/example.sh`：端到端示例，串起生成观测、生成报告、回放可视化

## 运行前准备

本项目统一用 `ai-football` 这个 conda 环境跑。先 source 一次环境脚本，它会设好 `$PY`
与上面那批路径变量——**别用裸 `python` / `pip`**（那是 conda base，里面没有 `rfdetr`）：

```bash
source tools/detection_finetune/env.sh
$PY -m pip install -r requirements.txt
```

1. 准备输入视频（数据目录下，例如 `$AI_FOOTBALL_DATA_DIR/素材/射门1-1080p60.mov`）
2. 如需 26 keypoints，准备本地 MMPose 配置和对应权重
3. 如需 ROI 过滤，先完成 ROI 标定，配置保存在 `config/roi.json`
4. 模型路径、检测阈值、过滤阈值和关键点平滑配置保存在 `config/app.yaml`

## 推荐流程

### 1. 标定 ROI

先打开标定界面，把感兴趣区域保存到 `config/roi.json`。

```bash
python code/app.py --video data/素材/射门1-1080p60.mov --draw-roi --roi-config config/roi.json
```

### 2. 生成单视角观测

这个步骤会完成：

- RF-DETR small 检测
- BoT-SORT 单镜头跟踪
- MMPose 26 keypoints
- 足球跟踪
- 输出 JSONL 观测文件

模型路径统一由 `config/app.yaml` 管理。如需手动下载 HRNet 权重：

```bash
python scripts/download_hrnet_pose_models.py --model hrnet-w48-dark
```

```bash
python code/app.py --video data/素材/传球-720p60.mov \
  --output-dir tmp/C1_hrnet_w48_dark_smooth \
  --device cuda:0 \
  --use-roi \
  --config config/app.yaml \
  --roi-config config/roi.json

python code/app.py \
  --video data/素材/传球-720p60.mov \
  --output-dir tmp/test_hrnet_w48_dark \
  --device cuda:0 \
  --use-roi \
  --config config/app.yaml \
  --roi-config config/roi.json


python code/app.py \
  --video data/素材/传球-720p60.mov \
  --output-dir tmp/C1 \
  --device cuda:0 \
  --use-roi \
  --debug \
  --config config/app.yaml \
  --roi-config config/roi.json
```

### 3. 生成评估报告

`code/evaluate_observations.py` 直接读取 JSONL，不再依赖视频本身。

```bash
python code/evaluate_observations.py \
  --input tmp/C1_hrnet_w48_dark_smooth/observations.jsonl \
  --output reports/C1_observations_quality_report.json \
  --markdown reports/C1_observations_quality_report.md
```

### 4. 回放可视化

`code/replay_observations.py` 基于 JSONL 重建回放画面。

```bash
python code/replay_observations.py \
  --observations tmp/C1_hrnet_w48_dark_smooth/observations.jsonl \
  --show \
  --app-config config/app.yaml
```

## 一键示例

`scripts/example.sh` 把上面的流程串起来了。它做的事情依次是：

1. 定义输入视频和输出目录
2. 调用 `code/app.py` 生成观测 JSONL
3. 调用 `code/evaluate_observations.py` 输出 JSON 和 Markdown 报告
4. 调用 `code/replay_observations.py` 回放结果

脚本中的变量含义如下：

- `input_video`：输入视频路径
- `output_dir`：观测 JSONL、渲染视频和运行元信息目录
- `output_report`：结构化质量报告
- `output_report_md`：可读版质量报告

脚本里的检测、姿态配置和权重都放在本地 `models/`。配置文件纳入 git，`.pt/.pth` 权重文件不纳入 git。

## 关键参数

### `code/app.py`

| 参数 | 作用 |
| --- | --- |
| `--video` | 输入视频 |
| `--roi-config` | ROI 配置 |
| `--config` | 模型路径、检测阈值、过滤阈值和关键点平滑配置 |
| `--output-dir` | 输出目录，固定写入 `observations.jsonl`、`rendered.mp4`、`run_args.json` 和 `app_config.yaml` |
| `--show` | 直接显示窗口 |
| `--use-roi` | 启用 ROI 过滤 |
| `--stride` | 按固定帧间隔抽帧；和 `--target-fps` 互斥 |
| `--target-fps` | 按目标 FPS 自动推导抽帧间隔；和 `--stride` 互斥 |

### `code/replay_observations.py`

| 参数 | 作用 |
| --- | --- |
| `--observations` | 输入 JSONL |
| `--show` | 显示回放窗口 |
| `--output-video` | 导出回放视频 |
| `--app-config` | 回放时使用的过滤配置 |

## 输出说明

- `tmp/<run>/observations.jsonl`：单视角观测结果，第一条记录是视频元信息
- `tmp/<run>/rendered.mp4`：带检测框、关键点和 debug 信息的渲染视频
- `tmp/<run>/run_args.json`：本次运行的 CLI 参数和解析后的输出路径
- `tmp/<run>/app_config.yaml`：本次运行使用的配置快照
- `reports/*_quality_report.json`：质量评估结果
- `reports/*_quality_report.md`：质量评估的 Markdown 版本
