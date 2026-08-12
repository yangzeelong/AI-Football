# AI-Football

用于足球视频单视角感知与观测输出的实验项目。

## 目录结构

```markdown
├── code
│   ├── app.py
│   ├── app_config.py
│   ├── app_pipeline.py
│   ├── replay_observations.py
│   └── ...
├── config
│   ├── app.yaml
│   ├── roi.json
│   └── ...
├── doc
│   └── ...
├── scripts
│   ├── example.sh
│   └── ...
├── data
│   └── ...
├── tmp
├── reports
└── README.md
```

## 目录说明

- `code`：核心代码
- `config`：ROI 和 app 过滤配置
- `doc`：任务说明、路线图和方案文档
- `scripts`：可直接参考的示例脚本
- `data`：输入视频和素材
- `tmp`：中间输出，如 JSONL 观测文件
- `reports`：质量评估报告输出

## 核心脚本

- `code/app.py`：主入口，完成检测、跟踪、关键点和 JSONL 输出
- `code/replay_observations.py`：根据 JSONL 回放并生成可视化结果
- `code/evaluate_observations.py`：根据 JSONL 生成质量评估报告
- `scripts/example.sh`：端到端示例，串起生成观测、生成报告、回放可视化

## 运行前准备

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 准备输入视频，例如 `data/素材/射门1-1080p60.mov`
3. 如需 26 keypoints，准备本地 MMPose 配置和对应权重
4. 如需 ROI 过滤，先完成 ROI 标定，配置保存在 `config/roi.json`
5. app 过滤阈值保存在 `config/app.yaml`

## 推荐流程

### 1. 标定 ROI

先打开标定界面，把感兴趣区域保存到 `config/roi.json`。

```bash
python code/app.py --video data/素材/射门1-1080p60.mov --draw-roi --roi-config config/roi.json
```

### 2. 生成单视角观测

这个步骤会完成：

- YOLO 检测
- BoT-SORT 单镜头跟踪
- MMPose 26 keypoints
- 足球跟踪
- 输出 JSONL 观测文件

```bash
python code/app.py --video data/素材/射门1-1080p60.mov \
  --detector yolo \
  --model models/yolo/yolov8n.pt \
  --device cuda:0 \
  --pose-device cuda:0 \
  --output-observations tmp/C1_observations.jsonl \
  --pose-config models/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py \
  --pose-checkpoint models/mmpose/rtmpose-wholebody/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth \
  --use-roi \
  --app-config config/app.yaml \
  --roi-config config/roi.json
```

### 3. 生成评估报告

`code/evaluate_observations.py` 直接读取 JSONL，不再依赖视频本身。

```bash
python code/evaluate_observations.py \
  --input tmp/C1_observations.jsonl \
  --output reports/C1_observations_quality_report.json \
  --markdown reports/C1_observations_quality_report.md
```

### 4. 回放可视化

`code/replay_observations.py` 基于 JSONL 重建回放画面。

```bash
python code/replay_observations.py \
  --observations tmp/C1_observations.jsonl \
  --show \
  --app-config config/app.yaml
```

## 一键示例

`scripts/example.sh` 把上面的流程串起来了。它做的事情依次是：

1. 定义输入视频和输出文件名
2. 调用 `code/app.py` 生成观测 JSONL
3. 调用 `code/evaluate_observations.py` 输出 JSON 和 Markdown 报告
4. 调用 `code/replay_observations.py` 回放结果

脚本中的变量含义如下：

- `input_video`：输入视频路径
- `tmp_file`：输出文件名前缀
- `output_observations`：观测 JSONL
- `output_report`：结构化质量报告
- `output_report_md`：可读版质量报告

脚本里的检测、姿态配置和权重都放在本地 `models/`，暂不纳入 git。

## 关键参数

### `code/app.py`

| 参数 | 作用 |
| --- | --- |
| `--video` | 输入视频 |
| `--roi-config` | ROI 配置 |
| `--app-config` | 人体/足球过滤配置 |
| `--output-observations` | 输出 JSONL |
| `--output-video` | 输出渲染视频 |
| `--show` | 直接显示窗口 |
| `--use-roi` | 启用 ROI 过滤 |
| `--pose-config` | MMPose 配置文件 |
| `--pose-checkpoint` | MMPose 权重文件 |
| `--pose-device` | MMPose 运行设备 |

### `code/replay_observations.py`

| 参数 | 作用 |
| --- | --- |
| `--observations` | 输入 JSONL |
| `--show` | 显示回放窗口 |
| `--output-video` | 导出回放视频 |
| `--app-config` | 回放时使用的过滤配置 |

## 输出说明

- `tmp/*.jsonl`：单视角观测结果，第一条记录是视频元信息
- `reports/*_quality_report.json`：质量评估结果
- `reports/*_quality_report.md`：质量评估的 Markdown 版本
- `tmp/*.mp4`：如果指定了 `--output-video`，会生成对应视频
