# TensorRT 模型转换

把 AI-Football 的 PyTorch 权重转成 Nexusflow C++ 侧可用的 TensorRT engine。

转换只有三步，但每一步都有坑，本文档记录**实际验证过的**流程、命令、产物命名和校验方法。

---

## 0. 流程总览

```
models/**/*.pth                      PyTorch 权重
      │
      │  A. 导出 fp32 ONNX（动态 batch）
      ▼
models/**/*.onnx (+ .onnx.data)      计算图，I/O = FLOAT，batch 维为符号
      │
      │  B. 转 fp16（保留 float32 I/O）
      ▼
models/**/*-fp16-fp32io.onnx         计算图，I/O = FLOAT，内部 = fp16
      │
      │  C. trtexec 编译
      ▼
models/**/*-fp16-b8-fp32io.engine    针对本机 GPU 的 engine
      │
      │  D. 数值校验（对比 PyTorch）
      ▼
                            与 PyTorch 输出一致性报告
```

ONNX 是**中间产物，必须保留**——engine 与 GPU/TRT 版本绑定，换机器必须从 ONNX 重建。

---

## 快速开始（step-by-step）

以「从零做出 HRNet-W48-DARK 的 engine」为例，7 步照抄即可。全程在仓库根目录执行。

```bash
cd /home/hx1/yzl/Work/AI-Football
PY=/root/miniconda3/envs/ai-football/bin/python
NEXUS=/home/hx1/yzl/Work/Nexusflow
```

### Step 1 · 装转换依赖，并把 numpy 压回 < 2

```bash
$PY -m pip install onnx onnxscript onnxruntime onnxsim onnx-graphsurgeon
$PY -m pip install "numpy<2"
$PY -c "import numpy; print(numpy.__version__)"
```

期望输出：`1.26.4`

> 这两条必须成对执行。上面的包会把 numpy 顶到 2.x，而 mmpose / ultralytics 要求 `<2`。

### Step 2 · 环境自检

```bash
bash tools/model_conversion/run_all.sh hrnet-w48-dark 2>/dev/null | head -12
```

或者直接手动确认：

```bash
$PY -c "import onnx, onnxscript, onnxruntime, onnxsim, onnx_graphsurgeon; print('deps ok')"
```

期望输出：`deps ok`

### Step 3 · 导出 fp32 动态 batch ONNX

导出脚本在 Nexusflow 仓库里，用**本仓库的 conda 环境**调用它：

```bash
(cd $NEXUS && $PY tools/model_conversion/export_hrnet_onnx.py \
    --config      /home/hx1/yzl/Work/AI-Football/models/mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py \
    --checkpoint  /home/hx1/yzl/Work/AI-Football/models/mmpose/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    --output      /home/hx1/yzl/Work/AI-Football/models/mmpose/hrnet/hrnet-w48-dark.onnx \
    --height 384 --width 288 \
    --opset 18 --device cuda:0 --dynamic-batch)
```

期望输出末尾：

```
input:  images  [batch, 3, 384, 288]  float32
output: heatmaps[batch, 133, 96, 72]  float32
```

**验收点**：`batch` 是符号（不是 `1`）。是 `1` 说明漏了 `--dynamic-batch`。

### Step 4 · 转 fp16（保留 float32 I/O）

```bash
$PY tools/model_conversion/convert_fp16_onnx.py --only hrnet-w48-dark
```

期望输出：

```
=== hrnet-w48-dark ===
   models/mmpose/hrnet/hrnet-w48-dark.onnx -> models/mmpose/hrnet/hrnet-w48-dark-fp16-fp32io.onnx
   dropped duplicate cast nodes: 25
   images   ['batch', 3, 384, 288] FLOAT
   heatmaps ['batch', 133, 96, 72] FLOAT
   saved (...  MB)
```

**验收点**：
- `dropped duplicate cast nodes` 出现（这是转换器的已知缺陷，脚本会自动修）；
- I/O 的 dtype 是 **FLOAT 而不是 FLOAT16** —— 这是刻意的，见第 2 节。

### Step 5 · 编译 engine

```bash
bash tools/model_conversion/build_engines.sh hrnet-w48-dark
```

期望输出：

```
########## hrnet-w48-dark ##########
Engine built in ... s
PASSED
  -> models/mmpose/hrnet/hrnet-w48-dark-fp16-b8-fp32io.engine (...  MB)
```

**验收点**：出现 `PASSED`，且同名 `.engine` 文件已生成。

### Step 6 · 数值校验（对比 PyTorch）

TensorRT 的 Python 绑定只装在系统解释器里，所以这一步换解释器：

```bash
/usr/bin/python3.12 tools/model_conversion/validate_engines.py \
    --engine models/mmpose/hrnet/hrnet-w48-dark-fp16-b8-fp32io.engine \
    --kind hrnet --height 384 --width 288
```

期望输出（节选）：

```
  解码后关键点位移（裁剪空间 288x384，共 ... 个）:
    mean=...  p99=...  max=...  逐位相同=0.9xxx
    仅项目使用的 23 个关键点（... 个）: mean=...  max=...  逐位相同=0.9xxx
```

**验收点**：23 个使用关键点的 `逐位相同` 接近 1.0、`max` 是个位数像素。
若 `max` 出现几十像素，先看 `score>=0.3` 那一层——无信号通道上的 argmax 抖动不是精度损失。

### Step 7 · 接到 C++ 配置

编辑 Nexusflow 的 `examples/aifootball_demo/config.yaml`：

```yaml
PoseEstimator:
  enginePath: /home/hx1/yzl/Work/AI-Football/models/mmpose/hrnet/hrnet-w48-dark-fp16-b8-fp32io.engine
  inputWidth: 288     # W，注意与 --width 一致
  inputHeight: 384    # H
  maxBatch: 8         # 必须 <= engine 的 profile max
```

然后重新编译并运行：

```bash
(cd $NEXUS && cmake --build build --parallel 4 && \
              ./build/bin/aifootball_demo examples/aifootball_demo/config.yaml)
```

**验收点**：日志出现 `input_mode=gpu`（说明 GPU 预处理仍生效，float32 I/O 的意义在这里）。

### 一步到位

上面 3~6 步可以一次跑完：

```bash
bash tools/model_conversion/run_all.sh hrnet-w48-dark     # 指定模型
bash tools/model_conversion/run_all.sh                    # 全部模型
STEPS=convert,build bash tools/model_conversion/run_all.sh # 只跑部分步骤
FORCE=1 bash tools/model_conversion/run_all.sh hrnet-w32   # 覆盖已有产物
```

---

## 1. 环境

所有转换都在 conda 环境 `ai-football`（Python 3.11）里做，不要用系统 Python。

除项目 `requirements.txt` 外，转换还需要 5 个包：

```bash
PY=/root/miniconda3/envs/ai-football/bin/python
$PY -m pip install onnx onnxscript onnxruntime onnxsim onnx-graphsurgeon
```

各包的作用：

| 包 | 用途 | 少了会怎样 |
| --- | --- | --- |
| `onnx` | 读写 ONNX、拓扑校验 | 无法加载/保存图 |
| `onnxscript` | PyTorch 2.6+ 的 `torch.onnx.export` 必需 | `ModuleNotFoundError: onnxscript` |
| `onnxruntime` | 提供 `convert_float_to_float16` | 缺 fp16 转换器 |
| `onnxsim` | RF-DETR exporter 内部简化图 | `ModuleNotFoundError: onnxsim` |
| `onnx-graphsurgeon` | RF-DETR exporter 内部改图 | `ModuleNotFoundError: onnx_graphsurgeon` |

### ⚠️ numpy 版本陷阱（每次装包都要检查）

上面这些包都会把 `numpy` 顶到 2.x，但本项目（`requirements.txt`、mmpose、mmcv-lite、ultralytics、supervision、roboflow）**要求 `numpy<2`**。装完必须压回来：

```bash
$PY -m pip install "numpy<2"
$PY -c "import numpy; print(numpy.__version__)"   # 期望 1.26.x
```

pip 还会报两条**可忽略**的警告：
- `opencv-python is not installed` —— 环境里用的是 conda 版 cv2，实际可用。
- `ml-dtypes 0.6.0 requires numpy>=2.0.0` —— onnxruntime 的传递依赖，实测 ORT 在 numpy 1.26.4 下工作正常。

---

## 2. 关键设计决策：I/O 保持 float32

**engine 的输入输出绑定是 `FLOAT`，只有图内部走 fp16。**

原因不是精度，是**兼容性**：Nexusflow 的 `RFDetrDetectorInfer::Init()` 用
`inputInfo.dtype == inference::DataType::kFloat32` 来决定是否启用 **GPU 预处理**
（`RFDetrCudaPreprocess.cu`，CUDA 上完成 resize + normalize + HWC→CHW，并直接写进
engine 的 device 输入缓冲）。一旦 engine 输入变成 HALF：

- GPU 预处理被禁用，自动退回 CPU 预处理（慢一个数量级）；
- `.cu` kernel、`HostFloatBuffer`、后处理解码全都得改成 half。

而这样做**没有任何性能损失**。同一模型两种 I/O 精度的实测对比（trtexec 纯计算时间）：

| engine | fp32 I/O + fp16 内部 | fp16 I/O | 差异 |
| --- | ---: | ---: | ---: |
| rf-detr-small 960 (b1 / b4 / b8) | 3.950 / 12.895 / 27.510 ms | 3.950 / 12.940 / 27.557 ms | < 0.2% |
| hrnet-w48-dark 384x288 (b1 / b4 / b8) | 2.990 / 3.910 / 7.339 ms | 2.976 / 3.912 / 7.347 ms | < 0.2% |
| hrnet-w32 256x192 (b1 / b4 / b8) | 0.682 / 1.001 / 1.868 ms | 0.676 / 1.002 / 1.870 ms | < 0.2% |

边界上那几个 Cast 的开销相对网络本身可以忽略，所以**统一选 float32 I/O**。

---

## 3. 步骤 A：导出 fp32 动态 batch ONNX

导出脚本在 **Nexusflow** 仓库里（`/home/hx1/yzl/Work/Nexusflow/tools/model_conversion/`），
不在本目录——它们是 C++ 侧部署契约的唯一真源，不要复制过来。

### RF-DETR

```bash
NEXUS=/home/hx1/yzl/Work/Nexusflow
PY=/root/miniconda3/envs/ai-football/bin/python

(cd "$NEXUS" && $PY tools/model_conversion/export_rfdetr_onnx.py \
    --size small \
    --checkpoint /home/hx1/yzl/Work/AI-Football/models/rfdetr/rf-detr-small.pth \
    --output-dir /home/hx1/yzl/Work/AI-Football/models/rfdetr/onnx \
    --device cpu --resolution 960 --dynamic-batch)
```

- `--resolution` 必须是 16 的正整数倍；常用 512 / 960。
- `--dynamic-batch` 会把 batch 维导出成符号，engine 侧再用优化 profile 限定范围。
- 输出文件名由脚本自动生成：`rf-detr-small-<res>-dynamic.onnx`。
- 脚本会把 `input/dets/labels` 重命名成部署契约名 `image/pred_boxes/pred_logits`。
- 内部强制走 legacy exporter（`dynamo=False`），因为 RF-DETR 1.3.0 的图在 dynamo 路径下会
  把 shape 推导固化成 batch=1 常量。

### HRNet

```bash
(cd "$NEXUS" && $PY tools/model_conversion/export_hrnet_onnx.py \
    --config  /home/hx1/yzl/Work/AI-Football/models/mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py \
    --checkpoint /home/hx1/yzl/Work/AI-Football/models/mmpose/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    --output /home/hx1/yzl/Work/AI-Football/models/mmpose/hrnet/hrnet-w48-dark.onnx \
    --height 384 --width 288 \
    --opset 18 --device cuda:0 --dynamic-batch)
```

注意 `--height/--width` 是 **(H, W)**，且必须与该模型 MMPose config 里的
`codec.input_size` 一致：

| 模型 | config `input_size` | `--height` (H) | `--width` (W) | heatmap 输出 |
| --- | --- | ---: | ---: | --- |
| HRNet-W48-DARK | `(288, 384)` | 384 | 288 | `[B, 133, 96, 72]` |
| HRNet-W32 | `(192, 256)` | 256 | 192 | `[B, 133, 64, 48]` |

脚本会校验 `heatmap == input / 4`，不一致时在 stderr 打 warning。

---

## 4. 步骤 B：fp32 → fp16

```bash
$PY tools/model_conversion/convert_fp16_onnx.py
```

默认对全部登记的模型做 **`keep_io_types=True`**（float32 I/O + fp16 内部）。
加 `--fp16-io` 则导出纯 fp16 I/O 版本（仅供对比实验，C++ 侧不能用）。

转换器有两个已知缺陷，脚本里已经修好：

1. **重复的 Cast 输出名**。`convert_float_to_float16` 对"被多个受保护算子共享的同一个输入
   张量"会重复插入同名 Cast 节点（RF-DETR 图里 15 个、HRNet 图里 25 个），产生 SSA 违规。
   这些节点语义等价，去重保留第一个即可。
2. **节点乱序**。插入 Cast 后图的拓扑顺序被破坏，需要重新排序。

另外 fp16 转换会插入大量 Cast，**必须自己修**，否则 `onnx.checker` 直接报错。

### 受保护算子（保持 fp32）

```python
EXTRA_BLOCK = ["Softmax", "LayerNormalization", "Erf", "ReduceSum", "Sqrt", "Resize"]
```

- `Softmax` / `LayerNormalization` / `Erf`：RF-DETR 的 transformer 侧对这几项在 fp16 下数值
  敏感，保持 fp32 可以避免精度塌陷。
- `Resize`：ORT 默认阻断列表里就有，这里显式列出以免被覆盖。

---

## 5. 步骤 C：编译 engine

```bash
bash tools/model_conversion/build_engines.sh
```

核心是 `trtexec`，profile 统一 **`min=1 / opt=4 / max=8`**：

```bash
trtexec \
  --onnx=<fp16 onnx> \
  --saveEngine=<engine> \
  --skipInference \
  --builderOptimizationLevel=3 \
  --minShapes=image:1x3x960x960 \
  --optShapes=image:4x3x960x960 \
  --maxShapes=image:8x3x960x960
```

### TRT 11.1 的两个行为

- **没有 `--fp16` 开关**。TRT 11 起改成 strongly-typed，精度完全由图决定，所以必须先做步骤 B。
  （`trtexec --help` 里 `--stronglyTyped` 已是 deprecated no-op。）
- **`--noTF32` 不是必须的**。help 原文：*"Disable tf32 precision (default is to enable tf32, in
  addition to fp32)"* —— TF32 默认开启。对已经是 fp16 的图，`--noTF32` 只影响残留的 fp32 算子，
  本目录脚本**不加**它。若要更保守的精度，可自行加上。

### profile 的选择

`opt` 是 TRT 做 kernel 选型和显存规划的目标点，应该贴合真实批量。
Nexusflow 的既有约定是 **`opt = max`**（其 README 写明 *"the engine builder optimizes the
TensorRT profile at max-batch, not batch 1"*）。本目录统一用 `1 / 4 / 8`：

- `opt=4` 与 C++ 侧 `maxBatchSize: 4`（检测）、`batchFrameCount: 4`（姿态）对齐；
- `max=8` 给 C++ 侧的 `maxBatch: 8` 留出 flip-test 合并的空间。

**`max` 必须 ≥ C++ 配置里的批量**，否则 `setInputShape` 会直接失败：
- `RFDetrDetectorInfer`：动态 engine 时 `effectiveMaxBatch = maxBatchSize`。
- `HRNetPoseEstimatorInfer`：动态 engine 时 `effectiveMaxBatch = maxBatch`。

---

## 6. 产物命名约定

| 后缀 | 含义 |
| --- | --- |
| `*.onnx` | 步骤 A 的 fp32 输出，**重建的唯一源** |
| `*.onnx.data` | ONNX 外部张量数据（大模型必需，两个文件必须成对保留） |
| `*-fp16-fp32io.onnx` | fp16 内部 + float32 I/O（**部署用**） |
| `*-fp16.onnx` | fp16 内部 + fp16 I/O（仅对比实验） |
| `*-fp16-b8-fp32io.engine` | 最终 engine，`b8` = profile max batch |
| `*-fp32tf32-b8.engine` | 未转 fp16、仅放开 TF32 的对照组 |
| `*-ch23*.onnx/.engine` | head 输出通道裁剪版（见下） |

> 历史遗留：`*-fp16int.onnx` 与 `*-fp16-fp32io.onnx` 语义完全相同，是早期命名不统一留下的
> 重复文件。新生成的一律用 `-fp16-fp32io`。

### `ch23` 变体

HRNet 的 heatmap head 输出 133 个通道（COCO-WholeBody 全量），但本项目只使用其中 **23 个**
（`observations.WHOLEBODY_TO_PROJECT_26`）。把 head 的输出通道裁到 23 之后：

- D2H 传输量降到 1/5.8（133→23 通道）；
- `PoseEstimator.Postprocess` 的解码量同步下降；
- 对项目实际使用的关键点**精度无损**。

代价是每次换关键点集都要重新裁剪。

---

## 7. 实测延迟

RTX 5070 Ti，TRT 11.1，纯 engine 计算时间（`trtexec`，profile 1/4/8）：

| engine | batch 1 | batch 4 | batch 8 |
| --- | ---: | ---: | ---: |
| `rf-detr-small-960` | 3.95 ms | 12.90 ms | 27.51 ms |
| `rf-detr-small-512` | 1.49 ms | 3.26 ms | 6.49 ms |
| `hrnet-w48-dark-384x288` | 2.99 ms | 3.91 ms | 7.34 ms |
| `hrnet-w32-256x192` | 0.68 ms | 1.00 ms | 1.87 ms |

按 batch=4 折算每帧：检测 2.12~3.22 ms/帧，姿态 0.25~0.98 ms/人。

---

## 8. 步骤 D：数值校验

```bash
$PY tools/model_conversion/validate_engines.py
```

方法：把**同一张预处理后的输入张量**分别喂给 PyTorch 和 TRT engine，比较原始输出，
再按各自的后处理做端到端对比。

判定标准：

- **RF-DETR**：对分数 > 0.25 的 query 比类别一致性 + 类别直方图。
  原始 query 的 `max|Δlogits|` 会有 4~5 的量级，但 **fp32 engine 也有同样量级**，说明这些偏差
  落在未被采用的 query 上，不是 fp16 引入的。
- **HRNet**：用**真实球员裁剪**（不要用随机噪声，噪声输入的 argmax 极不稳定），
  解码后比关键点在裁剪空间的位移。

已确认的结果：

| | 结果 |
| --- | --- |
| RF-DETR 960 fp16 vs PyTorch fp16 | 4 帧全部 `cls 一致率 = 1.0000`，类别直方图 `{1:13, 37:4, 62:1}` 完全一致 |
| HRNet-W32 fp16 的 23 个使用关键点 | 98.4% 逐位相同，p99 = 2.0 px，max = 2.0 px（192x256 裁剪空间） |
| HRNet-W32 fp16 全 133 点（score≥0.3） | 98.4% 逐位相同，max = 4.5 px |

---

## 9. 踩坑清单

1. **`onnx.save` 的外部数据是「追加」写入的**。目标文件已存在时，`.onnx.data` 会越滚越大
   （实测同一个小模型从 60MB 涨到 178MB）。**保存前必须先删掉目标 `.onnx` 和 `.onnx.data`。**
2. **fp16 转换器会留下重复 Cast 名和乱序节点**，必须自带去重 + 拓扑重排（见步骤 B）。
3. **装了转换依赖后必须把 numpy 压回 <2**（见步骤 1）。
4. **RF-DETR 的预处理契约与 Python pipeline 不同**。C++ 侧是 RGB → `/255` → ImageNet 归一化
   → **直接 resize 到 SxS（不做 letterbox）**；Python pipeline 是 letterbox（等比缩放 + 灰边 114）。
   同一份权重、同一个 engine，两条路的前处理不同，检测框不会逐框一致。
5. **换 GPU 或 TRT 大版本必须重新编译 engine**。ONNX 可以复用，engine 不行。
6. **静态 batch 的 engine 不能用动态 batch 喂**。`TensorRTEngine::PrepareInputDevice` 会在
   shape 校验阶段直接拒绝，不会静默出错。

---

## 10. 与 Nexusflow 的分工

| 环节 | 归属 |
| --- | --- |
| 导出脚本（部署契约的真源） | Nexusflow `tools/model_conversion/export_*.py` |
| fp32→fp16 转换、engine 编译、数值校验 | **本目录** |
| 运行时加载 / 前后处理 / 解码 | Nexusflow C++ SDK |

本目录文件一览：

| 文件 | 作用 | 何时用 |
| --- | --- | --- |
| `README.md` | 本文档 | — |
| `run_all.sh` | 全流程编排（自检 → 导出 → 转换 → 编译 → 校验） | 想一条命令走完 |
| `convert_fp16_onnx.py` | fp32 → fp16 转换（默认保留 float32 I/O） | 步骤 B |
| `build_engines.sh` | trtexec 批量编译，profile 统一 1/4/8 | 步骤 C |
| `validate_engines.py` | 与 PyTorch 的数值一致性校验 | 步骤 D |

C++ 侧配置（`examples/aifootball_demo/config.yaml`）里引用 engine 时注意：

```yaml
Detector:
  enginePath: .../rf-detr-small-960-fp16-b8-fp32io.engine
  inputSize: 960
  maxBatchSize: 4          # ≤ engine profile max
PoseEstimator:
  enginePath: .../hrnet-w48-dark-fp16-b8-fp32io.engine
  inputWidth: 288          # W
  inputHeight: 384         # H
  maxBatch: 8              # ≤ engine profile max
```

另外 demo 的 `algorithm` 段支持两个队列参数（SDK 已实现，仅需在 YAML 里配置）：

```yaml
algorithm:
  maxPendingFrames: 16     # 0 = 无界（默认）。每帧在途都会钉住一张解码帧（1080p 约 6MB）
  inputQueuePolicy: block  # 默认；配合上界=阻塞 ProcessAsync，形成背压
```

**不设 `maxPendingFrames` 是危险的**：解码器会远超最慢一级的消费速度，内存线性增长直到
被 OOM 杀掉（实测 3.3GB → 28.7GB / 24 秒）。
