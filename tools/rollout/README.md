# rollout —— 从数据到 C++ 部署的“一条流水线”

finetune 迭代会**反复**走同一条路：数据准备 → 训练 → 导出 engine → C++ 批量跑视频 → 汇总对比。
这个目录就是管这条路的：**只放编排与配置，不重写推理**（导出复用 `tools/detection_finetune/`，
推理用 Nexusflow 的 `aifootball_app` + TensorRT engine）。

## 为什么必须走 engine + C++

同一张卡、同一个模型，两条路的实测差 30–270×：

| 环节 | TRT fp16 engine（C++ 走这条） | Python 管线（PyTorch，无 engine） |
| --- | --- | --- |
| 检测 RF-DETR 960 | **3.91 ms/帧**（b1）／3.24 ms/帧（b4） | 114 ms/帧 |
| 姿态 HRNet-W48 384×288 | **2.96 ms/人**（b1）／**0.92 ms/人**（b8） | 250–330 ms/人 |
| 整段吞吐 | touch_eval 实测 **157.4 FPS / 6.35 ms**（无渲染）；开渲染 68.2 FPS | ~0.4 FPS / 2600 ms |

所以"全量逐帧"（35 段 ≈ 27.4 万帧/版本）在 C++ 侧是 **半小时～1.5 小时/版本**，
在 Python 侧是 **5 天/版本**。跨帧聚合（`batchFrameCount: 4`）与 `maxBatch: 8` 是关键，
别为了"简单"把姿态改成逐人调用。

### 吞吐由什么决定（实测口径，别误读上表）

上表是 **无渲染**口径（`--render` 没开时日志里没有 `render : … ms/frame` 行 = `renderedFrames=0`）。
实测（`射门1-1080p60`，16 核，机器空闲，`cpp_app/*/fastest.yaml`，测试命令见本节末）：

| 人/帧 | 姿态档 | 吞吐（无渲染） | 每帧 |
| --- | --- | --- | --- |
| 3.56（ROI 开，射门1 多边形匹配） | W32 192×256 | **108.0 FPS** | 9.26 ms |
| 10.70（ROI 关，假人全进姿态） | W32 192×256 | 46.5 FPS | 21.5 ms |
| 3.50（COCO 权重对照，ROI 开） | W32 192×256 | 105.2 FPS | 9.51 ms |

* **人的数量是主因**：每多 1 人/帧 ≈ **+1.6 ms/帧**（3.56→10.70 人 = +11.4 ms）。
  ROI 在射门1 上把 10.7 人/帧压到 3.5 → 吞吐 ×2.2。所以"90+ FPS"的前提是**每帧只有 ~3 个人被送进姿态**；
  换到人多、或关掉 ROI 的镜头，会直接掉到 40–50 FPS。
* **渲染不是免费的**：渲染跑在消费者线程（`examples/aifootball_app/main.cpp:111`），实测 **19.0 ms/帧**，
  其中约 1/3 与流水线重叠（21.5+19.0=40.5 vs 实测 33.6 ms）→ 46.5 → **29.8 FPS（-36%）**。
  离线 `Nexusflow/tools/render_jsonl.py` 是 **27.5 ms/帧**（要重新解码源视频，比内置还慢），
  但它是 CPU 侧、**可与下一段的 GPU 任务并行** → 对总墙钟影响小。
  报"端到端出视频"的吞吐必须用开渲染的数字，不要引用这里的无渲染值。
* 复测命令（三条对照即可复现上面每一行）：

```bash
cd /home/hx1/yzl/Work/Nexusflow
APP=./build/bin/aifootball_app; CFG=/home/hx1/yzl/Work/AI-Football/tools/rollout/cpp_app/v1-frozen-step2/fastest.yaml
V=/home/hx1/yzl/Work/data/射门1-1080p60.mov
$APP $CFG --video_path $V --output_dir /tmp/t_off --max_frames 1200                      # 无渲染
$APP $CFG --video_path $V --output_dir /tmp/t_on  --max_frames 1200 --render 1 --render_path /tmp/t_on.mp4   # 开渲染
# ROI 对照：rollout.py cpp-config --version v1-frozen-step2 --pose-profile fastest --keep-roi
```

## 一条命令的用法

```bash
cd /home/hx1/yzl/Work/AI-Football/tools/rollout && source ../detection_finetune/env.sh
$PY rollout.py doctor                 # 预检：app/姿态engine/类别契约/磁盘/各版本就绪（不过就退出非 0）
$PY rollout.py status                 # 版本 × (权重/ONNX/engine/软链) + 各版本任务进度
$PY rollout.py videos                 # 扫描到的视频（tag / 路径）
$PY rollout.py export --version <版本> # checkpoint → ONNX → fp16 → TRT engine → 模型卡
$PY rollout.py cpp-config --all       # 生成每个版本专用的 C++ config（三分类 + engine 路径）
$PY rollout.py run --all --stride 1   # 逐帧跑（可续跑：manifest 里 rc==0 的自动跳过）
$PY rollout.py run --version <版本> --video 颠球 --render   # 单段 + 出渲染视频
$PY rollout.py summary                # 汇总：逐视频 × 逐版本（帧数/FPS/球框）+ 各模块 ms/帧
$PY rollout.py all --dry-run          # 一把梭：doctor → 补缺 engine → cpp-config → run → summary（只看命令）
$PY rollout.py all                    # 真的跑一遍（缺 engine 会先导出；--no-export 可跳过）
```

停止：`kill -- -"$(cat /tmp/rollout.pid)"`（负号 = 整个进程组；**别用 pkill**，会误杀自己的 shell）。
导出进行中会在 `runs/.exporting` 留锁，跑批会拒绝启动（防"engine 还在写盘就被加载"）。

## 只看效果：抽帧 + 渲染 + 左右对照（review 用）

`runs/` 里那批是**逐帧、无渲染**的数值口径产物。要**人眼**看两个版本的差异，用下面这条线，
产物落在**另一个根目录**，不会覆盖数值产物：

```bash
cd tools/rollout && source ../detection_finetune/env.sh

# 1) 全场景抽帧 + 渲染（--max-frames 是"处理后帧数"，见下方坑）
$PY rollout.py --out-root tools/rollout/runs_review run --all \
      --stride 10 --max-frames 120 --render

# 2) 把两个版本在同一场景上的渲染结果拼成左右对照，N 段输入 → N 段输出
$PY stitch_compare.py --root tools/rollout/runs_review
```

产物：

| 东西 | 位置 |
| --- | --- |
| 单版本渲染视频 | `runs_review/<版本>/<场景tag>/rendered.mp4` |
| **左右对照视频（要看的）** | `runs_review/compare/<场景tag>.mp4` |
| 逐场景计数对照表 | `runs_review/compare/compare.md` |

`stitch_compare.py` 会把左版打在左半、右版打在右半（中间 4px 竖线、左上角标版本名），
并按场景汇总人/球计数——**球框差为正且偏大的场景优先人眼复核**，多半就是右版多出来的误报。
默认取 `rollout.yaml` 里前两个 `enabled` 版本，可用 `--left/--right` 换。

两个必须知道的坑：

* **`--stride` 不跳过解码。** app 仍解码每一帧，再丢掉 `frameId % stride != 0` 的。
  所以抽帧**不是线性提速**：stride 5→10 只快约 28%，5→20 只快约 39%（解码那部分恒定）。
  实测 1V2_2（1080p，4.3 人/帧）：检测 4.7 ms + 姿态 10.9 ms + 渲染 18.3 ms +
  **解码浪费 ≈35 ms** = 69.6 ms/帧。
* **`--max_frames` 按"处理后帧数"算**（main.cpp:261 在 stride 过滤之后判断），
  不是 `--help` 里写的 "decoded frames"。所以 `--stride 10 --max-frames 120`
  ≈ 源视频前 1200 帧 ≈ 输出 120 帧（按 `fps/stride` 写入，**实时播放**不慢放）。

## 目录与命名约定（与 `tools/detection_finetune/` 一致）

| 东西 | 位置 |
| --- | --- |
| 版本（= 血缘名） | `v1` / `v1-frozen-step1` / `v1-frozen-step2` / `v2`（小目标优化预留） |
| 训练产物 | `tools/detection_finetune/runs/<版本>/checkpoint_best_ema.pth` |
| 部署产物（ONNX/fp16/engine + 模型卡 + 权重软链） | `models/rfdetr/fine-tune/<版本>/` |
| 本流水线配置 | `tools/rollout/rollout.yaml` |
| C++ 专用 config（生成物，勿手改） | `tools/rollout/cpp_app/<版本>.yaml` |
| 跑批产物 | `tools/rollout/runs/<版本>/<视频tag>/{observations.jsonl, run.log, app.yaml}` |
| 跑批清单/汇总 | `tools/rollout/runs/{manifest.json, summary.md}` |

## 五个阶段（每一步都幂等、可重跑）

1. **数据**：`tools/detection_finetune/`（抽帧/标注/合成/体检）——本目录不管。
2. **训练**：`s4_train_launch.sh <版本>`（配置自带 `# EXTRA_ARGS:`，如 step1 的 `--freeze encoder`）。
3. **导出**：`rollout.py export` → 调 `s6_deploy_export.py`（ONNX/fp16/engine + 与 PyTorch 对拍）
   → 调 `s6_deploy_pin_model.py`（建 `rf-detr-small.pth` 软链 + 汇总模型卡 `meta.json`）。
4. **C++ 跑批**：`rollout.py cpp-config` + `rollout.py run` → 每个任务一个独立 `--output_dir`，
   日志、`observations.jsonl`、生效的 `app.yaml` 都留在该目录里，事后完全可复现。
5. **汇总**：`rollout.py summary` → 逐视频 × 逐版本的表（帧数、wall、FPS、人/球计数、观测大小）
   + **各模块 ms/帧**（`HRNetPoseEstimator` / `RFDetrDetector` / tracker / smoother，来自每帧的
   `module_timings_ms`——判断"慢在检测还是慢在姿态"就看这一节）。
6. **端到端评估（不在本目录，用 Nexusflow 既有工具）**：本流程只负责产出 `observations.jsonl`，
   触球判定/精度口径以 Nexusflow 那份为准：

   ```bash
   cd /home/hx1/yzl/Work/Nexusflow
   ./examples/touch_eval/run_touch_eval.sh data/移动传球.mov data/移动传球.touches.csv output/touch_eval_moving_pass
   # 或：cd examples/touch_eval/py && uv run touch-eval --observations <observations.jsonl> ...
   ```

## 加一个新版本（finetune 迭代的标准动作）

```bash
# 1) 训练（配置自带参数，如 step1 的 --freeze encoder）
cd /home/hx1/yzl/Work/AI-Football/tools/detection_finetune && ./s4_train_launch.sh v2

# 2) 登记：在 tools/rollout/rollout.yaml 的 versions: 里加一条（enabled: true + note 写清血缘）

# 3) 导出 + 上架（ONNX/fp16/engine + 模型卡 + 权重软链）
$PY tools/rollout/rollout.py export --version v2

# 4) 生成该版本专用的 C++ config
$PY tools/rollout/rollout.py cpp-config --version v2

# 5) 预检 + 跑批
$PY tools/rollout/rollout.py doctor && $PY tools/rollout/rollout.py run --version v2

# 6) 汇总
$PY tools/rollout/rollout.py summary
```

* 名字就是血缘：`runs/<版本>/`、`train_configs/<版本>.yaml`、`models/rfdetr/fine-tune/<版本>/`、
  `cpp_app/<版本>.yaml`、`runs/<版本>/` 五处同一个版本名，别处只引用它。
* `rollout.py doctor` 会把 `rollout.yaml` 的类别契约与训练侧真相源
  `tools/detection_finetune/classes.json` 对拍，不一致直接报错（这是最容易静默错的地方）。
* 换分辨率/换类别数：engine 文件名模板与 `classes:` 段要一起改，改完重跑 `doctor`。

## 硬性契约（踩过坑，别改）

* **classId = logits 列下标，不加 1**；列 0 = no-object。三分类模型写
  `numClasses: 4 / targetClasses: [1,2,3] / personClassId:1 ballClassId:2 coneClassId:3`。
* **tracker 的 `ballClassId` 必须跟着改**（COCO 是 37，三分类是 2），否则跟踪器一个球都收不到；
  `rollout.py` 会断言它和 `--version` 的类别号一致。
* **ROI**：C++ 配置里的 `algorithm.roi` 是**单个多边形**（按某一段视频标定的）。
  跨视频批量默认**关掉**（`roi.enabled: false`）；只在确定同机位时用 `--keep-roi` 打开。
* **姿态 engine 路径**固定用 `models/mmpose/hrnet/hrnet-w48-dark-ch23-fp16-b8-fp32io.engine`
  （23 通道 = 导出时保留的 body 通道，与 C++ 侧 `numKeypoints: 23` 对应）。
* **换 GPU / 换 TensorRT 版本必须重建 engine**；`rollout.py export` 会重建并记录 sha256。
* **别改** `Nexusflow` 的 C++ 源里与本流程无关的部分；本目录只生成配置、不生成代码。
