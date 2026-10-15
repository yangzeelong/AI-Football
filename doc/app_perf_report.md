# aifootball_app 性能测试报告

测试时间：2026-10-13 ｜ 被测程序：`Nexusflow/build/bin/aifootball_app` ｜ 测试人：AI 助手

## 〇、先分清：哪个数字来自哪个视频

| 视频（绝对路径） | 规格 | 角色 | 本次实测？ | 贡献了哪些数字 |
| --- | --- | --- | --- | --- |
| **`/home/hx1/yzl/Work/data/1V2 3.mov`** | 1280×720 h264 60fps，**8989 帧**，4.76 人/帧（前 3000 帧）/ 4.40（全片） | **主测视频** | ✅ **是**（7 次跑批） | 第一～四节、第六节的全部数字 |
| `/home/hx1/yzl/Work/data/射门1-1080p60.mov` | 1920×1080，**7201 帧**，3.5 人/帧 | 交叉验证用的历史基线 | ❌ 否（读自 `rollout.yaml`） | **仅** 第五节那一行：108.0 / 69.9 / 44.1 FPS |
| 7 段真实视频（射门2 / 趟球2 / 立定跳 / 拉伸1 / 右边比赛1 / 开场训练 / 左边比赛1） | 1080p 与 720p 混合 | 只提供「人 → 成本」的**斜率** | ❌ 否（既有报告） | **仅** 第七节的 2.56 ms/人 与「60 FPS → 5.1 人」 |

**读法**：除第五节、第七节显式标注的**借用值**外，本报告**每一个数字都来自 `1V2 3.mov`**。
下面每张表都有「视频」列，不靠上下文推断。

---

## 一、结论速览（视频 = `1V2 3.mov`）

**"aifootball_app 的性能是多少"没有单一答案——它在 `1V2 3.mov` 上跨 3 倍，取决于姿态档：**

| 姿态档 | 视频 | 帧数 | FPS | ms/帧 | 能不能实时（60 FPS） |
| --- | --- | --- | ---: | ---: | --- |
| `fastest`（HRNet-W32 192×256，关 flipTest） | `1V2 3.mov` | 3000 | **82.7** | 12.10 | ✅ 有余量（4.5 ms） |
| `fastest`（同上，跑全片） | `1V2 3.mov` | 8989 | **70.0 / 64.0** | 14.28 / 15.63 | ⚠️ 贴着 60 FPS 线 |
| `fast`（W48 288×384，关 flipTest） | `1V2 3.mov` | 3000 | **51.2** | 19.53 | ❌ 需把人/帧降到 ~3 |
| `accurate`（W48 288×384 + flipTest） | `1V2 3.mov` | 3000 | **28.0** | 35.70 | ❌ 不可能 |

三点必须一起说的结论（**均基于 `1V2 3.mov`**）：

1. **瓶颈是姿态，不是检测。** 检测器 TensorRT 只占 **0.3–0.5 ms/帧**；姿态（唯一**按人计费**的阶段）在大档位下吃掉几乎整个帧预算。降姿态档 / 减少跑姿态的人数，收益远大于任何其他改动。
2. **本视频只有 4.4–4.8 人/帧这一个水平，无法自算"每人成本"**（两个取值还被时长降级污染）。"约 2.56 ms/人"是**借用的斜率**（见第七节），不是本视频测出来的。
3. **同配置、同视频，跑 3000 帧 82.7 FPS，跑全片只有 70.0 / 64.0 FPS（两次全片差 9%）**，且**不是**因为人变多（前 3000 帧人更多：4.76 vs 全片 4.40）→ 时间相关降级（机器为共享容器，见第六节）→ 报数必须带**视频 + 跑多长**。

---

## 二、测试条件（口径）

| 项 | 值 |
| --- | --- |
| 机器 | **共享容器**：可见 16 逻辑核；GPU 与显存被宿主上其他租户占用（空闲基线 GPU 24–31%、显存 1.8 GB） |
| 被测程序 | `aifootball_app`（C++ SDK 管线，Nexusflow `build/bin`） |
| 检测器 | `rf-detr-small-v1-frozen-step2-960-fp16-b8-fp32io.engine`，`inputSize 960`，`numClasses 4`，`targetClasses [1,2,3]`，`maxBatchSize 4`，`useGpuPreprocess true` |
| 姿态 | `fastest` = `hrnet-w32-256x192-ch23-fp32tf32-b8.engine`；`fast`/`accurate` = `hrnet-w48-dark-ch23-fp16-b8.engine`（accurate 额外 `flipTest`） |
| **主测视频** | **`/home/hx1/yzl/Work/data/1V2 3.mov`**（rollout tag = `1V2_3`；193 MB），1280×720，**h264**，60 fps，8989 帧 |
| 主测视频的内容量 | 人/帧 **4.76**（前 3000 帧）/ **4.40**（全片）、球/帧 0.996、锥/帧 0.086 |
| 运行参数 | `--stride 1`，`--max_frames 3000`（除标注"全片"的行），**无渲染**，ROI 关闭 |
| **FPS 口径** | app 自报 **wall-clock `effective rate`**（`processing` 段，`frames / 墙钟`）。**不是** `observations.jsonl` 里的 `module_timings_ms`（那个含排队等待，相加会得到 12 FPS 的假值） |

---

## 三、逐档结果（视频 = `1V2 3.mov`，同会话背靠背）

| 跑次 | 视频 | 帧数 | FPS | ms/帧 | 相对 fastest |
| --- | --- | --- | ---: | ---: | ---: |
| `fastest`（基准） | `1V2 3.mov` | 3000 | 82.67 | 12.096 | — |
| `fastest`（同批重复一次，验复现性） | `1V2 3.mov` | 3000 | 82.26 | 12.157 | **−0.5%** ✅ |
| `fast` | `1V2 3.mov` | 3000 | 51.19 | 19.534 | −38.1% |
| `accurate` | `1V2 3.mov` | 3000 | 28.01 | 35.696 | −66.1% |
| `fastest` + 检测器预处理挪回 CPU | `1V2 3.mov` | 3000 | 30.36 | 32.944 | −63.3% |
| `fastest` + 全片（第 1 次） | `1V2 3.mov` | 8989 | 70.01 | 14.283 | −15.3% |
| `fastest` + 全片（复跑） | `1V2 3.mov` | 8989 | 63.98 | 15.629 | −22.6% |

- 同会话背靠背复现性 **0.5%** ✅
- 跨时长（3000 帧 vs 全片）差 **15–23%** ✗，两次全片之间也差 **9%** ✗
- 另有一条**不同时段**的既有全片记录（`runs/v1-frozen-step2/1V2_3/`，同视频同配置）：**70.4 FPS / 14.20 ms**（人/帧 4.40）
→ `fastest` 在 **`1V2 3.mov`** 上的可信区间是 **64–83 FPS**，报数必须连"视频 + 跑多长 + 哪个时段"一起说。

---

## 四、消融与分阶段拆解（视频 = `1V2 3.mov`，3000 帧）

### 4.1 姿态档消融（唯一变量 = 姿态模型）

| 项 | 视频 | fastest | fast | accurate |
| --- | --- | ---: | ---: | ---: |
| 姿态 TensorRT | `1V2 3.mov` | **13.59** ms/帧（2.92 ms/人） | 15.88 | **32.56**（6.91 ms/人） |
| 姿态预处理（CPU） | `1V2 3.mov` | 7.32 | **16.43** | 17.23 |
| 每帧总耗时 | `1V2 3.mov` | 12.10 | 19.53 | 35.70 |

- fastest → fast：**+7.44 ms/帧**（姿态输入从 192×256 涨到 288×384，GPU 与 CPU 预处理同时变贵）
- fastest → accurate：**+23.60 ms/帧**（再加 flipTest，姿态推理翻倍）
- 姿态是**每帧调用次数 = 人数**的阶段（实测 **4.5–4.7 次/帧**，与本视频 4.76 人/帧一致）→ 人和姿态成本**线性挂钩**

### 4.2 检测器消融（把 GPU 预处理挪回 CPU）

| 项 | 视频 | 基线 | 预处理挪回 CPU |
| --- | --- | ---: | ---: |
| FPS | `1V2 3.mov` | 82.26 | **30.36（−63%）** |
| `RFDetrDetector.Batch`（含预处理） | `1V2 3.mov` | 6.66 ms/帧 | **31.93 ms/帧** |
| 检测器 TensorRT | `1V2 3.mov` | 0.29 | 0.51 |

→ 检测器的 **GPU 预处理器（CUDA resize+normalize+HWC→CHW）值 20.8 ms/帧**。意义不在"该不该开"（当然该开），而在**它证明 CPU 侧没有余量**：任何把活儿挪回 CPU 的改动都会立刻崩掉吞吐。

### 4.3 分阶段计时（`fastest`，视频 = `1V2 3.mov`，3000 帧；ms/帧 = 阶段总耗时 ÷ 帧数）

| 阶段 | 视频 | ms/帧 | 类型 | 说明 |
| --- | --- | ---: | --- | --- |
| `Worker.Process.SdkInput` | `1V2 3.mov` | 11.73 | 解码/输入 | 720p h264 的读帧 |
| `HRNetPoseEstimator.TensorRT` | `1V2 3.mov` | 13.59 | **GPU** | 4.66 次/帧 × 2.92 ms/人 |
| `HRNetPoseEstimator.Preprocess` | `1V2 3.mov` | 7.32 | CPU | 每人的抠图/resize/normalize |
| `RFDetrDetector.Postprocess` | `1V2 3.mov` | 6.19 | CPU | 300 query × 4 类的后处理 |
| `RFDetrDetector.CopyOutput` | `1V2 3.mov` | 5.87 | CPU | 输出拷贝 |
| `RFDetrDetector.TensorRT` | `1V2 3.mov` | 0.36 | GPU | **检测本身几乎免费** |
| `Video.ColorConvert` | `1V2 3.mov` | 0.46 | CPU | |
| `KeypointSmoother` / `FootballTracker` / `SdkOutput` | `1V2 3.mov` | 0.06 | CPU | 可忽略 |

⚠️ **这张表不能相加**（实测 4 个 `Worker.Process.*` 项都 ≈ 每帧 11.7 ms，等于帧间隔，是"输入饥饿时把等待计进处理窗口"的特征）。**只能看相对大小**：姿态 ≫ 解码 ≈ 检测器 CPU 侧 > 其他。

---

## 五、借用数据（⚠️ **不是本次实测**，不参与上面任何结论）

这一节的数字**全部来自另一个视频**，只用于看趋势是否一致；**不要**把它们当 `1V2 3.mov` 的性能。

| 视频 | 是否本次实测 | 人/帧 | fastest | fast | accurate |
| --- | ---: | ---: | ---: | ---: | --- |
| `射门1-1080p60.mov`（既有基线，读自 `rollout.yaml`） | ❌ 否 | 3.5 | 108.0 FPS / 9.3 ms | 69.9 / 14.3 | 44.1 / 22.7 |
| `1V2 3.mov`（本次，3000 帧） | ✅ 是 | 4.76 | 82.7 / 12.10 | 51.2 / 19.53 | 28.0 / 35.70 |

- **趋势一致** ✅：三档排序 fastest > fast > accurate 完全一致。
- **fastest 可定量对上** ✅：按借用的 2.56 ms/人 校正人差（1.26 人 ≈ +3.2 ms）→ 9.3 + 3.2 = **12.5 ms**，`1V2 3.mov` 实测 **12.10 ms**（差 3.5%；720p 解码比 1080p 便宜，方向也对）。
- `fast` / `accurate` 在 `1V2 3.mov` 上偏低（−11% / −38%）：人更密（4.76 vs 3.5）、姿态预处理 CPU 侧在人更多时更吃亏、机器负载也不同。**绝对值不跨视频比较。**
- 绝对路径：`/home/hx1/yzl/Work/data/射门1-1080p60.mov`（`data/20260726/室外/射门1-1080p60.mov` 是同内容的另一份拷贝，同为 7201 帧/1080p）。

---

## 六、可信度与坑（这几条决定了怎么读上面的数字）

1. **这是共享容器，不能测机器的饱和度。** 我最初用 `/proc/stat` 得到"CPU 16 核 100%"、`nvidia-smi` 得到"GPU 76%"，并据此判定"CPU 受限"——**这是错的** ✗：无 app 时空闲基线同样是 CPU 100%、GPU 24%、显存 1823 MiB（`nvidia-smi` 却看不到任何计算进程）。`/proc/stat` 与 GPU 利用率反映的是**宿主机**，不是本进程。**要判本进程占用，只能读 `/proc/<pid>/stat` 与分阶段计时。**
2. **同视频同配置跨时长差 15–23%，两次全片之间还差 9%**（3000 帧 82.7 → 全片 70.01 → 全片复跑 63.98），且**不是**人变多（前 3000 帧人更多：4.76 vs 全片 4.40）→ 时间相关降级（邻居负载 / 热 / 分配器压力）。**所有结论必须同会话背靠背测，并注明视频 + 跑了多少帧。**
3. **`observations.jsonl` 的 `module_timings_ms` 不能当耗时**（含排队延迟，相加会得到 12 FPS 的假值）。
4. **日志里的 `Worker.Process.*` 只在饱和时才等于真实服务时间**：同一个 ByteTracker，82 FPS 跑里报 **11.7 ms/帧**，30 FPS 跑里只有 **0.083 ms/帧**。绝对值只信 wall-clock `effective rate`。
5. **GPU 侧数字（TensorRT）是上界**：GPU 与邻居共享，姿态 TRT 13.59 ms > 整帧 12.10 ms 就是"等 GPU 别人"造成的测量膨胀。独占机器上应当更好。

---

## 七、建议（按收益排序）

1. **姿态档是总开关**（本视频实测）：要实时就用 `fastest`（82.7 FPS @3000 帧）；需要 W48 质量（`fast`）就得接受 ~51 FPS 或降人/帧；`accurate` 只适合离线出报告。
2. **"人/帧"是第二杠杆**——但**本视频测不出斜率**：它只有 4.4–4.8 人/帧一个水平，且两个取值被时长降级污染，两点相减会得到负斜率（无意义）。可用的斜率是**借用值**（7 段真实视频的受控实测拟合 `ms/帧 ≈ 3.51 + 2.557 × 人/帧`，R²=0.9993，n=5），用它可以换算：
   - 60 FPS（16.7 ms）预算下，本视频 4.76 人（12.10 ms）还有 ~4.5 ms 余量 → **约可支撑 5 人/帧**（与 `runs/fps_vs_persons.md` 第三节互证：60 FPS → 5.1 人）
   - 30 FPS（33.3 ms）预算下，`fast` 档的人均成本 (19.53−3.51)/4.76 = **3.37 ms/人** → **约可支撑 8–9 人/帧**
   - 落地手段：开 ROI（比赛机位，既有实测可提速 2–3 倍）、`VideoReader.stride`、按人框尺寸跳过远景小人
   - ⚠️ 这些换算是**外推**，不是本视频实测；真要下结论，应在 `1V2 3.mov` 上再补 2–3 个人/帧档位（用 ROI 裁出不同人数）。
3. **检测器 CPU 侧**（本视频实测）：`Postprocess`(6.19) + `CopyOutput`(5.87) ≈ 12 ms/帧 的 CPU 开销，是"检测器以外"最大的一块。裁剪 query 数（300 → 100）或把后处理下沉到 GPU 值得试。
4. **解码 11.7 ms/帧**（本视频，720p h264，`SdkInput`）：已与 fastest 档的姿态同量级；开 `stride` 或硬件解码（NVDEC）可释放这部分 CPU。

---

## 附：复现命令

```bash
source AI-Football/tools/detection_finetune/env.sh    # 或直接用 conda 环境的 python

# 视频路径 ↔ rollout tag 对照（rollout 用 tag 索引 data/ 下的视频）
#   /home/hx1/yzl/Work/data/1V2 3.mov                  → tag: 1V2_3              ← 本报告主测
#   /home/hx1/yzl/Work/data/射门1-1080p60.mov           → tag: 射门1-1080p60      ← 第五节借用
#   tag 规则：文件相对 data/ 的路径，去掉扩展名，'/' 和空格都换成 '_'

# ① 全片基准（视频 = 1V2 3.mov；口径 = wall-clock effective rate）
cd AI-Football/tools/rollout
$PY rollout.py bench --version v1-frozen-step2 --video 1V2_3 --repeats 1

# ② 姿态档消融（同一视频，改 --pose-profile；配置由 rollout.py 生成）
$PY rollout.py bench --version v1-frozen-step2 --video 1V2_3 --pose-profile fast     --max-frames 3000
$PY rollout.py bench --version v1-frozen-step2 --video 1V2_3 --pose-profile accurate --max-frames 3000

# ③ 直接调 app（做配置级消融时用，输出目录要分开，别覆盖）
build/bin/aifootball_app tools/rollout/cpp_app/v1-frozen-step2/fastest.yaml \
  --video_path "/home/hx1/yzl/Work/data/1V2 3.mov" --output_dir /tmp/exp1 \
  --stride 1 --max_frames 3000

# 分阶段计时：加 --profile_timers 会额外打印 "Timer profile (per item)" 汇总表
# （不加也能从日志里的 Execute '<stage>' for N items cost: T ms ... 行解析）
```

原始产物（**均为主测视频 `1V2 3.mov`**，其 `--video_path` 可在各自 `run.log` 的命令行与 `observations.jsonl` 的 `video_path` 元数据里核到）：

- 全片基准：`tools/rollout/runs/bench/v1-frozen-step2/1V2_3/run1/{run.log, observations.jsonl, bench.md}`
- 消融日志：`tools/rollout/runs/perf_ablation/{pose_fastest,pose_fast,pose_accurate,det_base,det_cpu_preprocess}.log`（解析结果 `summary.json`）
- 借用值「2.56 ms/人」的出处：**7 段真实视频的受控实测拟合**（不是合成法），见 `tools/rollout/runs/fps_vs_persons.md` 第三节；同文件另有两条独立曲线：app 直跑 8 段 1080p → **2.31 ms/人**、合成法 → **2.36 ms/人**，三条一致在 **2.3–2.6 ms/人**
