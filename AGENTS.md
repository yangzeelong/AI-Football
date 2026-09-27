# AGENTS.md — 项目说明与开发流程

面向**在本仓库工作的 agent**。人类向的框架总览见 `README.md`；本文件只写"要动手做事时必须知道的东西"：环境、命令、约定、坑、以及和 Python 参考实现对照的方法。

---

## 1. 这是什么项目

**NexusFlow**：C++14 的数据流/流水线框架（`src/`、`include/nexusflow/`）。
框架上跑一条 **AI-Football 算法流水线**（`sdk/aifootball/`），demo app 在 `examples/aifootball_app/`。

```
视频 → 解码 → RF-DETR 检测 → ByteTracker(人/球) → HRNet 姿态 → 关键点平滑 → 足球跟踪 → JSONL / 渲染
```

| 路径 | 作用 |
| --- | --- |
| `include/nexusflow/`、`src/` | 框架本体（模块、消息、流水线、计时器 `src/common/TimerRegistry.*`） |
| `sdk/aifootball/` | 算法 SDK：检测/跟踪/姿态/平滑模块 + 公开 API `include/aifootball/AIFootball.hpp` |
| `examples/aifootball_app/` | demo app（`main.cpp`、`VideoRenderer.*`、`JsonlWriter.*`、`VideoReader.*`、`AppConfig.*`）、`config/` 配置目录；`TouchRuntime.*`/`TouchJsonlWriter.*` 是 `--detect_touches` 的在线触球判定与 `touches.jsonl` 输出 |
| `examples/touch_eval/` | 触球判定核心 + 离线精度评估（`touch_eval_core` 静态库被 app 复用）+ 720p 配置 + `run_touch_eval.sh`；`tools/plot_touch_sweep.py` 逐点重跑评估画 PR 曲线（`sweep_*.csv` / `pr_*.png`）、`tools/render_touch_review.py` 烧录复核视频，见其 `README.md` |
| `tools/` | `run_all_configs.sh`（批量跑配置）、`render_jsonl.py`（JSONL 回放成视频）、`model_conversion/`、`clang-format-all.sh` |
| `tests/test/` | gtest 用例（`nexusflow_tests`） |
| `benchmarks/` | 基准（含 `pose_benchmark`） |
| `data/` | 测试视频 `射门1-1080p60.mov`（1920×1080、60fps、7201 帧、120s） |
| **`/home/hx1/yzl/Work/AI-Football`** | **Python 参考实现（另一个仓库）**：`code/app.py`，我们做逐帧对照的基准 |

---

## 2. 环境与构建

- **C++14**（`CMakeLists.txt:5` → `set(CMAKE_CXX_STANDARD 14)`）。**禁止 C++17 及以上特性**（结构化绑定、`if constexpr`、`std::optional`、`string_view`、inline 变量、if-init 等）。
- CUDA/TensorRT 已就绪（`/usr/local/cuda`），引擎文件在 `/home/hx1/yzl/Work/AI-Football/models/`。
- **构建目录统一叫 `build`**（不要再出现 `build-release` 之类的名字）。Release / Unix Makefiles / `WITH_AIFOOTBALL_SDK=ON` `WITH_BENCHMARK=ON` `WITH_EXAMPLES=ON` `WITH_TESTING=ON`。

```bash
# 增量构建（日常就用这条）
cmake --build build --parallel 16
```

可执行文件在 `build/bin/`（`build/lib/` 放库）：

```
build/bin/aifootball_app       # demo
build/bin/nexusflow_tests      # 单测
```

### ⚠️ 本机无外网：从零配置必须用本地第三方源码

第三方依赖走 `FetchContent`（spdlog / yaml-cpp / jsoncpp / googletest / googlebenchmark，见 `cmake/3rdparty/`），默认从 GitHub 下载，**本机连不上**。已有的 `build/_deps/*-src` 就是它们的源码，配置时显式指过去：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DWITH_AIFOOTBALL_SDK=ON -DWITH_BENCHMARK=ON -DWITH_EXAMPLES=ON -DWITH_TESTING=ON \
      -DFETCHCONTENT_SOURCE_DIR_SPDLOG=$PWD/build/_deps/spdlog-src \
      -DFETCHCONTENT_SOURCE_DIR_YAML=$PWD/build/_deps/yaml-src \
      -DFETCHCONTENT_SOURCE_DIR_JSONCPP=$PWD/build/_deps/jsoncpp-src \
      -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$PWD/build/_deps/googletest-src \
      -DFETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK=$PWD/build/_deps/googlebenchmark-src
```

（当前 `build/` 的 cache 里这五项已经写好，日常 `cmake --build build` 不会再联网。`-DFETCHCONTENT_FULLY_DISCONNECTED=ON` **不行**：它会让 `spdlog::spdlog`/`GTest::gtest`/`benchmark::benchmark` 这些别名目标不生成。）

### ⚠️ 构建目录改名/搬迁的两个坑

1. cache 里的输出目录是**绝对路径**（`CMAKE_RUNTIME_OUTPUT_DIRECTORY` 等 → `<dir>/bin`、`<dir>/lib`）。换目录名（比如 `build-release` → `build`）后必须一并改，否则 `make` 会**重新创建旧目录**并把二进制写到那里：

   ```bash
   cmake -S . -B build \
     -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$PWD/build/bin \
     -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$PWD/build/lib \
     -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$PWD/build/lib \
     -DCMAKE_PDB_OUTPUT_DIRECTORY=$PWD/build/bin
   ```

2. 生成物里还留有旧绝对路径（如 `build/CMakeFiles/VerifyGlobs.cmake`），改名后直接 `make` 会报 `Not a file: .../build-release/...`。**先跑一次上面的 `cmake -S . -B build` 重新生成，再 `cmake --build build`**。

---

## 3. 跑 demo

### 3.1 配置目录

`examples/aifootball_app/config/` —— **放进目录就会被 `run_all_configs.sh` 跑到**：

| 文件 | 检测器 | 姿态 | 定位 |
| --- | --- | --- | --- |
| `config.yaml` | rfdetr-960 **fp16** | hrnet-w48-dark 288×384 | 最快；Python 无 fp16 检测器，**非严格对等** |
| `ref_rfdetr960_hrnetw48d384.yaml` | rfdetr-960 **fp32tf32** | hrnet-w48-dark 288×384 | **与 Python `config/app.yaml` 严格对等**（`half: false`）——做对照就用它 |
| `rfdetr960_hrnetw32_256.yaml` | rfdetr-960 | hrnet-w32 192×256 | 关键点比 w48 抖约 50%，用于对比 |

### 3.2 单次运行

```bash
./build/bin/aifootball_app examples/aifootball_app/config/config.yaml \
  --max_frames 300 --output_dir output/run --render
```

常用参数：`--video_path/-v`、`--max_frames/-n`、`--stride/-S`、`--output_dir/-o`、`--render/-r`、`--render_path/-R`、`--device/-d`、`--profile_timers`、`--verbose/-V`、`--quiet/-q`。
产物：`<output_dir>/observations.jsonl`（`--render` 时另加 `rendered.mp4`）。

### 3.3 批量（串行跑目录里所有配置）

```bash
PYTHON=/root/miniconda3/envs/ai-football/bin/python \
  ./tools/run_all_configs.sh 300 output/configs
```

- 扫描 `examples/aifootball_app/config/*.yaml`，**有几个跑几个**（按文件名排序）
- 串行执行（共享 GPU，并发会让计时与对比失真），每个配置独立输出目录 + `run log`
- 结束后拼成 **N 列并排视频** `side_by_side.mp4`（列宽自适应 `max(480, 3840/N)`），列上烧配置名

### 3.4 `--render` 画了什么

顶栏 `frame=/time=/det=`、`fps=<实际处理速率> processing`、青色 ROI 多边形（顶点带点）、person 绿框 + `person {conf:.2f} id=`、球检测框 + `sports ball {conf:.2f}`、被跟踪球橙圆（predicted 时蓝）+ `ball id=N observed`、被过滤球框按原因着色（红=置信度/品红=尺寸/紫=两者）、左下计数面板（`ball: raw/roi/kept/rescued/tracked/state`、`filter:`、`persons:`）。

- 渲染开销：1080p 约 **4 ms/帧（消费者线程）**，端到端只 +0.5 ms/帧左右；要**纯流水线**数字就跑不带 `--render` 的。
- 面板与过滤框**随 `--render` 常开**（Python 那边需 `--debug` 才画）。

### 3.5 回放工具

```bash
python3 tools/render_jsonl.py --observations <dir>/observations.jsonl --output <dir>/replay.mp4
# 可选：--video 覆盖源视频、--max-frames、--fps、--hide-boxes / --hide-keypoints / --hide-panel
```

> 注意：本仓库的 `observations.jsonl` 由 **C++ 侧**写出；`tools/render_jsonl.py` 也能直接回放 Python 侧产物（schema 相同），是"不重跑流水线就看结果"的最快路径。

---

## 4. 测试与性能

```bash
./build/bin/nexusflow_tests        # 当前 39 个用例
```

- 改完 SDK 算法/渲染**必须**跑；**不要为了让测试通过而改测试语义**。
- 性能/延迟：`--profile_timers` 打印每模块统计（`Execute '<name>' for N items cost ... avg/p95/min/max`，滑窗 1024 的 p95；单轴口径=每次调用）。

---

## 5. 与 Python 参考对照（最常用工作流）

### 5.1 Python 侧怎么跑

```bash
cd /home/hx1/yzl/Work/AI-Football            # 解释器：/root/miniconda3/envs/ai-football/bin/python
python code/app.py \
  --video /home/hx1/yzl/Work/Nexusflow/data/射门1-1080p60.mov \
  --output-dir tmp/<name> --config config/app.yaml \
  --roi-config config/roi.json --use-roi \
  --camera-id C1 --device cuda:0 --stride 1 \
  [--max-frames N] [--no-render]
```

模型与 preset 由 `--config` 指向的 app.yaml 决定（`config/app.yaml` = rfdetr 960 + `hrnet-w48-dark` + `half: false` ⇒ 对应我们的 **`ref_...yaml`**）。换 preset/检测分辨率就是换一份 app.yaml（参考 `tmp/pose_experiments_nexus/*/`）。

### 5.2 ⚠️ 不要动 Python 的工作区

`AI-Football` 工作区有**未提交改动**（8 文件，含 `pose_estimation.py` 的 `flip_test`、`football_vision.py` 的 YOLO backend/letterbox、`config/app.yaml`）。
**不要修改、回退、stash 它们**。要看旧版本用：

```bash
git -C /home/hx1/yzl/Work/AI-Football worktree add /tmp/pyrev <rev>
# 或
git -C /home/hx1/yzl/Work/AI-Football show <rev>:code/tracking.py > /tmp/tracking.py
```

### 5.3 对照方法与指标（照着做）

两边产物都是 `observations.jsonl`，schema 相同：

```json
{"type":"frame","frame_index":960,"timestamp_sec":16.0,"persons":[
  {"track_id":2,"bbox":[x0,y0,x1,y1],"confidence":0.89,
   "keypoints":[{"name":"nose","x":..,"y":..,"confidence":..,"state":"observed"}, ...]}],
 "balls":[{"track_id":-1,"bbox":[...],"center":[..],"confidence":..,"state":"observed"}],
 "raw_detection_counts":{"total":5,"person":3,"ball":2}}
```

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| 关键点抖动 | 同 track、**连续帧**、同一关节的帧间位移；报 mean / p50 / p95 / p99 / max | 越小越稳 |
| 真运动是否被削 | **净位移**（首末距离，必须保持）vs **路程**（累计距离，抖动会抬高） | 滤波/平滑类改动必看 |
| 框抖动 | bbox 中心帧间位移 | 检测器稳定性 |

**铁律**：抖动绝对值随样本长度剧烈变化（前 221 帧 0.20 px、前 300 帧 0.57 px、全片 7201 帧 1.06–1.39 px）——**只能同长度、同区间对比**。判定等价的口径是"同区间同长度差异 <1~2%"。

---

## 6. 硬规则（务必遵守）

1. **不要自动 `git commit`**。只有用户明确说"提交/commit"时才提交；否则改动一律留在工作区，并在汇报里说明未提交。
2. **C++14**，不用 C++17+ 语法。
3. **构建目录固定为 `build`**：增量构建 `cmake --build build --parallel 16`，可执行在 `build/bin/`。不要新建或沿用 `build-release` 等其他名字（换名/搬迁见第 2 节的两条坑）。
4. **GPU 共享**：渲染/推理实验**串行**跑，单次实验 ≤600 帧；**绝不 `kill`/`pkill` 他人进程**；不要动他人输出目录（例如 `tmp/backpressure/`）。
5. **不改动 `/home/hx1/yzl/Work/AI-Football` 的工作区**（见 5.2）。
6. 改完必须验证：构建退出码 0 + `build/bin/nexusflow_tests` 全绿；跑了就说跑了，**没跑就明确说没跑**，不要含糊。
7. **行尾**：提交前比对 `git diff --stat` 与 `git diff --stat --ignore-cr-at-eol`；`sdk/aifootball/src/detector/RFDetrDetectorInfer.hpp` 与 `src/pose/HRNetPoseEstimatorInfer.hpp` 在 HEAD 里是 CRLF/LF 混合，编辑器保存容易整文件改写（会产生几百行假 diff）。
8. 汇报带**数字 + 命令原文**；拿不准标"未确认"，不要猜。

---

## 7. 已知差异与坑（跟 Python 对齐时）

| 项 | 现状 | 说明 |
| --- | --- | --- |
| 关键点 `state` 词表 | 我们只有 `observed/virtual/missing` | Python 有 `smoothed/held`；**坐标已验证一致**，仅标签不同（会让 JSONL diff 满屏） |
| 球 track id | 我们的球是无跟踪透传（无 id） | Python 的 ByteTrack 跟踪 ball 类，标签带 `id=` |
| person 框 | `ProcessResult::persons[].bbox` = **姿态裁剪框**（= Python 的 JSONL ✅）；`detectionBbox` = 原始检测框，**仅渲染用** | Python 渲染画检测框、JSONL 存裁剪框，两边都已对齐 |
| 叠加层 | 面板/被过滤框随 `--render` 常开 | Python 需 `--debug` |
| 抖动成因 | 短命 track（检测间断→反复重建）+ 背向/遮挡的**远侧肢体**（膝/踝/趾/腕） | 调 smoother 参数只有 ±2~3% 收益；姿态 preset 影响更大（w32 比 w48 抖 ~50%）；检测分辨率 512→960 收益最大（人框抖 −57%） |

---

## 8. 代码地图（改哪里找哪里）

| 想改的东西 | 文件 |
| --- | --- |
| 检测器 | `sdk/aifootball/src/detector/RFDetrDetector*.{hpp,cpp}`（含 CUDA 预处理 `RFDetrCudaPreprocess.cu`） |
| 人的跟踪 / 球的跟踪 | `src/tracker/ByteTracker.*`（人；球只过滤+透传）、`src/tracker/FootballTracker.*`（球：打分选球、救援、`keepOnlyBestBall`） |
| 姿态 | `src/pose/HRNetPoseEstimator*.{hpp,cpp}`（裁剪扩展 `ExpandPoseBox`、flip-test、TensorRT batch） |
| 关键点平滑 | `src/smoother/KeypointSmoother.*`（static/moving 判定、骨架约束、虚拟关节） |
| 公开 API 与输出组装 | `include/aifootball/AIFootball.hpp`、`src/AIFootballPipeline.cpp` |
| 内部消息（数据流） | `src/common/MyMessage.hpp` |
| demo：渲染/JSONL/解码/配置解析 | `examples/aifootball_app/{VideoRenderer,JsonlWriter,VideoReader,AppConfig}.{hpp,cpp}`、`main.cpp` |
| 配置与批量脚本 | `examples/aifootball_app/config/`、`tools/run_all_configs.sh` |
| 计时器 | `src/common/TimerRegistry.{hpp,cpp}` |

---

## 9. 最近做过的变更（别重复劳动，也别踩回去）

- **球**：可输出全部候选（`keepOnlyBestBall: false` → `track_id=-1` 的未跟踪观测）；被过滤球按姿态支持救援（`rescued`）；选球改为打分（置信度+人脚支持−背景惩罚）。
- **渲染**：ROI 多边形、person/球标签、被过滤球按原因着色、计数面板、`fps=<处理速率>`、渲染耗时单列（`render: X ms/frame`，与端到端 `per frame` 区分）。
- **数据对齐**：`persons[].bbox` 改为姿态裁剪框（与 Python 比例 1.0000，关键点零变化）；`detectionBbox` 只服务渲染。
- **配置**：配置目录化 + `run_all_configs.sh` 扫目录跑；README/注释路径同步。
- **TimerRegistry**：单轴化（只报 items=每次调用）、p95 滑窗（1024、nearest-rank、仅在报表时计算）。
- **触球判定与精度评估**：新增 `examples/touch_eval/`（可复用判定核心 `touch_eval_core` + 离线评估 CLI + `run_touch_eval.sh` + 720p 配置）与 app 的 `--detect_touches`（`touches.jsonl` + 渲染打点）。方法、指标口径、标注格式与复现命令见 `examples/touch_eval/README.md`。
  - 标注升级为 **v2 一触一行 CSV**（`data/移动传球.touches.csv`，`# point_window_sec:` 声明点标注窗口、`no_touch` 行是负样本区间），旧 `.label` 继续支持，`--dump_gt_csv` 可把旧格式转成 v2（`(2次)` 展开并标注 SPLIT ME）。
  - **主口径 = 计数感知**：一条点标注 = 它代表的窗口（`.label` 秒=1.0 s、0.01 s CSV=0.5 s，居中）且发放 `count` 个名额；辅助口径保留一对一 ±0.25/0.5/1/2 s。报告新增误检三分类（同一次触球的另一帧 / 贴窗口 / 无人支持）+ `no_touch` 命中。
  - 标注 CSV 支持**可选帧号列**（`t_start_frame`/`t_end_frame`/`frame`，帧号优先于秒；需 `# fps: 60` 或 `--fps`，没有帧率时该列会被报告并忽略而不是瞎猜）；`index` 列会被采纳为报告里的 `gt#N`，重号自动顺延；`--renumber_touches` 可原地重排序号/补齐规范列（留 `<file>.bak`）。
  - **低质量行双口径**：`low_quality_data=yes` 标记"帧太糊/人太小、标注者也判断不了"的行；报告与 JSON 同时给出「全部标注」与「剔除这些行」两份主口径（后者把只由这些行解释的预测一并从分子分母去掉并报出剔除数），另有这些窗口的诊断。
  - 去伪门控由数据定：**不应期 21 帧（0.35 s，同人同脚去重）默认开启**；**球速突变门控与只信跟踪球默认关闭**（实测在本数据上掉召回），代码保留、拒绝计数打进报告。
  - 当前结果（`data/移动传球.mov` 全片 16037 帧 / **98 条标注触球**（30 条帧跨度区间 + 68 条点）+ 3 段确认无触球 + 5 行低质量）：主口径 **P/R/F1 = 0.573/0.684/0.623**（TP67 FP50 FN31），剔除低质量后 **0.565/0.699/0.625**（容量 93，随行剔除 2 个预测）；50 个误检 = 同一时刻报多了 19（同一次触球另一帧 8 + **落在窗口内但名额已满 11**）+ 确认无触球区间 3 + 贴窗口 26（窗口外 0.07~1.48 s）+ **无人支持 2**（126.73 s/131.62 s，待复核）。放宽窗口 `--window_pad 0.15/0.5` → F1 0.673/0.699。
  - **同一份运行的三种数法**（`tools/touch_view_report.py`）：事件级 F1 0.623；过程级（同人同脚 1 s 内合并）P 0.700/R 0.684/F1 0.692；覆盖式召回 ±0.5 s 83.7%、±1 s 89.8%、±2 s 98.0%。引用"准确率"时必须说明是哪种口径。
  - 数字比上一版（0.700）低是因为**标注变准了**：旧标注绝大多数是"整秒 + 1 s 窗口"，现标注 30 条写成 0.05~5 s 帧跨度，窗口变窄，同样的 117 个预测命中变少、贴窗口变多。
  - **PR 曲线**：`tools/plot_touch_sweep.py` 每个取值跑一次 `touch_eval` 再画图（纯 CPU，一条约 30 秒）。`min_contact_px` 在这条视频上峰值 **22 px → F1 0.639**（默认 20 px → 0.623；28/36 px 掉到 0.610/0.579），是当前真正的工作点旋钮（`contact_ratio` 被它夹住，扫出来是平的）。`window_pad` 曲线单调（放宽窗口只可能把误检变成命中），属于评估口径而非检测参数。
