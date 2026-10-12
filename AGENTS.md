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
| `examples/aifootball_app/` | demo app（`main.cpp`、`VideoRenderer.*`、`JsonlWriter.*`、`VideoReader.*`、`AppConfig.*`）、`config/` 配置目录；**只跑基模**：写 `observations.jsonl` + 可选渲染，触球判定与 `touches.jsonl` 已移到 Python |
| `examples/touch_eval/` | 触球业务逻辑（**Python**：`py/` uv 工程，包 `touch_eval`，含判定/评估/报告与差分测试）+ 720p 配置 + `run_touch_eval.sh`（app 只跑基模写 `observations.jsonl`，C++ 侧判定实现已删除）；`tools/plot_touch_sweep.py` 逐点重跑评估画 PR 曲线（`sweep_*.csv` / `pr_*.png`）、`tools/touch_view_report.py` 打印事件级/过程级（`--coverage` 才加覆盖式诊断）、`tools/render_touch_review.py` 烧录复核视频、`tools/render_pr_errors.py` 把每个 FP/FN 渲染成带中文原因的对照图 + `pr_review/index.md`；`--playback` 另出整段 1× 实时的 `pr_review/touch_playback.mp4`（顶部中文原因、底部 pred/gt 双时间轴，TP 绿/FP 红/FN 黄）。需要中文字体，缺失会提示 `apt-get install -y fonts-wqy-microhei`，见其 `README.md` |
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
| `coco_fp16_w48.yaml` | rfdetr-960 **fp16** | hrnet-w48-dark 288×384 | 最快；Python 无 fp16 检测器，**非严格对等** |
| `coco_tf32_w48.yaml` | rfdetr-960 **fp32tf32** | hrnet-w48-dark 288×384 | **与 Python `config/app.yaml` 严格对等**（`half: false`）——做对照就用它 |
| `coco_tf32_w32.yaml` | rfdetr-960 | hrnet-w32 192×256 | 关键点比 w48 抖约 50%，用于对比 |
| `coco_fp16_w32_fastest.yaml` | rfdetr-960 **fp16** | hrnet-w32 192×256，**无 flipTest** | **COCO 档里唯一 90+ FPS：97.1 FPS**；与 `coco_tf32_w32` 只差检测精度与 flipTest；`keepOnlyBestBall: true`（球观测数别跟别的份直接比） |
| `finetuned_v1-frozen-step2_fp16_w32_fastest.yaml` | v1-frozen-step2 **fp16**（三分类） | hrnet-w32 192×256，**无 flipTest** | **唯一 90+ FPS**：99.6 FPS（ROI 开、人/帧 4.2）——快在"4 列头 + W32 + 无 flip + 人少"四项同时成立 |

以上 6 份的**同口径实测吞吐**（`射门1-1080p60`、2000 帧、**无渲染**、ROI 按各自文件、人/帧 ≈4.2、16 核空闲；
每份文件的注释头里都写着同一组数字与复测命令）：

| 配置 | 实测 FPS | ms/帧 |
| --- | --- | --- |
| `finetuned_v1-frozen-step2_fp16_w32_fastest.yaml` | **99.6** | 10.0 |
| `coco_fp16_w32_fastest.yaml` | **97.1** | 10.3 |
| `finetuned_v1_fp16_w48.yaml` | 40.1 | 24.9 |
| `coco_tf32_w32.yaml` | 35.6 | 28.1 |
| `coco_fp16_w48.yaml` | 35.2 | 28.4 |
| `coco_tf32_w48.yaml`（对等参考档） | 26.2 | 38.1 |

* **人/帧 是主因，不是模型**：每多 1 人 ≈ +1.6 ms；同一条 W32 配置 ROI 关（10.7 人/帧）46.5 FPS ↔ ROI 开（3.6 人/帧）**108.0 FPS**。
* **渲染不在上面的数字里**：`--render` 实测 19.0 ms/帧（≈ -35% 吞吐）；离线 `tools/render_jsonl.py` 27.5 ms/帧，但可与下一段的 GPU 任务并行。
* 全片口径比 2000 帧口径高 8–20%（实测 `coco_fp16_w48` 42.7 vs 35.2、`coco_tf32_w32` 39.0 vs 35.6、`coco_fp16_w32_fastest` 105.2 vs 97.1）；短跑（600 帧）会因片段人数与机器状态波动，别用短跑数字下结论。

### 3.2 单次运行

```bash
./build/bin/aifootball_app examples/aifootball_app/config/coco_fp16_w48.yaml \
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
- **触球判定与精度评估**：`examples/touch_eval/`（Python 业务逻辑 `py/` + `run_touch_eval.sh` + 720p 配置）。C++ 的判定核心、离线评估 CLI、单测与 app 的 `--detect_touches`（`touches.jsonl`/渲染打点）**已删除**，app 只跑基模写 `observations.jsonl`。方法、指标口径、标注格式、差分口径与迁移缺口见 `examples/touch_eval/README.md` §0。
  - 标注升级为 **v2 一触一行 CSV**（`data/移动传球.touches.csv`，`# point_window_sec:` 声明点标注窗口、`no_touch` 行是负样本区间），旧 `.label` 继续支持，`--dump_gt_csv` 可把旧格式转成 v2（`(2次)` 展开并标注 SPLIT ME）。
  - **主口径 = 计数感知**：一条标注 = 它代表的窗口（区间行 = 它写的跨度 `[start, end)`；点行 = `point_window_sec` + `--window_anchor`，默认 `end`）且发放 `count` 个名额；辅助口径保留一对一容差扫描，默认按**帧**给：±10/30/60/120 帧（60 fps 下即 0.17/0.5/1/2 s），报告里帧与秒并列打印（`--tolerance_frames` / `--primary_tolerance_frames`；`--tolerance` 仍可按秒指定并覆盖帧）。报告新增误检三分类（同一次触球的另一帧 / 贴窗口 / 无人支持）+ `no_touch` 命中。
  - 标注 CSV 支持**可选帧号列**（`t_start_frame`/`t_end_frame`/`frame`，帧号优先于秒；需 `# fps: 60` 或 `--fps`，没有帧率时该列会被报告并忽略而不是瞎猜——**文件级警告会打进报告的 `ground truth warnings`**，不会再静默丢帧）。**帧号是闭区间**（`419–440` → `[419/60, 441/60)`），且**只有写下来的帧会写回文件**（从秒推出来的帧不写，避免"帧覆盖秒"的坑）。`index` 列会被采纳为报告里的 `gt#N`，重号自动顺延；`--renumber_touches` 可原地重排序号/补齐规范列、`--expand_whole_seconds` 把整秒点行原地展开成 `[S−1, S)`（都留 `<file>.bak`）。
  - **整秒的读法已经写进数据，不再是口径开关**：`data/移动传球.touches.csv` 里所有整秒行都用 `--expand_whole_seconds` 展开成它代表的那一秒（`00:29S` → `28.00–29.00` = `[S−1, S)`），**文件里没有点行**，于是 `--window_anchor end|centre|start`（对读列报告里并列打印）在同一份文件上给出**同一组数字**（0.546 / 0.546 / 0.546）。点行时代的账（点行 + 三种读法）：主口径 F1 **0.533 / 0.687 / 0.544**（legacy `.label` 0.541 / 0.724 / 0.561），**差 0.154**；数据更像 `centre`——① "离每条标注最近的事件"与写出时刻之差中位 **+0.000 s**、68/86 落在 ±0.5 s；② 标注者后来照视频抄准的 **25 行**里**只有 8 行**落在 `[S−1, S)`（`centre` 20、`start` 17）。**要不要把那 17 行"写早了一秒"的秒号 +1，是一个还没做的数据修改**（不改检测行为，只改分母怎么理解）；不要用检测参数去补这 0.15。窗口一律**半开 `[from, to)`**，计数表与主口径共用同一窗口。
  - **业务逻辑已由 Python 接管，并用差分测试钉住**：`examples/touch_eval/py/`（uv 工程，包 `touch_eval`；`cd examples/touch_eval/py && uv sync [--extra tools]`，然后 `uv run touch-eval --observations ... --touches_csv ... --output_dir ...`）逐行移植了 `ObservationJsonlReader` / `TouchDetector` / `TouchGroundTruth`（v2 CSV + legacy `.label`）/ `TouchMetrics` / 报告写出，C++ 侧的判定/评估实现与 app 的在线判定（`TouchRuntime`/`TouchJsonlWriter`/渲染器触球叠层/`--detect_touches`）**已删除**，app 只跑基模写 `observations.jsonl`。参考产物**冻结在 `py/tests/reference/`**，差分仍对着 C++ 比（不是自我比对）；未移植的东西（`--expand_whole_seconds`/`--renumber_touches`/`--dump_gt_csv`/`contact_signal.csv`/`touch_report.txt`）列在 README §0.1。差分测试 `uv run pytest -q -m parity`（整段 16037 帧约 7 s，**全绿**）：判定 97/97 事件 × 13 字段、两个不应期与两个 gate 的计数、`--dump_gt` 的全字段（97 touch + 3 no_touch + 6 low_quality + totals）、`touch_report.json` 的全字段（主口径 / 另一锚点视图 / 容差扫描 / 计数表 97 行 / 误检分解 / 诊断百分位 / 标定曲线 / 低质量第二视图 / failures 分组与计数 / 逐事件 `gt_index`+`verdict`）都逐字段一致（相对误差 < 1e-5）。**移植时必须照搬的两处精度细节**：`speedWindowSec` 在 C++ 里是 `double`（其余几何参数是 `float`，差 6e-9 s 就会多算/少算一个球样本），以及几何量全程 `float32`（用 `numpy.float32`）。两处有意差异：事件文件里 Python 写全精度（C++ 按 6 位有效数字取整）、`failures[].examples` 与 FP 文字说明是本实现自己的措辞（差分只比分组与计数）。性能：Python 跑完整段评估 **1.9 s**（其中读 89 MB JSONL 1.3 s）。
  - **`ignore=yes` 忽略某条标注**：该行所有口径都不计（容量/召回/误检分类/双口径），报告写 `N ignored row(s) (not scored anywhere)`，`touch_report.json` 有 `ignored` 段；行仍留在文件里（`--renumber_touches`/`--dump_gt*` 不丢），与"帧不可判"的 `low_quality_data=yes`（只在第二份分数里剔除）是两件事。
  - **低质量行双口径**：`low_quality_data=yes` 标记"帧太糊/人太小、标注者也判断不了"的行；报告与 JSON 同时给出「全部标注」与「剔除这些行」两份主口径（后者把只由这些行解释的预测一并从分子分母去掉并报出剔除数），另有这些窗口的诊断。
  - **匹配用最优分配**（最大基数匹配 Kuhn，容量>1 的行展开成多个槽位）：TP 是"最优一配一/一配名额"的结果，不是贪心扫一遍的结果——贪心在"近的检测占了两条标注都能配的那条、另一条检测只够得着它"时会吊死一条（真实数据上差 1 个 TP：±30 帧 0.738→0.749）。
  - 去伪门控由数据定：**不应期 30 帧（0.5 s，同人同脚）与同人任一脚 30 帧（`person_refractory_frames`）默认开启**；**球速突变门控与只信跟踪球默认关闭**（实测在本数据上掉召回），代码保留、拒绝计数打进报告。
  - 当前结果（**fp16 路径**，`data/移动传球.mov` 全片 16037 帧 / **97 条标注触球**（33 条帧跨度区间 + 64 条整秒展开成的 `[S−1,S)`，无点行）+ 3 段确认无触球 + 6 行低质量）：主口径 **P/R/F1 = 0.546/0.546/0.546**（TP53 FP44 FN44，容量 97，报出 97 个事件），剔除 6 行低质量后 **0.537/0.560/0.548**；**一对一默认容差 0（事件必须落在跨度内），与主口径同值 0.546/0.546/0.546**；诊断口径 ±30 帧 0.763、±10 帧 0.608、±60 帧 0.814、±120 帧 0.887；计数完全一致 51/97（52.6%）。44 个误检 = 落在跨度内但名额已满 6 + **贴着跨度 33** + 确认无触球区间 3 + 无人支持 2；自动重复上报 0。**默认容差 0 时一对一与主口径同值**；一旦放宽 T，它就不再是"时刻准不准"的度量（事件落进跨度就按 Δt=0 计，±10 帧因此从 0.410 涨到 0.608），只能当诊断读。放宽窗口 `--window_pad 0.15/0.5` → 主口径 F1 0.598/0.763（同样是改口径不是改检测）。
  - **精度路径只有 fp16 一条**（`examples/touch_eval/README.md` §1.4）：`config/moving_pass_720p.yaml` = fp16，**157.4 FPS / 6.35 ms**，是唯一维护的路径。曾经的 TF32 孪生（只差一行 `enginePath`，57.5 FPS / 17.4 ms）实测换路径只让触球分动 ±0.03 且方向不定，**已停用**（配置与产物删除）：所有"当前结果"、差分测试与扫描曲线只以 fp16 为准。引用速度/触球数字写明走的是哪条路。
  - **对外只报一对一 P/R**（`tools/touch_view_report.py`，默认路径）：主口径（计数感知，窗口=跨度）0.546/0.546/0.546；一对一（**默认容差 0**）同值 0.546/0.546/0.546；诊断 ±30 帧一对一 0.763/0.763/**0.763**；过程级（同人同脚 1 s 内合并）0.585/0.546/0.565。`--coverage` 的覆盖式（±30 帧 标注侧 82.7%、预测侧 81.4%）只是"这一秒有没有判到"的**诊断**，两边各自计数、不是 P/R，不作为答案引用。
  - **（诊断口径）一对一 ±0.5 s 到 0.80 还差 0.04，原因不在参数而在时间一致性**（`examples/touch_eval/README.md` §4.1）：当前 0.763，最优分配下 23 个预测无独立标注可配，绝对上限 0.865；要 0.80 需"多找出 ~8 条标注附近的触球且不多报"，而 **14 条标注在 ±0.5 s 内没有任何预测**（最近差 0.53~2.58 s；关掉两个不应期只多覆盖 2 条、主口径 0.546→0.516、±30 帧 0.763→0.715）。检测特征全部试过，与命中不可分：只信跟踪球 ±30 帧 0.738、球速门控 500 px/s 0.471，都不如默认；27 组参数网格里 ±30 帧最高 0.772（c22/s12/p30，该组主口径 0.538），主口径最高 0.567（c10，该组 ±30 帧 0.759）。**注意辅助 ±T 一对一已不是"时刻准不准"的度量**（事件落进跨度就按 Δt=0，±10 帧因此从 0.410 涨到 0.608）。逐条 FP/FN 已用 `tools/render_pr_errors.py` 渲染成中文标注的对照图（`output/<dir>/pr_review/`，含 `index.md`/`index.csv`；`--playback` 出整段实时播放视频 `touch_playback.mp4`，图上自动裁到"事件指定的人 + 那颗球"放大 2~3 倍），原因分 `slot_taken`/`adjacent`/`inside_no_touch`/`unannotated` 与 `contact_but_no_event`/`never_near`/`no_ball`/`no_feet`；复核后回写标注是唯一的提分路径。
  - **按曲线标定的默认值**（`tools/plot_touch_sweep.py`，每个点一次真实重放）：`min_contact_px` 20→**22**、`min_split_frames` 12→**24**、`refractory_frames` 21→**30**、新增 `person_refractory_frames`=**30**（同一人**任意脚** 0.5 s 内只报一次；球在两脚之间穿过时同一次触球会被左右脚各报一次，这是本轮收益最大的一刀；`centre` 读法下 30 帧在两个口径上同时最优，`end` 读法下主口径峰挪到 36 帧而锚点无关列仍是 30 帧）。当前（整秒展开成区间）主口径 0.546、±30 帧 0.763；主口径峰在 36 帧（0.558）而 ±30 帧峰在 30 帧（0.763），取 30 帧——**窗口宽容度会污染主口径的"最优值"，调参以"事件落进标注跨度且无系统偏移"的那一列为准**。`event_time_mode`：主口径峰在"接触中点"（0.558 vs 0.546），但 ±30 帧 0.741 vs 0.763 更差，保留开关、默认 0。
  - 数字比上一版（0.700）低是因为**标注变准了**：旧标注绝大多数是"整秒 + 1 s 窗口"，现标注 30 条写成 0.05~5 s 帧跨度，窗口变窄，同样的 117 个预测命中变少、贴窗口变多。
  - **PR 曲线**：`tools/plot_touch_sweep.py` 每个取值跑一次 `touch_eval` 再画图（纯 CPU，一条约 30 秒）。`min_contact_px` 在这条视频上峰值 **22 px → F1 0.639**（默认 20 px → 0.623；28/36 px 掉到 0.610/0.579），是当前真正的工作点旋钮（`contact_ratio` 被它夹住，扫出来是平的）。`window_pad` 曲线单调（放宽窗口只可能把误检变成命中），属于评估口径而非检测参数。
