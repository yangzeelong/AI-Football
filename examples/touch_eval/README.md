# touch_eval — 触球判定与精度评估

用一条带人工标注的视频衡量「AI-Football SDK 能不能准确判断球员触球」。
判定逻辑与评估工具都在 `examples/` 里，SDK 保持通用（`sdk/aifootball/**` 未改动）；
demo app 通过 `--detect_touches` 复用同一份判定核心（`touch_eval_core`），因此**在线看到的和报告里算的必然是同一套语义**。

```
标注（一触一行 CSV / 旧 .label）──┐
                                  ├─> GroundTruth ──┐
视频 ──aifootball_app──> observations.jsonl ─────────┼─> touch_eval ──> touch_report.txt / .json
             │                （流式读取 + 候选缓存） │
             └── touches.jsonl + 渲染触球标记 ────────┘
```

## 1. 一键复现

```bash
# 全片（16037 帧、267 s，约 5 分钟）。RENDER=1 会同时产出带触球标记的 rendered.mp4
./examples/touch_eval/run_touch_eval.sh
# 指定视频 / 标注 / 输出目录
./examples/touch_eval/run_touch_eval.sh data/移动传球.mov data/移动传球.touches.csv output/touch_eval_moving_pass
```

等价的单步命令：

```bash
# 1) 跑流水线并在线判定触球（1280x720 需用本目录的配置：ROI 关断）
./build/bin/aifootball_app examples/touch_eval/config/moving_pass_720p.yaml \
    --video_path data/移动传球.mov --output_dir output/touch_eval_moving_pass \
    --max_frames 0 --detect_touches --render

# 2) 纯 CPU 离线评估（读 observations.jsonl + 标注，1~2 秒出报告）
./build/bin/touch_eval --observations output/touch_eval_moving_pass/observations.jsonl \
    --label data/移动传球.touches.csv --output_dir output/touch_eval_moving_pass \
    --dump_gt output/touch_eval_moving_pass/ground_truth.json \
    --dump_gt_csv output/touch_eval_moving_pass/ground_truth.csv
```

产物：`touch_report.txt` / `touch_report.json`（指标）、`touch_events.jsonl`（每个预测事件，带是否命中、
**判定结论 `verdict`** 与球速诊断）、`contact_signal.csv`（逐帧接触信号，回看某帧为什么判/没判）、
`ground_truth.json` / `.csv`（可审计真值），在线侧 `touches.jsonl` 与 `rendered.mp4`。

### 1.1 肉眼复核视频（pred vs gt）

```bash
python3 examples/touch_eval/tools/render_touch_review.py --output-dir output/touch_eval_moving_pass
# 只看某一段 / 抽帧快速浏览
python3 examples/touch_eval/tools/render_touch_review.py --output-dir <dir> --from 30 --to 46
python3 examples/touch_eval/tools/render_touch_review.py --output-dir <dir> --stride 4
```

默认以 `<dir>/rendered.mp4`（SDK 叠加层）为底，在上面烧评估结论，输出 `<dir>/review_pred_vs_gt.mp4`：

- **画面上方**：当前处于哪条标注的窗口内（`GT #11  1 touch(es)  1/1 found` / `MISSED`），以及窗口附近的
  每个预测事件及其结论——`correct (inside an annotation window) -> GT #2`、`duplicate (same touch,
  another frame)`、`outside the window by a fraction of a second`、`inside a no_touch span`、
  `UNANNOTATED (nothing in the file supports it)`，并附人/脚别/距离/阈值。
- **左边框**：有标注时按"找到=绿 / 漏掉=红"着色，扫一眼就知道当前有没有漏。
- **底部整片时间条**：下沿是标注（绿=找到、红=漏掉、紫=确认无触球区间，次数 2 的更高），上沿是预测
  （绿=命中、橙=重复上报、黄=贴窗口、紫=落在无触球区间、红=无人支持），白线是播放头——不用拖进度条
  就能看出哪一段密集、哪一段漏得多。

全片渲染约 46 秒（纯 CPU），体积较大（720p60 约 400 MB），快速浏览用 `--stride`。

想在别的机器（例如 Windows）上看，用仓库自带的 Range 服务器 + IDE 端口转发：

```bash
python3 tools/serve_range.py --directory output/touch_eval_moving_pass --port 8080
# 浏览器打开 http://localhost:8080/review_pred_vs_gt.mp4 —— 支持边播边拖进度条
```

不要用 `python3 -m http.server`：它不支持 Range（每次返回整个文件、没有 `Accept-Ranges`），
播放器无法跳转，只能把几百 MB 一次性拉完，中途断开还会刷一屏 `ConnectionResetError` 栈。

### 1.2 换个口径看同一个结果（三种数法一次打印）

```bash
python3 examples/touch_eval/tools/touch_view_report.py --output-dir output/touch_eval_moving_pass
```

把已写出的 `touch_events.jsonl` + `ground_truth.json` 按三种数法各算一遍：**事件级**（主口径，逐帧算）、
**过程级**（同人同脚 1.0 s / 0.5 s 内合并成一次触球）、**覆盖式召回**（±T 内至少有一个预测就算判到）。
用途见 §4 的"同一份运行的三种数法"——引用数字时先说清是哪一种。

### 1.3 旋钮扫描与 PR 曲线

```bash
python3 examples/touch_eval/tools/plot_touch_sweep.py \
    --observations output/touch_eval_moving_pass/observations.jsonl \
    --annotations data/移动传球.touches.csv \
    --output-dir output/touch_eval_moving_pass
# 只重画（不重跑评估）：加 --render-only
# 自定取值 / 换旋钮：--sweep min_contact_px=16,20,24,28 --sweep refractory_frames
```

每个取值都是一次真实的 `touch_eval` 运行（同一份 `observations.jsonl`，纯 CPU，一次扫描约 30 秒、不占 GPU），
所以图上的点就是报告里的数。产物：

- `sweep_<knob>.csv`：一行一个取值，两套口径（全部标注 / 剔除低质量）的 TP/FP/FN/P/R/F1；
- `pr_<knob>.png`：左边是 precision-recall（点上的标签是旋钮取值，红星是当前默认，灰线是等 F1 线，坐标轴
  按数据自动缩放——整幅 0~1 的话所有点会挤成一坨），右边是 P/R/F1 随旋钮的变化。

**哪个旋钮值得扫**：`min_contact_px` 是当前真正的工作点（720p 下人体尺度项被它夹住）。实测：

| min_contact_px | 10 | 16 | **20（默认）** | 22 | 25 | 32 | 40 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P | 0.575 | 0.579 | 0.573 | 0.579 | 0.566 | 0.548 | 0.504 |
| R | 0.663 | 0.673 | 0.684 | 0.714 | 0.704 | 0.694 | 0.663 |
| F1 | 0.616 | 0.623 | **0.623** | **0.639** | 0.627 | 0.613 | 0.573 |

即 22 px 是这条曲线上的峰（F1 0.639，剔除低质量后 0.642），再往上就是"多报换召回"的下降段。**默认值目前
仍是 20 px**（要和已跑过的在线产物保持一致；改默认会让 `output/` 里的 `touches.jsonl` / `rendered.mp4`
变成旧参数下的结果）。`contact_ratio` 因为被 `min_contact_px` 夹住，扫出来几乎是平的——别用它调灵敏度。

`window_pad` 也可以扫，但它的曲线**单调**（放宽窗口只能把误检变成命中，反向不可能），所以它是"评估口径"
而不是检测参数，见 §4 的敏感性表。

## 2. 判定算法

1. **单帧接触度量**：对每个 `state=observed` 的球（优先跟踪球 `track_id>=1`，没有它时用最近的候选）与每个
   人的**脚部关键点**（project-26 的 15..22：左右踝 + 大趾/小趾/脚跟，共 8 个）算
   `d = min ||ball_center - foot_point||`。
   - `virtual`（平滑插值）与 `predicted`（球外推）**一律不作为证据**。
   - 脚部关键点还需 `confidence >= keypointConfidence`（默认 0.2，与 `FootballTracker::BallSupportScore` 同口径）。
2. **阈值**：`tau = min( clamp(min(W,H) * contactDistanceRatio, minContactPx, maxContactPx),
   clamp(personHeight * personHeightRatio, minContactPx, maxContactPx) )`；720p 下人体尺度项通常被
   `minContactPx` 盖住，调灵敏度先调它。
3. **滞回**：进入 `d <= tau`（contact）、`d <= tau*releaseRatio`（band，簇继续保持）、其余（none，
   容忍 `maxGapFrames` 帧后结束簇）。
4. **事件**：簇内取距离局部极小作为事件时刻；两个极小之间必须有 ≥ `minRiseFrames` 帧明显超出退出阈值、
   且相隔 ≥ `minSplitFrames`，才拆成两次触球（对应「同一秒 2 次」）；每个事件还要满足 `minContactFrames`
   帧接触，归属该帧最近脚点的人 `track_id` 与左/右脚。
5. **跨簇去重（不应期）**：同一人、同一脚在 `refractoryFrames`（默认 **21 帧 = 0.35 s**）内再次报出的
   触球视为同一次，直接丢弃后者。这一条是实测收益最大的去伪手段（见 §4）。
6. 三个可选门控（**默认关闭**，因为它们在本数据上无增益，见 §4）：球速突变门控
   `minSpeedChangePxPerSec / minSpeedRatio`、只信被跟踪球 `requireTrackedBall`。它们的拒绝计数会
   写进报告，避免"悄悄吃掉召回"。

## 3. 指标怎么读

标注是粗的：`.label` 的一行只说明「这一秒里有一次触球」，还可能写「2 次」。把一行当成一个点、只给一个
匹配名额，等于拿标注的精度去罚检测器。所以报告分四层：

**① 主口径：计数感知（窗口 + 名额）**

- 一条点标注 = 它代表的**窗口**（`.label` 的秒 = 1.0 s，0.01 s 写的 CSV 瞬时值 = 0.5 s，由文件头的
  `point_window_sec` 声明），窗口**居中**在该时刻上；
- 一条标注发放 `count` 个名额（`(2次)` 给 2 个）；
- 预测事件落在窗口内且名额未用完才算命中；`TP=分配数`、`FN=Σcount−TP`、`FP=预测数−TP`。

**② 误检分类**（只做归因，不参与上面的 P/R）：把每个未命中事件按位置归类——

| 分类 | 含义 |
| --- | --- |
| same touch, another frame | 与某个已命中事件同人同脚、相距 `duplicateSec`(0.5 s) 内：同一次触球的另一帧 |
| same moment, window full | **落在某条标注窗口内，但这个窗口的名额已经用完**：那行只声明了一次触球，我们报了两/三次——不是"看错时刻"，而是"同一时刻报了多次" |
| inside a no_touch span | 落在标注者明确写了「这段无触球」的区间里（**唯一被确认的误检**） |
| next to an annotation | 距最近窗口 `adjacencySec`(1.5 s) 内但没落进去：标注分辨率/人眼误差造成 |
| unannotated | 附近没有任何标注：真正没人支持的检测 |

前两类合起来就是「同一时刻报多了」；报告末尾还会打印每个例子的**离窗口多少秒**，所以"差 0.1 秒"和"差 1.4 秒"能直接区分。

**③ 辅助口径：一对一事件级**（历史可比）：`±0.25/0.5/1/2 s` 容差下的一对一贪心匹配，
以及按标注的「每秒次数」计数一致性。

**④ 漏检归因**：每条漏检落到「窗口内没球 / 球只有外推 / 没有可用脚点 / 球脚都近但被事件规则否掉 /
从没近到阈值」，并附窗口内最近距离与当时的阈值。

**⑤ 双口径：低质量行另算一份**

`low_quality_data=yes` 的行表示「这些帧太糊、人太小，标注者自己也判断不了」。报告同时给出两份分数：

- **全部标注**：按上面四层算；
- **剔除低质量**：把这些行删掉，并且把「只有这些行能解释」的预测一并从分子分母里去掉——它们既不算命中
  也不算误检，报告里给出被剔除的事件数，这样第二份分数不会因为偷偷缩小考题而虚高。

两份的差距就是「有多少分数压在不可判的帧上」。若文件没有任何行被标，报告会提示这一列的存在。

诊断区还会给：标注窗口内「有球/有可用球/有可用脚」的比例、标注时刻球-脚最近距离分位数 vs 全片背景分位数
（能不能分开，决定这项任务值不值得做）、球员框高度分布；低质量行的窗口也会单独统计（有球/有脚比例、
球员框高度、最近距离），用来回检"这一行标得对不对"。

## 4. 在 `data/移动传球.mov` 上的结果（16037 帧 / 267 s）

当前标注：**98 次触球**（其中 **30 条写成帧跨度的区间**，p50 0.167 s、最长 5 s；其余 68 条是点标注，
整秒 → 1 s 窗口）+ **3 段确认无触球** + **5 行标为低质量**。共报出 117 个触球事件。

| 口径 | TP | FP | FN | precision | recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| **主口径，全部标注**（容量 98） | 67 | 50 | 31 | **0.573** | **0.684** | **0.623** |
| **主口径，剔除 5 行低质量**（容量 93，随行剔除 2 个预测） | 65 | 50 | 28 | **0.565** | **0.699** | **0.625** |
| 辅助 ±0.5 s（全部 / 剔除） | 74 / 71 | 43 / 44 | 24 / 22 | 0.632 / 0.617 | 0.755 / 0.763 | 0.688 / 0.683 |
| 辅助 ±1.0 s（剔除） | 73 | 42 | 20 | 0.635 | 0.785 | 0.702 |
| 辅助 ±2.0 s（剔除） | 82 | 33 | 11 | 0.713 | 0.882 | 0.788 |

- 时间误差（主口径，全部标注）：signed mean **0.007 s**（无系统偏差）、|Δt| mean 0.148 s、
  p95 0.433 s、max 0.483 s → 检测的定位精度优于标注本身的精度。
- 计数一致性：98 条里 **57 条完全一致（58.2%）**；剔除低质量后 93 条里 55 条一致（59.1%）。
- 误检 50 个的归属：**同一时刻报多了 19**（同一次触球另一帧 8 + 落在标注窗口内但名额已满 11）、
  落在确认无触球区间 3、贴着标注窗口 26、**无人支持 2**（126.73 s frame 7604 / 131.62 s frame 7897，见下）。
- 门控拒掉：不应期 16 个（有效）、球速门控 0（关闭）、只信跟踪球 0（关闭）。
- 诊断：98 个标注窗口 **100 %** 都有可用球与可用脚；标注时刻球-脚最近距离 p50 **5.2 px**，
  全片背景 p50 **86.2 px**（可分，任务成立）。

**同一份运行的三种数法**（`tools/touch_view_report.py --output-dir <dir>` 一条命令打印）：

| 口径 | 数法 | 结果 |
| --- | --- | --- |
| 事件级（主口径） | 117 个预测事件各自计分 | P 0.573 / R 0.684 / **F1 0.623** |
| 过程级 | 同人同脚 1.0 s 内的预测先合并成一次触球 | 117 → 90 次过程，P 0.700 / R 0.684 / **F1 0.692**（0.5 s 合并：105 次，P 0.638 → F1 0.660） |
| 覆盖式召回 | 每条标注 ±T 内**至少有一个**预测就算判到（允许多对一） | ±0.25 s **58.2 %**、±0.5 s **83.7 %**、±1 s **89.8 %**、±2 s **98.0 %** |

读法：**"这一秒里有没有触球"基本都判到了（±1 s 90 %）**，掉分主要掉在"一次触球报了两三次、或者报的时刻差 0.2 秒"上——
50 个误检里 **19 个是"同一时刻报多了"**（同一次触球另一帧 8 + 落在窗口内但名额已满 11，后者到窗口的距离就是 0），
**26 个"贴着窗口"**（窗口外 0.07~1.48 s，其中 11 个只差 ≤0.2 s、15 个 ≤0.3 s，而你的 30 条区间标注跨度只有 0.05~5 s、p50 0.167 s），
剩下 3 个落在 `no_touch` 区间、2 个无人支持。
而 58.2 % 的「计数完全一致」是三者里最苛刻的定义：每一行**报出的数量必须和你写的数量完全相等**，多 1 个也不算。
引用数字时请写清是哪一个口径，别把 ±0.25 s 的 58.2 % 当成"识别率"。

**为什么比上一版（F1 0.700）低**：上一版 100 行里绝大多数是「整秒 + 1 s 窗口」，现在 30 条写成了
0.05~5 s 的帧跨度、其余写到精确时刻，窗口整体变窄：同样 117 个预测能落进去的更少、贴着窗口的更多。
**是标注变准，不是检测变差。** 把窗口放宽一点就能看到这条曲线（同一份 `observations.jsonl`，纯 CPU）：

| `--window_pad` | 全部标注 F1 | 剔除低质量 F1 | 无人支持的误检 |
| --- | --- | --- | --- |
| 0（默认，严格） | 0.623 | 0.625 | 2 |
| 0.15 s | 0.670 | 0.673 | 0 |
| 0.25 s | 0.670 | 0.676 | 0 |
| 0.5 s | 0.698 | 0.699 | 0 |

人眼抄时间的误差本身就在 0.15~0.25 s 量级，所以 0.15~0.25 那一档更接近「真实能力」；默认给 0 是因为放宽
会同时放过一部分漏检。**这是一个口径选择，不是代码问题**，报告里会打印实际用的 pad（`point window : ... pad ...`）。
这两条曲线（pad 的画成线、`min_contact_px` 的画成 PR 曲线）见
`output/touch_eval_moving_pass/pr_window_pad.png` 与 `pr_min_contact_px.png`，生成方式见 §1.3。

**低质量行的影响很小**：5 行里 2 行被预测命中（TP），剔除后 P 略降、R 略升、F1 基本不动（0.623 → 0.625）。
这 5 个窗口 5/5 都有可用球与可用脚、人框高度 p50 124.5 px —— 难在「球运动模糊 / 人偏小」，而不是「检测失效」。
换句话说：SDK 当前的误差主要不落在这批不可判的帧上。

**两个「无人支持」的预测值得复核**（报告 `failures` 段的 `unannotated`）：

| 时刻 | 帧 | 证据 | 判断 |
| --- | --- | --- | --- |
| 126.73 s | 7604 | 连续 11 帧接触、d 从 69 px 收敛到 **3.1 px**，随后一帧内球跳到 700 px 外，球速 103 → **4848 px/s** | 像一次真实踢球；标注在 123.43 与 128.30 之间正好是空的 |
| 131.62 s | 7897 | 6 帧接触、最低 **2.7 px**，球速 259 → 422 px/s，球置信度 0.28 偏低 | 弱一些，但也不像误判 |

对照图已生成（渲染视频的 ±6 帧拼图）：`output/touch_eval_moving_pass/unannotated_check/s126_73.png`
与 `s131_62.png`。**确认是触球就补两行 `touch`**，P/R 会立刻上升——这就是「复核 → 回写标注 → 重跑」的闭环。

**门控消融**（同一份 `observations.jsonl` 离线重放，主口径全部标注）：

| 配置 | TP | FP | FN | P | R | F1 | 重复帧 | 贴窗口 | 无人支持 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无门控（refractory 0） | 68 | 65 | 30 | 0.511 | 0.694 | 0.589 | 14 | 44 | 2 |
| 不应期 12 帧 (0.2 s) | 68 | 58 | 30 | 0.540 | 0.694 | 0.607 | 11 | 41 | 2 |
| **不应期 21 帧 (0.35 s)（默认）** | **67** | **50** | **31** | **0.573** | **0.684** | **0.623** | **8** | **37** | **2** |
| 不应期 31 帧 (0.5 s) | 65 | 43 | 33 | 0.602 | 0.663 | 0.631 | 0 | 38 | 2 |
| 不应期 21 + 只信跟踪球 | 63 | 49 | 35 | 0.562 | 0.643 | 0.600 | — | — | — |
| 不应期 21 + 球速门控 500 px/s | 25 | 26 | 73 | 0.490 | 0.255 | 0.336 | — | — | — |

**为什么球速门控默认关着**：被标注支持的触球其球速变化中位数只有一百多 px/s，而「贴窗口」那批真实触球是
几百 px/s —— 用球速变化做门控会把真触球一起删掉（F1 0.623 → 0.336）。结论是数据给的，所以门控保留代码、
默认关闭，并在报告里打印它的拒绝计数。

**为什么只信跟踪球默认关着**：命中里有一部分来自未被跟踪的候选球，砍掉它丢的召回比换来的精度更多
（F1 0.623 → 0.600）。

**阈值标定**（报告末尾的曲线，`contactDistanceRatio` 扫描，主口径/辅助口径并列）：`ratio 0.045`
（τ=32.4 px）主口径最优（F1 0.623）；τ 在 20~43 px 之间 F1 只差 0.03（0.590~0.623）。但真正的工作点是
`min_contact_px`（人体尺度项被它夹住），按 §1.3 的扫描它在这条视频上的峰值是 **22 px：F1 0.639**
（剔除低质量 0.642），比默认的 20 px 高 0.016，再往上开始掉（28 px 0.610、36 px 0.579）。默认暂不改，理由见
§1.3。换机位/场景要重跑这两条曲线——读现成的 `observations.jsonl` 重算，一条约 30 秒。

**人工复核闭环（上一轮）**：库里原本 4 个「无人支持」的误检，逐帧复核后写成 3 段 `no_touch` +
1 行 `touch`（266.83 s，原标注漏了），`unannotated` 4 → 2。剩下的 2 个就是上表那两帧。

**人工复核闭环（上一轮，已写进当前标注）**：

| 时间 | 复核结论 | 记录方式 |
| --- | --- | --- |
| 65.67 s | 球高速掠过、未触球 | `no_touch` 区间 65.27–66.07（frames 3916–3964） |
| 109.57 s | 小孩奔跑、球在一边 | `no_touch` 区间 109.17–109.97（6550–6598） |
| 264.15 s | 小孩奔跑、球在一边 | `no_touch` 区间 263.75–264.55（15825–15873） |
| 266.83 s | **确实触球，原标注漏了** | `touch 266.83`（frame 16010） |

> 65.67 s 那段的判定说明：复核为「球高速掠过未触球」，已写成负样本；若实际是擦到脚，删掉 CSV 里那条
> `no_touch` 行即可（`note` 里留了提示）。

## 5. 标注格式（v2：一触一行 CSV）

`data/移动传球.touches.csv`（**一行一次触球**，彻底消除"同一秒多次"的歧义）：

```csv
# point_window_sec: 1.00
index,type,t_start_sec,t_end_sec,count,foot,player,low_quality_data,note
1,touch,1.00,1.00,1,,,,
2,touch,34.20,35.10,1,right,,,keeper throws it back
3,no_touch,12.00,15.00,0,,,,keeper picks the ball up
4,touch,221.00,221.00,1,both,,yes,太模糊了人员
```

- `t_end == t_start` = 瞬时；不相等 = 这次触球过程持续了这段时间（沿用原 `.label` 的规则）。
- 时间精确到 0.01 s：**渲染视频顶栏烧录了 `frame=` 与 `time=`**，暂停抄一下即可，不需要帧级标注工具。
- `type=no_touch` 是**负样本区间**（"这段时间没有触球"），是唯一能便宜地测 precision 的东西。
- `low_quality_data`：这一行太糊/人太小、标注者也无法判断（写 `yes/true/1/低质量/模糊` 生效，
  留空或写 `no/0/ok` 不生效）。它不改变你写的时间，只让评估**多出一份「剔除这些行」的分数**，
  见 §3 的 ⑤；5 行以内的实测影响见 §4。写错的取值会在报告里报出来（不会静默当否）。
- 文件头的 `# point_window_sec:` 声明"点标注代表多宽的窗口"：手写 0.01 s 的时间用 0.5，从 `.label`
  转换来的（仍是整秒）用 1.0，`--point_window` 可覆盖。
- 列名可换（`start/end/count/foot/player/low_quality_data/note`、`开始/结束/次数/脚别/备注` 都认），
  无表头时按"老 8 列"顺序解析（`low_quality_data` 需要表头）。
- `index` 列**会被采纳**：报告里的 `gt#N` 就是你文件里的这一列，手改出重号时后面的行自动顺延并在
  该行的 `warnings` 里说明。写完想重排一遍：

  ```bash
  # 原地重写：序号 1..N（按文件行序，负样本行不跳号）+ 补齐规范列 + 同目录留 <file>.bak
  ./build/bin/touch_eval --touches_csv data/移动传球.touches.csv --renumber_touches
  ```
- **帧号可选，且优先级高于秒**（用来把一帧钉死、又不用重算时间）：

  ```csv
  # fps: 60
  index,type,t_start_sec,t_end_sec,t_start_frame,t_end_frame,count,foot,player,low_quality_data,note
  4,touch,266.83,266.83,16010,16010,1,right,,,confirmed by frame
  ```
  `frame`（单列）同时设首尾；**声明了 fps 才能用帧号**，否则该列会被"报告并忽略"而不是被瞎猜；
  `--fps 60` 可以覆盖/补上文件里没写的帧率。只写秒的行在有 fps 时也会自动补出帧号。
  ⚠️ 写成区间（首尾帧不同）的行，窗口就**只有这段跨度**，不再额外给 1 s——所以"只知道大概一秒"的行
  请写成整秒的点，别写成十几帧的区间，否则那一条会突然变严。

> ⚠️ `--dump_gt_csv` 会**覆盖**目标文件。别把它指向你手改过的 `data/移动传球.touches.csv`（要重新转换
> 请先备份，或换个输出路径）；要原地整理手改文件请用 `--renumber_touches`，它会先留 `.bak`。

**旧 `.label` 继续支持**，两者归一到同一个真值结构。把 `.label` 转成 v2：

```bash
./build/bin/touch_eval --label data/移动传球.label --dump_gt_csv data/移动传球.touches.csv
# 纯转换模式（不需要 observations）：把 "(2次)" 展开成 2 行并标注 SPLIT ME，人工只需修这些行
```

解析容错（不静默改写，非标准行在 `warnings` 留痕）：全角/半角括号、不规则空格、`mm:ss`、区间、
`(N次)`、`左右脚各1次`，以及历史笔误 `0.39 → 0:39`、`0.01S → 0:01`。`--dump_gt` 的 JSON 里保留
`raw_line` 与 `source_line`，任何可疑解析都能回查。

> `data/` 在 `.gitignore` 里：`.label` 与 `.touches.csv` 都是本地数据，不进版本库。评估结果与
> `observations.jsonl` 在 `output/`（同样被忽略）。

## 6. 在线使用

```bash
./build/bin/aifootball_app <config> --detect_touches [--touch_output <path>] \
    [--contact_ratio r] [--min_contact_px px] [--person_height_ratio r] \
    [--kp_conf c] [--ball_conf c] [--release_ratio r] \
    [--min_contact_frames n] [--max_gap_frames n] [--min_split_frames n] [--min_rise_frames n] \
    [--refractory_frames n] [--min_speed_change px_s] [--min_speed_ratio r] [--tracked_ball_only 0|1]
```

- `touches.jsonl`：首行 metadata（视频、fps、**全部生效参数**），之后每行一个事件。
- `--render`：接触帧画球-脚连线 + 黄圈 + `touch p<id> <left|right> d=<距离>/<阈值>`（带底板保证可读）；
  事件被接受后徽标面板保留 40 帧。
- 离线 CLI 的阈值/门控开关与在线同名同义，标定出来的值可以直接搬到线上。

## 7. 已知限制

- **标注的分辨率仍是主要制约**：现在 30 条是 0.05~5 s 的帧跨度、其余是整秒时刻，两边窗口宽度不一致
  （跨度小的那批只有 0.05~0.3 s）。报告里的"贴窗口"误检（37 个）就是它的量级，`--window_pad` 那组
  数字说明放宽一点它们就能变成命中。想要更强的指标，只能把时刻抄得更细（帧号已经有了，抄一帧的成本
  和抄一秒一样）。
- `no_touch` 区间只有 3 段，precision 的分母仍然很薄；「无人支持」的判定依赖 `adjacencySec`(1.5 s) 这个
  人为半径。
- 主口径的容量来自你写的行数：少写一行 = recall 的分子分母同时少一块，**它无法发现"标注漏了"**。
  122 秒附近那两个预测就是例子（§4），只能靠人复核。
- 脚别核对（`foot hint`）只有 4 条标注带，样本太少，报告里只作占位。
- 只有一条视频的标注；阈值、不应期都是从它标定出来的，换场景请重跑曲线与消融。
- 只判「脚触球」（踝/趾/跟）；头/手/胸触球未纳入。
- 判定依赖「球被检出且脚部关键点可见」；报告的漏检归因会说明卡在哪一环。
