# 触球判定逻辑（TouchDetector）

> 汇报用的汇总（关键数字、参数表、实测明细、产物清单）见
> [`docs/touch_summary.md`](touch_summary.md)。

本文讲**判定层**：给定逐帧观测，怎么产出一个"触球事件"。代码在
`examples/touch_eval/py/src/touch_eval/detect.py`（原 C++ `TouchDetector.cpp` 的逐行移植，
行为已用差分测试钉住）。

判定层是**逐帧喂入的有状态状态机**：在线（app 的实时处理）和离线评估走的是同一套代码路径，
所以"实时出的事件"和"报告里的事件"是同一件事。

评估层（命中 / 误检 / 漏检怎么算、门槛与名额）不在这里，见 `examples/touch_eval/README.md`。

---

## 0. 在管线里的位置

```
视频 ──▶ C++ 基模（目标检测 + 姿态估计）
          │  逐帧写 observations.jsonl：
          │    · 球框 + 置信度 + 是否被跟踪
          │    · 人框 + track_id + 身高（像素）
          │    · 每人的姿态关键点（含 valid / 是否插值 / 置信度）
          ▼
      判定层 detect.py（本文）──▶ touch_events.jsonl（触球事件）
          ▼
      评估层（窗口 + 名额 + 最优分配）──▶ touch_report.json
```

判定层只回答"这一帧到这一帧之间有没有触球"；**命中/误检/漏检是评估层的事**，汇报时建议分开讲。

---

## 1. 帧内：候选 → 阈值 → 电平

### 1.1 候选：每帧每只脚只留"最近的那个点"

对每一帧：

1. 只考虑**可用球**：`observed == true` 且 `confidence ≥ ballMinConfidence`（0.15）。
2. 对每颗可用球 × 每个人 × 左/右脚，在该侧的脚部关键点里挑**最近的合法点**——
   必须 `valid`（插值补出来的点**不算**）且 `confidence ≥ keypointConfidence`（0.2）。
3. 这样得到"一球一人一脚一侧"的一个候选；按距离排序，只留前 `candidateCount`（6）个。
4. 若存在**被跟踪的球**（`track_id > 0`）且 `preferTrackedBall` 打开，则优先用那颗球的候选：
   管线跟的就是这颗球，别的球凑到脚边不能算它的触球。

实现：`collect_contact_candidates()` / `select_contact()`（`detect.py` 146–256）。

### 1.2 阈值随"人高"缩放，而不是固定像素

```
τ = min( 帧项, 人项 )
帧项 = clamp(画面短边 × contactDistanceRatio, minContactPx, maxContactPx)
人项 = clamp(人高(px)  × personHeightRatio,  minContactPx, maxContactPx)
```

- 当前 `contactDistanceRatio = 0.045`、`personHeightRatio = 0.12`、夹在 `[22, 64] px`。
- 720p 下帧项 = `720 × 0.045 ≈ 32.4 px`；**远景的人脚小，人项更小，所以球必须更近才算**——
  同一场比赛里远景和近景因此用同一套物理标准。
- 实现：`contact_threshold_px()`（`detect.py` 128–143）。

### 1.3 三级电平（迟滞）

每帧最终只留一个"主候选"，按下式定级：

| 电平 | 条件 | 含义 |
| --- | --- | --- |
| `CONTACT` | `distance ≤ τ` | 接触帧 |
| `BAND` | `τ < distance ≤ τ × releaseRatio`（1.4）| 迟滞带：不算接触，但也不算"离开" |
| `NONE` | 更远 | 没有可用证据 |

进入用 `τ`、退出用 `τ × 1.4`——**进/出两个阈值**，避免距离在阈值附近抖动时把一次触球切成好几段。
实现：`select_contact()`（`detect.py` 220–256）。

---

## 2. 跨帧：状态机 → "簇" → 事件

### 2.1 开簇 / 关簇 / 间隙容忍（`TouchEventAggregator.push()`，`detect.py` 332–357）

- 出现 `BAND` 或 `CONTACT` 就**开簇**。
- 簇里连续 `NONE` 帧数超过 `maxGapFrames`（3 帧 ≈ 0.05 s）才关簇——短暂丢球、被遮挡不会切断
  一次触球。注意 **`BAND` 帧不消耗这 3 帧预算**：球一直贴在脚边，簇就一直活着。
- 簇长保险丝 `maxClusterFrames = 300` 帧（5 s），超了强制关簇。

### 2.2 从簇里"劈"出事件（`close_cluster()`，`detect.py` 376–508）

关簇时按顺序做四件事：

1. **接触帧数 < `minContactFrames`（2）→ 整簇丢弃**：单帧噪点不成事件。
2. 在**接触帧的距离序列**里取**局部极小**（距离比左右两侧都近的帧）作为触球点候选。
3. **合并太近的极小**——满足任一条就合并，并保留更深（距离更小）的那个：
   - 两个极小间隔 < `minSplitFrames`（24 帧 = 0.4 s）；或
   - 两者之间**没有明显抬升**：抬升需要 ≥ `minRiseFrames`（2）帧满足"该帧没有可用球"或
     "距离 > 释放阈值 `τ × releaseRatio`"。
   这一步是"两次触球挨太近被并成一个"的边界。
4. **切分覆盖范围**：相邻两个保留的极小之间，用距离**最远的峰**（`_peak_index`）切开，
   保证前一次触球的接触帧不会漏进后一次；切出来的段还要再满足"接触帧 ≥ 2"。

### 2.3 事件时刻与事件字段

- 事件时刻由 `eventTimeMode` 决定：`0` = 极小那一帧（球离脚最近，**默认**）、`1` = 接触首帧、
  `2` = 接触末帧、`3` = 首末中点。接触起点帧另存 `first_frame_index / first_timestamp_sec`。
- 事件里带：帧号、时刻、人 `track_id`、哪只脚、最近距离 `distance_px`、当时的阈值 `threshold_px`、
  接触帧数 `contact_frames`、球置信度、是否跟踪球，以及触前/触后**球速**。

球速口径：在事件时刻前/后各 `speedWindowSec`（0.15 s）窗口内，用球位移求速度；球样本历史保留
`4 × speedWindowSec`（0.6 s）。实现：`_remember_ball()` / `_ball_speed()`（`detect.py` 288–330）。

---

## 3. 事件级：门控与不应期

### 3.1 两道门（`close_cluster()` 尾部，`detect.py` 501–526）

| 门 | 参数 | 当前 | 含义 |
| --- | --- | --- | --- |
| 球门控 | `requireTrackedBall` | **关** | 打开后只接受"被跟踪的球"的事件 |
| 速度门控 | `minSpeedChangePxPerSec` / `minSpeedRatio` | **0 / 1.0（都关）** | 要求触前触后球速"突变"（差值或比值超阈值），用于滤掉推球/带球 |

被拒的事件计入 `rejected_by_ball_gate` / `rejected_by_speed_gate`，写进报告。

### 3.2 两条不应期（`TouchDetector.apply_refractory()`，`detect.py` 555–604）

按**帧距**、**同一个人**比较，同脚优先：

- 同脚 `refractoryFrames`（30 帧 = 0.5 s）内 → 抑制，计入 `suppressed_by_refractory`；
- 同一个人（任意脚）`personRefractoryFrames`（30 帧）内 → 抑制，计入 `suppressed_by_person_refractory`；
- 两条都设成 0 就不起作用；历史只保留 `max(两条)` 帧。

作用是压掉"同一次接触被多帧重复上报"，同时它也定义了**连击的下限时间 ≈ 0.5 s**。

---

## 4. 参数表（当前值 · 作用 · 怎么调）

| 参数 | 当前值 | 作用 | 调大 / 调小的效果 |
| --- | --- | --- | --- |
| `ballMinConfidence` | 0.15 | 球的可用门槛 | 调低→更多证据也更噪 |
| `keypointConfidence` | 0.2 | 脚点可用门槛（插值点永不可用）| 同上 |
| `contactDistanceRatio` | 0.045 | 帧项阈值 = 短边 × 此值 | 调大→召回↑、误检↑ |
| `personHeightRatio` | 0.12 | 人项阈值 = 人高 × 此值 | 同上，主要影响远景 |
| `minContactPx` / `maxContactPx` | 22 / 64 | 阈值上下夹 | 决定最紧/最松 |
| `releaseRatio` | 1.4 | 迟滞释放倍数 | 调大→更难"松手"，一次触球更不易被切碎 |
| `minContactFrames` | 2 | 成事件所需接触帧数 | 调大→更稳、更易漏 |
| `maxGapFrames` | 3 | 容忍多少帧无证据 | 调大→遮挡时仍连续 |
| `minSplitFrames` | 24 | 两个极小至少隔多少帧才算两次 | 调大→更倾向合并 |
| `minRiseFrames` | 2 | 两次之间要抬升多少帧 | 调大→更倾向合并 |
| `refractoryFrames` | 30 | 同脚不应期（帧）| 调大→压重复上报、也可能吃掉真连击 |
| `personRefractoryFrames` | 30 | 同人不应期（帧）| 同上 |
| `candidateCount` | 6 | 每帧留几个候选 | — |
| `preferTrackedBall` | 开 | 优先被跟踪的球 | 关掉后按最近距离选 |
| `eventTimeMode` | 0 | 事件时刻取哪帧 | 0=最近帧 1=首帧 2=末帧 3=中点 |
| `speedWindowSec` | 0.15 | 球速窗口（秒）| — |
| `minSpeedChangePxPerSec` / `minSpeedRatio` | 0 / 1.0 | 速度门控 | 开启可滤推球 |
| `requireTrackedBall` | 关 | 只认被跟踪的球 | 开→更严 |
| `maxClusterFrames` | 300 | 簇长保险丝（帧）| — |

**按症状调参**：漏检多（球脚其实贴上了）→ 降 `minContactFrames`、升 `maxGapFrames`、降不应期；
一次触球被拆成两个 → 升 `refractoryFrames` / `personRefractoryFrames`、升 `minSplitFrames` /
`minRiseFrames` / `releaseRatio`；两次触球被并成一个 → 反向调这三个；远景漏、近景误 → 调
`personHeightRatio` / `minContactPx` / `maxContactPx`；误检多（球没真贴上）→ 收紧
`contactDistanceRatio` / `personHeightRatio`，或打开 `requireTrackedBall`。

---

## 5. 实测漏斗（moving_pass，fp16）

整段 **16037 帧 → 97 个事件**：

| 环节 | 计数 |
| --- | --- |
| 同脚不应期抑制 | 17 |
| 同人不应期抑制 | 10 |
| 球门控拒绝 | 0（该门关着）|
| 速度门控拒绝 | 0（该门关着）|

事件级阈值 τ 的分布：22 px × 41、32 px × 31、23–31 px 共 25 个 —— 正好落在
"人项/下限（22）"和"帧项（32.4）"两个台阶上。

同一份产物上的评估结果（名额级主口径，锚点 `end`，点行窗口 1.0 s）：

| 口径 | 容量 | TP | FP | FN | precision | recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 含 `low_quality_data` | 97 | 79 | 18 | 18 | 0.8144 | 0.8144 | 0.8144 |
| 不含 `low_quality_data` | 86 | 76 | 16 | 10 | 0.8261 | 0.8837 | 0.8539 |

---

## 6. 设计要点（三个权衡）

1. **"一次触球别被切碎" vs "两次触球别被并成一个"**：前者靠迟滞（`releaseRatio`）与间隙容忍
   （`maxGapFrames`），后者靠极小值 + 抬升要求（`minSplitFrames`、`minRiseFrames`、`releaseRatio`）。
   这三四个旋钮是一对反向的力，任何调参都是在这两端之间换位置。
2. **阈值随人高缩放**（而非固定像素）：远景与近景用同一物理标准；`minContactPx` / `maxContactPx`
   夹住极端尺度，避免近景过松、远景过严。
3. **两条不应期**专门压"同一次接触的重复上报"（本次实测压掉 27 个候选事件），顺带定义了连击下限
   ≈ 0.5 s。

---

## 7. 复现与自检

```bash
# 判定 + 评估（观测来自基模那一遍推理）
cd examples/touch_eval/py
uv run touch-eval --observations ../../../output/touch_eval_moving_pass-update2/observations.jsonl \
                  --touches_csv ../../../data/移动传球.touches.csv \
                  --output_dir ../../../output/touch_eval_moving_pass-update2
```

- 事件落 `touch_events.jsonl`，阈值/时间等参数串与全部计数落 `touch_report.json`
  （`primary_count_aware.suppressed_by_refractory` 等字段）。
- **与 C++ 的差分测试**：`uv run pytest -q -m parity` —— 逐事件 13 个字段、两个不应期与两个门控的
  计数、评估报告全字段都对拍冻结的 C++ 参考产物。
- 逐条复核图（TP/FP/FN 分三个目录 + 时间线汇总表）：
  `uv run --project examples/touch_eval/py --extra tools python3 examples/touch_eval/tools/render_review_folders.py --output-dir <产物目录>`。

---

## 附：移植时必须照搬的精度细节

- 除 `speedWindowSec` 外，几何参数在 C++ 里都是 `float`（Python 侧用 `numpy.float32`）；
  `speedWindowSec` 是 `double`——窄化成 float 会让球速窗口的远边移动 6e-9 s，足以多算/少算一个
  球样本，从而改变 `speed_after`。
- 阈值函数里的 `clamp` 也用 float32 做，`τ` 的最后一位会影响"贴着阈值的帧算不算接触"。
