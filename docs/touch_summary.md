# 触球判定与评估：汇报汇总

> 一页看结论，后面是明细与出处。算法机制的逐步说明在
> [`docs/touch_detection_logic.md`](touch_detection_logic.md)；评估口径与标注格式在
> [`examples/touch_eval/README.md`](../examples/touch_eval/README.md)。
> 本文所有数字来自产物 `output/touch_eval_moving_pass-update2/`（fp16 整段 16037 帧）；
> 该目录里另有一份**自动生成**的同口径快照 `metrics.md` / `metrics.csv`
> （`uv run --project examples/touch_eval/py python3 examples/touch_eval/tools/write_metrics_summary.py
> --output-dir output/touch_eval_moving_pass-update2`），标注改动后重跑即可同步。

## 1. 关键数字

| 口径 | 容量（名额数） | TP | FP | FN | precision | recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 含 `low_quality_data`（对外） | 97 | 79 | 18 | 18 | 0.8144 | 0.8144 | **0.8144** |
| 不含 `low_quality_data`（第二视图） | 86 | 76 | 16 | 10 | 0.8261 | 0.8837 | **0.8539** |

- 口径是**名额级**：一行标注写 `count` 次就有 `count` 个名额；含低质量的口径里 P = R 只是巧合（名额数 97 = 预测事件数 97，两个分母相同）。
- 复核包：`output/touch_eval_moving_pass-update2/pr_review/`（122 张图 + 播放视频 + 时间线清单）。

## 2. 管线与分工

```
视频 ──▶ C++：基模（目标检测 + 姿态估计）──▶ observations.jsonl（每帧球/人/关键点）
                                              │
                        Python：判定（簇 → 事件）──▶ touch_events.jsonl
                                              │
                        Python：标注解析 + 评估 + 报告 + 复核图/视频 ──▶ touch_report.json 等
```

- C++ 侧**只做感知**，不再含任何触球参数；判定、评估、渲染全在 Python 包 `touch_eval`（uv 工程，唯一 Python 环境）。
- 判定层是逐帧有状态状态机，在线与离线走同一条代码路径。

## 3. 判定逻辑（六步）——大白话

1. **每帧找"离球最近的合法脚点"**：球要有置信度、脚点不能是插值补的；若球在被跟踪，则优先用它。
2. **"够得着"的线不是固定的**：`τ = min(短边 × 0.045, 人高 × 0.12)`，夹在 22–64 px——近处宽松、远处严格，用同一把尺子量所有人。
3. **进门容易、出门难**：距离 ≤ τ 算接触；要退到 `1.4 × τ` 外才算松手（防抖）。
4. **连成一小段才算数**：接触帧 ≥ 2 帧（单帧噪点丢掉）；中间最多容忍 3 帧完全无证据，迟滞带帧不算断。
5. **一段里可能藏着两次触球**：看"球离脚多远"曲线的谷底；两个谷底隔不到 0.4 s、或中间球没抬起来 → 合成一次；明显抬起来 → 算两次。事件时刻取谷底那一帧。
6. **两道橡皮擦**：同脚 0.5 s 内、同人（换脚也算）0.5 s 内的事件被抑制——压重复上报。

精确规则（含代码行号）见 `docs/touch_detection_logic.md` §1–§3。

## 4. 参数（21 个，全部可命令行覆盖，也会写进报告）

| 组 | 旗标（当前值）| 作用 |
| --- | --- | --- |
| 证据 | `--ball_min_confidence 0.15`、`--keypoint_confidence 0.2` | 球/脚点可用门槛（插值脚点永不可用）|
| 证据 | `--candidate_count 6`、`--prefer_tracked_ball` | 每帧留几个候选；优先被跟踪的球 |
| 阈值 | `--contact_distance_ratio 0.045`、`--person_height_ratio 0.12` | 帧项 / 人项，取小者为 τ |
| 阈值 | `--min_contact_px 22`、`--max_contact_px 64`、`--release_ratio 1.4` | 上下夹；迟滞释放倍数 |
| 聚簇 | `--min_contact_frames 2`、`--max_gap_frames 3`、`--max_cluster_frames 300` | 成事件下限；空档容忍；簇长保险丝 |
| 分裂 | `--min_split_frames 24`、`--min_rise_frames 2` | 两次触球要隔多远、要抬升几帧 |
| 门控 | `--require_tracked_ball`（关）、`--min_speed_change 0`、`--min_speed_ratio 1.0`（都关）| 预留：只认跟踪球 / 球速突变 |
| 时间 | `--speed_window 0.15`（唯一 double）、`--event_time_mode 0` | 球速窗口；事件时刻取哪帧 |
| 去重 | `--refractory_frames 30`、`--person_refractory_frames 30` | 同脚 / 同人不应期（帧）|

## 5. 实测明细（16037 帧）

### 5.1 漏斗

| 环节 | 数量 |
| --- | --- |
| 帧 | 16037 |
| 簇（候选接触片段） | 161 |
| 丢弃：接触帧 < 2 | −43（25 个整簇零接触、18 个只有 1 帧）|
| 剩下的簇 | 118 |
| 候选事件（6 个簇各出 2 个） | 124 |
| 抑制：同脚不应期 | −17 |
| 抑制：同人不应期 | −10 |
| 门控拒绝 | −0（球门控、速度门控都关着）|
| **最终事件** | **97** |

### 5.2 簇的形状

| 统计量 | 最小 / 中位 / 最大 |
| --- | --- |
| 簇长（帧） | 4 / 14（≈0.23 s）/ 234（≈3.9 s）|
| 簇内接触帧数 | 0 / 6 / 199 |
| 一个簇出 2 个事件的簇 | 6 个（"一次接近里其实是两次触球"）|

最典型的被丢弃形状是 4 帧（1 帧接触 + 3 帧空档）：23 个。

### 5.3 事件的接触帧数

最小 2 / 中位 13 / 均值 18.9 / 最大 199；**恰好压线 2 帧的有 10 个**（约 10%）。

### 5.4 为什么"至少 2 帧"（同一份数据的消融）

| 设置 | 事件数 | TP / FP / FN | P | R | F1 |
| --- | --- | --- | --- | --- | --- |
| 2 帧（现状） | 97 | 79 / 18 / 18 | **0.8144** | 0.8144 | **0.8144** |
| 放宽到 1 帧 | 107 | 80 / 27 / 17 | 0.7477 | 0.8247 | 0.7843 |

放宽到 1 帧：多 10 个事件 = 多 9 个误检、只换回 1 个漏检；F1 掉 3 个点。

### 5.5 阈值 τ 的分布（事件级）

| τ (px) | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 事件数 | 41 | 1 | 1 | 2 | 2 | 5 | 3 | 3 | 3 | 5 | 31 |

两个台阶分别对应"人项/下限压住（22）"与"帧项（720 × 0.045 ≈ 32.4）"。

### 5.6 计数表与误差构成

| 项 | 值 |
| --- | --- |
| 行数 | 72（97 次触球）|
| 计数完全正确 | 54 行 |
| 计数偏多 / 偏少 | 3 / 15 |
| 平均计数误差 | 0.306 |

| FP 18 条 | 条数 | | FN 18 个名额 | 数量 |
| --- | --- | --- | --- | --- |
| `adjacent`（贴着标注窗口）| 9 | | 整行未命中：13 行 | 14 个名额 |
| `in_window_no_slot`（窗口内名额已满）| 4 | | ↳ 其中"球脚贴进了阈值但没成事件" | 12 行 |
| `inside_no_touch`（落在确认未触球区间）| 4 | | ↳ 其中"球从未近到阈值" | 1 行 |
| `unannotated`（附近无标注）| 1 | | 部分命中：2 行 | 4 个名额 |
| `duplicates` | 0 | | | |

### 5.7 低质量标注（10 行，单独一套图）

标注者自己标了 `low_quality_data=yes`，图上带原始备注：

| 备注类型 | 行数 | 行号 |
| --- | --- | --- |
| 脚踩球 | 3 | 13、14、21 |
| 人球太小了 | 2 | 23、27 |
| 球太小了 | 2 | 31、32 |
| 脚和球平行 | 1 | 9 |
| 太模糊了人员 | 1 | 66 |
| 不像踢球 | 1 | 67 |

其中 **8 行同时也算漏检**（13、14、21、23、31、32、66、67），2 行是命中/名额已满（9、27）。它们从主口径里**不剔除**，只影响第二视图（容量 97 → 86）。

## 6. 复核产物清单

```
output/touch_eval_moving_pass-update2/
├── observations.jsonl          基模逐帧输出（fp16 那一遍）
├── touch_report.json           评估报告（全部口径与计数）
├── touch_events.jsonl          97 个事件（含 gt_index / verdict / matched）
├── contact_signal.csv          逐帧接触信号
├── rendered.mp4                基模渲染视频（复核视频的底片）
├── timeline.csv                每个标注一行的时间线核对表
├── tp_fp_fn_detail.csv         逐条 TP/FP/FN 明细
└── pr_review/
    ├── tp/            79 张   命中事件
    ├── fp/            18 张   误检（文件名带原因）
    ├── fn/            15 张   漏检（含"缺名额"行）
    ├── low_quality/   10 张   低质量行（落在球脚最近的那一刻，带标注备注）
    ├── timeline.md / timeline.csv   按时间排的汇总清单（含低质量一节）
    ├── low_quality.csv              低质量行清单
    └── touch_playback.mp4           整段实时播放：底部 pred/gt 两条轨道 + 顶部原因
                                     16037 帧 / 60 fps / 1280×720 / 428 MB（本机无 ffmpeg，用 mp4v）
```

图的画法：每个条目切 6 帧拼成一张接触片（以事发帧为中心 ±0.2 s），顶部烧上"是什么 / 为什么 / 证据 / 怎么修"，中文字体 wqy-microhei。

## 7. 可信度（为什么这些数可以信）

| 项 | 做法 | 结果 |
| --- | --- | --- |
| 与 C++ 参考一致 | 判定层与评估层的差分测试（逐事件 13 个字段、两个不应期与两个门控的计数、报告全字段）| 全部一致（`uv run pytest -q -m parity`，22/22 绿）|
| 观测量可复现 | 同一条 fp16 管线跑两遍 | 除计时字段外只有 37/16038 帧末位小数有差，检出结果一致、指标完全相同 |
| 复核产物自洽 | 出图前对拍：TP/FP/FN 数 = 报告主口径、FP 分类 = 报告 verdict、FN 原因分组 = 报告 failures；播放视频的标记数也核对报告 | 全部通过；本次另用插桩重放独立复算得 97 事件 / 抑制 17、10，与报告一致 |

## 8. 数字出处（哪个数字去哪里看）

| 数字 | 出处 |
| --- | --- |
| P/R/F1、容量、TP/FP/FN | `touch_report.json` → `primary_count_aware` / `low_quality.primary_count_aware` |
| 不应期与门控计数、窗口统计 | `touch_report.json` → `primary_count_aware`、`diagnostics` |
| 计数表（54/72 等） | `touch_report.json` → `counts` |
| 簇统计（161 / 43 丢 / 6 处拆分） | 插桩重放 `TouchEventAggregator`（见 §7 第三行），与报告的事件数交叉验证 |
| 接触帧分布、minContactFrames 消融 | `touch_report.json` → `events[].contact_frames`；消融为 `--min_contact_frames 1` 另跑一次 |
| 逐条明细与图 | `pr_review/timeline.md`、`tp_fp_fn_detail.csv`、`pr_review/{tp,fp,fn,low_quality}/` |

复现整条链路：

```bash
cd examples/touch_eval/py
uv run touch-eval --observations ../../../output/touch_eval_moving_pass-update2/observations.jsonl \
                  --touches_csv ../../../data/移动传球.touches.csv \
                  --output_dir ../../../output/touch_eval_moving_pass-update2
uv run --project . --extra tools python3 ../tools/render_review_folders.py \
       --output-dir ../../../output/touch_eval_moving_pass-update2 --playback
```

## 9. 结论与下一步

- **结论**：判定与评估链路已经收敛，对外口径 **P/R/F1 = 81.4%**（含低质量数据），剔除标注者自认不可判读的 10 行后是 **82.6% / 88.4% / 85.4%**；每一步都有图可复核、有原因和修改建议。
- **剩余误差的构成很清楚**：漏检的 14 个名额里 12 个是"球脚确实贴进了阈值、但没形成事件"（多数是接触太短或被合并），误检的 18 条里 9 条是"贴着标注窗口"（多是整秒粒度带来的时刻差）。
- **下一步可选**：① 逐条过 `pr_review/`（尤其 `low_quality/` 那 10 行：能判读就改写、判不了就保留标记）；② 对"贴上了却没成事件"的一批做小范围参数扫描（`minContactFrames` / 两个不应期），每次全量评估约 2 秒，可分钟级迭代；③ 若要更稳的工作点，把阈值之类从"单视频拟合"改成带区间的标定（报告里已有标定曲线与容差扫描）。
