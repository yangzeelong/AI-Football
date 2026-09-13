# AI-Football 960 分辨率性能对比报告

日期：2026-09-13

## 结论

在同一段 `1920x1080 / 60 FPS` 视频上，当前 TensorRT 工程化程序在启用
ROI、关闭渲染时完成 7200 帧耗时 `259.777 s`，有效处理速度约 `27.716 FPS`。
Python 版本在启用 ROI、关闭渲染时完成 7201 帧耗时 `557.134 s`，速度约
`12.925 FPS`。按 wall-clock 计算，当前 TensorRT 结果约为 Python 的
`2.145x`，节省 `297.357 s`，吞吐提升约 `114.5%`。

本报告中的 TensorRT ROI 结果来自当前 SDK/demo 版本的完整视频运行；Python
结果来自 AI-Football 的 `run_detector_resolution_compare.py` 流程。两者的
ROI 过滤时机、tracking 细节和输出人数仍有实现差异，因此该结果用于工程
吞吐对比，不作为严格精度等价结论。

## 测试条件

| 项目 | 配置 |
| --- | --- |
| 输入视频 | `data/射门1-1080p60.mov` |
| 原始视频 | 1920x1080，60 FPS |
| RF-DETR | small，960x960，TensorRT，dynamic batch 4 |
| Pose | HRNet-W48-DARK，288x384，TensorRT |
| GPU | NVIDIA RTX 4070 SUPER |
| 渲染 | 性能主结果均关闭；渲染仅作为单独开销对比 |
| ROI | 四边形，源坐标 1920x1080；点为 `[1579,1067]`、`[56,733]`、`[950,416]`、`[1796,493]` |

## 主结果

| 实现 | ROI | 渲染 | 帧数 | 总耗时(s) | 有效 FPS | 单帧耗时(ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Python | 是 | 否 | 7201 | 557.134 | 12.925 | 77.37 |
| TensorRT/C++ | 否 | 否 | 7200 | 544.655 | 13.219 | 75.647 |
| TensorRT/C++ | 是 | 否 | 7200 | 259.777 | 27.716 | 36.08 |

Python 无渲染 ROI 数据来源：
`/home/hx1/yzl/Work/AI-Football/reports/performance_960_stride1_no_render_20260913.md`

Python 旧的带渲染数据来源：
`/home/hx1/yzl/Work/AI-Football/reports/performance_960_stride1.md`

TensorRT 结果目录：

- 无 ROI：`output/sdk_no_render_full`
- 有 ROI：`output/sdk_roi_full`

## ROI 影响

在当前 TensorRT/C++ 流程中，加入 ROI 后：

| 指标 | 无 ROI | 有 ROI | 变化 |
| --- | ---: | ---: | ---: |
| 总耗时 | 544.655 s | 259.777 s | 节省 284.878 s |
| 有效 FPS | 13.219 | 27.716 | 2.097x |
| 单帧耗时 | 75.647 ms | 36.08 ms | 降低约 52.3% |

主要原因是 ROI 减少了进入 PoseEstimator 的人员数量，降低了 HRNet
crop、batch 和 TensorRT 推理负载。当前 ROI 主要作用在 PersonTracker 的
人员结果上，Detector 仍然处理整帧图像。

## 渲染开销

已有运行数据表明，渲染会显著增加端到端耗时：

| 实现 | 渲染 | 总耗时(s) | 有效 FPS |
| --- | ---: | ---: | ---: |
| Python | 是 | 681.358 | 10.569 |
| Python | 否 | 557.134 | 12.925 |
| TensorRT/C++ | 是 | 591.190 | 12.179 |
| TensorRT/C++ | 否，ROI | 259.777 | 27.716 |

Python 渲染额外耗时约 `124.224 s`。TensorRT/C++ 的带渲染运行对应的是
此前无 ROI 的输出链路，不能与有 ROI 的主结果直接相减；它仍然说明渲染、
颜色转换和编码不应放入推理吞吐基准。

## TensorRT 阶段计时

当前 ROI 完整运行中，TimerRegistry 的代表性统计如下。Detector 的时间
是 batch 级平均；item 时间是按 work units 折算后的平均时间。

| 阶段 | 平均 batch(ms) | 平均 item(ms) | batch QPS |
| --- | ---: | ---: | ---: |
| Detector.Preprocess | 约 1.35 | 约 0.34 | 约 740 |
| Detector.TensorRT | 75.579 | 18.895 | 13.231 |
| Detector.Batch | 79.990 | 19.997 | 12.502 |
| PoseEstimator.Preprocess | 约 5.2 | 约 1.86 | 约 192 |
| PoseEstimator.TensorRT | 约 31.1 | 约 11.1 | 约 32 |
| PoseEstimator.Batch | 约 37.08 | 约 13.26 | 约 27 |

Detector GPU 预处理已经从此前约 `60.607 ms/batch` 降到约 `1.35 ms/batch`；
当前瓶颈主要在 TensorRT 执行、HRNet 推理和端到端 actor 调度，而不是
Detector 的 resize/normalize。

## 输出与差异

- Python ROI 运行平均人员数约 `3.498/frame`，球出现比例约 `0.886`。
- TensorRT/C++ ROI 完整运行平均输出人员数约 `2.612/frame`；短烟测约
  `3.395/frame`。这说明当前 C++ tracker/ROI 结果与 Python 尚未完全对齐。
- TensorRT/C++ 无 ROI 平均人员数约 `10.271/frame`，原始人员检测平均约
  `14.081/frame`。
- Python GPU 采样平均利用率约 `70.3%`，峰值 `100%`，平均显存约
  `2559 MiB`，峰值约 `5255 MiB`。
- Python 使用 7201 帧，C++ 使用 7200 帧，差异为 1 帧，不影响吞吐结论，
  但后续严格对比应统一帧数和 warm-up 规则。

## SDK 重构后的验证

SDK 现在只提供算法运行时：

```text
decoded RGB24 frame -> AIFootballPipeline::Process -> PollResult/callback -> ProcessResult
```

`VideoReader`、`VideoDecoder`、`VideoRenderer`、`ObservationWriter` 和
`AlarmPusher` 不再加入 SDK 内部 pipeline。离线 demo 自己做 FFmpeg 解码、
JSONL 输出和可选 MP4 渲染；也可以使用 `tools/render_jsonl.py` 对 JSONL
做离线重放。`DecodedFrameView`
当调用方提供 `dataOwner` 时 SDK 可异步借用输入 buffer；未提供 owner 时，
SDK 会在 `Process()` 返回前复制 RGB 像素，调用方可安全复用解码 buffer。
SDK 输入队列默认使用 `QueuePolicy::Block`，队列满时 `Process()` 会阻塞形成
背压；实时优先场景可切换为 `DropOldest` 或 `DropNew`。

## 后续 TODO

1. 用新的异步 `Process()` API 重跑 Python/C++ 的同帧数基准，统一 warm-up、
   ROI 过滤时机和统计口径。
2. 对齐 C++ 与 Python 的 tracker 生命周期、检测过滤和 ROI 语义，重新检查
   人员数、球出现率及轨迹一致性。
3. 分别测量 Detector、Pose 的 `instanceCount=1/2` 和 batch profile，记录
   显存、GPU 利用率与端到端 FPS。
4. 在线接入时由上游负责解码格式转换；当前 SDK 输入契约为 packed RGB24，
   后续可按需要扩展 NV12、BGR 或带 stride 的 plane view。
