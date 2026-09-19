# AI-Football C++ TensorRT 工程化阶段总结

日期：2026-09-13

## 1. 背景

这两天主要做的是 AI-Football 的第一版 C++ TensorRT 工程化。

底层使用的是我前期自研的 NexusFlow 框架。NexusFlow 是一个基于 Actor / Dataflow 思路实现的 C++ pipeline 框架，可以把视频算法链路拆成多个独立 `Module`，再通过异步消息队列串起来。

这次工作不是从零写一个完整业务系统，而是在 NexusFlow 这个框架基础上，把 AI-Football 原来的 Python 验证流程迁移到 C++ TensorRT，并先做出一条可以运行、可以测试、可以继续优化的工程化链路。

## 2. 当前已经完成的内容

目前已经完成了一版可以运行的 C++ 程序，整体流程是：

```text
本地视频读取 -> 解码成 RGB 帧 -> 调用 AI-Football SDK -> 输出 JSONL -> 可选渲染 MP4
```

SDK 侧做了第一轮接口整理：

- 对外入口收敛为 `AIFootballPipeline`。
- SDK 只负责算法 pipeline，不再绑定本地视频读取、解码、渲染和文件输出。
- 线上接入时，可以直接把已经解码后的帧传给 SDK。
- `Process()` 改成异步提交接口。
- 增加了输入队列策略：`Block`、`DropOldest`、`DropNew`。
- ROI 改成通过 YAML 解析后传入 SDK context。
- demo 程序负责本地视频读取、JSONL 输出和可选 MP4 渲染。
- JsonCpp、yaml-cpp 等依赖通过 CMake 管理。

目前可以比较稳妥地说：

```text
已经完成了 TensorRT C++ SDK 的第一版工程化，程序可以端到端跑通，并且具备后续线上接入和性能调优的基础。
```

## 3. 性能相关工作

这两天完成的性能优化和性能观测主要包括：

- RF-DETR detector 接入 TensorRT。
- HRNet pose estimator 接入 TensorRT。
- detector 支持 960 输入分辨率。
- detector 支持 dynamic batch engine。
- detector preprocess 完成 CUDA 版本第一版接入。
- 增加了模块级耗时统计，可以看到 detector、pose、tracker 等模块耗时。
- 增加了模块内部耗时统计，可以拆分查看 preprocess、TensorRT、postprocess 等阶段。
- SDK 输入支持 `DecodedFrameView::dataOwner`，demo 中已经使用 owned RGB buffer，避免 SDK 内部再复制一份 RGB 输入数据。
- 渲染逻辑从主算法路径中拆出来，默认关闭，避免影响正式性能测试。

当前比较稳的性能口径如下：

```text
Python no-render + ROI: 约 12.9 FPS
C++ TensorRT no-render + ROI: Release 短测约 25.7 FPS
之前 C++ TensorRT no-render + ROI 全量测试: 约 27.7 FPS

【目标】：
目标： 2 * 60 = 120fps
Python  约 60 FPS
C++ 100FPS
```

可以保守描述为：

```text
在 Release 模式、关闭渲染、开启 ROI 的情况下，C++ TensorRT 版本相比 Python no-render baseline 有比较明显提升，目前大概是 2 倍左右。
```

这里需要说明一点：Debug build 的性能没有参考意义。我测过一组 600 帧数据：

```text
Debug 600 frames: 44.224s，约 13.57 FPS
Release 600 frames: 23.323s，约 25.73 FPS
```

所以后续正式性能测试都会固定使用 Release build。

## 4. 当前遇到的问题

目前主要问题不是程序跑不通，而是结果和性能还需要进一步对齐。

### 4.1 CUDA preprocess 还需要继续验证

CUDA preprocess 已经接入到 detector 路径，但目前还没有完成和 Python baseline 的一致性验证。

后续需要重点排查：

- resize 策略是否一致。
- normalize 参数和顺序是否一致。
- RGB / BGR 通道顺序是否一致。
- HWC / CHW layout 转换是否一致。
- 坐标还原逻辑是否一致。
- TensorRT 输入 tensor 和 Python 侧输入 tensor 是否完全一致。

汇报时可以这样说：

```text
CUDA preprocess 已经完成第一版接入，但目前仍处于验证阶段。下一阶段会重点排查和 Python baseline 的输入一致性问题。
```

### 4.2 C++ TensorRT 结果和 Python 结果还没完全对齐

当前 C++ TensorRT pipeline 已经可以跑通，但 ROI 之后的人员数、目标过滤结果和 Python baseline 还有差异。

后续需要继续检查：

- ROI 坐标缩放。
- tracker 生命周期。
- 检测阈值。
- class id 过滤。
- 同帧数、同 stride、同 warm-up 条件。

这部分现在还不能说已经完全完成，只能说已经跑通，正在进入结果对齐阶段。

### 4.3 渲染只作为 debug 能力

demo 现在支持 `--render` 输出 MP4，但渲染会引入 CPU 绘制和视频编码开销。

所以后续性能测试默认关闭渲染；需要展示效果或排查问题时再开启。

## 5. 下一阶段计划

后续一周主要继续做工程化收敛和性能优化：

1. 用最新 SDK 版本重新跑完整 Release 性能测试。
2. 对齐 Python 和 C++ TensorRT 的检测结果。
3. 重点排查 CUDA preprocess 与 Python preprocess 的一致性。
4. 继续测试 detector 和 pose 的 batch / instanceCount 配置。
5. 补充 `QueuePolicy` 的边界行为测试。
6. 补一个 callback 模式的 SDK 接入示例。
7. 继续减少线上接入时的数据拷贝，后续可以考虑 NV12、BGR、plane view 或 GPU buffer 输入。

## 6. 汇报总结口径

可以用下面这段作为总结：

```text
这两天主要完成了 C++ TensorRT SDK 的第一版工程化闭环，程序已经可以端到端跑通，并且相比 Python baseline 有明显性能提升。目前还处在结果对齐和性能细化阶段，后续重点会排查 CUDA preprocess 和 Python baseline 的一致性，同时继续做 batch、多实例和输入数据拷贝优化。
```

更保守一点也可以说：

```text
目前 C++ TensorRT 版本已经跑通了第一版主流程，性能上已经看到提升，但还没有到最终交付状态。接下来主要是做结果对齐、Release 性能复测和进一步工程化优化。
```
