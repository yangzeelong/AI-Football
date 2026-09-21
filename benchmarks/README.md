# 简介

## 1、Message(Inheritance-based) vs Message(Type-erasure)
```bash
Running ./benchmark/gbenchmark
Run on (28 X 3417.6 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x14)
  L1 Instruction 32 KiB (x14)
  L2 Unified 2048 KiB (x14)
  L3 Unified 33792 KiB (x1)
Load Average: 0.67, 0.35, 0.31
-------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
BM_Inheritance_Create          13.5 ns         13.5 ns     51900065
BM_TypeErasure_Create          13.4 ns         13.4 ns     52228718
BM_Inheritance_Broadcast       2.41 ns         2.41 ns    291988191
BM_TypeErasure_Broadcast       2.40 ns         2.40 ns    291383569
BM_Inheritance_Process         21.9 ns         21.9 ns     31925824
BM_TypeErasure_Process         5.28 ns         5.28 ns    132702795
```

## AI-Football Pose Benchmark

`pose_benchmark` directly calls `HRNetPoseEstimatorInfer` and measures pose
preprocess, TensorRT, and postprocess time for different batch sizes:

The benchmark baseline requires a dynamic-batch engine. Static batch-1 engines
are rejected by default so their throughput is not mixed into multi-batch
results.

```bash
build-release/bin/pose_benchmark \
  --engine /home/hx1/yzl/Work/AI-Football/models/mmpose/hrnet/hrnet-w48-dark.engine \
  --video data/射门1-1080p60.mov \
  --iterations 30 \
  --flip 0 \
  --batches 1,4,8,16
```

Use `--flip 1` to measure the additional cost of MMPose flip-test TTA. This
benchmark excludes detector, tracker, and pipeline scheduling overhead.
