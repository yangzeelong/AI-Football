# AI-Football TensorRT Integration Status

Updated: 2026-09-13

## Current Pipeline

The current C++ pipeline is:

```text
VideoReader
  -> VideoDecoder
  -> RFDetrDetector
  -> ByteTracker
  -> HRNetPoseEstimator
  -> KeypointSmoother
  -> FootballTracker
  -> VideoRenderer
  -> ObservationWriter / AlarmPusher
```

The application uses the RF-DETR small 960 model and the HRNet-W48-DARK
wholebody TensorRT engine. The source video remains in its original format;
the RF-DETR model receives a square 960x960 tensor internally, matching the
Python resolution comparison implementation.

## Verified Test

Test input:

```text
data/射门1-1080p60.mov
```

Video properties:

```text
1920x1080, 60 FPS, RGB24 after decoding
```

Model configuration:

```text
RF-DETR: small, 960x960, TensorRT
HRNet: W48-DARK, 288x384 input, TensorRT
```

Build and test results:

```text
cmake --build build --parallel 4       PASS
ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 77
```

The application has also been run successfully on the test video. The
rendered output is written to:

```text
output/rendered.mp4
```

The generated video was verified with OpenCV as readable, with 1920x1080
frames and approximately 60 FPS. Observation output is written to:

```text
output/observations.jsonl
```

## Current Performance

The framework measures the processing time of each module centrally in
`nexusflow::Module::ProcessTimed()`. Each output message carries a
`MessageMeta::moduleTimingMs` map, and the observation writer emits it as
`module_timings_ms`. These values represent the elapsed time of the module's
`Process()` invocation. Batch-specific amortization is reported separately by
TimerRegistry.

## TimerRegistry

`nexusflow::TimerRegistry` is a thread-safe global aggregation service in the
framework. `TIMER_SCOPE` prints every invocation and is intended for focused
debugging. The average timers aggregate samples and print by elapsed time,
sample count, or either condition:

```cpp
TIMER_SCOPE("Detector.DebugStep");
TIMER_SCOPE_UNITS("Detector.DebugBatch", batchSize);

TIMER_START_AVERAGE_MS("Detector.TensorRT", 5000);
TIMER_INCREMENT_AVERAGE("Detector.TensorRT", batchSize);
TIMER_END_AVERAGE("Detector.TensorRT");

TIMER_START_AVERAGE_N("Detector.Postprocess", 100);
TIMER_START_AVERAGE("Detector.Postprocess", 100, 5000);

// or:
TIMER_SCOPE_AVERAGE_MS("Detector.Postprocess", batchSize, 5000);
TIMER_SCOPE_AVERAGE_N("Detector.Postprocess", batchSize, 100);
TIMER_SCOPE_AVERAGE("Detector.Postprocess", batchSize, 100, 5000);
```

Each timer reports total samples, work units, average batch time, average
item time, batch QPS, item QPS, and min/max time. `Module::ProcessTimed()`
is unchanged by this instrumentation. `StartAverage*()` controls the print
policy; work units are accumulated by `IncrementAverage()` or RAII scope
construction. Detector and pose inference paths currently use
`TIMER_SCOPE_AVERAGE_MS(..., 5000)` for preprocess, TensorRT, output-copy,
postprocess, and whole-batch timings.

Representative measurements from the earlier 960-resolution run with CPU
preprocessing:

| Module | Mean time per message |
| --- | ---: |
| VideoReader | 0.01 ms |
| VideoDecoder | 3.15 ms |
| RFDetrDetector | 43.49 ms |
| ByteTracker | 3.37 ms |
| HRNetPoseEstimator | 72.38 ms |
| KeypointSmoother | 0.34 ms |
| FootballTracker | 0.33 ms |
| VideoRenderer | 4.08 ms |

These are module service times, not end-to-end FPS. The actors run in
parallel, so the values must not be added together. The current dominant
stage is HRNet pose inference, followed by RF-DETR detection. The renderer
adds CPU drawing, color conversion, and H.264 encoding work.

The output video's 60 FPS is its playback rate and must not be interpreted as
the actual processing throughput. A strict throughput benchmark still needs
to measure processed frames divided by wall-clock runtime under fixed test
conditions.

## GPU Preprocessing

RF-DETR preprocessing can run on CUDA when `useGpuPreprocess: true`, the
TensorRT input is `float32`, and the application is built with the CUDA kernel
target. The implementation reuses a device-side RGB staging buffer, uploads
each decoded RGB24 frame on the TensorRT stream, and performs bilinear resize,
normalization, and HWC-to-NCHW conversion directly into the TensorRT input
buffer. The CPU implementation remains as a fallback for unsupported builds or
runtime CUDA failures.

The current 960-resolution test was run with batch size 4 and rendering
enabled. Its steady-state TimerRegistry measurements were:

| Timer | Mean batch time | Mean item time |
| --- | ---: | ---: |
| Detector.Preprocess | 1.394 ms | 0.348 ms |
| Detector.TensorRT | 86.691 ms | 21.673 ms |
| Detector.Batch | 91.352 ms | 22.838 ms |

The corresponding CPU-preprocessing run reported approximately 60.607 ms for
`Detector.Preprocess` and 152.938 ms for `Detector.Batch`. The GPU path removes
the large host resize and normalization cost, while the current synchronous
TensorRT execution still determines most of the remaining detector latency.

## Instance and Batch Support

The detector and pose modules now support independent model instances. Each
instance owns its TensorRT execution context, CUDA stream, and device/host
scratch buffers. Chunks assigned to different instances run concurrently and
are merged back in input order. Configure this with `instanceCount`; the
default remains `1` because every additional instance consumes GPU memory and
may contend for the same GPU.

The runtime reads the input batch dimension from the serialized engine. Static
batch-1 engines are automatically clamped to an effective batch of `1`, while
larger requests are split into chunks instead of failing or dropping pose
inputs. The RF-DETR detector also copies each batch output tensor once before
decoding frame slices. This fixes the previous batch-output offset bug, where
each frame could read the first frame's device output.

RF-DETR export now accepts `--dynamic-batch`. Build the resulting ONNX with
`build_tensorrt_engines.sh --max-batch N`, then set `maxBatchSize: N` in the
runtime configuration. The generated 960 model was validated with TensorRT
11.1 at batch 1 and batch 4 on the current GPU. A static existing engine
remains valid and continues to run with effective batch `1`.

Decoded frames are now held by `std::shared_ptr<const VideoFrame>`. Copying a
frame through the actor graph copies the pointer instead of the approximately
6.2 MB pixel payload. The decoder still performs one required pack from
FFmpeg's potentially padded image planes into contiguous RGB24 storage, and
the renderer creates a writable overlay buffer when drawing is enabled.

## Known Problems

1. Detector and pose engines share one GPU. Model-instance parallelism does not
   guarantee GPU kernel overlap; the engines may contend for GPU resources.
2. Rendering introduces an additional RGB buffer copy and H.264 encoding
   cost. It should be disabled when measuring inference-only throughput.
3. The current sample configuration contains machine-specific model paths and
   a sample video path. These should be moved to a portable runtime config or
   CLI overrides before deployment.
4. `--stop_frame` is parsed by the executable but is not yet propagated to
   `VideoReader`, so it does not currently stop input after the requested
   frame.

## TODO

### Measurement

- Add a wall-clock throughput summary: processed frames, elapsed seconds,
  effective FPS, dropped frames, and queue depth.
- Add warm-up and steady-state measurement modes.
- Benchmark the same input, model resolution, sampling stride, and rendering
  setting for Python and C++.
- Record GPU utilization and memory usage during the benchmark.

### Memory and Pipeline

- Keep writable overlays separate from the original frame buffer.
- Make queue sizes and worker batch policies configurable.
- Propagate `--stop_frame`, `--stride`, and `--target_fps` into the source
  module instead of only parsing them in `main.cpp`.

### Inference

- Benchmark different `instanceCount` and batch profiles against GPU memory
  usage and end-to-end throughput.
- Split detector preprocessing measurement into host upload, CUDA kernel, and
  stream synchronization costs; keep TensorRT execution and postprocessing as
  separate timers.
- Evaluate whether the 960 RF-DETR engine should remain fixed or use the new
  dynamic-batch profile in production.

### Output and Deployment

- Add an inference-only mode that skips rendering and video encoding.
- Make the renderer codec, bitrate, and pixel format configurable.
- Add portable model path resolution and a deployment configuration example.
- Add automated validation for rendered video dimensions, FPS, and frame count.
