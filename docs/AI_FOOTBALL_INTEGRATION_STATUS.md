# AI-Football TensorRT Integration Status

Updated: 2026-09-12

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

The framework now measures the processing time of each module centrally in
`nexusflow::Module::ProcessTimed()`. Each output message carries a
`MessageMeta::moduleTimingMs` map, and the observation writer emits it as
`module_timings_ms`.

Representative measurements from the current 960-resolution run:

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
runtime configuration. A static existing engine remains valid and continues
to run with effective batch `1`.

## Known Problems

1. `VideoFrame::frameData` is currently a `std::string`. Assigning a
   `VideoFrame` between pipeline messages copies the full RGB frame instead of
   sharing the pixel buffer. A 1920x1080 RGB24 frame is about 6.2 MB, and this
   copy happens across several modules.
2. Detector and pose engines share one GPU. Model-instance parallelism does not
   guarantee GPU kernel overlap; the engines may contend for GPU resources.
3. Rendering introduces an additional RGB buffer copy and H.264 encoding
   cost. It should be disabled when measuring inference-only throughput.
4. The current sample configuration contains machine-specific model paths and
   a sample video path. These should be moved to a portable runtime config or
   CLI overrides before deployment.
5. `--stop_frame` is parsed by the executable but is not yet propagated to
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

- Replace per-message RGB string copies with a shared immutable frame buffer.
- Keep writable overlays separate from the original frame buffer.
- Make queue sizes and worker batch policies configurable.
- Propagate `--stop_frame`, `--stride`, and `--target_fps` into the source
  module instead of only parsing them in `main.cpp`.

### Inference

- Benchmark different `instanceCount` and batch profiles against GPU memory
  usage and end-to-end throughput.
- Measure detector and pose TensorRT execution separately from host
  preprocessing and postprocessing.
- Evaluate whether the 960 RF-DETR engine should remain fixed or use the new
  dynamic-batch profile in production.

### Output and Deployment

- Add an inference-only mode that skips rendering and video encoding.
- Make the renderer codec, bitrate, and pixel format configurable.
- Add portable model path resolution and a deployment configuration example.
- Add automated validation for rendered video dimensions, FPS, and frame count.
