# AI-Football SDK

`aifootball_sdk` packages the AI-Football algorithm modules on top of the
Nexusflow actor pipeline. Applications use the public facade instead of
including detector, pose, tracker, or TensorRT implementation headers.

The SDK does not open files, demux/decode video, render frames, or write
observations. Those responsibilities stay in the embedding application or in
the offline app.

## Public API

Include:

```cpp
#include <aifootball/AIFootball.hpp>
#include <chrono>
```

The main entry point is `aifootball::AIFootballPipeline`:

```cpp
aifootball::AIFootballContext context;
context.configPath = "config.yaml";
context.deviceId = 0;
context.inputQueuePolicy = aifootball::QueuePolicy::Block;
context.roi.enabled = true;
context.roi.width = 1920;
context.roi.height = 1080;
context.roi.points = {{1579.0f, 1067.0f}, {56.0f, 733.0f},
                     {950.0f, 416.0f}, {1796.0f, 493.0f}};

auto pipeline = aifootball::AIFootballPipeline::Create(context);
if (pipeline->Init() != nexusflow::SUCCESS) return false;

aifootball::DecodedFrameView frame;
frame.frameId = frameIndex;
frame.timestampSec = timestampSec;
frame.width = width;
frame.height = height;
frame.strideBytes = width * 3;
frame.data = rgb24Bytes;
frame.dataBytes = static_cast<std::size_t>(width) * height * 3;

auto resultFuture = pipeline->ProcessAsync(frame);
if (!resultFuture.WaitFor(std::chrono::milliseconds(30))) {
    // Keep submitting frames or poll the future later.
}
// At end of stream only:
pipeline->Drain();
const auto output = resultFuture.Get();
if (output.status == nexusflow::SUCCESS) {
    // Consume output.result.
}
pipeline->DeInit();
```

`ProcessAsync()` enqueues work and returns a move-only `ProcessFuture`. The
future supports `IsReady()`, `Wait()`, `WaitFor(timeout)`, and `Get()`.
`Get()` consumes the result once. `ProcessFutureResult::status` reports whether
the frame was processed, dropped, or rejected; the detailed output is in
`ProcessFutureResult::result` when the status is `nexusflow::SUCCESS`.
`Drain()` is still only an end-of-stream drain operation and should not be used
as a per-frame synchronization point.

`DecodedFrameView` accepts packed RGB24. If `dataOwner` is set, the SDK retains
that shared owner until the result is delivered and can use the buffer without
copying. If it is empty, the SDK copies the bytes before returning from
`ProcessAsync()` so the caller may immediately reuse its decode buffer.

The input queue is unbounded by default (`maxPendingFrames == 0`). Set
`AIFootballContext::maxPendingFrames` to a positive value to enable a bound.
`QueuePolicy::Block` is the default and applies backpressure by blocking
`ProcessAsync()` until the internal source queue has room.
`QueuePolicy::DropOldest` drops the oldest queued input frame and completes its
future with a failure status before accepting the new frame.
`QueuePolicy::DropNew` rejects the current frame and returns an already-ready
future with a failure status.

Results are completed in input order while preserving the configured detector
batch policy. `DeInit()` stops the internal actor graph and completes any
remaining futures with a failure status; callers should drain and consume
results before de-initializing.

`AIFootballContext` contains execution and algorithm-level settings. ROI points use the coordinate
system described by `roi.width` and `roi.height`; the runtime scales them to
the decoded frame dimensions when necessary.

ROI is supplied by the embedding application through `AIFootballContext`, normally by
parsing `algorithm.roi` from YAML. ROI coordinates are expressed in the source
frame coordinate system described by `roi.width` and `roi.height`.

## Build

```bash
cmake -S . -B build -DWITH_AIFOOTBALL_SDK=ON
cmake --build build --target aifootball_sdk aifootball_app --parallel 4
```

The SDK target links the locally detected FFmpeg, CUDA, and TensorRT
dependencies. The public headers are installed under `include/aifootball`.

## App

`examples/aifootball_app` is an offline integration executable. It owns
FFmpeg decoding, writes `observations.jsonl` after calling the SDK, and can
optionally render detections to MP4. Its `VideoReader` creates an owned RGB24
buffer per decoded frame and passes it through `DecodedFrameView::dataOwner`,
so the SDK does not need an additional input copy. Rendering is disabled by
default:

```bash
python3 tools/render_jsonl.py \
  --observations output/sdk_app/observations.jsonl \
  --output output/sdk_app/rendered.mp4
```

Example:

```bash
build/examples/aifootball_app/aifootball_app \
  examples/aifootball_app/config.yaml \
  --video_path data/射门1-1080p60.mov \
  --output_dir output/sdk_app

build/examples/aifootball_app/aifootball_app \
  examples/aifootball_app/config.yaml \
  --output_dir output/sdk_debug \
  --render
```
