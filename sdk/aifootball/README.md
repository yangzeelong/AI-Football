# AI-Football SDK

`aifootball_sdk` packages the AI-Football algorithm modules on top of the
Nexusflow actor pipeline. Applications use the public facade instead of
including detector, pose, tracker, or TensorRT implementation headers.

The SDK does not open files, demux/decode video, render frames, or write
observations. Those responsibilities stay in the embedding application or in
the offline demo.

## Public API

Include:

```cpp
#include <aifootball/AIFootball.hpp>
```

The main entry point is `aifootball::Runtime`:

```cpp
aifootball::RuntimeOptions options;
options.configPath = "config.yaml";
options.deviceId = 0;

aifootball::AlgoConfig algoConfig;
algoConfig.roi.enabled = true;
algoConfig.roi.width = 1920;
algoConfig.roi.height = 1080;
algoConfig.roi.points = {{1579.0f, 1067.0f}, {56.0f, 733.0f},
                         {950.0f, 416.0f}, {1796.0f, 493.0f}};

auto runtime = aifootball::Runtime::Create(options, algoConfig);
if (runtime->Init() != nexusflow::SUCCESS) return false;

aifootball::DecodedFrameView frame;
frame.frameId = frameIndex;
frame.timestampSec = timestampSec;
frame.width = width;
frame.height = height;
frame.strideBytes = width * 3;
frame.data = rgb24Bytes;
frame.dataBytes = static_cast<std::size_t>(width) * height * 3;

const auto status = runtime->Process(frame); // asynchronous enqueue

aifootball::ProcessResult result;
runtime->PollResult(result, 0); // non-blocking; or install a callback
// At end of stream only:
runtime->Flush();
runtime->DeInit();
```

`Process()` only enqueues work and returns without waiting for inference.
Results are retrieved asynchronously with `PollResult()` or delivered through
`SetResultCallback()`. `PollResult(result, 0)` is a non-blocking poll. Call
`Flush()` once at the end of a stream to drain the final partial detector
batch; it should not be used as a per-frame synchronization point.

`DecodedFrameView` accepts packed RGB24. If `dataOwner` is set, the SDK retains
that shared owner until the result is delivered and can use the buffer without
copying. If it is empty, the SDK copies the bytes before returning from
`Process()` so the caller may immediately reuse its decode buffer.

Results are emitted in input order when polled, while preserving the configured
detector batch policy. `DeInit()` stops the internal actor graph; callers should
flush and drain results before de-initializing.

`AlgoConfig` contains algorithm-level settings. ROI points use the coordinate
system described by `roi.width` and `roi.height`; the runtime scales them to
the decoded frame dimensions when necessary.

ROI is supplied by the embedding application through `AlgoConfig`, normally by
parsing `algorithm.roi` from YAML. ROI coordinates are expressed in the source
frame coordinate system described by `roi.width` and `roi.height`.

## Build

```bash
cmake -S . -B build -DWITH_AIFOOTBALL_SDK=ON
cmake --build build --target aifootball_sdk aifootball_demo --parallel 4
```

The SDK target links the locally detected FFmpeg, CUDA, and TensorRT
dependencies. The public headers are installed under `include/aifootball`.

## Demo

`examples/aifootball_demo` is an offline integration executable. It owns
FFmpeg decoding and writes `observations.jsonl` after calling the SDK. It does
not render video. Render that JSONL for debugging with:

```bash
python3 tools/render_jsonl.py \
  --observations output/sdk_demo/observations.jsonl \
  --output output/sdk_demo/rendered.mp4
```

Example:

```bash
build/examples/aifootball_demo/aifootball_demo \
  examples/aifootball_demo/config.yaml \
  --video_path data/射门1-1080p60.mov \
  --output_dir output/sdk_demo
```
