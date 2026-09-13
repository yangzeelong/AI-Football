# AI-Football SDK

`aifootball_sdk` packages the AI-Football business modules on top of the
Nexusflow actor pipeline. Applications use the public facade instead of
including detector, pose, tracker, video, or TensorRT implementation headers.

## Public API

Include:

```cpp
#include <aifootball/AIFootball.hpp>
```

The main entry point is `aifootball::Runtime`:

```cpp
aifootball::RuntimeOptions options;
options.configPath = "config.yaml";
options.videoPath = "input.mp4";
options.outputDir = "output";
options.deviceId = 0;

auto runtime = aifootball::Runtime::Create(options);
const auto status = runtime->Run();
```

`Runtime::Run()` performs initialization, starts the actor graph, waits for
EOF or the configured timeout, then stops and de-initializes the pipeline.
`Init()`, `Start()`, `Wait()`, `Stop()`, and `DeInit()` are available when an
embedding application needs explicit lifecycle control.

## Build

```bash
cmake -S . -B build -DWITH_AIFOOTBALL_SDK=ON
cmake --build build --target aifootball_sdk aifootball_demo --parallel 4
```

The SDK target links the locally detected FFmpeg, CUDA, and TensorRT
dependencies. The public headers are installed under `include/aifootball`.

## Demo

`examples/aifootball_demo` is an integration smoke-test executable. It uses
only the public SDK facade and supports video, output-directory, CUDA-device,
timeout, stride, verbose, and quiet command-line options.
