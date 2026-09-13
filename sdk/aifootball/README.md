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
// Debug rendering is disabled by default.
options.enableRendering = false;

auto runtime = aifootball::Runtime::Create(options);
const auto status = runtime->Run();
```

`Runtime::Run()` performs initialization, starts the actor graph, waits for
EOF or the configured timeout, then stops and de-initializes the pipeline.
`Init()`, `Start()`, `Wait()`, `Stop()`, and `DeInit()` are available when an
embedding application needs explicit lifecycle control.

Video rendering is a debug side effect and is disabled by default through
`RuntimeOptions::enableRendering`. The pipeline still forwards messages to
downstream output modules. Set this option to `true` when an application needs
`rendered.mp4`.

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
timeout, stride, debug rendering, verbose, and quiet command-line options.
Rendering can be enabled with `--render`; it is disabled by default.
