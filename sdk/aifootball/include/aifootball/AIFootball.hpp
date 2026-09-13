#pragma once

#include <nexusflow/ErrorCode.hpp>

#include <memory>
#include <string>

namespace aifootball {

/** Runtime options for the configuration-driven AI-Football pipeline. */
struct RuntimeOptions {
    /// YAML graph configuration containing model and pipeline settings.
    std::string configPath;
    /// Optional override for the VideoReader videoPath setting.
    std::string videoPath;
    /// Optional output directory for observations.jsonl, result.txt, and
    /// rendered.mp4 when rendering is enabled.
    std::string outputDir;
    /// CUDA device selected before model initialization.
    int deviceId = 0;
    /// Stop waiting after this many seconds; zero waits for EOF.
    int maxSeconds = 0;
    /// Optional override for VideoReader stride; zero keeps the YAML value.
    int stride = 0;
    /// Enable debug video rendering. Disabled by default for SDK workloads.
    bool enableRendering = false;
};

/**
 * Stable facade for embedding the AI-Football pipeline in another program.
 *
 * The facade owns module registration, YAML loading, pipeline lifecycle, and
 * completion handling. Callers do not need to depend on app-internal module
 * headers or the command-line executable.
 */
class Runtime {
public:
    static std::unique_ptr<Runtime> Create(const RuntimeOptions& options);

    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    nexusflow::ErrorCode Init();
    nexusflow::ErrorCode Start();
    nexusflow::ErrorCode Wait();
    nexusflow::ErrorCode Stop();
    nexusflow::ErrorCode DeInit();

    /// Convenience lifecycle: Init -> Start -> Wait -> Stop -> DeInit.
    nexusflow::ErrorCode Run();

    /// Wake Wait() and let the caller proceed to Stop().
    void RequestStop();

private:
    explicit Runtime(RuntimeOptions options);

    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace aifootball
