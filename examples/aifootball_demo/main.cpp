#include "CommandParser.hpp"
#include "DemoConfig.hpp"
#include "JsonlWriter.hpp"
#include "VideoReader.hpp"
#include "VideoRenderer.hpp"

#include <aifootball/AIFootball.hpp>
#include <nexusflow/Logging.hpp>
#include <nexusflow/TimerRegistry.hpp>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <deque>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

std::atomic<bool> g_stop{false};

void HandleSignal(int) { g_stop.store(true, std::memory_order_relaxed); }

struct RunOptions {
    std::string configPath;
    std::string videoPath;
    std::string outputDir;
    std::string renderPath;
    int maxFrames = 0;
    int stride = 1;
    int deviceId = 0;
    bool render = false;
    bool profileTimers = false;
};

struct DemoResources {
    aifootball_demo::VideoReader reader;
    aifootball_demo::JsonlWriter writer;
    aifootball_demo::VideoRenderer renderer;
    std::string observationsPath;
};

RunOptions ReadOptions(const app::CommandParser& cli) {
    RunOptions options;
    options.configPath = cli.Get("config");
    options.videoPath = cli.Get("video_path").empty()
        ? aifootball_demo::LoadVideoPath(options.configPath)
        : cli.Get("video_path");
    options.outputDir = cli.Get("output_dir");
    options.renderPath = cli.Get("render_path");
    options.maxFrames = cli.GetInt("max_frames");
    options.stride = std::max(1, cli.GetInt("stride", 1));
    options.deviceId = cli.GetInt("device", 0);
    options.render = cli.IsFlagSet("render");
    options.profileTimers = cli.IsFlagSet("profile_timers");
    return options;
}

void PrintTimerSnapshot() {
    auto stats = nexusflow::TimerRegistry::Instance().Snapshot();
    std::sort(stats.begin(), stats.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.totalMs > rhs.totalMs;
    });

    std::cout << "\nTimer profile:\n";
    std::cout << std::left << std::setw(32) << "name"
              << std::right << std::setw(10) << "samples"
              << std::setw(12) << "work"
              << std::setw(14) << "avg_ms"
              << std::setw(14) << "item_ms"
              << std::setw(14) << "total_ms"
              << std::setw(12) << "min_ms"
              << std::setw(12) << "max_ms" << '\n';
    std::cout << std::fixed << std::setprecision(3);
    for (const auto& item : stats) {
        std::cout << std::left << std::setw(32) << item.name
                  << std::right << std::setw(10) << item.samples
                  << std::setw(12) << item.workUnits
                  << std::setw(14) << item.AvgBatchMs()
                  << std::setw(14) << item.AvgItemMs()
                  << std::setw(14) << item.totalMs
                  << std::setw(12) << item.minMs
                  << std::setw(12) << item.maxMs << '\n';
    }
}

bool PrepareResources(const RunOptions& options, DemoResources& resources) {
    if (options.videoPath.empty()) {
        LOG_ERROR("No video path provided and graph.modules has no VideoReader path");
        return false;
    }
    if (!aifootball_demo::EnsureDirectory(options.outputDir)) {
        LOG_ERROR("Failed to create output directory '{}'", options.outputDir);
        return false;
    }
    if (!resources.reader.Open(options.videoPath)) {
        LOG_ERROR("Failed to open/decode video '{}'", options.videoPath);
        return false;
    }

    const std::string cameraId =
        aifootball_demo::LoadCameraId(options.configPath);
    resources.observationsPath =
        aifootball_demo::JoinPath(options.outputDir, "observations.jsonl");
    if (!resources.writer.Open(resources.observationsPath, options.videoPath,
                               resources.reader.fps(), resources.reader.width(),
                               resources.reader.height(), options.stride,
                               cameraId)) {
        LOG_ERROR("Failed to open observations output '{}'",
                  resources.observationsPath);
        return false;
    }

    const std::string renderPath = options.renderPath.empty()
        ? aifootball_demo::JoinPath(options.outputDir, "rendered.mp4")
        : options.renderPath;
    const double renderFps =
        resources.reader.fps() / static_cast<double>(options.stride);
    if (!resources.renderer.Open(renderPath, renderFps,
                                 resources.reader.width(),
                                 resources.reader.height(), options.render)) {
        LOG_ERROR("Failed to initialize video renderer '{}'", renderPath);
        return false;
    }
    return true;
}

using ResultFuture = aifootball::ProcessFuture;

bool HandleResult(aifootball::ProcessFutureResult value,
                  aifootball_demo::JsonlWriter& writer,
                  aifootball_demo::VideoRenderer& renderer) {
    if (value.status != nexusflow::SUCCESS) {
        LOG_ERROR("AI-Football SDK failed for frame {} with status {}",
                  value.result.frameId, value.status);
        return false;
    }
    writer.Write(value.result);
    if (!renderer.Write(value.result)) {
        LOG_ERROR("Failed to render SDK result for frame {}",
                  value.result.frameId);
        return false;
    }
    return true;
}

bool HandleFutureResult(ResultFuture& future,
                        aifootball_demo::JsonlWriter& writer,
                        aifootball_demo::VideoRenderer& renderer) {
    try {
        return HandleResult(future.Get(), writer, renderer);
    } catch (const std::exception& error) {
        LOG_ERROR("Failed to consume AI-Football SDK future: {}", error.what());
        return false;
    }
}

bool WriteReadyResults(std::deque<ResultFuture>& pendingResults,
                       aifootball_demo::JsonlWriter& writer,
                       aifootball_demo::VideoRenderer& renderer) {
    while (!pendingResults.empty() && pendingResults.front().IsReady()) {
        if (!HandleFutureResult(pendingResults.front(), writer, renderer)) {
            return false;
        }
        pendingResults.pop_front();
    }
    return true;
}

bool DrainResults(std::deque<ResultFuture>& pendingResults,
                  aifootball_demo::JsonlWriter& writer,
                  aifootball_demo::VideoRenderer& renderer) {
    while (!pendingResults.empty()) {
        pendingResults.front().Wait();
        if (!HandleFutureResult(pendingResults.front(), writer, renderer)) {
            return false;
        }
        pendingResults.pop_front();
    }
    return true;
}

bool ProcessVideo(const RunOptions& options, DemoResources& resources,
                  aifootball::AIFootballPipeline& pipeline,
                  int& processedFrames) {
    std::deque<ResultFuture> pendingResults;
    bool stoppedByLimit = false;
    bool failed = false;
    const bool decoded = resources.reader.Decode(
        [&](const aifootball::DecodedFrameView& frame) {
            if (g_stop.load(std::memory_order_relaxed)) return false;
            if (frame.frameId % static_cast<uint64_t>(options.stride) != 0) {
                return true;
            }
            if (options.maxFrames > 0 && processedFrames >= options.maxFrames) {
                stoppedByLimit = true;
                return false;
            }

            auto future = pipeline.ProcessAsync(frame);
            if (future.IsReady()) {
                try {
                    auto value = future.Get();
                    if (value.status != nexusflow::SUCCESS) {
                        LOG_ERROR("AI-Football SDK failed for frame {} with status {}",
                                  frame.frameId, value.status);
                        failed = true;
                        return false;
                    }
                    if (!resources.renderer.SubmitFrame(frame) ||
                        !HandleResult(std::move(value), resources.writer,
                                      resources.renderer)) {
                        failed = true;
                        return false;
                    }
                } catch (const std::exception& error) {
                    LOG_ERROR("Failed to consume AI-Football SDK future: {}",
                              error.what());
                    failed = true;
                    return false;
                }
            } else {
                if (!resources.renderer.SubmitFrame(frame)) {
                    failed = true;
                    return false;
                }
                pendingResults.push_back(std::move(future));
            }

            ++processedFrames;
            if (!WriteReadyResults(pendingResults, resources.writer,
                                   resources.renderer)) {
                failed = true;
                return false;
            }
            return true;
        });

    if (failed) {
        LOG_ERROR("AI-Football demo processing failed for '{}'", options.videoPath);
        return false;
    }
    if (!decoded && !stoppedByLimit) {
        LOG_ERROR("VideoReader failed while decoding '{}'", options.videoPath);
        return false;
    }
    if (pipeline.Drain() != nexusflow::SUCCESS) {
        LOG_ERROR("Failed to drain AI-Football SDK pipeline");
        return false;
    }
    return DrainResults(pendingResults, resources.writer, resources.renderer);
}

int RunDemo(const RunOptions& options) {
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    DemoResources resources;
    if (!PrepareResources(options, resources)) return 2;

    try {
        aifootball::AIFootballContext context =
            aifootball_demo::LoadContext(options.configPath);
        context.configPath = options.configPath;
        context.deviceId = options.deviceId;

        auto pipeline = aifootball::AIFootballPipeline::Create(context);
        if (!pipeline || pipeline->Init() != nexusflow::SUCCESS) {
            LOG_ERROR("AI-Football SDK initialization failed");
            return 2;
        }

        int processedFrames = 0;
        if (!ProcessVideo(options, resources, *pipeline, processedFrames)) {
            pipeline->DeInit();
            return 2;
        }
        if (pipeline->DeInit() != nexusflow::SUCCESS) {
            LOG_ERROR("Failed to deinitialize AI-Football SDK pipeline");
            return 2;
        }
        if (options.profileTimers) PrintTimerSnapshot();
        LOG_INFO("AI-Football SDK demo processed {} frames, observations='{}'",
                 processedFrames, resources.observationsPath);
        return 0;
    } catch (const std::exception& error) {
        LOG_ERROR("Fatal: {}", error.what());
        return 2;
    }
}

} // namespace

int main(int argc, char* argv[]) {
    app::CommandParser cli(argc, argv);
    cli.AddPositional("config", "Path to config.yaml", true);
    cli.AddArgument("--video_path", "-v", "Override video path from config");
    cli.AddArgument("--output_dir", "-o", "Output directory", false,
                    "output/sdk_demo");
    cli.AddArgument("--max_frames", "-n", "Stop after N decoded frames (0 = all)",
                    false, "0");
    cli.AddArgument("--stride", "-S", "Process every Nth decoded frame", false,
                    "1");
    cli.AddArgument("--device", "-d", "CUDA device id", false, "0");
    cli.AddArgument("--render", "-r", "Render detections to an MP4", false,
                    app::CommandParser::FlagMarker());
    cli.AddArgument("--render_path", "-R", "Rendered MP4 path", false, "");
    cli.AddArgument("--verbose", "-V", "Enable DEBUG-level logging", false,
                    app::CommandParser::FlagMarker());
    cli.AddArgument("--quiet", "-q", "Suppress INFO-level logging", false,
                    app::CommandParser::FlagMarker());
    cli.AddArgument("--profile_timers", "", "Print internal timer profile at exit", false,
                    app::CommandParser::FlagMarker());
    if (!cli.Parse()) {
        cli.PrintHelp();
        return 1;
    }
    if (cli.IsHelpRequested()) {
        cli.PrintHelp();
        return 0;
    }

    nexusflow::logger::LoggerParam logParams;
    logParams.logLevel = cli.IsFlagSet("verbose")
        ? nexusflow::logger::LogLevel::DEBUG
        : cli.IsFlagSet("quiet") ? nexusflow::logger::LogLevel::WARN
                                  : nexusflow::logger::LogLevel::INFO;
    nexusflow::logger::InitializeGlobalLogger(logParams);
    return RunDemo(ReadOptions(cli));
}
