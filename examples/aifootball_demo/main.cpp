#include "CommandParser.hpp"
#include "DemoConfig.hpp"
#include "JsonlWriter.hpp"
#include "VideoReader.hpp"
#include "VideoRenderer.hpp"

#include <aifootball/AIFootball.hpp>
#include <nexusflow/Logging.hpp>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <exception>
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
    return options;
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

bool WriteReadyResults(aifootball::AIFootballPipeline& pipeline,
                       aifootball_demo::JsonlWriter& writer,
                       aifootball_demo::VideoRenderer& renderer,
                       int& pendingFrames) {
    while (pendingFrames > 0) {
        aifootball::ProcessResult result;
        if (pipeline.PollResult(result, 0) != nexusflow::SUCCESS) break;
        writer.Write(result);
        if (!renderer.Write(result)) return false;
        --pendingFrames;
    }
    return true;
}

bool DrainResults(aifootball::AIFootballPipeline& pipeline,
                  aifootball_demo::JsonlWriter& writer,
                  aifootball_demo::VideoRenderer& renderer,
                  int& pendingFrames) {
    while (pendingFrames > 0) {
        aifootball::ProcessResult result;
        if (pipeline.PollResult(result, 300000) != nexusflow::SUCCESS) {
            LOG_ERROR("Failed to poll pending SDK result");
            return false;
        }
        writer.Write(result);
        if (!renderer.Write(result)) {
            LOG_ERROR("Failed to render SDK result for frame {}", result.frameId);
            return false;
        }
        --pendingFrames;
    }
    return true;
}

bool ProcessVideo(const RunOptions& options, DemoResources& resources,
                  aifootball::AIFootballPipeline& pipeline,
                  int& processedFrames) {
    int pendingFrames = 0;
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

            if (!resources.renderer.SubmitFrame(frame)) {
                failed = true;
                return false;
            }
            if (pipeline.Process(frame) != nexusflow::SUCCESS) {
                LOG_ERROR("AI-Football SDK input queue is full or closed at frame {}",
                          frame.frameId);
                failed = true;
                return false;
            }
            ++pendingFrames;
            ++processedFrames;
            if (!WriteReadyResults(pipeline, resources.writer,
                                   resources.renderer, pendingFrames)) {
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
    if (pipeline.Flush() != nexusflow::SUCCESS) {
        LOG_ERROR("Failed to flush AI-Football SDK pipeline");
        return false;
    }
    return DrainResults(pipeline, resources.writer, resources.renderer,
                        pendingFrames);
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
