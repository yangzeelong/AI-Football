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
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    const std::string configPath = cli.Get("config");
    const std::string videoPath = cli.Get("video_path").empty()
        ? aifootball_demo::LoadVideoPath(configPath)
        : cli.Get("video_path");
    const std::string outputDir = cli.Get("output_dir");
    const int maxFrames = cli.GetInt("max_frames");
    const int stride = std::max(1, cli.GetInt("stride", 1));
    const bool render = cli.IsFlagSet("render");

    if (videoPath.empty()) {
        LOG_ERROR("No video path provided and graph.modules has no VideoReader path");
        return 2;
    }
    if (!aifootball_demo::EnsureDirectory(outputDir)) {
        LOG_ERROR("Failed to create output directory '{}'", outputDir);
        return 2;
    }

    aifootball_demo::VideoReader reader;
    if (!reader.Open(videoPath)) {
        LOG_ERROR("Failed to open/decode video '{}'", videoPath);
        return 2;
    }

    const std::string cameraId = aifootball_demo::LoadCameraId(configPath);
    const std::string observationsPath =
        aifootball_demo::JoinPath(outputDir, "observations.jsonl");
    aifootball_demo::JsonlWriter writer;
    if (!writer.Open(observationsPath, videoPath, reader.fps(), reader.width(),
                     reader.height(), stride, cameraId)) {
        LOG_ERROR("Failed to open observations output '{}'", observationsPath);
        return 2;
    }
    const std::string renderPath = cli.Get("render_path").empty()
        ? aifootball_demo::JoinPath(outputDir, "rendered.mp4")
        : cli.Get("render_path");
    aifootball_demo::VideoRenderer renderer;
    if (!renderer.Open(renderPath, reader.fps(), reader.width(), reader.height(),
                       render)) {
        LOG_ERROR("Failed to initialize video renderer '{}'", renderPath);
        return 2;
    }

    try {
        aifootball::AIFootballContext context =
            aifootball_demo::LoadContext(configPath);
        context.configPath = configPath;
        context.deviceId = cli.GetInt("device", 0);

        auto pipeline = aifootball::AIFootballPipeline::Create(context);
        if (!pipeline || pipeline->Init() != nexusflow::SUCCESS) {
            LOG_ERROR("AI-Football SDK initialization failed");
            return 2;
        }

        int pendingFrames = 0;
        int processedFrames = 0;
        bool stoppedByLimit = false;
        const bool decoded = reader.Decode(
            [&](const aifootball::DecodedFrameView& frame) {
                if (g_stop.load(std::memory_order_relaxed)) return false;
                if (frame.frameId % static_cast<uint64_t>(stride) != 0) return true;
                if (maxFrames > 0 && processedFrames >= maxFrames) {
                    stoppedByLimit = true;
                    return false;
                }

                if (!renderer.SubmitFrame(frame)) return false;
                if (pipeline->Process(frame) != nexusflow::SUCCESS) {
                    LOG_ERROR("AI-Football SDK input queue is full or closed at frame {}",
                              frame.frameId);
                    return false;
                }
                ++pendingFrames;
                ++processedFrames;
                return WriteReadyResults(*pipeline, writer, renderer,
                                         pendingFrames);
            });

        if (!decoded && !stoppedByLimit) {
            LOG_ERROR("VideoReader failed while decoding '{}'", videoPath);
            pipeline->DeInit();
            return 2;
        }
        if (pipeline->Flush() != nexusflow::SUCCESS) {
            LOG_ERROR("Failed to flush AI-Football SDK pipeline");
            pipeline->DeInit();
            return 2;
        }
        while (pendingFrames > 0) {
            aifootball::ProcessResult result;
            if (pipeline->PollResult(result, 300000) != nexusflow::SUCCESS) {
                LOG_ERROR("Failed to poll pending SDK result");
                pipeline->DeInit();
                return 2;
            }
            writer.Write(result);
            if (!renderer.Write(result)) {
                LOG_ERROR("Failed to render SDK result for frame {}", result.frameId);
                pipeline->DeInit();
                return 2;
            }
            --pendingFrames;
        }
        pipeline->DeInit();
        LOG_INFO("AI-Football SDK demo processed {} frames, observations='{}'",
                 processedFrames, observationsPath);
    } catch (const std::exception& error) {
        LOG_ERROR("Fatal: {}", error.what());
        return 2;
    }
    writer.Close();
    return 0;
}
