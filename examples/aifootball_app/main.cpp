#include "CommandParser.hpp"
#include "AppConfig.hpp"
#include "JsonlWriter.hpp"
#include "TouchRuntime.hpp"
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
#include <memory>
#include <string>

namespace {

std::atomic<bool> g_stop{false};

void HandleSignal(int) { g_stop.store(true, std::memory_order_relaxed); }

struct RunOptions {
    std::string configPath;
    std::string videoPath;
    std::string outputDir;
    std::string renderPath;
    std::string touchOutputPath;
    int maxFrames = 0;
    int stride = 1;
    int deviceId = 0;
    bool render = false;
    bool profileTimers = false;
    bool detectTouches = false;
    touch_eval::TouchDetectorParams touchParams;
};

struct AppResources {
    aifootball_app::VideoReader reader;
    aifootball_app::JsonlWriter writer;
    aifootball_app::VideoRenderer renderer;
    std::unique_ptr<aifootball_app::TouchRuntime> touches;
    std::string observationsPath;
};

RunOptions ReadOptions(const app::CommandParser& cli,
                       const aifootball_app::AppConfig& config) {
    RunOptions options;
    options.configPath = config.Path();
    options.videoPath = cli.Get("video_path").empty()
        ? config.VideoPath()
        : cli.Get("video_path");
    options.outputDir = cli.Get("output_dir");
    options.renderPath = cli.Get("render_path");
    options.touchOutputPath = cli.Get("touch_output");
    options.maxFrames = cli.GetInt("max_frames");
    options.stride = std::max(1, cli.GetInt("stride", 1));
    options.deviceId = cli.GetInt("device", 0);
    options.render = cli.IsFlagSet("render");
    options.profileTimers = cli.IsFlagSet("profile_timers");
    options.detectTouches = cli.IsFlagSet("detect_touches");

    // Touch thresholds mirror the offline evaluator's flags, so a value found
    // by the calibration sweep can be pasted straight into a run.
    touch_eval::TouchDetectorParams& touch = options.touchParams;
    if (cli.Has("contact_ratio")) {
        touch.contactDistanceRatio = cli.GetFloat("contact_ratio");
    }
    if (cli.Has("person_height_ratio")) {
        touch.personHeightRatio = cli.GetFloat("person_height_ratio");
    }
    if (cli.Has("min_contact_px")) touch.minContactPx = cli.GetFloat("min_contact_px");
    if (cli.Has("max_contact_px")) touch.maxContactPx = cli.GetFloat("max_contact_px");
    if (cli.Has("release_ratio")) touch.releaseRatio = cli.GetFloat("release_ratio");
    if (cli.Has("kp_conf")) touch.keypointConfidence = cli.GetFloat("kp_conf");
    if (cli.Has("ball_conf")) touch.ballMinConfidence = cli.GetFloat("ball_conf");
    if (cli.Has("min_contact_frames")) {
        touch.minContactFrames = cli.GetInt("min_contact_frames");
    }
    if (cli.Has("max_gap_frames")) touch.maxGapFrames = cli.GetInt("max_gap_frames");
    if (cli.Has("min_split_frames")) {
        touch.minSplitFrames = cli.GetInt("min_split_frames");
    }
    if (cli.Has("min_rise_frames")) touch.minRiseFrames = cli.GetInt("min_rise_frames");
    if (cli.Has("refractory_frames")) {
        touch.refractoryFrames = cli.GetInt("refractory_frames");
    }
    if (cli.Has("min_speed_change")) {
        touch.minSpeedChangePxPerSec = cli.GetFloat("min_speed_change");
    }
    if (cli.Has("min_speed_ratio")) {
        touch.minSpeedRatio = cli.GetFloat("min_speed_ratio");
    }
    if (cli.Has("tracked_ball_only")) {
        touch.requireTrackedBall = cli.GetInt("tracked_ball_only") != 0;
    }
    return options;
}

void PrintTimerSnapshot() {
    auto stats = nexusflow::TimerRegistry::Instance().Snapshot();
    std::sort(stats.begin(), stats.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.totalMs > rhs.totalMs;
    });

    std::cout << "\nTimer profile (per item):\n";
    std::cout << std::left << std::setw(32) << "name"
              << std::right << std::setw(12) << "items"
              << std::setw(14) << "avg_ms"
              << std::setw(12) << "p95_ms"
              << std::setw(14) << "total_ms"
              << std::setw(12) << "min_ms"
              << std::setw(12) << "max_ms"
              << ' ' << std::left << "where" << '\n';
    std::cout << std::fixed << std::setprecision(3);
    for (const auto& item : stats) {
        std::cout << std::left << std::setw(32) << item.name
                  << std::right << std::setw(12) << item.items
                  << std::setw(14) << item.AvgItemMs()
                  << std::setw(12) << item.p95Ms
                  << std::setw(14) << item.totalMs
                  << std::setw(12) << item.minMs
                  << std::setw(12) << item.maxMs
                  << ' ' << std::left << item.Where() << '\n';
    }
}

/// Wall-clock summary. The per-module timers report service time; only this
/// number reflects the rate a consumer of the pipeline actually gets, so the
/// demo prints it even when the timer profile is off.
void PrintThroughput(
    const std::chrono::steady_clock::time_point& runStarted,
    const std::chrono::steady_clock::time_point& processingStarted,
    const std::chrono::steady_clock::time_point& processingEnded,
    int processedFrames, const AppResources& resources) {
    auto secondsBetween = [](const std::chrono::steady_clock::time_point& from,
                             const std::chrono::steady_clock::time_point& to) {
        return std::chrono::duration<double>(to - from).count();
    };
    const double startupSec = secondsBetween(runStarted, processingStarted);
    const double processingSec = secondsBetween(processingStarted, processingEnded);

    const double perFrameMs = processedFrames > 0
        ? processingSec * 1000.0 / static_cast<double>(processedFrames)
        : 0.0;
    // The renderer runs on the consumer thread inside this window. Its own time
    // is reported as consumer occupancy rather than subtracted from per frame:
    // the pipeline keeps working while it draws, so the subtraction would
    // understate the throughput the run actually achieved. Run without
    // --render for the pipeline-only figure.
    const double renderMs = resources.renderer.RenderMsPerFrame();
    const uint64_t renderedFrames = resources.renderer.RenderedFrames();

    std::cout << "\nThroughput:\n";
    std::cout << std::fixed << std::setprecision(3)
              << "  startup        : " << startupSec << " s\n"
              << "  processing     : " << processingSec << " s\n"
              << "  frames         : " << processedFrames << '\n'
              << "  effective rate : "
              << (processingSec > 0.0
                      ? static_cast<double>(processedFrames) / processingSec
                      : 0.0)
              << " FPS\n"
              << "  per frame      : " << perFrameMs << " ms\n";
    if (renderedFrames > 0) {
        std::cout << "  render         : " << renderMs << " ms/frame over "
                  << renderedFrames << " frames (consumer thread)\n";
    }
    std::cout << "  observations   : " << resources.observationsPath << '\n';
    if (resources.touches) {
        std::cout << "  touches        : " << resources.touches->OutputPath()
                  << " (" << resources.touches->EventCount() << " events)\n";
    }
}

bool PrepareResources(const RunOptions& options,
                      const aifootball_app::AppConfig& config,
                      AppResources& resources) {
    if (options.videoPath.empty()) {
        LOG_ERROR("No video path provided and graph.modules has no VideoReader path");
        return false;
    }
    if (!aifootball_app::EnsureDirectory(options.outputDir)) {
        LOG_ERROR("Failed to create output directory '{}'", options.outputDir);
        return false;
    }
    if (config.RgbRingSize() > 0) {
        resources.reader.SetRgbRingSize(
            static_cast<std::size_t>(config.RgbRingSize()));
    }
    if (!resources.reader.Open(options.videoPath)) {
        LOG_ERROR("Failed to open/decode video '{}'", options.videoPath);
        return false;
    }

    const std::string& cameraId = config.CameraId();
    resources.observationsPath =
        aifootball_app::JoinPath(options.outputDir, "observations.jsonl");
    if (!resources.writer.Open(resources.observationsPath, options.videoPath,
                               resources.reader.fps(), resources.reader.width(),
                               resources.reader.height(), options.stride,
                               cameraId)) {
        LOG_ERROR("Failed to open observations output '{}'",
                  resources.observationsPath);
        return false;
    }

    const std::string renderPath = options.renderPath.empty()
        ? aifootball_app::JoinPath(options.outputDir, "rendered.mp4")
        : options.renderPath;
    const double renderFps =
        resources.reader.fps() / static_cast<double>(options.stride);
    if (!resources.renderer.Open(renderPath, renderFps,
                                 resources.reader.width(),
                                 resources.reader.height(), options.render)) {
        LOG_ERROR("Failed to initialize video renderer '{}'", renderPath);
        return false;
    }
    // Same polygon the tracker filters against, so the overlay shows what the
    // pipeline actually kept.
    resources.renderer.SetRoi(config.Context().roi);

    if (!options.detectTouches) return true;
    resources.touches.reset(new aifootball_app::TouchRuntime(
        options.touchParams, resources.reader.width(), resources.reader.height()));
    const std::string touchPath = options.touchOutputPath.empty()
        ? aifootball_app::JoinPath(options.outputDir, "touches.jsonl")
        : options.touchOutputPath;
    if (!resources.touches->Open(touchPath, options.videoPath,
                                 resources.reader.fps(), cameraId)) {
        LOG_ERROR("Failed to open touch output '{}'", touchPath);
        return false;
    }
    return true;
}

using ResultFuture = aifootball::ProcessFuture;

bool HandleResult(aifootball::ProcessFutureResult value,
                  aifootball_app::TouchRuntime* touches,
                  aifootball_app::JsonlWriter& writer,
                  aifootball_app::VideoRenderer& renderer) {
    if (value.status != nexusflow::SUCCESS) {
        LOG_ERROR("AI-Football SDK failed for frame {} with status {}",
                  value.result.frameId, value.status);
        return false;
    }
    // Touch detection runs before the renderer so the overlay of this frame is
    // the state of this frame.
    if (touches) renderer.SetTouchOverlay(touches->Push(value.result));
    writer.Write(value.result);
    if (!renderer.Write(value.result)) {
        LOG_ERROR("Failed to render SDK result for frame {}",
                  value.result.frameId);
        return false;
    }
    return true;
}

bool HandleFutureResult(ResultFuture& future,
                        aifootball_app::TouchRuntime* touches,
                        aifootball_app::JsonlWriter& writer,
                        aifootball_app::VideoRenderer& renderer) {
    try {
        return HandleResult(future.Get(), touches, writer, renderer);
    } catch (const std::exception& error) {
        LOG_ERROR("Failed to consume AI-Football SDK future: {}", error.what());
        return false;
    }
}

bool WriteReadyResults(std::deque<ResultFuture>& pendingResults,
                       aifootball_app::TouchRuntime* touches,
                       aifootball_app::JsonlWriter& writer,
                       aifootball_app::VideoRenderer& renderer) {
    // Futures are consumed strictly in submission order, which is also the
    // frame order the touch aggregation relies on.
    while (!pendingResults.empty() && pendingResults.front().IsReady()) {
        if (!HandleFutureResult(pendingResults.front(), touches, writer,
                                renderer)) {
            return false;
        }
        pendingResults.pop_front();
    }
    return true;
}

bool DrainResults(std::deque<ResultFuture>& pendingResults,
                  aifootball_app::TouchRuntime* touches,
                  aifootball_app::JsonlWriter& writer,
                  aifootball_app::VideoRenderer& renderer) {
    while (!pendingResults.empty()) {
        pendingResults.front().Wait();
        if (!HandleFutureResult(pendingResults.front(), touches, writer,
                                renderer)) {
            return false;
        }
        pendingResults.pop_front();
    }
    return true;
}

bool ProcessVideo(const RunOptions& options, AppResources& resources,
                  aifootball::AIFootballPipeline& pipeline,
                  int& processedFrames) {
    std::deque<ResultFuture> pendingResults;
    bool stoppedByLimit = false;
    bool failed = false;
    // Producer-side budget: packet read + software decode + color conversion +
    // pipeline submit + JSONL write for results that are already ready. It is
    // measured between two callbacks, so it includes the decoder loop.
    std::chrono::steady_clock::time_point lastFrameEnd =
        std::chrono::steady_clock::now();
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
                        !HandleResult(std::move(value), resources.touches.get(),
                                      resources.writer, resources.renderer)) {
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
            if (!WriteReadyResults(pendingResults, resources.touches.get(),
                                   resources.writer, resources.renderer)) {
                failed = true;
                return false;
            }
            const auto frameEnd = std::chrono::steady_clock::now();
            const double frameMs = std::chrono::duration<double, std::milli>(
                frameEnd - lastFrameEnd).count();
            TIMER_ADD_ITEMS("App.ProduceInterval", frameMs, 1);
            lastFrameEnd = frameEnd;
            return true;
        });

    if (failed) {
        LOG_ERROR("AI-Football app processing failed for '{}'", options.videoPath);
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
    return DrainResults(pendingResults, resources.touches.get(),
                        resources.writer, resources.renderer);
}

int RunApp(const RunOptions& options, const aifootball_app::AppConfig& config) {
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    AppResources resources;
    if (!PrepareResources(options, config, resources)) return 2;

    const auto runStarted = std::chrono::steady_clock::now();

    try {
        aifootball::AIFootballContext context = config.Context();
        context.configPath = options.configPath;
        context.deviceId = options.deviceId;

        auto pipeline = aifootball::AIFootballPipeline::Create(context);
        if (!pipeline || pipeline->Init() != nexusflow::SUCCESS) {
            LOG_ERROR("AI-Football SDK initialization failed");
            return 2;
        }

        // Everything above is startup (video open, engine load, actor launch);
        // the wall clock below covers only frame processing.
        const auto processingStarted = std::chrono::steady_clock::now();
        int processedFrames = 0;
        if (!ProcessVideo(options, resources, *pipeline, processedFrames)) {
            pipeline->DeInit();
            return 2;
        }
        const auto processingEnded = std::chrono::steady_clock::now();
        if (pipeline->DeInit() != nexusflow::SUCCESS) {
            LOG_ERROR("Failed to deinitialize AI-Football SDK pipeline");
            return 2;
        }
        if (resources.touches) {
            // Close the last event: it only ends when the stream does.
            resources.touches->Flush();
            resources.touches->Close();
            LOG_INFO("Touch detection: {} event(s) written to '{}'", 
                     resources.touches->EventCount(),
                     resources.touches->OutputPath());
            LOG_INFO("Touch gates: {} suppressed by the refractory window, "
                     "{} refused by the speed gate, {} by tracked-ball-only",
                     resources.touches->SuppressedByRefractory(),
                     resources.touches->RejectedBySpeedGate(),
                     resources.touches->RejectedByBallGate());
        }
        if (options.profileTimers) PrintTimerSnapshot();
        PrintThroughput(runStarted, processingStarted, processingEnded,
                        processedFrames, resources);
        LOG_INFO("AI-Football SDK app processed {} frames, observations='{}'",
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
                    "output/sdk_app");
    cli.AddArgument("--max_frames", "-n", "Stop after N decoded frames (0 = all)",
                    false, "0");
    cli.AddArgument("--stride", "-S", "Process every Nth decoded frame", false,
                    "1");
    cli.AddArgument("--device", "-d", "CUDA device id", false, "0");
    cli.AddArgument("--render", "-r", "Render detections to an MP4", false,
                    app::CommandParser::FlagMarker());
    cli.AddArgument("--render_path", "-R", "Rendered MP4 path", false, "");
    cli.AddArgument("--detect_touches", "", "Detect ball touches and write touches.jsonl",
                    false, app::CommandParser::FlagMarker());
    cli.AddArgument("--touch_output", "", "Touch events JSONL path", false,
                    "");
    cli.AddArgument("--contact_ratio", "", "Touch radius as a fraction of min(width,height)",
                    false, "0.045");
    cli.AddArgument("--person_height_ratio", "", "Touch radius cap as a fraction of person height",
                    false, "0.06");
    cli.AddArgument("--min_contact_px", "", "Lower bound of the touch radius", false,
                    "18");
    cli.AddArgument("--max_contact_px", "", "Upper bound of the touch radius", false,
                    "64");
    cli.AddArgument("--release_ratio", "", "Hysteresis: exit = enter * ratio", false,
                    "1.4");
    cli.AddArgument("--kp_conf", "", "Minimum foot keypoint confidence", false, "0.2");
    cli.AddArgument("--ball_conf", "", "Minimum ball confidence", false, "0.15");
    cli.AddArgument("--min_contact_frames", "", "Contact frames needed for one touch",
                    false, "2");
    cli.AddArgument("--max_gap_frames", "", "Frames without evidence tolerated inside a touch",
                    false, "3");
    cli.AddArgument("--min_split_frames", "", "Frame gap needed to split two touches",
                    false, "12");
    cli.AddArgument("--min_rise_frames", "", "Frames above the exit threshold needed to split",
                    false, "2");
    cli.AddArgument("--refractory_frames", "",
                    "Same person and foot: touches closer than this are one touch",
                    false, "21");
    cli.AddArgument("--min_speed_change", "",
                    "Minimum ball speed change at the touch, px/s (0 = off)", false,
                    "0");
    cli.AddArgument("--min_speed_ratio", "",
                    "Minimum ratio between the two ball speeds (1 = off)", false,
                    "1");
    cli.AddArgument("--tracked_ball_only", "",
                    "Only accept touches of the tracked ball (1) or any candidate (0)",
                    false, "0");
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

    try {
        // Parse the configuration once and share it with every consumer.
        const aifootball_app::AppConfig config(cli.Get("config"));
        return RunApp(ReadOptions(cli, config), config);
    } catch (const std::exception& error) {
        LOG_ERROR("Fatal: {}", error.what());
        return 2;
    }
}
