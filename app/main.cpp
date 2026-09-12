#include "module/common/CommandParser.hpp"
#include "module/common/Module.hpp"
#include "module/common/PipelineCompletionSignal.hpp"
#include "nexusflow/Logging.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <string>

using namespace nexusflow;

// ---------------------------------------------------------------------------
// Graceful shutdown on SIGINT / SIGTERM
// ---------------------------------------------------------------------------

static std::atomic<bool> g_stopRequested{false};

static void handleSignal(int sig) {
    (void)sig;
    g_stopRequested = true;
    PipelineCompletionSignal::Instance().NotifyComplete();
}

static void installSignalHandlers() {
    std::signal(SIGINT,  handleSignal);
    std::signal(SIGTERM, handleSignal);
}

// ---------------------------------------------------------------------------
// Pipeline execution
// ---------------------------------------------------------------------------

void registerAllModules() {
    // Register explicitly from the executable. Header-level static
    // registration is convenient for examples, but can be discarded or
    // reordered when application modules are linked into a large target.
    auto& factory = ModuleFactory::GetInstance();
#ifdef WITH_FFMPEG
    factory.Register<VideoReader>("VideoReader");
    factory.Register<VideoDecoder>("VideoDecoder");
#endif
    factory.Register<RFDetrDetector>("RFDetrDetector");
    factory.Register<ByteTracker>("ByteTracker");
    factory.Register<HRNetPoseEstimator>("HRNetPoseEstimator");
    factory.Register<KeypointSmoother>("KeypointSmoother");
    factory.Register<FootballTracker>("FootballTracker");
    factory.Register<VideoRenderer>("VideoRenderer");
    factory.Register<ObservationWriter>("ObservationWriter");
    factory.Register<AlarmPusher>("AlarmPusher");
}

void executePipeline(Pipeline& pipeline, int maxSeconds) {
    LOG_INFO("Initializing pipeline...");
    if (pipeline.Init() != ErrorCode::SUCCESS) {
        throw std::runtime_error("Pipeline initialization failed.");
    }

    LOG_INFO("Pipeline starting...");
    pipeline.Start();

    PipelineCompletionSignal::Instance().SetOnComplete([] {
        LOG_INFO("Pipeline completion signal received (EOF or SIGINT)");
    });

    if (maxSeconds > 0) {
        LOG_INFO("Pipeline running (max {} seconds, or until EOF)...", maxSeconds);
        bool done = PipelineCompletionSignal::Instance()
                        .WaitForCompletion(std::chrono::seconds(maxSeconds));
        if (!done) {
            LOG_WARN("Pipeline did not reach EOF within {} seconds; stopping anyway",
                     maxSeconds);
        }
    } else {
        LOG_INFO("Pipeline running until EOF (no timeout)...");
        PipelineCompletionSignal::Instance().WaitForCompletion(std::chrono::seconds(0));
    }

    LOG_INFO("Pipeline stopping...");
    pipeline.Stop();

    LOG_INFO("De-initializing pipeline...");
    pipeline.DeInit();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // --- CLI (Python argparse style) ---
    app::CommandParser cli(argc, argv);
    cli.AddPositional("config", "Path to config.yaml", true);
    cli.AddArgument("--video_path",   "-v", "Override video path from config");
    cli.AddArgument("--output_dir",   "-o", "Override output directory");
    cli.AddArgument("--stop_frame",   "-s", "Stop after N frames (0 = no limit)", false, "0");
    cli.AddArgument("--max_seconds",  "-t", "Cap run time in seconds (0 = wait for EOF)", false, "0");
    cli.AddArgument("--stride",       "-S", "Process every Nth frame (1 = all frames)", false, "1");
    cli.AddArgument("--target_fps",   "-f", "Target processing FPS (0 = native speed)", false, "0");
    cli.AddArgument("--device",       "-d", "CUDA device id", false, "0");
    cli.AddArgument("--verbose",      "-V", "Enable DEBUG-level logging", false, app::CommandParser::FlagMarker());
    cli.AddArgument("--quiet",        "-q", "Suppress INFO-level logging", false, app::CommandParser::FlagMarker());

    if (!cli.Parse()) {
        cli.PrintHelp();
        return -1;
    }
    if (cli.IsHelpRequested()) {
        cli.PrintHelp();
        return 0;
    }

    // --- Logger setup ---
    logger::LoggerParam logParams;
    if (cli.IsFlagSet("verbose"))     logParams.logLevel = logger::LogLevel::DEBUG;
    else if (cli.IsFlagSet("quiet"))  logParams.logLevel = logger::LogLevel::WARN;
    else                              logParams.logLevel = logger::LogLevel::INFO;
    logger::InitializeGlobalLogger(logParams);

    installSignalHandlers();

    // --- Extract params ---
    std::string configPath = cli.Get("config");
    int maxSeconds  = cli.GetInt("max_seconds");
    int stopFrame   = cli.GetInt("stop_frame");
    int stride      = cli.GetInt("stride");
    int targetFps   = cli.GetInt("target_fps");
    int deviceId    = cli.GetInt("device");
    std::string videoPath = cli.Get("video_path");
    std::string outputDir = cli.Get("output_dir");

    LOG_INFO("=== AI-Football Pipeline ===");
    LOG_INFO("  config:       {}", configPath);
    if (!videoPath.empty())  LOG_INFO("  video_path:   {}", videoPath);
    if (!outputDir.empty())  LOG_INFO("  output_dir:   {}", outputDir);
    if (stopFrame > 0)       LOG_INFO("  stop_frame:   {}", stopFrame);
    if (maxSeconds > 0)      LOG_INFO("  max_seconds:  {}", maxSeconds);
    if (stride > 1)          LOG_INFO("  stride:       {}", stride);
    if (targetFps > 0)       LOG_INFO("  target_fps:   {}", targetFps);
    LOG_INFO("  device:       cuda:{}", deviceId);

    // --- Run pipeline ---
    try {
        registerAllModules();
        auto pipeline = Pipeline::CreateFromYaml(configPath);
        if (pipeline == nullptr) {
            throw std::runtime_error("Failed to create pipeline from YAML config.");
        }
        executePipeline(*pipeline, maxSeconds);
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal: {}", e.what());
        return -2;
    }

    LOG_INFO("Execution finished successfully.");
    return 0;
}
