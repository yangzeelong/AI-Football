#include "CommandParser.hpp"
#include <aifootball/AIFootball.hpp>
#include "nexusflow/Logging.hpp"

#include <atomic>
#include <csignal>
#include <string>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace {

aifootball::AlgoConfig LoadAlgoConfig(const std::string& configPath) {
    aifootball::AlgoConfig algoConfig;
    const YAML::Node root = YAML::LoadFile(configPath);
    const YAML::Node roi = root["algorithm"]["roi"];
    if (!roi) return algoConfig;

    if (roi["enabled"]) algoConfig.roi.enabled = roi["enabled"].as<bool>();
    if (roi["width"]) algoConfig.roi.width = roi["width"].as<int>();
    if (roi["height"]) algoConfig.roi.height = roi["height"].as<int>();

    const YAML::Node points = roi["points"];
    if (points) {
        if (!points.IsSequence()) {
            throw std::runtime_error("algorithm.roi.points must be a sequence");
        }
        for (const auto& point : points) {
            if (!point.IsSequence() || point.size() < 2) {
                throw std::runtime_error("each algorithm.roi point must be [x, y]");
            }
            aifootball::RoiPoint parsed;
            parsed.x = point[0].as<float>();
            parsed.y = point[1].as<float>();
            algoConfig.roi.points.push_back(parsed);
        }
    }

    if (algoConfig.roi.enabled && algoConfig.roi.points.size() < 3) {
        throw std::runtime_error(
            "algorithm.roi.enabled requires at least 3 points");
    }
    return algoConfig;
}

} // namespace

// ---------------------------------------------------------------------------
// Graceful shutdown on SIGINT / SIGTERM
// ---------------------------------------------------------------------------

static std::atomic<aifootball::Runtime*> g_runtime{nullptr};

static void handleSignal(int sig) {
    (void)sig;
    auto* runtime = g_runtime.load(std::memory_order_relaxed);
    if (runtime) runtime->RequestStop();
}

static void installSignalHandlers() {
    std::signal(SIGINT,  handleSignal);
    std::signal(SIGTERM, handleSignal);
}

// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // --- CLI (Python argparse style) ---
    app::CommandParser cli(argc, argv);
    cli.AddPositional("config", "Path to config.yaml", true);
    cli.AddArgument("--video_path",   "-v", "Override video path from config");
    cli.AddArgument("--output_dir",   "-o", "Override output directory");
    cli.AddArgument("--max_seconds",  "-t", "Cap run time in seconds (0 = wait for EOF)", false, "0");
    cli.AddArgument("--stride",       "-S", "Process every Nth frame (1 = all frames)", false, "1");
    cli.AddArgument("--device",       "-d", "CUDA device id", false, "0");
    cli.AddArgument("--render",       "",  "Enable debug video rendering", false,
                    app::CommandParser::FlagMarker());
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
    nexusflow::logger::LoggerParam logParams;
    if (cli.IsFlagSet("verbose"))     logParams.logLevel = nexusflow::logger::LogLevel::DEBUG;
    else if (cli.IsFlagSet("quiet"))  logParams.logLevel = nexusflow::logger::LogLevel::WARN;
    else                              logParams.logLevel = nexusflow::logger::LogLevel::INFO;
    nexusflow::logger::InitializeGlobalLogger(logParams);

    installSignalHandlers();

    // --- Extract params ---
    std::string configPath = cli.Get("config");
    int maxSeconds  = cli.GetInt("max_seconds");
    int stride      = cli.GetInt("stride");
    int deviceId    = cli.GetInt("device");
    std::string videoPath = cli.Get("video_path");
    std::string outputDir = cli.Get("output_dir");
    const bool enableRendering = cli.IsFlagSet("render");

    LOG_INFO("=== AI-Football SDK demo ===");
    LOG_INFO("  config:       {}", configPath);
    if (!videoPath.empty())  LOG_INFO("  video_path:   {}", videoPath);
    if (!outputDir.empty())  LOG_INFO("  output_dir:   {}", outputDir);
    if (maxSeconds > 0)      LOG_INFO("  max_seconds:  {}", maxSeconds);
    if (stride > 1)          LOG_INFO("  stride:       {}", stride);
    LOG_INFO("  device:       cuda:{}", deviceId);
    LOG_INFO("  rendering:    {}", enableRendering ? "enabled" : "disabled");

    // --- Run through the public SDK facade ---
    try {
        aifootball::RuntimeOptions options;
        options.configPath = configPath;
        options.videoPath = videoPath;
        options.outputDir = outputDir;
        options.enableRendering = enableRendering;
        options.deviceId = deviceId;
        options.maxSeconds = maxSeconds;
        options.stride = stride > 0 ? stride : 0;
        const auto algoConfig = LoadAlgoConfig(configPath);

        auto runtime = aifootball::Runtime::Create(options, algoConfig);
        g_runtime.store(runtime.get(), std::memory_order_relaxed);
        const auto result = runtime->Run();
        g_runtime.store(nullptr, std::memory_order_relaxed);
        if (result != nexusflow::SUCCESS) {
            LOG_ERROR("AI-Football SDK demo failed, error={}", result);
            return 2;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal: {}", e.what());
        g_runtime.store(nullptr, std::memory_order_relaxed);
        return 2;
    }

    LOG_INFO("AI-Football SDK demo finished successfully.");
    return 0;
}
