#include "module/common/Module.hpp"
#include "module/common/PipelineCompletionSignal.hpp"
#include "nexusflow/Logging.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

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
// Helper Functions
// ---------------------------------------------------------------------------

void registerAllModules() {}

void executePipeline(Pipeline& pipeline, int maxSeconds) {
    LOG_INFO("Initializing pipeline...");
    if (pipeline.Init() != ErrorCode::SUCCESS) {
        throw std::runtime_error("Pipeline initialization failed.");
    }

    LOG_INFO("Pipeline starting...");
    pipeline.Start();

    // Register a callback so we can log the moment the terminal module fires.
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

void runWithYamlConfig(const std::string& configPath, int maxSeconds) {
    LOG_INFO("--- Running in Declarative Mode (from YAML) ---");
    registerAllModules();
    auto pipeline = Pipeline::CreateFromYaml(configPath);
    if (pipeline == nullptr) {
        throw std::runtime_error("Failed to create pipeline from YAML config.");
    }
    executePipeline(*pipeline, maxSeconds);
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

static void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <config.yaml> [--max-seconds N]\n"
              << "  --max-seconds N   Cap the run time (default: wait for EOF indefinitely)\n"
              << "  --help            Show this message\n";
}

int main(int argc, char* argv[]) {
    logger::LoggerParam params;
    params.logLevel = logger::LogLevel::INFO;
    logger::InitializeGlobalLogger(params);

    installSignalHandlers();

    if (argc < 2) {
        printUsage(argv[0]);
        return -1;
    }

    std::string yamlConfigPath;
    int maxSeconds = 0;  // 0 => wait forever

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--max-seconds" && i + 1 < argc) {
            maxSeconds = std::atoi(argv[++i]);
        } else if (arg.rfind("--max-seconds=", 0) == 0) {
            maxSeconds = std::atoi(arg.c_str() + std::strlen("--max-seconds="));
        } else if (!arg.empty() && arg[0] != '-') {
            yamlConfigPath = arg;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return -1;
        }
    }

    if (yamlConfigPath.empty()) {
        std::cerr << "Error: config.yaml path is required\n";
        printUsage(argv[0]);
        return -1;
    }

    try {
        runWithYamlConfig(yamlConfigPath, maxSeconds);
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal: {}", e.what());
        return -2;
    }

    LOG_INFO("Execution finished successfully.");
    return 0;
}
