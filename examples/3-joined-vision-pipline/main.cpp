#include "../src/utils/logging.hpp" // TODO: remove
#include <chrono>
#include <iostream>

#include "module/module.hpp"

#include <nexusflow/Nexusflow.hpp>
#include <thread>

using namespace nexusflow;

// --- Helper Functions ---

void waitFor(const std::chrono::seconds& duration) { std::this_thread::sleep_for(duration); }

void registerAllModules() {
    NEXUSFLOW_REGISTER_MODULE(MyStreamPullerModule);
    NEXUSFLOW_REGISTER_MODULE(MyDecoderModule);
    NEXUSFLOW_REGISTER_MODULE(MyHeadDetectorModule);
    NEXUSFLOW_REGISTER_MODULE(MyPersonDetectorModule);
    NEXUSFLOW_REGISTER_MODULE(MyHeadPersonFusionModule);
    NEXUSFLOW_REGISTER_MODULE(MyBehaviorAnalyzerModule);
    NEXUSFLOW_REGISTER_MODULE(MyAlarmPusherModule);
}

void executePipeline(Pipeline& pipeline) {
    LOG_INFO("Initializing pipeline...");
    if (pipeline.Init() != ErrorCode::SUCCESS) {
        throw std::runtime_error("Pipeline initialization failed.");
    }

    LOG_INFO("Pipeline starting...");
    pipeline.Start();

    LOG_INFO("Pipeline running for 10 seconds...");
    std::this_thread::sleep_for(std::chrono::seconds(10));

    LOG_INFO("Pipeline stopping...");
    pipeline.Stop();

    LOG_INFO("De-initializing pipeline...");
    pipeline.DeInit();
}

void runWithYamlConfig(const std::string& configPath) {
    LOG_INFO("--- Running in Declarative Mode (from YAML) ---");
    registerAllModules();
    auto pipeline = Pipeline::CreateFromYaml(configPath);
    if (pipeline == nullptr) {
        throw std::runtime_error("Failed to create pipeline from YAML config.");
    }
    executePipeline(*pipeline);
}

int main(int argc, char* argv[]) {
    utils::logger::LoggerParam params;
    // params.logLevel = utils::logger::LogLevel::DEBUG;
    params.logLevel = utils::logger::LogLevel::INFO;
    utils::logger::InitializeGlobalLogger(params);

    if (argc < 2) {
        LOG_ERROR("Usage: {} <config.yaml>");
        return -1;
    }

    auto yamlConfigPath = argv[1];
    runWithYamlConfig(yamlConfigPath);

    LOG_INFO("Execution finished successfully.");

    return 0;
}