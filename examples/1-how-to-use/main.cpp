#include "../src/utils/logging.hpp" // TODO: remove
#include "my_module/MockInputModule.hpp"
#include "my_module/MockOutputModule.hpp"
#include "my_module/MockProcessModule.hpp"
#include "nexusflow/Module.hpp"
#include "nexusflow/ModuleFactory.hpp"
#include <chrono>
#include <iostream>

#include <nexusflow/Nexusflow.hpp>
#include <thread>

using namespace nexusflow;

// --- Helper Functions ---

void waitFor(const std::chrono::seconds& duration) { std::this_thread::sleep_for(duration); }

void registerAllModules() {
    NEXUSFLOW_REGISTER_MODULE(MockInputModule);
    NEXUSFLOW_REGISTER_MODULE(MockProcessModule);
    NEXUSFLOW_REGISTER_MODULE(MockOutputModule);
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

// --- Main Logic for two modes ---

void runWithBuildModule() {
    LOG_INFO("--- Running in Programmatic Mode (PipelineBuilder) ---");
    auto inputModule = std::make_shared<MockInputModule>("InputNode");
    auto process1Module = std::make_shared<MockProcessModule>("ProcessNode1");
    auto process2Module = std::make_shared<MockProcessModule>("ProcessNode2");
    auto outputModule = std::make_shared<MockOutputModule>("OutputNode");

    auto pipeline = PipelineBuilder()
                        .AddModule(inputModule)
                        .AddModule(process1Module)
                        .AddModule(process2Module)
                        .AddModule(outputModule)
                        .Connect("InputNode", "ProcessNode1")
                        .Connect("InputNode", "ProcessNode2")
                        .Connect("ProcessNode1", "OutputNode")
                        .Connect("ProcessNode2", "OutputNode")
                        .Build();

    if (pipeline == nullptr) {
        throw std::runtime_error("Failed to build pipeline.");
    }

    executePipeline(*pipeline);
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

    try {
        if (argc < 2) {
            runWithBuildModule();
        } else {
            runWithYamlConfig(argv[1]);
        }
    } catch (const std::exception& e) {
        LOG_CRITICAL("An exception occurred: {}", e.what());
        return 1;
    }

    LOG_INFO("Execution finished successfully.");

    return 0;
}