#include <aifootball/AIFootball.hpp>

#include "common/Module.hpp"
#include "common/PipelineCompletionSignal.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/ModuleFactory.hpp>
#include <nexusflow/Pipeline.hpp>

#include <yaml-cpp/yaml.h>

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace aifootball {
namespace {

std::string JoinPath(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    if (dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

bool EnsureDirectory(const std::string& path) {
    if (path.empty()) return true;
    std::string current;
    size_t begin = 0;
    if (path[0] == '/') {
        current = "/";
        begin = 1;
    }
    while (begin < path.size()) {
        const size_t slash = path.find('/', begin);
        const size_t end = slash == std::string::npos ? path.size() : slash;
        if (end > begin) {
            if (!current.empty() && current.back() != '/') current.push_back('/');
            current.append(path, begin, end - begin);
            if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
        if (slash == std::string::npos) break;
        begin = slash + 1;
    }
    return true;
}

void RegisterBuiltInModules() {
    static std::once_flag once;
    std::call_once(once, [] {
        auto& factory = nexusflow::ModuleFactory::GetInstance();
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
    });
}

} // namespace

class Runtime::Impl {
public:
    explicit Impl(RuntimeOptions runtimeOptions)
        : options(std::move(runtimeOptions)) {}

    ~Impl() {
        if (pipeline && started) {
            pipeline->Stop();
            started = false;
        }
        if (pipeline && initialized) {
            pipeline->DeInit();
            initialized = false;
        }
        completion.SetOnComplete(nullptr);
        if (!generatedConfig.empty()) std::remove(generatedConfig.c_str());
    }

    bool PrepareConfig() {
        if (options.configPath.empty()) {
            LOG_ERROR("AI-Football SDK: configPath is empty");
            return false;
        }

        try {
            YAML::Node root = YAML::LoadFile(options.configPath);
            YAML::Node modules = root["graph"]["modules"];
            if (!modules || !modules.IsSequence()) {
                LOG_ERROR("AI-Football SDK: config has no graph.modules sequence");
                return false;
            }

            for (auto module : modules) {
                const std::string name = module["name"].as<std::string>("");
                YAML::Node config = module["config"];
                if (!config || !config.IsMap()) {
                    config = YAML::Node(YAML::NodeType::Map);
                    module["config"] = config;
                }
                if (name == "VideoReader") {
                    if (!options.videoPath.empty()) config["videoPath"] = options.videoPath;
                    if (options.stride > 0) config["stride"] = options.stride;
                } else if (name == "VideoRenderer") {
                    // Rendering is a debug side effect and is explicitly
                    // controlled by the SDK, independent of the YAML default.
                    config["enabled"] = options.enableRendering;
                    if (!options.outputDir.empty()) {
                        config["outputPath"] = JoinPath(options.outputDir, "rendered.mp4");
                    }
                } else if (name == "ObservationWriter") {
                    if (!options.videoPath.empty()) config["videoPath"] = options.videoPath;
                    if (!options.outputDir.empty()) {
                        config["outputPath"] = JoinPath(options.outputDir, "observations.jsonl");
                    }
                } else if (name == "AlarmPusher" && !options.outputDir.empty()) {
                    config["savePath"] = JoinPath(options.outputDir, "result.txt");
                }
            }

            if (!options.outputDir.empty() && !EnsureDirectory(options.outputDir)) {
                LOG_ERROR("AI-Football SDK: failed to create output directory '{}'",
                          options.outputDir);
                return false;
            }

            std::ostringstream path;
            path << "/tmp/nexusflow_aifootball_" << static_cast<long>(::getpid())
                 << "_" << reinterpret_cast<std::uintptr_t>(this) << ".yaml";
            generatedConfig = path.str();
            YAML::Emitter emitter;
            emitter << root;
            std::ofstream file(generatedConfig);
            if (!file.is_open()) {
                LOG_ERROR("AI-Football SDK: failed to create temporary config '{}'",
                          generatedConfig);
                return false;
            }
            file << emitter.c_str() << "\n";
            effectiveConfig = generatedConfig;
            return true;
        } catch (const YAML::Exception& e) {
            LOG_ERROR("AI-Football SDK: failed to prepare config '{}': {}",
                      options.configPath, e.what());
            return false;
        }
    }

    RuntimeOptions options;
    std::unique_ptr<nexusflow::Pipeline> pipeline;
    std::string effectiveConfig;
    std::string generatedConfig;
    bool initialized = false;
    bool started = false;
    PipelineCompletionSignal& completion = PipelineCompletionSignal::Instance();
};

Runtime::Runtime(RuntimeOptions options)
    : m_impl(std::make_unique<Impl>(std::move(options))) {}

Runtime::~Runtime() = default;

std::unique_ptr<Runtime> Runtime::Create(const RuntimeOptions& options) {
    return std::unique_ptr<Runtime>(new Runtime(options));
}

nexusflow::ErrorCode Runtime::Init() {
    if (!m_impl || m_impl->initialized) return nexusflow::SUCCESS;
    if (!m_impl->PrepareConfig()) return nexusflow::FAILURE;

#ifdef WITH_CUDA
    if (m_impl->options.deviceId < 0 ||
        cudaSetDevice(m_impl->options.deviceId) != cudaSuccess) {
        LOG_ERROR("AI-Football SDK: failed to select CUDA device {}",
                  m_impl->options.deviceId);
        return nexusflow::FAILURE;
    }
#endif

    RegisterBuiltInModules();
    m_impl->completion.Reset();
    m_impl->completion.SetOnComplete([] {
        LOG_INFO("AI-Football SDK: pipeline completed");
    });
    m_impl->pipeline = nexusflow::Pipeline::CreateFromYaml(m_impl->effectiveConfig);
    if (!m_impl->pipeline) {
        LOG_ERROR("AI-Football SDK: failed to create pipeline from '{}'",
                  m_impl->effectiveConfig);
        return nexusflow::FAILURE;
    }
    m_impl->initialized = true;
    return nexusflow::SUCCESS;
}

nexusflow::ErrorCode Runtime::Start() {
    if (!m_impl || !m_impl->pipeline || !m_impl->initialized) {
        return nexusflow::UNINITIALIZED_ERROR;
    }
    if (m_impl->started) return nexusflow::FAILED_ALREADY_START;
    const auto result = m_impl->pipeline->Init();
    if (result != nexusflow::SUCCESS) return result;
    const auto startResult = m_impl->pipeline->Start();
    if (startResult == nexusflow::SUCCESS) m_impl->started = true;
    return startResult;
}

nexusflow::ErrorCode Runtime::Wait() {
    if (!m_impl || !m_impl->started) return nexusflow::UNINITIALIZED_ERROR;
    if (m_impl->options.maxSeconds > 0) {
        const bool completed = m_impl->completion.WaitForCompletion(
            std::chrono::seconds(m_impl->options.maxSeconds));
        if (!completed) {
            LOG_WARN("AI-Football SDK: timeout after {} seconds",
                     m_impl->options.maxSeconds);
        }
    } else {
        m_impl->completion.WaitForCompletion(std::chrono::seconds(0));
    }
    return nexusflow::SUCCESS;
}

nexusflow::ErrorCode Runtime::Stop() {
    if (!m_impl || !m_impl->pipeline || !m_impl->started) return nexusflow::SUCCESS;
    const auto result = m_impl->pipeline->Stop();
    m_impl->started = false;
    return result;
}

nexusflow::ErrorCode Runtime::DeInit() {
    if (!m_impl || !m_impl->pipeline || !m_impl->initialized) return nexusflow::SUCCESS;
    const auto result = m_impl->pipeline->DeInit();
    m_impl->initialized = false;
    m_impl->completion.SetOnComplete(nullptr);
    return result;
}

nexusflow::ErrorCode Runtime::Run() {
    auto result = Init();
    if (result != nexusflow::SUCCESS) return result;
    result = Start();
    if (result != nexusflow::SUCCESS) {
        DeInit();
        return result;
    }
    result = Wait();
    const auto stopResult = Stop();
    const auto deinitResult = DeInit();
    if (result != nexusflow::SUCCESS) return result;
    if (stopResult != nexusflow::SUCCESS) return stopResult;
    return deinitResult;
}

void Runtime::RequestStop() {
    if (m_impl) m_impl->completion.NotifyComplete();
}

} // namespace aifootball
