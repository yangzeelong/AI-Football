#include <aifootball/AIFootball.hpp>

#include "common/Module.hpp"
#include "common/MyMessage.hpp"
#include "runtime/Source.hpp"
#include "runtime/Sink.hpp"

#include "base/GraphUtils.hpp"
#include <nexusflow/Logging.hpp>
#include <nexusflow/ModuleFactory.hpp>
#include <nexusflow/Pipeline.hpp>
#include <nexusflow/PipelineBuilder.hpp>

#include <yaml-cpp/yaml.h>

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aifootball {
namespace {

constexpr const char* kInputModuleName = "SdkInput";
constexpr const char* kOutputModuleName = "SdkOutput";

bool IsOfflineModule(const std::string& className) {
    return className == "VideoReader" || className == "VideoDecoder" ||
           className == "VideoRenderer" || className == "ObservationWriter" ||
           className == "AlarmPusher";
}

void RegisterBuiltInModules() {
    static std::once_flag once;
    std::call_once(once, [] {
        auto& factory = nexusflow::ModuleFactory::GetInstance();
        factory.Register<RFDetrDetector>("RFDetrDetector");
        factory.Register<ByteTracker>("ByteTracker");
        factory.Register<HRNetPoseEstimator>("HRNetPoseEstimator");
        factory.Register<KeypointSmoother>("KeypointSmoother");
        factory.Register<FootballTracker>("FootballTracker");
    });
}

struct ModuleSpec {
    std::string name;
    std::string className;
    nexusflow::Config config;
};

std::vector<ModuleSpec> LoadAlgorithmSpecs(const std::string& configPath,
                                           const AIFootballContext& context) {
    YAML::Node root = YAML::LoadFile(configPath);
    const YAML::Node modules = root["graph"]["modules"];
    if (!modules || !modules.IsSequence()) {
        throw std::runtime_error("config must contain graph.modules sequence");
    }

    std::vector<ModuleSpec> specs;
    for (const auto& module : modules) {
        const std::string name = module["name"].as<std::string>("");
        const std::string className = module["class"].as<std::string>("");
        if (name.empty() || className.empty()) {
            throw std::runtime_error("every graph module requires name and class");
        }
        if (IsOfflineModule(className)) continue;

        ModuleSpec spec{name, className, nexusflow::Config()};
        const YAML::Node config = module["config"];
        if (config && config.IsMap()) {
            for (const auto& entry : config) {
                spec.config.Add(entry.first.as<std::string>(),
                                graphutils::convertYamlNodeToAny(entry.second));
            }
        }

        if (name == "ByteTracker" || name == "PersonTracker") {
            spec.config.Add("roiEnabled", context.roi.enabled);
            spec.config.Add("roiWidth", context.roi.width);
            spec.config.Add("roiHeight", context.roi.height);
            std::vector<nexusflow::Any> points;
            points.reserve(context.roi.points.size());
            for (const auto& point : context.roi.points) {
                std::vector<nexusflow::Any> pair;
                pair.emplace_back(point.x);
                pair.emplace_back(point.y);
                points.emplace_back(std::move(pair));
            }
            spec.config.Add("roiPoints", std::move(points));
        }

        specs.push_back(std::move(spec));
    }

    if (specs.empty()) {
        throw std::runtime_error("config contains no algorithm modules");
    }
    return specs;
}

KeypointState ConvertKeypointState(::KeypointState state) {
    switch (state) {
        case ::KeypointState::Observed: return KeypointState::Observed;
        case ::KeypointState::Virtual: return KeypointState::Virtual;
        case ::KeypointState::Missing: return KeypointState::Missing;
    }
    return KeypointState::Missing;
}

BallTrackState ConvertBallState(::BallTrackState state) {
    switch (state) {
        case ::BallTrackState::Observed: return BallTrackState::Observed;
        case ::BallTrackState::Predicted: return BallTrackState::Predicted;
        case ::BallTrackState::Lost: return BallTrackState::Lost;
    }
    return BallTrackState::Lost;
}

ProcessResult ConvertResult(const ResultPacket& packet) {
    const auto& input = packet.message;
    ProcessResult output;
    output.frameId = input.videoFrame ? input.videoFrame->frameId : 0;
    output.timestampSec = input.timestampSec;
    output.rawCounts.total = input.rawCounts.total;
    output.rawCounts.person = input.rawCounts.person;
    output.rawCounts.ball = input.rawCounts.ball;
    output.moduleTimingsMs = packet.moduleTimingsMs;
    // SdkInput measures the framework worker wait, not algorithm work. Keep
    // only timings belonging to the public algorithm modules.
    output.moduleTimingsMs.erase(kInputModuleName);

    output.persons.reserve(input.persons.size());
    for (const auto& person : input.persons) {
        PersonResult result;
        result.trackId = person.trackId;
        result.bbox = {{person.x0, person.y0, person.x1, person.y1}};
        result.confidence = person.detectionConfidence;
        for (int i = 0; i < kProjectKeypointCount; ++i) {
            result.keypoints[static_cast<std::size_t>(i)] = {
                person.keypoints[i].x,
                person.keypoints[i].y,
                person.keypoints[i].confidence,
                ConvertKeypointState(person.keypoints[i].state),
            };
        }
        output.persons.push_back(std::move(result));
    }

    output.balls.reserve(input.balls.size());
    for (const auto& ball : input.balls) {
        BallResult result;
        result.trackId = ball.trackId;
        result.bbox = {{ball.x0, ball.y0, ball.x1, ball.y1}};
        result.center = {{ball.centerX, ball.centerY}};
        result.confidence = ball.confidence;
        result.state = ConvertBallState(ball.state);
        output.balls.push_back(std::move(result));
    }
    return output;
}

ProcessFuture ReadyFuture(nexusflow::ErrorCode status) {
    Promise<ProcessFutureResult> promise;
    auto future = promise.GetFuture();
    ProcessFutureResult result;
    result.status = status;
    promise.SetValue(std::move(result));
    return future;
}

bool IsValidFrame(const DecodedFrameView& frame) {
    return frame.frameId <= std::numeric_limits<uint32_t>::max() &&
           frame.pixelFormat == PixelFormat::RGB24 && frame.width > 0 &&
           frame.height > 0 && frame.data != nullptr && frame.dataBytes != 0 &&
           (frame.strideBytes == 0 || frame.strideBytes == frame.width * 3) &&
           frame.dataBytes >= static_cast<std::size_t>(frame.width) *
                                  static_cast<std::size_t>(frame.height) * 3;
}

uint64_t FrameIdOf(const FrameMessage& message) {
    return message.videoFrame ? message.videoFrame->frameId : 0;
}

} // namespace

class AIFootballPipeline::Impl {
public:
    explicit Impl(AIFootballContext runtimeContext)
        : context(std::move(runtimeContext)) {}

    ~Impl() {
        if (initialized || pipeline || frameSource || resultSink) DeInit();
    }

    nexusflow::ErrorCode DeInit() {
        if (!initialized && !pipeline && !frameSource && !resultSink) {
            return nexusflow::SUCCESS;
        }
        if (frameSource) frameSource->Close();
        if (resultSink) resultSink->Close();

        nexusflow::ErrorCode result = nexusflow::SUCCESS;
        if (pipeline) {
            const auto stopResult = pipeline->Stop();
            if (stopResult != nexusflow::SUCCESS) result = stopResult;
            const auto deinitResult = pipeline->DeInit();
            if (deinitResult != nexusflow::SUCCESS) result = deinitResult;
        }
        pipeline.reset();
        frameSource.reset();
        resultSink.reset();
        initialized = false;
        CompleteAllPending(nexusflow::FAILURE);
        return result;
    }

    AIFootballContext context;
    std::unique_ptr<nexusflow::Pipeline> pipeline;
    std::shared_ptr<Source> frameSource;
    std::shared_ptr<Sink> resultSink;
    // Serializes submission and Drain so frame order matches caller order.
    std::mutex processMutex;
    std::mutex promiseMutex;
    std::unordered_map<uint64_t, Promise<ProcessFutureResult>> pendingPromises;
    bool initialized = false;

    void ReservePendingPromises() {
        const std::size_t reserveCount = context.maxPendingFrames == 0
            ? 64 : std::max<std::size_t>(context.maxPendingFrames, 64);
        pendingPromises.reserve(reserveCount);
    }

    bool RegisterPending(uint64_t frameId,
                         ProcessFuture& future) {
        Promise<ProcessFutureResult> promise;
        future = promise.GetFuture();
        std::lock_guard<std::mutex> lock(promiseMutex);
        const auto inserted = pendingPromises.emplace(frameId, std::move(promise));
        return inserted.second;
    }

    void CompletePending(uint64_t frameId, nexusflow::ErrorCode status,
                         ProcessResult result = ProcessResult{}) {
        Promise<ProcessFutureResult> promise;
        {
            std::lock_guard<std::mutex> lock(promiseMutex);
            const auto it = pendingPromises.find(frameId);
            if (it == pendingPromises.end()) {
                LOG_WARN("AI-Football SDK: no pending future for frame {}", frameId);
                return;
            }
            promise = std::move(it->second);
            pendingPromises.erase(it);
        }
        ProcessFutureResult value;
        value.status = status;
        value.result = std::move(result);
        promise.TrySetValue(std::move(value));
    }

    void CompleteAllPending(nexusflow::ErrorCode status) {
        std::vector<Promise<ProcessFutureResult>> promises;
        {
            std::lock_guard<std::mutex> lock(promiseMutex);
            promises.reserve(pendingPromises.size());
            for (auto& entry : pendingPromises) {
                promises.push_back(std::move(entry.second));
            }
            pendingPromises.clear();
        }
        for (auto& promise : promises) {
            ProcessFutureResult value;
            value.status = status;
            promise.TrySetValue(std::move(value));
        }
    }

    void Deliver(ResultPacket packet) {
        ProcessResult result = ConvertResult(packet);
        const uint64_t frameId = result.frameId;
        CompletePending(frameId, nexusflow::SUCCESS, std::move(result));
    }
};

AIFootballPipeline::AIFootballPipeline(AIFootballContext context)
    : m_impl(std::make_unique<Impl>(std::move(context))) {}

AIFootballPipeline::~AIFootballPipeline() = default;

std::unique_ptr<AIFootballPipeline> AIFootballPipeline::Create(
    const AIFootballContext& context) {
    return std::unique_ptr<AIFootballPipeline>(
        new AIFootballPipeline(context));
}

nexusflow::ErrorCode AIFootballPipeline::Init() {
    if (!m_impl) return nexusflow::UNINITIALIZED_ERROR;
    if (m_impl->initialized) return nexusflow::SUCCESS;
    if (m_impl->context.configPath.empty()) {
        LOG_ERROR("AI-Football SDK: configPath is empty");
        return nexusflow::FAILURE;
    }

#ifdef WITH_CUDA
    if (m_impl->context.deviceId < 0 ||
        cudaSetDevice(m_impl->context.deviceId) != cudaSuccess) {
        LOG_ERROR("AI-Football SDK: failed to select CUDA device {}",
                  m_impl->context.deviceId);
        return nexusflow::FAILURE;
    }
#endif

    try {
        RegisterBuiltInModules();
        const auto specs = LoadAlgorithmSpecs(m_impl->context.configPath,
                                              m_impl->context);

        m_impl->frameSource = std::make_shared<Source>(
            kInputModuleName, m_impl->context.maxPendingFrames,
            m_impl->context.inputQueuePolicy);
        m_impl->resultSink = std::make_shared<Sink>(kOutputModuleName);
        m_impl->ReservePendingPromises();
        m_impl->resultSink->SetResultHandler([impl = m_impl.get()](ResultPacket packet) {
            impl->Deliver(std::move(packet));
        });

        nexusflow::PipelineBuilder builder;
        builder.AddModule(m_impl->frameSource);
        std::string previous = kInputModuleName;
        for (const auto& spec : specs) {
            auto module = nexusflow::ModuleFactory::GetInstance().CreateModule(
                spec.className, spec.name, spec.config);
            if (!module) {
                LOG_ERROR("AI-Football SDK: failed to create module '{}' ({})",
                          spec.name, spec.className);
                return nexusflow::FAILURE;
            }
            builder.AddModule(module).Connect(previous, spec.name);
            previous = spec.name;
        }
        builder.AddModule(m_impl->resultSink).Connect(previous, kOutputModuleName);

        m_impl->pipeline = builder.Build();
        if (!m_impl->pipeline) {
            LOG_ERROR("AI-Football SDK: failed to build algorithm pipeline");
            return nexusflow::FAILURE;
        }
        auto result = m_impl->pipeline->Init();
        if (result != nexusflow::SUCCESS) return result;
        result = m_impl->pipeline->Start();
        if (result != nexusflow::SUCCESS) {
            m_impl->pipeline->DeInit();
            m_impl->pipeline.reset();
            return result;
        }
        m_impl->initialized = true;
        LOG_INFO("AI-Football SDK: initialized algorithm-only pipeline, roi={}, modules={}",
                 m_impl->context.roi.enabled, specs.size());
        return nexusflow::SUCCESS;
    } catch (const YAML::Exception& error) {
        LOG_ERROR("AI-Football SDK: failed to load config '{}': {}",
                  m_impl->context.configPath, error.what());
    } catch (const std::exception& error) {
        LOG_ERROR("AI-Football SDK: initialization failed: {}", error.what());
    }
    return nexusflow::FAILURE;
}

ProcessFuture AIFootballPipeline::ProcessAsync(
    const DecodedFrameView& frame) {
    if (!m_impl || !m_impl->initialized || !m_impl->frameSource ||
        !m_impl->resultSink) {
        return ReadyFuture(nexusflow::UNINITIALIZED_ERROR);
    }
    if (!IsValidFrame(frame)) {
        LOG_ERROR("AI-Football SDK: invalid RGB24 frame id={}, size={}x{}, stride={}, bytes={}",
                  frame.frameId, frame.width, frame.height,
                  frame.strideBytes, frame.dataBytes);
        return ReadyFuture(nexusflow::FAILURE);
    }

    std::lock_guard<std::mutex> processLock(m_impl->processMutex);
    ProcessFuture future;
    if (!m_impl->RegisterPending(frame.frameId, future)) {
        LOG_ERROR("AI-Football SDK: duplicate pending frame id={}", frame.frameId);
        return ReadyFuture(nexusflow::FAILURE);
    }

    auto videoFrame = std::make_shared<VideoFrame>();
    videoFrame->frameId = static_cast<uint32_t>(frame.frameId);
    if (frame.dataOwner) {
        videoFrame->externalData = frame.data;
        videoFrame->externalDataBytes = frame.dataBytes;
        videoFrame->externalOwner = frame.dataOwner;
    } else {
        videoFrame->frameData.assign(
            reinterpret_cast<const char*>(frame.data), frame.dataBytes);
    }
    videoFrame->width = frame.width;
    videoFrame->height = frame.height;
    videoFrame->channels = 3;

    FrameMessage message;
    message.videoFrame = std::move(videoFrame);
    message.timestampSec = frame.timestampSec;
    FrameMessage droppedMessage;
    if (!m_impl->frameSource->Submit(std::move(message), &droppedMessage)) {
        m_impl->CompletePending(frame.frameId, nexusflow::FAILURE);
        return future;
    }
    if (droppedMessage.videoFrame) {
        m_impl->CompletePending(FrameIdOf(droppedMessage), nexusflow::FAILURE);
    }
    return future;
}

nexusflow::ErrorCode AIFootballPipeline::Drain() {
    if (!m_impl || !m_impl->initialized || !m_impl->frameSource ||
        !m_impl->resultSink) {
        return nexusflow::UNINITIALIZED_ERROR;
    }

    std::lock_guard<std::mutex> processLock(m_impl->processMutex);
    m_impl->resultSink->PrepareForEnd();
    FrameMessage endMessage;
    endMessage.isEnd = true;
    if (!m_impl->frameSource->Submit(std::move(endMessage))) {
        m_impl->CompleteAllPending(nexusflow::FAILURE);
        return nexusflow::FAILURE;
    }
    if (!m_impl->resultSink->WaitForEnd(std::chrono::minutes(5))) {
        m_impl->CompleteAllPending(nexusflow::FAILURE);
        return nexusflow::FAILURE;
    }
    return nexusflow::SUCCESS;
}

nexusflow::ErrorCode AIFootballPipeline::DeInit() {
    if (!m_impl) return nexusflow::SUCCESS;
    std::lock_guard<std::mutex> processLock(m_impl->processMutex);
    return m_impl->DeInit();
}

} // namespace aifootball
