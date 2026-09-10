#include "HRNetPoseEstimator.hpp"
#include "common/Defer.hpp"
#include "common/MyMessage.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/Message.hpp>

// ---------------------------------------------------------------------------

HRNetPoseEstimator::HRNetPoseEstimator(const std::string& name) : Module(name) {
    LOG_TRACE("HRNetPoseEstimator constructor, name={}", name);
}

HRNetPoseEstimator::~HRNetPoseEstimator() = default;

ns::ErrorCode HRNetPoseEstimator::Configure(const ns::Config& config) {
    LOG_TRACE("HRNetPoseEstimator::Configure");
    m_inferParam.enginePath        = config.GetValueOrDefault<std::string>("enginePath", "");
    m_inferParam.inputBindingName  = config.GetValueOrDefault<std::string>("inputBindingName", "images");
    m_inferParam.outputBindingName = config.GetValueOrDefault<std::string>("outputBindingName", "heatmaps");
    m_inferParam.inputWidth        = config.GetValueOrDefault<int>("inputWidth", 288);
    m_inferParam.inputHeight       = config.GetValueOrDefault<int>("inputHeight", 384);
    m_inferParam.heatmapWidth      = config.GetValueOrDefault<int>("heatmapWidth", 72);
    m_inferParam.heatmapHeight     = config.GetValueOrDefault<int>("heatmapHeight", 96);
    m_inferParam.numKeypoints      = config.GetValueOrDefault<int>("numKeypoints", 133);
    m_inferParam.stride            = config.GetValueOrDefault<int>("stride", 4);
    m_inferParam.useDark           = config.GetValueOrDefault<bool>("useDark", true);
    m_inferParam.poseBoxExpansion  = config.GetValueOrDefault<bool>("poseBoxExpansion", true);
    m_inferParam.xPadRatio         = config.GetValueOrDefault<float>("xPadRatio", 0.15f);
    m_inferParam.yPadRatio         = config.GetValueOrDefault<float>("yPadRatio", 0.25f);
    m_inferParam.minPadPx          = config.GetValueOrDefault<float>("minPadPx", 12.0f);
    m_inferParam.maxBatch          = config.GetValueOrDefault<int>("maxBatch", 16);
    m_inferParam.keypointConfThr   = config.GetValueOrDefault<float>("keypointConfThr", 0.05f);
    m_inferParam.meanR             = config.GetValueOrDefault<float>("meanR", 123.675f);
    m_inferParam.meanG             = config.GetValueOrDefault<float>("meanG", 116.28f);
    m_inferParam.meanB             = config.GetValueOrDefault<float>("meanB", 103.53f);
    m_inferParam.stdR              = config.GetValueOrDefault<float>("stdR", 58.395f);
    m_inferParam.stdG              = config.GetValueOrDefault<float>("stdG", 57.12f);
    m_inferParam.stdB              = config.GetValueOrDefault<float>("stdB", 57.375f);

    LOG_INFO("HRNetPoseEstimator: engine={}, input={}x{}, K={}, dark={}",
             m_inferParam.enginePath, m_inferParam.inputWidth, m_inferParam.inputHeight,
             m_inferParam.numKeypoints, m_inferParam.useDark);
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode HRNetPoseEstimator::Init() {
    LOG_TRACE("HRNetPoseEstimator::Init");

    if (m_inferParam.enginePath.empty()) {
        LOG_WARN("HRNetPoseEstimator: enginePath empty, poses will be empty");
        return ns::ErrorCode::SUCCESS;
    }

    m_infer = std::make_unique<pose::HRNetPoseEstimatorInfer>();
    if (!m_infer->Init(m_inferParam)) {
        LOG_ERROR("HRNetPoseEstimator: HRNetPoseEstimatorInfer::Init failed");
        m_infer.reset();
        return ns::ErrorCode::FAILURE;
    }

    LOG_INFO("HRNetPoseEstimator initialized");
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode HRNetPoseEstimator::DeInit() {
    LOG_TRACE("HRNetPoseEstimator::DeInit");
    if (m_infer) { m_infer->Release(); m_infer.reset(); }
    return ns::ErrorCode::SUCCESS;
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

void HRNetPoseEstimator::Process(ns::Message& inputMessage) {
    if (!inputMessage.HasData()) {
        LOG_WARN("HRNetPoseEstimator: empty message, ignoring");
        return;
    }
    auto* trkMsg = inputMessage.MutPtr<TrackedDetectionMessage>();
    if (!trkMsg) {
        LOG_WARN("HRNetPoseEstimator: not TrackedDetectionMessage, ignoring");
        return;
    }

    PoseMessage out;
    out.videoFrame       = trkMsg->videoFrame;
    out.balls            = trkMsg->balls;
    out.rawCounts        = trkMsg->rawCounts;
    out.activeTrackCount = trkMsg->activeTrackCount;
    out.lostTrackCount   = trkMsg->lostTrackCount;
    out.isEnd            = trkMsg->isEnd;
    out.timestamp        = trkMsg->timestamp;
    out.timestampSec     = trkMsg->timestampSec;

    defer { Broadcast(nexusflow::Message(std::move(out))); };

    if (trkMsg->isEnd) return;

    const auto& vf = trkMsg->videoFrame;

    // Fallback: emit persons with missing keypoints if engine not ready.
    if (!m_infer || !m_infer->IsReady() || trkMsg->persons.empty() ||
        vf.frameData.empty() || vf.width <= 0 || vf.height <= 0 || vf.channels != 3) {
        for (const auto& d : trkMsg->persons) {
            PersonPose pp;
            pp.trackId = d.trackId;
            pp.x0 = d.x0; pp.y0 = d.y0; pp.x1 = d.x1; pp.y1 = d.y1;
            pp.detectionConfidence = d.score;
            out.persons.push_back(pp);
        }
        return;
    }

    // Build input list for the infer class.
    const uint8_t* rgb = reinterpret_cast<const uint8_t*>(vf.frameData.data());
    std::vector<pose::HRNetPoseEstimatorInfer::PersonInput> inputs;
    inputs.reserve(trkMsg->persons.size());
    for (const auto& d : trkMsg->persons) {
        inputs.push_back({rgb, vf.width, vf.height, d.x0, d.y0, d.x1, d.y1, d.trackId, d.score});
    }

    // Run inference.
    std::vector<PersonPose> results;
    if (m_infer->InferBatch(inputs, results)) {
        out.persons = std::move(results);
    } else {
        // Fallback on failure.
        for (const auto& d : trkMsg->persons) {
            PersonPose pp;
            pp.trackId = d.trackId;
            pp.x0 = d.x0; pp.y0 = d.y0; pp.x1 = d.x1; pp.y1 = d.y1;
            pp.detectionConfidence = d.score;
            out.persons.push_back(pp);
        }
    }

    LOG_DEBUG("HRNetPoseEstimator: frame={} persons={}", vf.frameId, out.persons.size());
}
