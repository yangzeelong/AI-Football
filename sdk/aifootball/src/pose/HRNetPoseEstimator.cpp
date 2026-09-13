#include "HRNetPoseEstimator.hpp"
#include "common/Defer.hpp"
#include "common/MyMessage.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/Message.hpp>

#include <algorithm>
#include <future>

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
    m_inferParam.maxBatch          = std::max(1, m_inferParam.maxBatch);
    m_instanceCount                = std::max(1, config.GetValueOrDefault<int>("instanceCount", 1));
    m_inferParam.keypointConfThr   = config.GetValueOrDefault<float>("keypointConfThr", 0.05f);
    m_inferParam.meanR             = config.GetValueOrDefault<float>("meanR", 123.675f);
    m_inferParam.meanG             = config.GetValueOrDefault<float>("meanG", 116.28f);
    m_inferParam.meanB             = config.GetValueOrDefault<float>("meanB", 103.53f);
    m_inferParam.stdR              = config.GetValueOrDefault<float>("stdR", 58.395f);
    m_inferParam.stdG              = config.GetValueOrDefault<float>("stdG", 57.12f);
    m_inferParam.stdB              = config.GetValueOrDefault<float>("stdB", 57.375f);

    LOG_INFO("HRNetPoseEstimator: engine={}, input={}x{}, K={}, maxBatch={}, instances={}, dark={}",
             m_inferParam.enginePath, m_inferParam.inputWidth, m_inferParam.inputHeight,
             m_inferParam.numKeypoints, m_inferParam.maxBatch,
             m_instanceCount, m_inferParam.useDark);
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode HRNetPoseEstimator::Init() {
    LOG_TRACE("HRNetPoseEstimator::Init");

    if (m_inferParam.enginePath.empty()) {
        LOG_WARN("HRNetPoseEstimator: enginePath empty, poses will be empty");
        return ns::ErrorCode::SUCCESS;
    }

    m_inferPool.reserve(static_cast<size_t>(m_instanceCount));
    for (int i = 0; i < m_instanceCount; ++i) {
        auto infer = std::make_unique<pose::HRNetPoseEstimatorInfer>();
        if (!infer->Init(m_inferParam)) {
            LOG_ERROR("HRNetPoseEstimator: HRNetPoseEstimatorInfer::Init failed for instance {}", i);
            for (auto& item : m_inferPool) item->Release();
            m_inferPool.clear();
            return ns::ErrorCode::FAILURE;
        }
        m_inferPool.push_back(std::move(infer));
    }

    LOG_INFO("HRNetPoseEstimator initialized: instances={}, effectiveBatchPerInstance={}",
             m_inferPool.size(), m_inferPool.front()->MaxBatch());
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode HRNetPoseEstimator::DeInit() {
    LOG_TRACE("HRNetPoseEstimator::DeInit");
    for (auto& infer : m_inferPool) infer->Release();
    m_inferPool.clear();
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
    if (m_inferPool.empty() || !m_inferPool.front()->IsReady() || trkMsg->persons.empty() ||
        !vf || vf->DataSize() == 0 || vf->width <= 0 || vf->height <= 0 || vf->channels != 3) {
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
    const uint8_t* rgb = vf->Data();
    std::vector<pose::HRNetPoseEstimatorInfer::PersonInput> inputs;
    inputs.reserve(trkMsg->persons.size());
    for (const auto& d : trkMsg->persons) {
        inputs.push_back({rgb, vf->width, vf->height, d.x0, d.y0, d.x1, d.y1, d.trackId, d.score});
    }

    // Run inference.
    std::vector<PersonPose> results;
    if (InferPersons(inputs, results)) {
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

    LOG_DEBUG("HRNetPoseEstimator: frame={} persons={}",
              vf ? vf->frameId : 0, out.persons.size());
}

bool HRNetPoseEstimator::InferPersons(
    const std::vector<pose::HRNetPoseEstimatorInfer::PersonInput>& inputs,
    std::vector<PersonPose>& results) {
    results.clear();
    if (inputs.empty()) return true;
    if (m_inferPool.empty()) return false;

    const int batchSize = m_inferPool.front()->MaxBatch();
    if (batchSize <= 0) return false;
    results.resize(inputs.size());

    if (m_inferPool.size() == 1) {
        if (inputs.size() <= static_cast<size_t>(batchSize)) {
            return m_inferPool.front()->IsReady() &&
                   m_inferPool.front()->InferBatch(inputs, results);
        }

        // Keep the single-instance path synchronous. Creating a temporary
        // std::async worker for every frame would add overhead to the default
        // configuration without providing any parallelism.
        size_t next = 0;
        while (next < inputs.size()) {
            const size_t count = std::min<size_t>(
                static_cast<size_t>(batchSize), inputs.size() - next);
            std::vector<pose::HRNetPoseEstimatorInfer::PersonInput> persons(
                inputs.begin() + static_cast<std::ptrdiff_t>(next),
                inputs.begin() + static_cast<std::ptrdiff_t>(next + count));
            std::vector<PersonPose> chunkResults;
            if (!m_inferPool.front()->IsReady() ||
                !m_inferPool.front()->InferBatch(persons, chunkResults) ||
                chunkResults.size() != count) {
                return false;
            }
            for (size_t i = 0; i < count; ++i) {
                results[next + i] = std::move(chunkResults[i]);
            }
            next += count;
        }
        return true;
    }

    if (m_inferPool.size() > 1 && inputs.size() > 1) {
        LOG_DEBUG("HRNetPoseEstimator: dispatching {} person(s) across {} instance(s), maxChunk={}",
                  inputs.size(), m_inferPool.size(), batchSize);
    }

    // A frame can contain more people than one engine profile supports. Split
    // all people into chunks and execute one chunk per model instance in each
    // wave, preserving the input order in the merged result.
    size_t next = 0;
    while (next < inputs.size()) {
        struct ChunkResult {
            size_t offset = 0;
            bool ok = false;
            std::vector<PersonPose> poses;
        };

        std::vector<std::future<ChunkResult>> futures;
        const size_t waveStart = next;
        const size_t remaining = inputs.size() - next;
        const size_t targetChunk = std::min<size_t>(
            static_cast<size_t>(batchSize),
            (remaining + m_inferPool.size() - 1) / m_inferPool.size());
        for (size_t instance = 0;
             instance < m_inferPool.size() && next < inputs.size();
             ++instance) {
            const size_t offset = next;
            const size_t count = std::min<size_t>(
                targetChunk, inputs.size() - next);
            next += count;

            futures.emplace_back(std::async(
                std::launch::async,
                [this, instance, &inputs, offset, count]() {
                    ChunkResult chunk;
                    chunk.offset = offset;
                    std::vector<pose::HRNetPoseEstimatorInfer::PersonInput> persons(
                        inputs.begin() + static_cast<std::ptrdiff_t>(offset),
                        inputs.begin() + static_cast<std::ptrdiff_t>(offset + count));
                    chunk.ok = m_inferPool[instance]->IsReady() &&
                               m_inferPool[instance]->InferBatch(persons, chunk.poses);
                    return chunk;
                }));
        }

        bool waveOk = true;
        for (auto& future : futures) {
            try {
                ChunkResult chunk = future.get();
                if (!chunk.ok || chunk.poses.size() !=
                    std::min<size_t>(targetChunk,
                                     inputs.size() - chunk.offset)) {
                    waveOk = false;
                    continue;
                }
                for (size_t i = 0; i < chunk.poses.size(); ++i) {
                    results[chunk.offset + i] = std::move(chunk.poses[i]);
                }
            } catch (const std::exception& exc) {
                LOG_ERROR("HRNetPoseEstimator: inference instance failed: {}", exc.what());
                waveOk = false;
            }
        }
        if (!waveOk) return false;
        if (next == waveStart) return false;
    }
    return true;
}
