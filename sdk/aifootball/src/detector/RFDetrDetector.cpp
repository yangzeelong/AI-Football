#include "RFDetrDetector.hpp"
#include "common/Defer.hpp"
#include "common/MyMessage.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/Message.hpp>

#include <algorithm>
#include <chrono>
#include <future>

// ---------------------------------------------------------------------------

RFDetrDetector::RFDetrDetector(const std::string& name) : Module(name) {
    LOG_TRACE("RFDetrDetector constructor, name={}", name);
}

RFDetrDetector::~RFDetrDetector() = default;

ns::ErrorCode RFDetrDetector::Configure(const ns::Config& config) {
    LOG_TRACE("RFDetrDetector::Configure");

    // Model params → forwarded to RFDetrDetectorInfer
    m_inferParam.enginePath          = config.GetValueOrDefault<std::string>("enginePath", "");
    m_inferParam.inputBindingName    = config.GetValueOrDefault<std::string>("inputBindingName", "image");
    m_inferParam.outputBindingName   = config.GetValueOrDefault<std::string>("outputBindingName", "detections");
    m_inferParam.logitsBindingName   = config.GetValueOrDefault<std::string>("logitsBindingName", "pred_logits");
    m_inferParam.boxesBindingName    = config.GetValueOrDefault<std::string>("boxesBindingName", "pred_boxes");
    m_inferParam.inputSize           = config.GetValueOrDefault<int>("inputSize", 560);
    m_inferParam.confidenceThreshold = config.GetValueOrDefault<float>("confidenceThreshold", 0.25f);
    m_inferParam.numClasses          = config.GetValueOrDefault<int>("numClasses", 80);
    m_inferParam.numQueries          = config.GetValueOrDefault<int>("numQueries", 300);
    m_inferParam.meanR               = config.GetValueOrDefault<float>("meanR", 0.485f);
    m_inferParam.meanG               = config.GetValueOrDefault<float>("meanG", 0.456f);
    m_inferParam.meanB               = config.GetValueOrDefault<float>("meanB", 0.406f);
    m_inferParam.stdR                = config.GetValueOrDefault<float>("stdR", 0.229f);
    m_inferParam.stdG                = config.GetValueOrDefault<float>("stdG", 0.224f);
    m_inferParam.stdB                = config.GetValueOrDefault<float>("stdB", 0.225f);
    m_inferParam.useGpuPreprocess    = config.GetValueOrDefault<bool>(
        "useGpuPreprocess", true);

    // GraphUtils stores YAML sequences as vector<Any>. Convert the integer
    // class ids explicitly instead of silently falling back to an empty set.
    auto classValues = config.GetValueOrDefault<std::vector<nexusflow::Any>>(
        "targetClasses", std::vector<nexusflow::Any>{});
    m_inferParam.targetClasses.clear();
    for (const auto& value : classValues) {
        if (const auto* id = value.get<int>()) {
            m_inferParam.targetClasses.insert(*id);
        }
    }

    // Batch params → used by this Module
    m_batchParam.maxBatchSize   = std::max(1, config.GetValueOrDefault<int>("maxBatchSize", 1));
    m_batchParam.batchTimeoutMs = config.GetValueOrDefault<int>("batchTimeoutMs", 0);
    m_batchParam.maxRetryNum    = config.GetValueOrDefault<int>("maxRetryNum", 0);
    m_instanceCount             = std::max(1, config.GetValueOrDefault<int>("instanceCount", 1));
    m_inferParam.maxBatchSize   = m_batchParam.maxBatchSize;

    LOG_INFO("RFDetrDetector: engine={}, inputSize={}, maxBatch={}, instances={}, timeout={}ms, maxRetry={}, gpuPreprocess={}",
             m_inferParam.enginePath, m_inferParam.inputSize,
             m_batchParam.maxBatchSize, m_instanceCount,
             m_batchParam.batchTimeoutMs, m_batchParam.maxRetryNum,
             m_inferParam.useGpuPreprocess);
    LOG_INFO("RFDetrDetector: targetClasses count={}", m_inferParam.targetClasses.size());
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode RFDetrDetector::Init() {
    LOG_TRACE("RFDetrDetector::Init");

    if (m_inferParam.enginePath.empty()) {
        LOG_WARN("RFDetrDetector: enginePath empty, will emit empty results");
        return ns::ErrorCode::SUCCESS;
    }

    m_inferPool.reserve(static_cast<size_t>(m_instanceCount));
    for (int i = 0; i < m_instanceCount; ++i) {
        auto infer = std::make_unique<detector::RFDetrDetectorInfer>();
        if (!infer->Init(m_inferParam)) {
            LOG_ERROR("RFDetrDetector: RFDetrDetectorInfer::Init failed for instance {}", i);
            for (auto& item : m_inferPool) item->Release();
            m_inferPool.clear();
            return ns::ErrorCode::FAILURE;
        }
        m_inferPool.push_back(std::move(infer));
    }

    LOG_INFO("RFDetrDetector initialized: instances={}, effectiveBatchPerInstance={}",
             m_inferPool.size(), m_inferPool.front()->MaxBatch());
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode RFDetrDetector::DeInit() {
    LOG_TRACE("RFDetrDetector::DeInit");
    DrainBatch();
    for (auto& infer : m_inferPool) infer->Release();
    m_inferPool.clear();
    return ns::ErrorCode::SUCCESS;
}

// ---------------------------------------------------------------------------
// Batch policy
// ---------------------------------------------------------------------------

bool RFDetrDetector::ShouldDrain() const {
    if (m_batchBuffer.empty()) return false;
    if (static_cast<int>(m_batchBuffer.size()) >= m_batchParam.maxBatchSize) return true;
    if (m_batchParam.maxRetryNum > 0 && m_framesSinceFlush >= m_batchParam.maxRetryNum) return true;
    if (m_batchParam.batchTimeoutMs <= 0) return false;

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - m_batchBuffer.front().enqueueTime).count();
    return elapsed >= m_batchParam.batchTimeoutMs;
}

void RFDetrDetector::DrainBatch() {
    if (m_batchBuffer.empty()) return;

    const int B = static_cast<int>(m_batchBuffer.size());
    LOG_DEBUG("RFDetrDetector: draining {} frame(s)", B);

    defer {
        m_batchBuffer.clear();
        m_framesSinceFlush = 0;
    };

    // Build FrameInput list for the infer class.
    std::vector<detector::RFDetrDetectorInfer::FrameInput> inputs(B);
    for (int i = 0; i < B; ++i) {
        const auto& bf = m_batchBuffer[i];
        inputs[i].rgb = bf.videoFrame
            ? bf.videoFrame->Data() : nullptr;
        inputs[i].width  = bf.videoFrame ? bf.videoFrame->width : 0;
        inputs[i].height = bf.videoFrame ? bf.videoFrame->height : 0;
        inputs[i].channels = bf.videoFrame ? bf.videoFrame->channels : 0;
        inputs[i].dataBytes = bf.videoFrame ? bf.videoFrame->DataSize() : 0;
        inputs[i].frameId = bf.videoFrame ? bf.videoFrame->frameId : 0;
        if (!bf.videoFrame || bf.videoFrame->DataSize() == 0 ||
            bf.videoFrame->channels != 3) {
            inputs[i].rgb = nullptr;
        }
    }

    // Run inference, including host preprocessing and postprocessing.
    std::vector<std::vector<detector::Detection>> results;
    bool ok = InferFrames(inputs, results);

    // Broadcast per-frame results.
    for (int i = 0; i < B; ++i) {
        const auto& bf = m_batchBuffer[i];
        DetectionMessage out;
        out.videoFrame   = bf.videoFrame;
        out.isEnd        = bf.isEnd;
        out.timestamp    = bf.timestamp;
        out.timestampSec = bf.timestampSec;

        if (ok && i < static_cast<int>(results.size())) {
            out.detections.reserve(results[i].size());
            for (const auto& det : results[i]) {
                Detection d;
                d.x0 = det.x0; d.y0 = det.y0;
                d.x1 = det.x1; d.y1 = det.y1;
                d.score   = det.score;
                d.classId = det.classId;
                d.trackId = -1;
                out.detections.push_back(d);
                out.rawCounts.total++;
                if (det.classId == 1) out.rawCounts.person++;
                else if (det.classId == 37) out.rawCounts.ball++;
            }
        }

        LOG_DEBUG("RFDetrDetector: frame {} -> {} detections",
                  bf.videoFrame ? bf.videoFrame->frameId : 0, out.detections.size());
        Broadcast(nexusflow::Message(std::move(out)));
    }
}

bool RFDetrDetector::InferFrames(
    const std::vector<detector::RFDetrDetectorInfer::FrameInput>& inputs,
    std::vector<std::vector<detector::Detection>>& results) {
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
            std::vector<detector::RFDetrDetectorInfer::FrameInput> frames(
                inputs.begin() + static_cast<std::ptrdiff_t>(next),
                inputs.begin() + static_cast<std::ptrdiff_t>(next + count));
            std::vector<std::vector<detector::Detection>> chunkResults;
            if (!m_inferPool.front()->IsReady() ||
                !m_inferPool.front()->InferBatch(frames, chunkResults) ||
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
        LOG_DEBUG("RFDetrDetector: dispatching {} frame(s) across {} instance(s), maxChunk={}",
                  inputs.size(), m_inferPool.size(), batchSize);
    }

    // Each TensorRT execution context is owned by one pool entry. Chunks in a
    // wave therefore run concurrently without sharing CUDA streams or host
    // scratch buffers; subsequent waves reuse the same instances.
    size_t next = 0;
    while (next < inputs.size()) {
        struct ChunkResult {
            size_t offset = 0;
            bool ok = false;
            std::vector<std::vector<detector::Detection>> detections;
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
                    std::vector<detector::RFDetrDetectorInfer::FrameInput> frames(
                        inputs.begin() + static_cast<std::ptrdiff_t>(offset),
                        inputs.begin() + static_cast<std::ptrdiff_t>(offset + count));
                    chunk.ok = m_inferPool[instance]->IsReady() &&
                               m_inferPool[instance]->InferBatch(frames, chunk.detections);
                    return chunk;
                }));
        }

        bool waveOk = true;
        for (auto& future : futures) {
            try {
                ChunkResult chunk = future.get();
                if (!chunk.ok || chunk.detections.size() !=
                    std::min<size_t>(targetChunk,
                                     inputs.size() - chunk.offset)) {
                    waveOk = false;
                    continue;
                }
                for (size_t i = 0; i < chunk.detections.size(); ++i) {
                    results[chunk.offset + i] = std::move(chunk.detections[i]);
                }
            } catch (const std::exception& exc) {
                LOG_ERROR("RFDetrDetector: inference instance failed: {}", exc.what());
                waveOk = false;
            }
        }
        if (!waveOk) return false;
        if (next == waveStart) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

void RFDetrDetector::Process(ns::Message& inputMessage) {
    if (!inputMessage.HasData()) {
        LOG_WARN("RFDetrDetector: empty message, ignoring");
        return;
    }
    auto* frameMsg = inputMessage.MutPtr<FrameMessage>();
    if (!frameMsg) {
        LOG_WARN("RFDetrDetector: not FrameMessage, ignoring");
        return;
    }

    // EOF: drain the pending batch, then propagate.
    if (frameMsg->isEnd) {
        DrainBatch();
        DetectionMessage out;
        out.videoFrame   = frameMsg->videoFrame;
        out.isEnd        = true;
        out.timestamp    = frameMsg->timestamp;
        out.timestampSec = frameMsg->timestampSec;
        Broadcast(nexusflow::Message(std::move(out)));
        return;
    }

    // Buffer frame.
    BufferedFrame bf;
    bf.videoFrame   = frameMsg->videoFrame;
    bf.isEnd        = false;
    bf.timestamp    = frameMsg->timestamp;
    bf.timestampSec = frameMsg->timestampSec;
    bf.enqueueTime  = std::chrono::steady_clock::now();
    m_batchBuffer.push_back(std::move(bf));
    m_framesSinceFlush++;

    if (ShouldDrain()) DrainBatch();
}
