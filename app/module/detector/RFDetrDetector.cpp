#include "RFDetrDetector.hpp"
#include "common/Defer.hpp"
#include "common/MyMessage.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/Message.hpp>

#include <chrono>

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

    auto classes = config.GetValueOrDefault<std::vector<int>>("targetClasses", std::vector<int>{});
    m_inferParam.targetClasses.clear();
    for (int c : classes) m_inferParam.targetClasses.insert(c);

    // Batch params → used by this Module
    m_batchParam.maxBatchSize   = config.GetValueOrDefault<int>("maxBatchSize", 1);
    m_batchParam.batchTimeoutMs = config.GetValueOrDefault<int>("batchTimeoutMs", 0);
    m_batchParam.maxRetryNum    = config.GetValueOrDefault<int>("maxRetryNum", 0);

    LOG_INFO("RFDetrDetector: engine={}, inputSize={}, maxBatch={}, timeout={}ms, maxRetry={}",
             m_inferParam.enginePath, m_inferParam.inputSize,
             m_batchParam.maxBatchSize, m_batchParam.batchTimeoutMs, m_batchParam.maxRetryNum);
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode RFDetrDetector::Init() {
    LOG_TRACE("RFDetrDetector::Init");

    if (m_inferParam.enginePath.empty()) {
        LOG_WARN("RFDetrDetector: enginePath empty, will emit empty results");
        return ns::ErrorCode::SUCCESS;
    }

    m_infer = std::make_unique<detector::RFDetrDetectorInfer>();
    if (!m_infer->Init(m_inferParam)) {
        LOG_ERROR("RFDetrDetector: RFDetrDetectorInfer::Init failed");
        m_infer.reset();
        return ns::ErrorCode::FAILURE;
    }

    LOG_INFO("RFDetrDetector initialized");
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode RFDetrDetector::DeInit() {
    LOG_TRACE("RFDetrDetector::DeInit");
    FlushBatch();
    if (m_infer) { m_infer->Release(); m_infer.reset(); }
    return ns::ErrorCode::SUCCESS;
}

// ---------------------------------------------------------------------------
// Batch policy
// ---------------------------------------------------------------------------

bool RFDetrDetector::ShouldFlush() const {
    if (m_batchBuffer.empty()) return false;
    if (static_cast<int>(m_batchBuffer.size()) >= m_batchParam.maxBatchSize) return true;
    if (m_batchParam.maxRetryNum > 0 && m_framesSinceFlush >= m_batchParam.maxRetryNum) return true;
    if (m_batchParam.batchTimeoutMs <= 0) return false;

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - m_batchBuffer.front().enqueueTime).count();
    return elapsed >= m_batchParam.batchTimeoutMs;
}

void RFDetrDetector::FlushBatch() {
    if (m_batchBuffer.empty()) return;

    const int B = static_cast<int>(m_batchBuffer.size());
    LOG_DEBUG("RFDetrDetector: flushing {} frame(s)", B);

    defer {
        m_batchBuffer.clear();
        m_framesSinceFlush = 0;
    };

    // Build FrameInput list for the infer class.
    std::vector<detector::RFDetrDetectorInfer::FrameInput> inputs(B);
    for (int i = 0; i < B; ++i) {
        const auto& bf = m_batchBuffer[i];
        inputs[i].rgb    = reinterpret_cast<const uint8_t*>(bf.videoFrame.frameData.data());
        inputs[i].width  = bf.videoFrame.width;
        inputs[i].height = bf.videoFrame.height;
        if (bf.videoFrame.frameData.empty() || bf.videoFrame.channels != 3) {
            inputs[i].rgb = nullptr;
        }
    }

    // Run inference.
    std::vector<std::vector<detector::Detection>> results;
    bool ok = m_infer && m_infer->IsReady() && m_infer->InferBatch(inputs, results);

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
                if (det.classId == 0) out.rawCounts.person++;
                else if (det.classId == 32) out.rawCounts.ball++;
            }
        }

        LOG_DEBUG("RFDetrDetector: frame {} -> {} detections",
                  bf.videoFrame.frameId, out.detections.size());
        Broadcast(nexusflow::Message(std::move(out)));
    }
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

    // EOF: flush pending batch, then propagate.
    if (frameMsg->isEnd) {
        FlushBatch();
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

    if (ShouldFlush()) FlushBatch();
}
