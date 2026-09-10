#pragma once

#include "nexusflow/ErrorCode.hpp"
#include <nexusflow/Nexusflow.hpp>
#include "common/MyMessage.hpp"
#include "RFDetrDetectorInfer.hpp"

#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace ns = nexusflow;

/**
 * @brief RFDetrDetector - Nexusflow Module wrapper for RF-DETR inference.
 *
 * Responsibilities:
 *   - Message routing (FrameMessage in → DetectionMessage out)
 *   - Batch accumulation policy (maxBatchSize / batchTimeoutMs / maxRetryNum)
 *   - EOF propagation
 *
 * All model logic (pre/post-processing, engine) lives in RFDetrDetectorInfer.
 */
class RFDetrDetector : public ns::Module {
public:
    RFDetrDetector(const std::string& name);
    ~RFDetrDetector() override;

    ns::ErrorCode Configure(const ns::Config& config) override;
    ns::ErrorCode Init() override;
    ns::ErrorCode DeInit() override;

protected:
    void Process(ns::Message& inputMessage) override;

private:
    struct BatchParam {
        int maxBatchSize   = 1;
        int batchTimeoutMs = 0;
        int maxRetryNum    = 0;
    } m_batchParam;

    struct BufferedFrame {
        VideoFrame videoFrame;
        bool isEnd = false;
        uint64_t timestamp = 0;
        double timestampSec = 0.0;
        std::chrono::steady_clock::time_point enqueueTime;
    };

    void FlushBatch();
    bool ShouldFlush() const;

    // Model inference (has-a)
    detector::RFDetrDetectorInfer::Param m_inferParam;
    std::unique_ptr<detector::RFDetrDetectorInfer> m_infer;

    // Batch state
    std::deque<BufferedFrame> m_batchBuffer;
    int m_framesSinceFlush = 0;
};

NEXUSFLOW_REGISTER_MODULE(RFDetrDetector);
