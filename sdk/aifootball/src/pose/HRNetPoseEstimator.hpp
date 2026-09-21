#pragma once

#include "nexusflow/ErrorCode.hpp"
#include <nexusflow/Nexusflow.hpp>
#include "common/MyMessage.hpp"
#include "HRNetPoseEstimatorInfer.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ns = nexusflow;

/**
 * @brief HRNetPoseEstimator - Nexusflow Module wrapper for HRNet-W48-DARK pose.
 *
 * Responsibilities:
 *   - Message routing (TrackedDetectionMessage in → PoseMessage out)
 *   - EOF propagation
 *
 * All model logic (crop, normalize, DARK decode, engine) lives in
 * HRNetPoseEstimatorInfer.
 */
class HRNetPoseEstimator : public ns::Module {
public:
    HRNetPoseEstimator(const std::string& name);
    ~HRNetPoseEstimator() override;

    ns::ErrorCode Configure(const ns::Config& config) override;
    ns::ErrorCode Init() override;
    ns::ErrorCode DeInit() override;

protected:
    void Process(ns::Message& inputMessage) override;

private:
    struct PendingFrame {
        TrackedDetectionMessage message;
        nexusflow::MessageMeta metadata;
    };

    bool InferPersons(
        const std::vector<pose::HRNetPoseEstimatorInfer::PersonInput>& inputs,
        std::vector<PersonPose>& results);
    void DrainPending();

    pose::HRNetPoseEstimatorInfer::Param m_inferParam;
    std::vector<std::unique_ptr<pose::HRNetPoseEstimatorInfer>> m_inferPool;
    int m_instanceCount = 1;
    int m_batchFrameCount = 4;
    std::vector<PendingFrame> m_pendingFrames;
    std::size_t m_pendingPersons = 0;
};

NEXUSFLOW_REGISTER_MODULE(HRNetPoseEstimator);
