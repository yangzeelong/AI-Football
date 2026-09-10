#include "AlarmPusher.hpp"
#include <nexusflow/Logging.hpp>
#include "common/MyMessage.hpp"
#include "common/PipelineCompletionSignal.hpp"
#include "nexusflow/Message.hpp"

AlarmPusher::AlarmPusher(const std::string& name) : Module(name) { LOG_TRACE("AlarmPusher constructor, name={}", name); }

AlarmPusher::~AlarmPusher() { LOG_TRACE("AlarmPusher destructor, name={}", GetModuleName()); }

// --- Lifecycle ---
ns::ErrorCode AlarmPusher::Configure(const ns::Config& config) {
    LOG_TRACE("AlarmPusher::Configure");
    return ns::ErrorCode::SUCCESS;
}

void AlarmPusher::Process(nexusflow::Message& inputMessage) {
    if (!inputMessage.HasData()) {
        LOG_WARN("AlarmPusher: received empty message");
        return;
    }

    // Terminal sink: accept the final BallTrackMessage from FootballTracker.
    auto* ballMsg = inputMessage.MutPtr<BallTrackMessage>();
    if (ballMsg) {
        if (ballMsg->isEnd) {
            LOG_INFO("AlarmPusher: end-of-stream (BallTrackMessage)");
            PipelineCompletionSignal::Instance().NotifyComplete();
            return;
        }
        LOG_INFO("AlarmPusher: {}", ballMsg->toString());
        return;
    }

    // Fallback: still accept FrameMessage for simple pipelines.
    auto* frameMsg = inputMessage.MutPtr<FrameMessage>();
    if (frameMsg) {
        if (frameMsg->isEnd) {
            LOG_INFO("AlarmPusher: end-of-stream (FrameMessage)");
            PipelineCompletionSignal::Instance().NotifyComplete();
            return;
        }
        LOG_INFO("AlarmPusher: {}", frameMsg->toString());
        return;
    }

    LOG_WARN("AlarmPusher: unsupported message type, ignoring");
}
