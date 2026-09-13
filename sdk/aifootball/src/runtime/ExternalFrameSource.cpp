#include "ExternalFrameSource.hpp"

#include <nexusflow/Logging.hpp>

namespace aifootball {

ExternalFrameSource::ExternalFrameSource(const std::string& name,
                                         std::size_t maxPendingFrames)
    : Module(name), m_maxPendingFrames(maxPendingFrames) {}

bool ExternalFrameSource::Submit(FrameMessage message) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_closed || (m_maxPendingFrames > 0 &&
                         m_pending.size() >= m_maxPendingFrames)) {
            if (!m_closed) {
                LOG_WARN("ExternalFrameSource: input queue is full ({})",
                         m_maxPendingFrames);
            }
            return false;
        }
        m_pending.push_back(std::move(message));
    }
    m_condition.notify_one();
    return true;
}

void ExternalFrameSource::Close() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_closed = true;
        m_pending.clear();
    }
    m_condition.notify_all();
}

void ExternalFrameSource::Process(nexusflow::Message& inputMessage) {
    if (inputMessage.HasData()) {
        LOG_WARN("ExternalFrameSource: unexpected input message, ignoring");
        return;
    }

    FrameMessage frame;
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_condition.wait(lock, [this] {
            return m_closed || !m_pending.empty();
        });
        if (m_pending.empty()) return;
        frame = std::move(m_pending.front());
        m_pending.pop_front();
    }
    Broadcast(nexusflow::Message(std::move(frame)));
}

} // namespace aifootball
