#include "Source.hpp"

#include <nexusflow/Logging.hpp>

namespace aifootball {

Source::Source(const std::string& name, std::size_t maxPendingFrames,
               QueuePolicy queuePolicy)
    : Module(name),
      m_maxPendingFrames(maxPendingFrames),
      m_queuePolicy(queuePolicy) {}

bool Source::Submit(FrameMessage message, FrameMessage* droppedMessage) {
    auto hasRoom = [this] {
        return m_maxPendingFrames == 0 || m_pending.size() < m_maxPendingFrames;
    };

    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_closed) return false;
        if (!hasRoom()) {
            if (message.isEnd || m_queuePolicy == QueuePolicy::Block) {
                m_roomCondition.wait(lock, [this, &hasRoom] {
                    return m_closed || hasRoom();
                });
                if (m_closed) return false;
            } else if (m_queuePolicy == QueuePolicy::DropOldest) {
                LOG_WARN("Source: input queue is full ({}), dropping oldest frame",
                         m_maxPendingFrames);
                if (droppedMessage) *droppedMessage = std::move(m_pending.front());
                m_pending.pop_front();
            } else {
                LOG_WARN("Source: input queue is full ({}), dropping current frame",
                         m_maxPendingFrames);
                return false;
            }
        }
        m_pending.push_back(std::move(message));
    }
    m_dataCondition.notify_one();
    return true;
}

void Source::Close() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_closed = true;
        m_pending.clear();
    }
    m_dataCondition.notify_all();
    m_roomCondition.notify_all();
}

void Source::Process(nexusflow::Message& inputMessage) {
    if (inputMessage.HasData()) {
        LOG_WARN("Source: unexpected input message, ignoring");
        return;
    }

    FrameMessage frame;
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_dataCondition.wait(lock, [this] {
            return m_closed || !m_pending.empty();
        });
        if (m_pending.empty()) return;
        frame = std::move(m_pending.front());
        m_pending.pop_front();
    }
    m_roomCondition.notify_one();
    Broadcast(nexusflow::Message(std::move(frame)));
}

} // namespace aifootball
