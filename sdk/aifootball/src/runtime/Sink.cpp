#include "Sink.hpp"

#include <nexusflow/Logging.hpp>

namespace aifootball {

Sink::Sink(const std::string& name) : Module(name) {}

void Sink::SetCallback(Callback callback) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_callback = std::move(callback);
}

bool Sink::WaitNext(ResultPacket& packet, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (!m_condition.wait_for(lock, timeout, [this] {
            return m_closed || !m_pending.empty();
        })) {
        return false;
    }
    if (m_pending.empty()) return false;
    packet = std::move(m_pending.front());
    m_pending.pop_front();
    return true;
}

void Sink::PrepareForEnd() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_endSeen = false;
}

bool Sink::WaitForEnd(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(m_mutex);
    return m_condition.wait_for(lock, timeout, [this] {
        return m_closed || m_endSeen;
    }) && m_endSeen;
}

void Sink::Close() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_closed = true;
        m_pending.clear();
    }
    m_condition.notify_all();
}

void Sink::Process(nexusflow::Message& inputMessage) {
    const auto* message = inputMessage.BorrowPtr<BallTrackMessage>();
    if (!message) {
        LOG_WARN("Sink: message is not BallTrackMessage, ignoring");
        return;
    }

    if (message->isEnd) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_endSeen = true;
        }
        m_condition.notify_all();
        return;
    }

    ResultPacket packet;
    packet.message = *message;
    packet.moduleTimingsMs = inputMessage.GetMetaData().moduleTimingMs;
    Callback callback;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_closed) return;
        callback = m_callback;
        if (!callback) m_pending.push_back(packet);
    }
    if (callback) {
        callback(std::move(packet));
    } else {
        m_condition.notify_one();
    }
}

} // namespace aifootball
