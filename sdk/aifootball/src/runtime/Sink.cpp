#include "Sink.hpp"

#include <nexusflow/Logging.hpp>

namespace aifootball {

Sink::Sink(const std::string& name) : Module(name) {}

void Sink::SetResultHandler(ResultHandler handler) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_resultHandler = std::move(handler);
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
        m_resultHandler = nullptr;
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
    ResultHandler handler;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_closed) return;
        handler = m_resultHandler;
    }
    if (handler) handler(std::move(packet));
}

} // namespace aifootball
