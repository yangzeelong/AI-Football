#pragma once

#include "common/MyMessage.hpp"
#include <nexusflow/Module.hpp>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>

namespace aifootball {

struct ResultPacket {
    BallTrackMessage message;
    std::map<std::string, double> moduleTimingsMs;
};

/** Internal sink used to return algorithm results to AIFootballPipeline. */
class Sink final : public nexusflow::Module {
public:
    using Callback = std::function<void(ResultPacket)>;

    explicit Sink(const std::string& name);

    bool WaitNext(ResultPacket& packet, std::chrono::milliseconds timeout);
    void PrepareForEnd();
    bool WaitForEnd(std::chrono::milliseconds timeout);
    void SetCallback(Callback callback);
    void Close();

protected:
    void Process(nexusflow::Message& inputMessage) override;

private:
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::deque<ResultPacket> m_pending;
    Callback m_callback;
    bool m_endSeen = false;
    bool m_closed = false;
};

} // namespace aifootball
