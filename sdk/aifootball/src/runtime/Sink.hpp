#pragma once

#include "common/MyMessage.hpp"
#include <nexusflow/Module.hpp>

#include <chrono>
#include <condition_variable>
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
    using ResultHandler = std::function<void(ResultPacket)>;

    explicit Sink(const std::string& name);

    void PrepareForEnd();
    bool WaitForEnd(std::chrono::milliseconds timeout);
    void SetResultHandler(ResultHandler handler);
    void Close();

protected:
    void Process(nexusflow::Message& inputMessage) override;

private:
    std::mutex m_mutex;
    std::condition_variable m_condition;
    ResultHandler m_resultHandler;
    bool m_endSeen = false;
    bool m_closed = false;
};

} // namespace aifootball
