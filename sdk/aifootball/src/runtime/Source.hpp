#pragma once

#include <aifootball/AIFootball.hpp>

#include "common/MyMessage.hpp"
#include <nexusflow/Module.hpp>

#include <condition_variable>
#include <deque>
#include <mutex>

namespace aifootball {

/** Internal source used to bridge caller-owned decoded frames into the graph. */
class Source final : public nexusflow::Module {
public:
    explicit Source(const std::string& name, std::size_t maxPendingFrames,
                    QueuePolicy queuePolicy);

    bool Submit(FrameMessage message);
    void Close();

protected:
    void Process(nexusflow::Message& inputMessage) override;

private:
    std::mutex m_mutex;
    std::condition_variable m_dataCondition;
    std::condition_variable m_roomCondition;
    std::deque<FrameMessage> m_pending;
    const std::size_t m_maxPendingFrames;
    const QueuePolicy m_queuePolicy;
    bool m_closed = false;
};

} // namespace aifootball
