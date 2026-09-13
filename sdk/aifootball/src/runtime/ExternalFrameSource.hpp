#pragma once

#include "common/MyMessage.hpp"
#include <nexusflow/Module.hpp>

#include <condition_variable>
#include <deque>
#include <mutex>

namespace aifootball {

/** Internal source used to bridge caller-owned decoded frames into the graph. */
class ExternalFrameSource final : public nexusflow::Module {
public:
    explicit ExternalFrameSource(const std::string& name,
                                 std::size_t maxPendingFrames);

    bool Submit(FrameMessage message);
    void Close();

protected:
    void Process(nexusflow::Message& inputMessage) override;

private:
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::deque<FrameMessage> m_pending;
    const std::size_t m_maxPendingFrames;
    bool m_closed = false;
};

} // namespace aifootball
