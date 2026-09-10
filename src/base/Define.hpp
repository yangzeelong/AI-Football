#ifndef NEXUSFLOW_BASE_DEFINE_HPP
#define NEXUSFLOW_BASE_DEFINE_HPP

#include "common/ConcurrentQueue.hpp"
#include "nexusflow/Message.hpp"
#include <memory>
#include <unordered_map>

namespace nexusflow {

// Type alias for the internal message queue.
// clang-format off
using MessageQueue     = ConcurrentQueue<Message>;
using MessageQueuePtr  = std::shared_ptr<MessageQueue>;
using MessageQueueUPtr = std::unique_ptr<MessageQueue>;

// clang-format on

} // namespace nexusflow

#endif // NEXUSFLOW_BASE_DEFINE_HPP
