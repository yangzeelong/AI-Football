#ifndef NEXUSFLOW_DISPATCHER_HPP
#define NEXUSFLOW_DISPATCHER_HPP

#include "base/Define.hpp"
#include "common/ViewPtr.hpp"
#include "nexusflow/Config.hpp"
#include "nexusflow/Message.hpp"
#include "utils/logging.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nexusflow { namespace dispatcher {

/**
 * @class Dispatcher
 * @brief An internal helper class responsible for dispatching messages to downstream queues.
 *
 * This class encapsulates the logic for broadcasting or sending messages to a set
 * of pre-configured output queues. It holds non-owning "view" pointers to the queues,
 * which are owned and managed by the Pipeline.
 *
 * This is an implementation detail of the framework and is not part of the public API.
 */
class Dispatcher {
public:
    Dispatcher(const ViewPtr<Config>& configView);

    ~Dispatcher();

    /**
     * @brief Broadcasts a message to all configured output queues.
     * @details To optimize performance, this method copies the message for the
     * first N-1 queues and moves the original message into the last queue.
     * @param msg The message to broadcast.
     */
    void Broadcast(const Message& msg);

    /**
     * @brief Sends a message to a specific output queue.
     * @param outputName The name of the output queue to send the message to.
     * @param msg The message to send.
     * @throws std::invalid_argument If the outputName is not found in the output queue map.
     */
    void SendTo(const std::string& outputName, const Message& msg);

    /**
     * @brief Adds a new output queue to the dispatcher.
     * @param name The name of the output queue.
     * @param queue The output queue to add.
     */
    void AddSubscriber(const std::string& name, ViewPtr<MessageQueue> queue) {
        if (m_subscriberMap.find(name) != m_subscriberMap.end()) {
            LOG_ERROR("Output queue with name {} already exists", name);
            throw std::invalid_argument("Output queue with name " + name + " already exists");
        }
        m_subscriberMap[name] = queue;
    }

private:
    ViewPtr<Config> m_configView;
    std::unordered_map<std::string, ViewPtr<MessageQueue>> m_subscriberMap;
};

}} // namespace nexusflow::dispatcher

#endif // NEXUSFLOW_DISPATCHER_HPP