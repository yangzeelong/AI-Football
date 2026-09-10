#include "Dispatcher.hpp"
#include "nexusflow/Config.hpp"

namespace nexusflow { namespace dispatcher {

Dispatcher::Dispatcher(const ViewPtr<Config>& configView) { m_configView = configView; };

Dispatcher::~Dispatcher() = default;

void Dispatcher::Broadcast(const Message& message) {
    for (auto& pair : m_subscriberMap) {
        auto& subscriber = pair.second;
        subscriber->tryPush(message);
    }
}

void Dispatcher::SendTo(const std::string& outputName, const Message& msg) {
    auto it = m_subscriberMap.find(outputName);
    if (it != m_subscriberMap.end()) {
        auto& subscriber = it->second;
        subscriber->tryPush(msg);
    }
}

}} // namespace nexusflow::dispatcher