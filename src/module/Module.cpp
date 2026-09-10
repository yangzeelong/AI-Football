#include "nexusflow/Module.hpp"
#include "dispatcher/Dispatcher.hpp"
#include "nexusflow/Config.hpp"
#include "nexusflow/ErrorCode.hpp"
#include "nexusflow/Message.hpp"
#include "utils/logging.hpp"

namespace nexusflow {

Module::Module(std::string name) : m_moduleName(std::move(name)) {
    // Initialize the module with the given name.
    LOG_TRACE("Module '{}' created.", m_moduleName);
}

Module::~Module() {
    // Ensure the module is stopped before destruction.
    LOG_TRACE("Module '{}' destroying...", m_moduleName);
};

ErrorCode Module::Configure(const Config& config) {
    // Initialize the module.
    LOG_TRACE("Module '{}' initializing...", m_moduleName);
    return ErrorCode::SUCCESS;
}

ErrorCode Module::Init() {
    // Initialize the module.
    LOG_TRACE("Module '{}' initializing...", m_moduleName);
    return ErrorCode::SUCCESS;
}

ErrorCode Module::DeInit() {
    // De-initialize the module.
    LOG_TRACE("Module '{}' de-initializing...", m_moduleName);
    return ErrorCode::SUCCESS;
}

void Module::ProcessBatch(std::vector<Message>& inputBatchMessages) {
    // Process the batch of input messages.
    LOG_DEBUG("Module '{}' processing batch of {} messages.", m_moduleName, inputBatchMessages.size());
    for (auto& message : inputBatchMessages) {
        Process(message);
    }
}

void Module::Broadcast(const Message& message) {
    if (m_dispatcherPtr != nullptr) {
        LOG_DEBUG("Module '{}' broadcasting message.", m_moduleName);
        m_dispatcherPtr->Broadcast(message);
    } else {
        LOG_WARN("Module '{}' has no handle, cannot broadcast message.", m_moduleName);
    }
}

void Module::SendTo(const std::string& outputName, const Message& msg) {
    if (m_dispatcherPtr != nullptr) {
        LOG_DEBUG("Module '{}' sending message to '{}'.", m_moduleName, outputName);
        m_dispatcherPtr->SendTo(outputName, msg);
    } else {
        LOG_WARN("Module '{}' has no handle, cannot send message.", m_moduleName);
    }
}

const std::string& Module::GetModuleName() const { return m_moduleName; }

void Module::SetDispatcher(const std::shared_ptr<dispatcher::Dispatcher>& dispatcher) { m_dispatcherPtr = dispatcher; }

} // namespace nexusflow
