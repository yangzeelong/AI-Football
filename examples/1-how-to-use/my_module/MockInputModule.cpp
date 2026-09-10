
#include "MockInputModule.hpp"
#include "../src/utils/logging.hpp" // TODO: remove
#include "MyMessage.hpp"
#include "nexusflow/ErrorCode.hpp"
#include "nexusflow/Message.hpp"
#include <thread>
#include <type_traits>

MockInputModule::MockInputModule(const std::string& name) : Module(name) { LOG_TRACE("MockInputModule constructor, name={}", name); }

MockInputModule::~MockInputModule() { LOG_TRACE("MockInputModule destructor, name={}", GetModuleName()); }

nexusflow::ErrorCode MockInputModule::Configure(const nexusflow::Config& config) {
    LOG_INFO("Configure init, module={}", GetModuleName());
    m_sendIntervalMs = config.GetValueOrDefault("send_interval_ms", 1000);
    LOG_INFO("Configure done, m_sendIntervalMs={}", m_sendIntervalMs);
    for (auto& pair : config.GetConfigMap()) {
        LOG_INFO("param key={}, type={}", pair.first, pair.second.getType().name());
    }

    return nexusflow::ErrorCode::SUCCESS;
}

void MockInputModule::Process(nexusflow::Message& inputMessage) {
    // no input message
    if (inputMessage.HasData()) {
        LOG_WARN("MockInputModule: inputMessage is not nullptr");
        return;
    }

    // mock 5 fps messages.
    std::this_thread::sleep_for(std::chrono::milliseconds(m_sendIntervalMs));
    static int counter = 0;
    // 因为SeqMessage有std::mutex, 所以包一层shared_ptr
    auto seqMsg = std::make_shared<SeqMessage>();
    seqMsg->addData(GetModuleName() + "_" + std::to_string(counter++));
    LOG_INFO(GetModuleName() + ": send message: {}", seqMsg->toString());

    auto newMsg = nexusflow::MakeMessage(std::move(seqMsg));
    Broadcast(newMsg);
}
