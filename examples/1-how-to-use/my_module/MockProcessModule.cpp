#include "MockProcessModule.hpp"
#include "../src/utils/logging.hpp"
#include "MyMessage.hpp"
#include "nexusflow/Message.hpp"

MockProcessModule::MockProcessModule(const std::string& name) : Module(name) {
    LOG_TRACE("MockProcessModule constructor, name={}", name);
}

MockProcessModule::~MockProcessModule() { LOG_TRACE("MockProcessModule destructor, name={}", GetModuleName()); }

void MockProcessModule::Process(nexusflow::Message& inputMessage) {
    if (auto& seqMsg = inputMessage.Mut<std::shared_ptr<SeqMessage>>()) {
        LOG_DEBUG("Received message is {}", seqMsg->toString());
        // Add some data to the message
        seqMsg->addData(GetModuleName() + "_" + std::to_string(m_count++));
        LOG_INFO(GetModuleName() + ": send message: {}", seqMsg->toString());
        Broadcast(nexusflow::MakeMessage(std::move(seqMsg)));
    }
}

// REGISTER_MODULE(MockProcessModule);
