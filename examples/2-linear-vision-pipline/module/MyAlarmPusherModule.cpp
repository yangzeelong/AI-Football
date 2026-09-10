
#include "MyAlarmPusherModule.hpp"
#include "../src/utils/logging.hpp" // TODO: remove
#include "MyMessage.hpp"
#include "nexusflow/ErrorCode.hpp"
#include "nexusflow/Message.hpp"
#include <thread>

MyAlarmPusherModule::MyAlarmPusherModule(const std::string& name) : Module(name) {
    LOG_TRACE("MyAlarmPusherModule constructor, name={}", name);
}

MyAlarmPusherModule::~MyAlarmPusherModule() { LOG_TRACE("MyAlarmPusherModule destructor, name={}", GetModuleName()); }

nexusflow::ErrorCode MyAlarmPusherModule::Configure(const nexusflow::Config& config) {
    m_savePath = config.GetValueOrDefault("savePath", std::string("default-result.txt"));
    LOG_INFO("MyAlarmPusherModule::Configure, name={}, savePath={}", GetModuleName(), m_savePath);

    return nexusflow::ErrorCode::SUCCESS;
}

nexusflow::ErrorCode MyAlarmPusherModule::Init() {
    m_outFile.open(m_savePath, std::ios::out);
    if (!m_outFile.is_open()) {
        LOG_ERROR("MyAlarmPusherModule::Init, name={}, open file failed, path={}", GetModuleName(), m_savePath);
        return nexusflow::ErrorCode::FAILED_TO_OPEN_FILE;
    }

    LOG_INFO("MyAlarmPusherModule::Init, name={}, savePath={}", GetModuleName(), m_savePath);
    return nexusflow::ErrorCode::SUCCESS;
}

nexusflow::ErrorCode MyAlarmPusherModule::DeInit() {
    if (m_outFile.is_open()) m_outFile.close();
    LOG_INFO("MyAlarmPusherModule::DeInit, name={}", GetModuleName());
    return nexusflow::ErrorCode::SUCCESS;
}

void MyAlarmPusherModule::Process(nexusflow::Message& inputMessage) {
    if (auto* msg = inputMessage.BorrowPtr<InferenceMessage>()) {
        m_outFile << msg->toString() << std::endl;
    }
}
