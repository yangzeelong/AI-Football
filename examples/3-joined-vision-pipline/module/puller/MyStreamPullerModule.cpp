
#include "MyStreamPullerModule.hpp"
#include "../MyMessage.hpp"
#include "../src/utils/logging.hpp" // TODO: remove
#include "nexusflow/Message.hpp"
#include <chrono>
#include <thread>

namespace {

DecoderMessage CreateMessage(uint32_t fps) {
    const int sendIntervalMs = 1000 / fps; // 25fps
    std::this_thread::sleep_for(std::chrono::milliseconds(sendIntervalMs));

    static int frameIdx = 0;
    DecoderMessage msg;
    msg.videoPackage = "package-" + std::to_string(frameIdx++);

    auto now = std::chrono::system_clock::now();
    msg.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    return msg;
}

} // namespace

MyStreamPullerModule::MyStreamPullerModule(const std::string& name) : Module(name) {
    LOG_TRACE("MyStreamPullerModule constructor, name={}", name);
}

MyStreamPullerModule::~MyStreamPullerModule() { LOG_TRACE("MyStreamPullerModule destructor, name={}", GetModuleName()); }

void MyStreamPullerModule::Process(nexusflow::Message& inputMessage) {
    // no input message
    if (inputMessage.HasData()) {
        LOG_WARN("MyStreamPullerModule: inputMessage is not nullptr");
        return;
    }

    constexpr uint32_t kFPS = 25;
    auto msg = CreateMessage(kFPS);
    nexusflow::Message dispatchMsg(msg);
    Broadcast(dispatchMsg);
}
