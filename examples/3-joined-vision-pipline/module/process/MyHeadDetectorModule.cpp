
#include "MyHeadDetectorModule.hpp"
#include "../MyMessage.hpp"
#include "../src/utils/logging.hpp" // TODO: remove
#include "nexusflow/ErrorCode.hpp"
#include "nexusflow/Message.hpp"

namespace {

std::vector<Box> DetectInfer(uint64_t id) {
    std::vector<Box> boxes;
    // Generate boxes
    for (int i = 0; i < 10; i++) {
        Box box;
        box.label = 333;
        box.labelName = "HEAD-" + std::to_string(id);
        box.score = 1.0;
        box.rect.x0 = i * 10;
        box.rect.y0 = i * 10;
        box.rect.x1 = i * 10 + 100;
        box.rect.y1 = i * 10 + 100;
        boxes.push_back(box);
    }
    return boxes;
}

} // namespace

MyHeadDetectorModule::MyHeadDetectorModule(const std::string& name) : Module(name) {
    LOG_TRACE("MyHeadDetectorModule constructor, name={}", name);
}

MyHeadDetectorModule::~MyHeadDetectorModule() { LOG_TRACE("MyHeadDetectorModule destructor, name={}", GetModuleName()); }

nexusflow::ErrorCode MyHeadDetectorModule::Configure(const nexusflow::Config& config) {
    m_modelPath = config.GetValueOrDefault("modelPath", std::string(""));
    LOG_INFO("MyHeadDetectorModule::Configure, name={}, modelPath={}", GetModuleName(), m_modelPath);
    return nexusflow::ErrorCode::SUCCESS;
}

nexusflow::ErrorCode MyHeadDetectorModule::Init() {
    LOG_INFO("Tring to load model from {}", m_modelPath);
    LOG_INFO("MyHeadDetectorModule::Init, name={}, modelPath={}", GetModuleName(), m_modelPath);
    return nexusflow::ErrorCode::SUCCESS;
}

void MyHeadDetectorModule::Process(nexusflow::Message& inputMessage) {
    if (auto* msg = inputMessage.MutPtr<InferenceMessage>()) {
        msg->boxes = DetectInfer(msg->videoFrame.frameId);
        LOG_INFO("'{}' Send message to next module, data={}", GetModuleName(), msg->toString());

        inputMessage.MetaData().sourceName = GetModuleName();
        Broadcast(inputMessage);
    }
}
