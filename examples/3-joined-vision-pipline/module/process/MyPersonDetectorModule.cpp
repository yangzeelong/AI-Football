
#include "MyPersonDetectorModule.hpp"
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
        box.label = 666;
        box.labelName = "PERSON-" + std::to_string(id);
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

MyPersonDetectorModule::MyPersonDetectorModule(const std::string& name) : Module(name) {
    LOG_TRACE("MyPersonDetectorModule constructor, name={}", name);
}

MyPersonDetectorModule::~MyPersonDetectorModule() { LOG_TRACE("MyPersonDetectorModule destructor, name={}", GetModuleName()); }

nexusflow::ErrorCode MyPersonDetectorModule::Configure(const nexusflow::Config& config) {
    m_modelPath = config.GetValueOrDefault("modelPath", std::string(""));
    LOG_INFO("MyPersonDetectorModule::Configure, name={}, modelPath={}", GetModuleName(), m_modelPath);
    return nexusflow::ErrorCode::SUCCESS;
}

nexusflow::ErrorCode MyPersonDetectorModule::Init() {
    LOG_INFO("Tring to load model from {}", m_modelPath);

    LOG_INFO("MyPersonDetectorModule::Init, name={}, modelPath={}", GetModuleName(), m_modelPath);
    return nexusflow::ErrorCode::SUCCESS;
}

void MyPersonDetectorModule::Process(nexusflow::Message& inputMessage) {
    if (auto* msg = inputMessage.MutPtr<InferenceMessage>()) {
        msg->boxes = DetectInfer(msg->videoFrame.frameId);
        LOG_INFO("'{}' Send message to next module, data={}", GetModuleName(), msg->toString());

        inputMessage.MetaData().sourceName = GetModuleName();
        Broadcast(inputMessage);
    }
}
