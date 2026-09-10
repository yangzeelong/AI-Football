
#include "MyHeadPersonFusionModule.hpp"
#include "../MyMessage.hpp"
#include "../src/utils/logging.hpp" // TODO: remove
#include "nexusflow/ErrorCode.hpp"
#include "nexusflow/Message.hpp"
#include <unordered_map>

MyHeadPersonFusionModule::MyHeadPersonFusionModule(const std::string& name) : Module(name) {
    LOG_TRACE("MyHeadPersonFusionModule constructor, name={}", name);
}

MyHeadPersonFusionModule::~MyHeadPersonFusionModule() { LOG_TRACE("MyHeadPersonFusionModule destructor, name={}", GetModuleName()); }

nexusflow::ErrorCode MyHeadPersonFusionModule::Configure(const nexusflow::Config& config) {
    m_modelPath = config.GetValueOrDefault("modelPath", std::string(""));
    LOG_INFO("MyHeadPersonFusionModule::Configure, name={}, modelPath={}", GetModuleName(), m_modelPath);
    return nexusflow::ErrorCode::SUCCESS;
}

nexusflow::ErrorCode MyHeadPersonFusionModule::Init() {
    LOG_INFO("Tring to load model from {}", m_modelPath);

    LOG_INFO("MyHeadPersonFusionModule::Init, name={}, modelPath={}", GetModuleName(), m_modelPath);
    return nexusflow::ErrorCode::SUCCESS;
}

void MyHeadPersonFusionModule::Process(nexusflow::Message& inputMessage) {
    const std::string headKey = "HeadDetector";
    const std::string personKey = "PersonDetector";

    using InputType = std::unordered_map<std::string, nexusflow::Message>;
    if (auto* msg = inputMessage.MutPtr<InputType>()) {
        for (auto& pair : *msg) {
            LOG_DEBUG("'{}' Receive message from previous module, data={}", pair.first,
                      pair.second.BorrowPtr<InferenceMessage>()->toString());
        }

        // Check if the input message contains the required keys
        if (msg->find(headKey) == msg->end() || msg->find(personKey) == msg->end()) {
            LOG_ERROR("Input message does not contain the required keys");
            return;
        }

        InferenceMessage* headMessage = msg->at(headKey).MutPtr<InferenceMessage>();
        InferenceMessage* personMessage = msg->at(personKey).MutPtr<InferenceMessage>();

        InferenceMessage fusedMessage = DoFusion(*headMessage, *personMessage);

        LOG_INFO("'{}' Send message to next module, data={}", GetModuleName(), fusedMessage.toString());

        Broadcast(nexusflow::MakeMessage(std::move(fusedMessage)));
    }
}

InferenceMessage MyHeadPersonFusionModule::DoFusion(const InferenceMessage& headMessage, const InferenceMessage& personMessage) const {
    InferenceMessage resultMessage;

    // or
    resultMessage.videoFrame = headMessage.videoFrame; // Assuming the video frame is the same for both messages
    // resultMessage.videoFrame = personMessage.videoFrame;

    // mock do fusion

    auto minSize = std::min(headMessage.boxes.size(), personMessage.boxes.size());
    resultMessage.boxes.resize(minSize);

    for (size_t i = 0; i < minSize; ++i) {
        auto& headBox = headMessage.boxes[i];
        auto& personBox = personMessage.boxes[i];
        auto& resultBox = resultMessage.boxes[i];

        resultBox.labelName = "Fusion(" + headBox.labelName + ", " + personBox.labelName + ")";
        resultBox.score = 0;
        resultBox.label = 0;
        resultBox.rect.x0 = resultBox.rect.y0 = resultBox.rect.x1 = resultBox.rect.y1 = 0;
    }

    return resultMessage;
}
