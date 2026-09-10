#ifndef MY_MESSAGE_HPP
#define MY_MESSAGE_HPP

#include <nexusflow/Message.hpp>
#include <sstream>

// --- Base ---
struct VideoFrame {
    uint32_t frameId;
    std::string frameData;
};

struct Rect {
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
};

struct Box {
    // --- Detection
    Rect rect;
    int label = 0;
    float score = 0.0f;
    std::string labelName;

    // --- Classification
    int clsLabel = 0;
    float clsScore = 0.0f;
    std::string clsLabelName;
};

// --- Message ---
struct DecoderMessage {
    std::string videoPackage;
    VideoFrame videoFrame;
    bool isKeyFrame = false;
    bool isEnd = false;
    uint64_t timestamp = 0; // ms

    std::string toString() const {
        std::ostringstream oss;
        oss << "[DecoderMessage] = {videoPackage=" << videoPackage << ", frameId=" << videoFrame.frameId
            << ", isKeyFrame=" << isKeyFrame << ", isEnd=" << isEnd << ", timestamp=" << timestamp
            << ", videoFrame=\n\tframeId=" << videoFrame.frameId << ", frameData=" << videoFrame.frameData << "\n}";
        return oss.str();
    }
};

struct InferenceMessage {
    VideoFrame videoFrame;
    std::vector<Box> boxes; // {x0, y0, x1, y1, score, label}

    std::string toString() const {
        std::ostringstream oss;
        oss << "[InferenceMessage] = {frameId=" << videoFrame.frameId << ", boxes=[" << std::endl;
        for (const auto& box : boxes) {
            oss << "\tx0=" << box.rect.x0 << ", y0=" << box.rect.y0 << ", x1=" << box.rect.x1 << ", y1=" << box.rect.y1
                << ", score=" << box.score << ", label=" << box.label << ", labelName=" << box.labelName
                << ", clsScore=" << box.clsScore << ", clsLabel=" << box.clsLabel << ", clsLabelName=" << box.clsLabelName
                << std::endl;
        }
        oss << "}";
        return oss.str();
    }
};

// --- Message Convert ---
static InferenceMessage ConvertDecoderMessageToInferenceMessage(const DecoderMessage& decoderMessage) {
    InferenceMessage inferenceMessage;
    inferenceMessage.videoFrame = decoderMessage.videoFrame;
    inferenceMessage.boxes = std::vector<Box>();
    return inferenceMessage;
}

#endif