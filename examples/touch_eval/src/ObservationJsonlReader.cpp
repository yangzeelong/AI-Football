#include "ObservationJsonlReader.hpp"

#include <json/json.h>

#include <memory>
#include <sstream>

namespace touch_eval {
namespace {

bool IsObservedState(const Json::Value& state) {
    return state.isString() && state.asString() == "observed";
}

float NumberAt(const Json::Value& array, Json::ArrayIndex index) {
    if (!array.isArray() || array.size() <= index) return 0.0f;
    if (!array[index].isNumeric()) return 0.0f;
    return static_cast<float>(array[index].asDouble());
}

} // namespace

bool ObservationJsonlReader::Open(const std::string& path, std::string* error) {
    m_file.open(path.c_str(), std::ios::in | std::ios::binary);
    if (!m_file.is_open()) {
        if (error) *error = "cannot open observations '" + path + "'";
        return false;
    }
    m_path = path;
    return true;
}

bool ObservationJsonlReader::ParseMetadataLine(const std::string& line) {
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    std::string errors;
    if (!reader->parse(line.data(), line.data() + line.size(), &root, &errors)) {
        return false;
    }
    const Json::Value& video = root["video"];
    if (!video.isObject()) return false;
    m_metadata.videoPath = video.get("video_path", std::string()).asString();
    m_metadata.cameraId = video.get("camera_id", std::string()).asString();
    m_metadata.fps = video.get("fps", 0.0).asDouble();
    m_metadata.width = video.get("width", 0).asInt();
    m_metadata.height = video.get("height", 0).asInt();
    m_metadata.stride = video.get("stride", 1).asInt();
    return true;
}

bool ObservationJsonlReader::ParseFrameLine(const std::string& line,
                                            TouchFrameInput* frame) {
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    std::string errors;
    if (!reader->parse(line.data(), line.data() + line.size(), &root, &errors)) {
        return false;
    }

    TouchFrameInput out;
    out.frameIndex = root.get("frame_index", 0).asUInt64();
    out.timestampSec = root.get("timestamp_sec", 0.0).asDouble();
    out.frameWidth = m_metadata.width;
    out.frameHeight = m_metadata.height;

    const Json::Value& persons = root["persons"];
    for (Json::ArrayIndex p = 0; p < persons.size(); ++p) {
        const Json::Value& person = persons[p];
        TouchFrameInput::Person parsedPerson;
        parsedPerson.trackId = person.get("track_id", -1).asInt();

        const Json::Value& box = person["bbox"];
        if (box.isArray() && box.size() >= 4) {
            const float height = NumberAt(box, 3) - NumberAt(box, 1);
            parsedPerson.heightPx = height > 0.0f ? height : 0.0f;
        }

        const Json::Value& keypoints = person["keypoints"];
        for (Json::ArrayIndex k = 0; k < keypoints.size(); ++k) {
            const int slot = static_cast<int>(k) - kFootKeypointIndices[0];
            if (slot < 0 || slot >= kFootKeypointCount) continue;
            const Json::Value& keypoint = keypoints[k];
            TouchFrameInput::FootPoint& foot = parsedPerson.feet[slot];
            foot.confidence =
                static_cast<float>(keypoint.get("confidence", 0.0).asDouble());
            // Interpolated (virtual) and missing keypoints are not evidence of
            // a touch: only a real observation may move the ball to a foot.
            foot.valid = IsObservedState(keypoint["state"]) &&
                         keypoint["x"].isNumeric() && keypoint["y"].isNumeric();
            if (!foot.valid) continue;
            foot.x = static_cast<float>(keypoint["x"].asDouble());
            foot.y = static_cast<float>(keypoint["y"].asDouble());
        }
        out.persons.push_back(parsedPerson);
    }

    const Json::Value& balls = root["balls"];
    for (Json::ArrayIndex b = 0; b < balls.size(); ++b) {
        const Json::Value& ball = balls[b];
        TouchFrameInput::Ball parsedBall;
        parsedBall.trackId = ball.get("track_id", -1).asInt();
        parsedBall.confidence =
            static_cast<float>(ball.get("confidence", 0.0).asDouble());
        parsedBall.observed = IsObservedState(ball["state"]);
        const Json::Value& center = ball["center"];
        if (center.isArray() && center.size() >= 2) {
            parsedBall.x = NumberAt(center, 0);
            parsedBall.y = NumberAt(center, 1);
        } else {
            const Json::Value& box = ball["bbox"];
            parsedBall.x = 0.5f * (NumberAt(box, 0) + NumberAt(box, 2));
            parsedBall.y = 0.5f * (NumberAt(box, 1) + NumberAt(box, 3));
        }
        out.balls.push_back(parsedBall);
    }

    *frame = out;
    return true;
}

bool ObservationJsonlReader::Next(TouchFrameInput* frame) {
    if (!frame) return false;
    std::string line;
    while (std::getline(m_file, line)) {
        ++m_lineNumber;
        if (line.empty() || line == "\r") continue;

        // Peek at "type" without building a full document twice: the metadata
        // line is a one-off, so a cheap substring test is enough here.
        const bool looksLikeMetadata =
            line.find("\"type\":\"metadata\"") != std::string::npos ||
            line.find("\"type\": \"metadata\"") != std::string::npos;
        if (looksLikeMetadata) {
            if (ParseMetadataLine(line)) {
                ++m_metadataLines;
            } else {
                ++m_skippedLines;
                m_lastSkipReason = "malformed metadata line " +
                                   std::to_string(m_lineNumber);
            }
            continue;
        }

        if (ParseFrameLine(line, frame)) {
            ++m_framesRead;
            return true;
        }
        std::ostringstream reason;
        reason << "skipped line " << m_lineNumber << " (unparseable or missing fields)";
        m_lastSkipReason = reason.str();
        ++m_skippedLines;
    }
    return false;
}

} // namespace touch_eval
