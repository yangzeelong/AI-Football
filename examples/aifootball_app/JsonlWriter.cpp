#include "JsonlWriter.hpp"

#include <json/json.h>

namespace aifootball_app {
namespace {

Json::Value FloatArray(const std::array<float, 4>& values) {
    Json::Value result(Json::arrayValue);
    for (float value : values) result.append(value);
    return result;
}

const char* KeypointStateName(aifootball::KeypointState state) {
    switch (state) {
        case aifootball::KeypointState::Observed: return "observed";
        case aifootball::KeypointState::Virtual: return "virtual";
        case aifootball::KeypointState::Missing: return "missing";
    }
    return "missing";
}

const char* BallStateName(aifootball::BallTrackState state) {
    switch (state) {
        case aifootball::BallTrackState::Observed: return "observed";
        case aifootball::BallTrackState::Predicted: return "predicted";
        case aifootball::BallTrackState::Lost: return "lost";
    }
    return "lost";
}

Json::StreamWriterBuilder& CompactWriter() {
    static Json::StreamWriterBuilder builder = [] {
        Json::StreamWriterBuilder value;
        value["indentation"] = "";
        value["commentStyle"] = "None";
        return value;
    }();
    return builder;
}

} // namespace

bool JsonlWriter::Open(const std::string& path, const std::string& videoPath,
                       double fps, int width, int height, int stride,
                       const std::string& cameraId) {
    m_file.open(path, std::ios::out | std::ios::trunc);
    if (!m_file.is_open()) return false;
    m_cameraId = cameraId;
    Json::Value metadata;
    metadata["type"] = "metadata";
    metadata["schema_version"] = 1;
    metadata["video"]["video_path"] = videoPath;
    metadata["video"]["fps"] = fps;
    metadata["video"]["width"] = width;
    metadata["video"]["height"] = height;
    metadata["video"]["frame_count"] = 0;
    metadata["video"]["stride"] = stride;
    metadata["video"]["camera_id"] = cameraId;
    m_file << Json::writeString(CompactWriter(), metadata) << '\n';
    return true;
}

void JsonlWriter::Write(const aifootball::ProcessResult& result) {
    Json::Value frame;
    frame["type"] = "frame";
    frame["frame_index"] = Json::UInt64(result.frameId);
    frame["timestamp_sec"] = result.timestampSec;
    frame["camera_id"] = m_cameraId;
    frame["persons"] = Json::arrayValue;
    for (const auto& person : result.persons) {
        Json::Value item;
        item["track_id"] = person.trackId;
        item["bbox"] = FloatArray(person.bbox);
        item["confidence"] = person.confidence;
        item["keypoints"] = Json::arrayValue;
        static const char* const names[26] = {
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
            "left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
            "right_small_toe", "right_heel", "neck", "pelvis", "thorax"};
        for (size_t i = 0; i < person.keypoints.size(); ++i) {
            const auto& keypoint = person.keypoints[i];
            Json::Value point;
            point["name"] = names[i];
            if (keypoint.state == aifootball::KeypointState::Missing) {
                point["x"] = Json::nullValue;
                point["y"] = Json::nullValue;
            } else {
                point["x"] = keypoint.x;
                point["y"] = keypoint.y;
            }
            point["confidence"] = keypoint.confidence;
            point["state"] = KeypointStateName(keypoint.state);
            item["keypoints"].append(point);
        }
        item["state"] = "observed";
        frame["persons"].append(item);
    }
    frame["balls"] = Json::arrayValue;
    for (const auto& ball : result.balls) {
        Json::Value item;
        item["track_id"] = ball.trackId;
        item["bbox"] = FloatArray(ball.bbox);
        item["center"] = Json::arrayValue;
        item["center"].append(ball.center[0]);
        item["center"].append(ball.center[1]);
        item["confidence"] = ball.confidence;
        item["state"] = BallStateName(ball.state);
        frame["balls"].append(item);
    }
    // 锥框（三分类版才有；COCO 版是空数组）——给调试渲染画红框用，别当成跟踪输出。
    frame["cones"] = Json::arrayValue;
    for (const auto& cone : result.cones) {
        Json::Value item;
        item["bbox"] = FloatArray(cone.bbox);
        item["confidence"] = cone.confidence;
        frame["cones"].append(item);
    }
    frame["raw_detection_counts"]["total"] = result.rawCounts.total;
    frame["raw_detection_counts"]["person"] = result.rawCounts.person;
    frame["raw_detection_counts"]["ball"] = result.rawCounts.ball;
    // 锥（classId 3）**只有检测器级**统计：跟踪阶段没有锥通道，所以 filteredCounts
    // 永远不含它（ByteTracker.cpp 只数 total/person/ball）。为了不污染
    // `raw_detection_counts`（对外口径 = 过滤后计数，与 Python observations 对齐），
    // 检测器原始计数单独出一块，四类都在，便于算"锥被判成球"的代理指标。
    frame["detector_counts"]["total"] = result.detectorCounts.total;
    frame["detector_counts"]["person"] = result.detectorCounts.person;
    frame["detector_counts"]["ball"] = result.detectorCounts.ball;
    frame["detector_counts"]["cone"] = result.detectorCounts.cone;
    for (const auto& timing : result.moduleTimingsMs) {
        frame["module_timings_ms"][timing.first] = timing.second;
    }
    m_file << Json::writeString(CompactWriter(), frame) << '\n';
}

void JsonlWriter::Close() {
    if (m_file.is_open()) {
        m_file.flush();
        m_file.close();
    }
}

} // namespace aifootball_app
