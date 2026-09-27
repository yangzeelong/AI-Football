#include "TouchJsonlWriter.hpp"

#include <json/json.h>

namespace aifootball_app {
namespace {

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

bool TouchJsonlWriter::Open(const std::string& path,
                            const std::string& videoPath, double fps,
                            const std::string& cameraId,
                            const touch_eval::TouchDetectorParams& params) {
    m_file.open(path, std::ios::out | std::ios::trunc);
    if (!m_file.is_open()) return false;

    Json::Value metadata;
    metadata["type"] = "metadata";
    metadata["schema_version"] = 1;
    metadata["video_path"] = videoPath;
    metadata["camera_id"] = cameraId;
    metadata["fps"] = fps;
    metadata["params"] = touch_eval::DescribeTouchDetectorParams(params);
    m_file << Json::writeString(CompactWriter(), metadata) << '\n';
    return true;
}

void TouchJsonlWriter::Write(const touch_eval::TouchEvent& event,
                             int eventIndex) {
    if (!m_file.is_open()) return;
    Json::Value item;
    item["type"] = "touch";
    item["event_index"] = eventIndex;
    item["frame_index"] = Json::UInt64(event.frameIndex);
    item["first_frame_index"] = Json::UInt64(event.firstFrameIndex);
    item["timestamp_sec"] = event.timestampSec;
    item["first_timestamp_sec"] = event.firstTimestampSec;
    item["person_track_id"] = event.personTrackId;
    item["foot"] = event.leftFoot ? "left" : "right";
    item["distance_px"] = event.distancePx;
    item["threshold_px"] = event.thresholdPx;
    item["ball_confidence"] = event.ballConfidence;
    item["ball_tracked"] = event.ballTracked;
    item["ball_track_id"] = event.ballTrackId;
    item["contact_frames"] = event.contactFrames;
    item["speed_before_px_per_sec"] = event.speedBeforePxPerSec;
    item["speed_after_px_per_sec"] = event.speedAfterPxPerSec;
    m_file << Json::writeString(CompactWriter(), item) << '\n';
}

void TouchJsonlWriter::Close() {
    if (m_file.is_open()) {
        m_file.flush();
        m_file.close();
    }
}

} // namespace aifootball_app
