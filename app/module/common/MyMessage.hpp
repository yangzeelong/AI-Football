#ifndef MY_MESSAGE_HPP
#define MY_MESSAGE_HPP

#include <nexusflow/Message.hpp>
#include <sstream>
#include <vector>

// --- Base ---
struct VideoFrame {
    uint32_t frameId = 0;
    std::string frameData; // Raw pixel data (RGB24 by default)
    int width = 0;
    int height = 0;
    int channels = 3; // RGB24 = 3 channels
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

// --- Stream info (sent with the first PacketMessage for decoder initialization) ---
struct StreamInfo {
    int codecId = 0;      // AVCodecID value
    int width = 0;
    int height = 0;
    int pixFmt = 0;       // AVPixelFormat value
    std::string extradata; // Codec extradata (SPS/PPS for H.264, etc.)
    double fps = 25.0;
    int timeBaseNum = 0;
    int timeBaseDen = 1;
};

// --- Messages ---

// PacketMessage: output of VideoReader (demux), input of VideoDecoder
struct PacketMessage {
    std::string packetData; // Compressed NAL units
    int64_t pts = 0;        // Presentation timestamp (in stream time_base)
    int64_t dts = 0;        // Decode timestamp
    bool isKeyFrame = false;
    bool isEnd = false;     // EOF marker
    uint32_t packetIndex = 0;

    // Stream info: populated on first packet only
    bool hasStreamInfo = false;
    StreamInfo streamInfo;

    std::string toString() const {
        std::ostringstream oss;
        oss << "[PacketMessage] = {idx=" << packetIndex
            << ", size=" << packetData.size() << "B"
            << ", pts=" << pts << ", dts=" << dts
            << ", key=" << isKeyFrame
            << ", end=" << isEnd;
        if (hasStreamInfo) {
            oss << ", stream=" << streamInfo.width << "x" << streamInfo.height
                << " fps=" << streamInfo.fps;
        }
        oss << "}";
        return oss.str();
    }
};

// FrameMessage: output of VideoDecoder (decoded frame), input of Detector etc.
struct FrameMessage {
    VideoFrame videoFrame;
    bool isKeyFrame = false;
    bool isEnd = false;
    uint64_t timestamp = 0; // Wall-clock ms
    double timestampSec = 0.0; // PTS-based seconds

    std::string toString() const {
        std::ostringstream oss;
        oss << "[FrameMessage] = {frameId=" << videoFrame.frameId
            << ", " << videoFrame.width << "x" << videoFrame.height
            << ", ch=" << videoFrame.channels
            << ", dataSize=" << videoFrame.frameData.size() << "B"
            << ", isKeyFrame=" << isKeyFrame
            << ", isEnd=" << isEnd
            << ", ts=" << timestamp
            << ", tsSec=" << timestampSec << "}";
        return oss.str();
    }
};

// --- Detection (output of Detector module) ---
//
// Uses float coordinates in the *original frame* space so that downstream
// modules (tracker, pose estimator, ROI filter) do not need to know about
// the resize transform applied inside the detector.
struct Detection {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float score = 0.0f;
    int   classId = -1;   // RF-DETR COCO index (1 = person, 37 = sports ball, ...)
    int   trackId = -1;   // Filled by tracker; -1 for raw detections

    float width()  const { return x1 - x0; }
    float height() const { return y1 - y0; }
    float centerX() const { return 0.5f * (x0 + x1); }
    float centerY() const { return 0.5f * (y0 + y1); }
    float area()    const { return width() * height(); }
};

// Raw per-class detection counts, useful for downstream diagnostics /
// JSONL 'raw_detection_counts' field (mirrors Python observations.py).
struct DetectionCounts {
    int total = 0;
    int person = 0;
    int ball = 0;
};

struct DetectionMessage {
    VideoFrame videoFrame;                 // Decoded frame (COW string, cheap to copy)
    std::vector<Detection> detections;     // Detections in *original frame* coords
    DetectionCounts rawCounts;             // Counts before threshold / class filtering
    bool isEnd = false;
    double timestampSec = 0.0;
    uint64_t timestamp = 0;

    std::string toString() const {
        std::ostringstream oss;
        oss << "[DetectionMessage] = {frameId=" << videoFrame.frameId
            << ", " << videoFrame.width << "x" << videoFrame.height
            << ", dets=" << detections.size()
            << ", person=" << rawCounts.person
            << ", ball=" << rawCounts.ball
            << ", isEnd=" << isEnd
            << ", tsSec=" << timestampSec << "}";
        return oss.str();
    }
};

// Backward compatibility alias
using DecoderMessage = FrameMessage;

// --- Tracked detection (output of PersonTracker / ByteTracker) ---
//
// Same as DetectionMessage but each Detection carries a valid trackId (>= 0)
// and track state info. Non-person detections (e.g. sports ball) are passed
// through with trackId = -1 unless a dedicated ball tracker assigns one.
enum class TrackState : int {
    New      = 0,
    Tracked  = 1,
    Lost     = 2,
    Removed  = 3,
};

struct TrackedDetectionMessage {
    VideoFrame videoFrame;
    std::vector<Detection> persons;   // person detections with trackId assigned
    std::vector<Detection> balls;     // ball detections (trackId may be -1 here)
    DetectionCounts rawCounts;        // copied from upstream Detector
    int   activeTrackCount = 0;       // # of tracks in Tracked state this frame
    int   lostTrackCount   = 0;       // # of tracks in Lost state this frame
    bool  isEnd = false;
    double timestampSec = 0.0;
    uint64_t timestamp = 0;

    std::string toString() const {
        std::ostringstream oss;
        oss << "[TrackedDetectionMessage] = {frameId=" << videoFrame.frameId
            << ", persons=" << persons.size()
            << ", balls=" << balls.size()
            << ", active=" << activeTrackCount
            << ", lost=" << lostTrackCount
            << ", isEnd=" << isEnd
            << ", tsSec=" << timestampSec << "}";
        return oss.str();
    }
};

// --- Pose estimation types ---
//
// Project 26-keypoint layout (mirrors AI-Football observations.py):
//   0..22  : wholebody keypoints from RTMPose (COCO-WholeBody 23)
//   23     : neck    (virtual, midpoint of left_shoulder/right_shoulder)
//   24     : pelvis  (virtual, midpoint of left_hip/right_hip)
//   25     : thorax  (virtual, midpoint of neck/pelvis)
//
// Names array is exposed via GetProject26Names() for downstream JSONL writer.
constexpr int kProjectKeypointCount = 26;
constexpr int kWholebodyKeypointCount = 23;

enum class KeypointState : int {
    Missing  = 0,
    Observed = 1,
    Virtual  = 2,
};

struct Keypoint2D {
    float x = 0.0f;
    float y = 0.0f;
    float confidence = 0.0f;
    KeypointState state = KeypointState::Missing;
};

struct PersonPose {
    int   trackId = -1;
    // Detection bbox (original frame coords)
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    float detectionConfidence = 0.0f;
    // 26 keypoints in *original frame* coordinates.
    Keypoint2D keypoints[kProjectKeypointCount];
};

struct PoseMessage {
    VideoFrame videoFrame;
    std::vector<PersonPose> persons;
    std::vector<Detection>  balls;      // pass-through from upstream tracker
    DetectionCounts rawCounts;
    int   activeTrackCount = 0;
    int   lostTrackCount   = 0;
    bool  isEnd = false;
    double timestampSec = 0.0;
    uint64_t timestamp = 0;

    std::string toString() const {
        std::ostringstream oss;
        oss << "[PoseMessage] = {frameId=" << videoFrame.frameId
            << ", persons=" << persons.size()
            << ", balls=" << balls.size()
            << ", isEnd=" << isEnd
            << ", tsSec=" << timestampSec << "}";
        return oss.str();
    }
};

// Returns the canonical 26-keypoint name list.
inline const char* const* GetProject26Names() {
    static const char* const names[kProjectKeypointCount] = {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_big_toe", "left_small_toe", "left_heel",
        "right_big_toe", "right_small_toe", "right_heel",
        "neck", "pelvis", "thorax",
    };
    return names;
}

// Wholebody(23) -> Project(26) index mapping. -1 means "no source" (virtual).
inline const int* GetWholebodyToProject26() {
    static const int map[kProjectKeypointCount] = {
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
        -1, -1, -1,   // neck / pelvis / thorax (virtual)
    };
    return map;
}

// --- Smoothed pose (output of KeypointSmoother) ---
//
// Same shape as PoseMessage but keypoints have been temporally smoothed.
// Kept as a distinct type so downstream modules can distinguish raw vs.
// smoothed poses at compile time.
struct SmoothedPoseMessage {
    VideoFrame videoFrame;
    std::vector<PersonPose> persons;
    std::vector<Detection>  balls;
    DetectionCounts rawCounts;
    int   activeTrackCount = 0;
    int   lostTrackCount   = 0;
    bool  isEnd = false;
    double timestampSec = 0.0;
    uint64_t timestamp = 0;

    std::string toString() const {
        std::ostringstream oss;
        oss << "[SmoothedPoseMessage] = {frameId=" << videoFrame.frameId
            << ", persons=" << persons.size()
            << ", balls=" << balls.size()
            << ", isEnd=" << isEnd
            << ", tsSec=" << timestampSec << "}";
        return oss.str();
    }
};

// --- Football (ball) tracking ---
//
// BallTrackState mirrors Python tracking.py:
//   Observed  : a detection was matched this frame
//   Predicted : no detection, position extrapolated from velocity
//   Lost      : missed for too long, no prediction emitted
enum class BallTrackState : int {
    Lost      = 0,
    Predicted = 1,
    Observed  = 2,
};

struct BallTrack {
    int   trackId = -1;
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;   // bbox (original frame coords)
    float centerX = 0, centerY = 0;
    float confidence = 0.0f;                 // decayed when predicted
    float vx = 0, vy = 0;                    // estimated velocity (px/frame)
    int   missedFrames = 0;                  // consecutive frames without detection
    BallTrackState state = BallTrackState::Lost;
};

struct BallTrackMessage {
    VideoFrame videoFrame;
    std::vector<PersonPose> persons;   // pass-through from smoother
    std::vector<BallTrack>  balls;     // tracked ball(s) this frame
    DetectionCounts rawCounts;
    int   activeTrackCount = 0;
    int   lostTrackCount   = 0;
    bool  isEnd = false;
    double timestampSec = 0.0;
    uint64_t timestamp = 0;

    std::string toString() const {
        std::ostringstream oss;
        oss << "[BallTrackMessage] = {frameId=" << videoFrame.frameId
            << ", persons=" << persons.size()
            << ", balls=" << balls.size()
            << ", isEnd=" << isEnd
            << ", tsSec=" << timestampSec << "}";
        return oss.str();
    }
};

struct InferenceMessage {
    VideoFrame videoFrame;
    std::vector<Box> boxes;

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
static InferenceMessage ConvertFrameMessageToInferenceMessage(const FrameMessage& frameMessage) {
    InferenceMessage inferenceMessage;
    inferenceMessage.videoFrame = frameMessage.videoFrame;
    inferenceMessage.boxes = std::vector<Box>();
    return inferenceMessage;
}

#endif
