#include "ObservationWriter.hpp"
#include "common/MyMessage.hpp"
#include "common/PipelineCompletionSignal.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/Message.hpp>

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

// ---------------------------------------------------------------------------

ObservationWriter::ObservationWriter(const std::string& name) : Module(name) {
    LOG_TRACE("ObservationWriter constructor, name={}", name);
}
ObservationWriter::~ObservationWriter() {
    if (m_file.is_open()) m_file.close();
}

ns::ErrorCode ObservationWriter::Configure(const ns::Config& config) {
    LOG_TRACE("ObservationWriter::Configure");
    m_param.outputPath = config.GetValueOrDefault<std::string>("outputPath", "output/observations.jsonl");
    m_param.cameraId   = config.GetValueOrDefault<std::string>("cameraId", "C1");
    m_param.videoPath  = config.GetValueOrDefault<std::string>("videoPath", "");
    m_param.fps        = config.GetValueOrDefault<double>("fps", 25.0);
    m_param.width      = config.GetValueOrDefault<int>("width", 0);
    m_param.height     = config.GetValueOrDefault<int>("height", 0);
    m_param.frameCount = config.GetValueOrDefault<int>("frameCount", 0);
    m_param.stride     = config.GetValueOrDefault<int>("stride", 1);
    LOG_INFO("ObservationWriter config: out={}, camera={}", m_param.outputPath, m_param.cameraId);
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode ObservationWriter::Init() {
    LOG_TRACE("ObservationWriter::Init");
    // Ensure parent directory exists.
    size_t slash = m_param.outputPath.find_last_of('/');
    if (slash != std::string::npos && slash > 0) {
        std::string dir = m_param.outputPath.substr(0, slash);
        // Best-effort mkdir -p via system() is avoided; use mkdir for single level.
        ::mkdir(dir.c_str(), 0755);
    }
    m_file.open(m_param.outputPath, std::ios::out | std::ios::trunc);
    if (!m_file.is_open()) {
        LOG_ERROR("ObservationWriter: failed to open {}", m_param.outputPath);
        return ns::ErrorCode::FAILURE;
    }
    m_metadataWritten = false;
    m_framesWritten = 0;
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode ObservationWriter::DeInit() {
    LOG_TRACE("ObservationWriter::DeInit");
    if (m_file.is_open()) {
        m_file.flush();
        m_file.close();
    }
    LOG_INFO("ObservationWriter: wrote {} frames to {}", m_framesWritten, m_param.outputPath);
    return ns::ErrorCode::SUCCESS;
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

void ObservationWriter::JsonEscape(std::ostream& os, const std::string& s) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '"':  os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\b': os << "\\b";  break;
            case '\f': os << "\\f";  break;
            case '\n': os << "\\n";  break;
            case '\r': os << "\\r";  break;
            case '\t': os << "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    os << buf;
                } else {
                    os << c;
                }
        }
    }
    os << '"';
}

void ObservationWriter::JsonFloat(std::ostream& os, float v) {
    if (!std::isfinite(v)) { os << "null"; return; }
    char buf[32]; std::snprintf(buf, sizeof(buf), "%.4f", v);
    os << buf;
}

void ObservationWriter::JsonDouble(std::ostream& os, double v) {
    if (!std::isfinite(v)) { os << "null"; return; }
    char buf[32]; std::snprintf(buf, sizeof(buf), "%.6f", v);
    os << buf;
}

// ---------------------------------------------------------------------------
// Metadata + frame records
// ---------------------------------------------------------------------------

void ObservationWriter::WriteMetadata() {
    std::ostringstream os;
    os << "{\"type\":\"metadata\",\"schema_version\":1,\"video\":{";
    os << "\"video_path\":"; JsonEscape(os, m_param.videoPath); os << ",";
    os << "\"fps\":";       JsonDouble(os, m_param.fps);        os << ",";
    os << "\"width\":";     os << m_param.width;                os << ",";
    os << "\"height\":";    os << m_param.height;               os << ",";
    os << "\"frame_count\":"; os << m_param.frameCount;         os << ",";
    os << "\"stride\":";    os << m_param.stride;               os << ",";
    os << "\"camera_id\":"; JsonEscape(os, m_param.cameraId);
    os << "}}\n";
    m_file << os.str();
    m_metadataWritten = true;
}

static const char* KeypointStateStr(KeypointState s) {
    switch (s) {
        case KeypointState::Observed: return "observed";
        case KeypointState::Virtual:  return "virtual";
        case KeypointState::Missing:  return "missing";
    }
    return "missing";
}

static const char* BallStateStr(BallTrackState s) {
    switch (s) {
        case BallTrackState::Observed:  return "observed";
        case BallTrackState::Predicted: return "predicted";
        case BallTrackState::Lost:      return "lost";
    }
    return "lost";
}

void ObservationWriter::WriteFrame(const BallTrackMessage& msg) {
    const char* const* names = GetProject26Names();
    std::ostringstream os;
    os << "{\"type\":\"frame\",";
    os << "\"frame_index\":" << msg.videoFrame.frameId << ",";
    os << "\"timestamp_sec\":"; JsonDouble(os, msg.timestampSec); os << ",";
    os << "\"camera_id\":"; JsonEscape(os, m_param.cameraId); os << ",";

    // persons
    os << "\"persons\":[";
    for (size_t i = 0; i < msg.persons.size(); ++i) {
        if (i) os << ",";
        const auto& p = msg.persons[i];
        os << "{\"track_id\":" << p.trackId << ",";
        os << "\"bbox\":[";
        JsonFloat(os, p.x0); os << ","; JsonFloat(os, p.y0); os << ",";
        JsonFloat(os, p.x1); os << ","; JsonFloat(os, p.y1); os << "],";
        os << "\"confidence\":"; JsonFloat(os, p.detectionConfidence); os << ",";
        os << "\"keypoints\":[";
        for (int k = 0; k < kProjectKeypointCount; ++k) {
            if (k) os << ",";
            const auto& kp = p.keypoints[k];
            os << "{\"name\":"; JsonEscape(os, names[k]); os << ",";
            os << "\"x\":";
            if (kp.state == KeypointState::Missing) os << "null";
            else JsonFloat(os, kp.x);
            os << ",\"y\":";
            if (kp.state == KeypointState::Missing) os << "null";
            else JsonFloat(os, kp.y);
            os << ",\"confidence\":"; JsonFloat(os, kp.confidence);
            os << ",\"state\":\"" << KeypointStateStr(kp.state) << "\"}";
        }
        os << "],\"state\":\"observed\"}";
    }
    os << "],";

    // balls
    os << "\"balls\":[";
    for (size_t i = 0; i < msg.balls.size(); ++i) {
        if (i) os << ",";
        const auto& b = msg.balls[i];
        os << "{\"track_id\":" << b.trackId << ",";
        os << "\"bbox\":[";
        JsonFloat(os, b.x0); os << ","; JsonFloat(os, b.y0); os << ",";
        JsonFloat(os, b.x1); os << ","; JsonFloat(os, b.y1); os << "],";
        os << "\"center\":["; JsonFloat(os, b.centerX); os << ","; JsonFloat(os, b.centerY); os << "],";
        os << "\"confidence\":"; JsonFloat(os, b.confidence); os << ",";
        os << "\"state\":\"" << BallStateStr(b.state) << "\"}";
    }
    os << "],";

    // raw_detection_counts
    os << "\"raw_detection_counts\":{";
    os << "\"total\":"  << msg.rawCounts.total  << ",";
    os << "\"person\":" << msg.rawCounts.person << ",";
    os << "\"ball\":"   << msg.rawCounts.ball;
    os << "}}\n";

    m_file << os.str();
    m_framesWritten++;
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

void ObservationWriter::Process(ns::Message& inputMessage) {
    if (!inputMessage.HasData()) {
        LOG_WARN("ObservationWriter: empty message, ignoring"); return;
    }
    auto* msg = inputMessage.MutPtr<BallTrackMessage>();
    if (!msg) {
        LOG_WARN("ObservationWriter: message is not BallTrackMessage, ignoring"); return;
    }

    if (!m_file.is_open()) {
        LOG_WARN("ObservationWriter: file not open, dropping frame");
        return;
    }

    if (!m_metadataWritten) {
        // Try to infer width/height/fps from the first frame if not configured.
        if (m_param.width  == 0) m_param.width  = msg->videoFrame.width;
        if (m_param.height == 0) m_param.height = msg->videoFrame.height;
        WriteMetadata();
    }

    if (msg->isEnd) {
        m_file.flush();
        // Signal main() that the pipeline has drained the last frame.
        PipelineCompletionSignal::Instance().NotifyComplete();
        LOG_INFO("ObservationWriter: EOF reached, completion signal fired");
        return;
    }

    WriteFrame(*msg);

    // Flush periodically to avoid losing data on crash.
    if ((m_framesWritten & 0x3F) == 0) m_file.flush();
}
