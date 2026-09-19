#include "FootballTracker.hpp"
#include "common/MyMessage.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/Message.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

// ---------------------------------------------------------------------------

FootballTracker::FootballTracker(const std::string& name) : Module(name) {
    LOG_TRACE("FootballTracker constructor, name={}", name);
}
FootballTracker::~FootballTracker() = default;

ns::ErrorCode FootballTracker::Configure(const ns::Config& config) {
    LOG_TRACE("FootballTracker::Configure");
    m_param.maxMissedFrames = config.GetValueOrDefault<int>("maxMissedFrames", 12);
    m_param.maxAssociationDistancePx = config.GetValueOrDefault<float>("maxAssociationDistancePx", 180.0f);
    m_param.confidenceDecay = config.GetValueOrDefault<float>("confidenceDecay", 0.75f);
    m_param.ballClassId = config.GetValueOrDefault<int>("ballClassId", 37);
    LOG_INFO("FootballTracker config: maxMissed={}, maxAssocDist={}px, decay={}",
             m_param.maxMissedFrames, m_param.maxAssociationDistancePx, m_param.confidenceDecay);
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode FootballTracker::Init() {
    LOG_TRACE("FootballTracker::Init");
    m_track = TrackState{};
    m_nextTrackId = 1;
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode FootballTracker::DeInit() {
    LOG_TRACE("FootballTracker::DeInit");
    m_track = TrackState{};
    return ns::ErrorCode::SUCCESS;
}

// ---------------------------------------------------------------------------
// Track update / predict
// ---------------------------------------------------------------------------

void FootballTracker::UpdateFromDetection(const Detection& d, double ts) {
    float cx = 0.5f * (d.x0 + d.x1);
    float cy = 0.5f * (d.y0 + d.y1);
    float w  = d.x1 - d.x0;
    float h  = d.y1 - d.y0;

    if (!m_track.active) {
        m_track.active = true;
        m_track.trackId = (d.trackId >= 0) ? d.trackId : m_nextTrackId++;
        m_track.cx = cx; m_track.cy = cy;
        m_track.vx = 0;  m_track.vy = 0;
        m_track.w = w;   m_track.h = h;
        m_track.confidence = d.score;
        m_track.timestampSec = ts;
        m_track.missedFrames = 0;
        if (m_track.trackId >= m_nextTrackId) m_nextTrackId = m_track.trackId + 1;
        return;
    }

    double dt = ts - m_track.timestampSec;
    if (dt > 1e-6) {
        m_track.vx = static_cast<float>((cx - m_track.cx) / dt);
        m_track.vy = static_cast<float>((cy - m_track.cy) / dt);
    }
    m_track.cx = cx; m_track.cy = cy;
    m_track.w = w;   m_track.h = h;
    m_track.confidence = d.score;
    m_track.timestampSec = ts;
    m_track.missedFrames = 0;
}

void FootballTracker::Predict(double ts) {
    if (!m_track.active) return;
    m_track.missedFrames++;
    if (m_track.missedFrames > m_param.maxMissedFrames) {
        m_track.active = false;
        return;
    }
    double dt = std::max(0.0, ts - m_track.timestampSec);
    m_track.cx += m_track.vx * static_cast<float>(dt);
    m_track.cy += m_track.vy * static_cast<float>(dt);
    m_track.timestampSec = ts;
    m_track.confidence *= m_param.confidenceDecay;
}

void FootballTracker::Emit(BallTrackMessage& out) const {
    if (!m_track.active) return;
    BallTrack b;
    b.trackId = m_track.trackId;
    b.centerX = m_track.cx;
    b.centerY = m_track.cy;
    b.x0 = m_track.cx - m_track.w * 0.5f;
    b.y0 = m_track.cy - m_track.h * 0.5f;
    b.x1 = m_track.cx + m_track.w * 0.5f;
    b.y1 = m_track.cy + m_track.h * 0.5f;
    b.confidence = m_track.confidence;
    b.vx = m_track.vx;
    b.vy = m_track.vy;
    b.missedFrames = m_track.missedFrames;
    b.state = (m_track.missedFrames == 0) ? BallTrackState::Observed
                                          : BallTrackState::Predicted;
    out.balls.push_back(b);
}

bool FootballTracker::IsRescuableBall(
    const Detection& detection, const SmoothedPoseMessage& message) const {
    // Match Python's _rescue_ball_detections(): rejected balls are allowed
    // back into the candidate set only when pose support is strong enough and
    // the box is still plausibly a ball-sized object.
    if (detection.score < 0.05f || !message.videoFrame) return false;

    if (BallSupportScore(detection, message) < 0.35f) return false;

    const float imageWidth = static_cast<float>(message.videoFrame->width);
    const float imageHeight = static_cast<float>(message.videoFrame->height);
    const float maxBallSize = std::min(imageWidth, imageHeight) * 0.18f;
    return std::max(detection.width(), detection.height()) <= maxBallSize;
}

float FootballTracker::BallSupportScore(
    const Detection& detection, const SmoothedPoseMessage& message) const {
    if (message.persons.empty() || !message.videoFrame) return 0.0f;

    const float imageWidth = static_cast<float>(message.videoFrame->width);
    const float imageHeight = static_cast<float>(message.videoFrame->height);
    const float supportRadius = std::max(
        36.0f, std::min(imageWidth, imageHeight) * 0.08f);
    const float ballX = detection.centerX();
    const float ballY = detection.centerY();
    float best = 0.0f;

    // These are the project-26 indices used by the Python implementation:
    // left/right ankle, toes, and heel occupy indices 15..22.
    constexpr int kFootBegin = 15;
    constexpr int kFootEnd = 23;
    for (const auto& person : message.persons) {
        std::vector<std::pair<float, float>> footPoints;
        for (int i = kFootBegin; i < kFootEnd; ++i) {
            const auto& keypoint = person.keypoints[i];
            if (keypoint.confidence < 0.2f ||
                keypoint.state == KeypointState::Missing) {
                continue;
            }
            footPoints.emplace_back(keypoint.x, keypoint.y);
        }

        if (!footPoints.empty()) {
            float nearest = std::numeric_limits<float>::infinity();
            for (const auto& point : footPoints) {
                nearest = std::min(nearest,
                                   std::hypot(ballX - point.first,
                                              ballY - point.second));
            }
            if (nearest <= supportRadius) {
                best = std::max(best, 1.0f - nearest / supportRadius);
                // Match Python's rescue rule: when valid foot keypoints are
                // close enough, do not also apply the bbox fallback.
                continue;
            }
        }

        const float width = person.x1 - person.x0;
        const float height = person.y1 - person.y0;
        const float xMargin = std::max(12.0f, width * 0.08f);
        const float upperY = person.y1 - std::max(18.0f, height * 0.35f);
        const float lowerY = person.y1 + std::max(12.0f, height * 0.1f);
        if (person.x0 - xMargin <= ballX && ballX <= person.x1 + xMargin &&
            upperY <= ballY && ballY <= lowerY) {
            best = std::max(best, 0.7f);
        }
    }
    return best;
}

float FootballTracker::BackgroundPenalty(
    const Detection& detection, float supportScore, float associationScore,
    const SmoothedPoseMessage& message) const {
    if (supportScore > 0.0f || associationScore > 0.2f ||
        !message.videoFrame || message.videoFrame->height <= 0) {
        return 0.0f;
    }
    const float yRatio = detection.centerY() /
                         static_cast<float>(message.videoFrame->height);
    if (yRatio < 0.25f) return 0.55f;
    if (yRatio < 0.4f) return 0.25f;
    return 0.0f;
}

float FootballTracker::ScoreDetection(
    const Detection& detection, const std::pair<float, float>* predictedCenter,
    const SmoothedPoseMessage& message) const {
    const float supportScore = BallSupportScore(detection, message);
    float associationScore = 0.0f;
    if (predictedCenter) {
        float radius = m_param.maxAssociationDistancePx;
        if (m_track.active) {
            radius += std::min(120.0f,
                               static_cast<float>(m_track.missedFrames) * 20.0f);
        }
        if (radius > 0.0f) {
            const float distance = std::hypot(
                predictedCenter->first - detection.centerX(),
                predictedCenter->second - detection.centerY());
            associationScore = std::max(0.0f, 1.0f - distance / radius);
        }
    }
    return detection.score + 0.45f * supportScore +
           0.35f * associationScore -
           BackgroundPenalty(detection, supportScore, associationScore,
                             message);
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

void FootballTracker::Process(ns::Message& inputMessage) {
    if (!inputMessage.HasData()) {
        LOG_WARN("FootballTracker: empty message, ignoring"); return;
    }
    auto* smMsg = inputMessage.MutPtr<SmoothedPoseMessage>();
    if (!smMsg) {
        LOG_WARN("FootballTracker: message is not SmoothedPoseMessage, ignoring"); return;
    }

    BallTrackMessage out;
    out.videoFrame       = smMsg->videoFrame;
    out.persons          = smMsg->persons;   // pass-through
    // Public raw_detection_counts follows Python's post-ROI/application
    // detection count. Keep detector-level rawCounts internal to diagnostics.
    out.rawCounts        = smMsg->filteredCounts;
    out.activeTrackCount = smMsg->activeTrackCount;
    out.lostTrackCount   = smMsg->lostTrackCount;
    out.isEnd            = smMsg->isEnd;
    out.timestamp        = smMsg->timestamp;
    out.timestampSec     = smMsg->timestampSec;

    if (smMsg->isEnd) { Broadcast(nexusflow::Message(std::move(out))); return; }

    // Python performs this rescue after pose estimation. Keep the rejected
    // detections separate until now so the normal ByteTracker filtering and
    // raw count semantics remain unchanged.
    std::vector<Detection> candidates = smMsg->balls;
    for (const auto& detection : smMsg->rejectedBalls) {
        if (!IsRescuableBall(detection, *smMsg)) continue;
        candidates.push_back(detection);
        ++out.rawCounts.total;
        ++out.rawCounts.ball;
    }

    // --- Select best ball detection ---
    const Detection* best = nullptr;
    float bestScore = -std::numeric_limits<float>::infinity();
    std::pair<float, float> predictedCenter;
    const std::pair<float, float>* predictedCenterPtr = nullptr;
    if (m_track.active) {
        double dt = std::max(0.0, smMsg->timestampSec - m_track.timestampSec);
        predictedCenter = {
            m_track.cx + m_track.vx * static_cast<float>(dt),
            m_track.cy + m_track.vy * static_cast<float>(dt),
        };
        predictedCenterPtr = &predictedCenter;
    }
    for (const auto& d : candidates) {
        if (d.classId != m_param.ballClassId) continue;
        const float score = ScoreDetection(d, predictedCenterPtr, *smMsg);
        if (score > bestScore) {
            bestScore = score;
            best = &d;
        }
    }

    const float minimumScore = m_track.active ? 0.15f : 0.18f;
    if (bestScore < minimumScore) best = nullptr;

    if (best) {
        UpdateFromDetection(*best, smMsg->timestampSec);
    } else {
        Predict(smMsg->timestampSec);
    }

    Emit(out);

    LOG_DEBUG("FootballTracker: frame={} ballsIn={} active={} state={}",
              smMsg->videoFrame ? smMsg->videoFrame->frameId : 0, candidates.size(),
              m_track.active ? 1 : 0,
              m_track.active ? (m_track.missedFrames == 0 ? "observed" : "predicted") : "none");

    Broadcast(nexusflow::Message(std::move(out)));
}
