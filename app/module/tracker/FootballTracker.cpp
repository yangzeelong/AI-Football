#include "FootballTracker.hpp"
#include "common/MyMessage.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/Message.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

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
    out.rawCounts        = smMsg->rawCounts;
    out.activeTrackCount = smMsg->activeTrackCount;
    out.lostTrackCount   = smMsg->lostTrackCount;
    out.isEnd            = smMsg->isEnd;
    out.timestamp        = smMsg->timestamp;
    out.timestampSec     = smMsg->timestampSec;

    if (smMsg->isEnd) { Broadcast(nexusflow::Message(std::move(out))); return; }

    // --- Select best ball detection ---
    const Detection* best = nullptr;
    if (!m_track.active) {
        // Highest confidence.
        for (const auto& d : smMsg->balls) {
            if (d.classId != m_param.ballClassId) continue;
            if (!best || d.score > best->score) best = &d;
        }
    } else {
        // Predicted center for this frame.
        double dt = std::max(0.0, smMsg->timestampSec - m_track.timestampSec);
        float px = m_track.cx + m_track.vx * static_cast<float>(dt);
        float py = m_track.cy + m_track.vy * static_cast<float>(dt);

        float bestDist = std::numeric_limits<float>::infinity();
        const Detection* fallback = nullptr;  // highest-confidence fallback
        for (const auto& d : smMsg->balls) {
            if (d.classId != m_param.ballClassId) continue;
            if (!fallback || d.score > fallback->score) fallback = &d;
            float dcx = 0.5f * (d.x0 + d.x1);
            float dcy = 0.5f * (d.y0 + d.y1);
            float dist = std::hypot(dcx - px, dcy - py);
            if (dist <= m_param.maxAssociationDistancePx && dist < bestDist) {
                bestDist = dist;
                best = &d;
            }
        }
        if (!best) best = fallback;
    }

    if (best) {
        UpdateFromDetection(*best, smMsg->timestampSec);
    } else {
        Predict(smMsg->timestampSec);
    }

    Emit(out);

    LOG_DEBUG("FootballTracker: frame={} ballsIn={} active={} state={}",
              smMsg->videoFrame.frameId, smMsg->balls.size(),
              m_track.active ? 1 : 0,
              m_track.active ? (m_track.missedFrames == 0 ? "observed" : "predicted") : "none");

    Broadcast(nexusflow::Message(std::move(out)));
}
