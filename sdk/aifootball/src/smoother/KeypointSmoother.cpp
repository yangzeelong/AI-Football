#include "KeypointSmoother.hpp"
#include "common/MyMessage.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/Message.hpp>

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------

KeypointSmoother::KeypointSmoother(const std::string& name) : Module(name) {
    LOG_TRACE("KeypointSmoother constructor, name={}", name);
}
KeypointSmoother::~KeypointSmoother() = default;

ns::ErrorCode KeypointSmoother::Configure(const ns::Config& config) {
    LOG_TRACE("KeypointSmoother::Configure");
    m_param.enabled = config.GetValueOrDefault<bool>("enabled", true);
    m_param.method  = config.GetValueOrDefault<std::string>("method", "skeleton");
    m_param.staticWindow  = config.GetValueOrDefault<int>("staticWindow", 8);
    m_param.staticMotionPx = config.GetValueOrDefault<float>("staticMotionPx", 2.5f);
    m_param.deadbandPx    = config.GetValueOrDefault<float>("deadbandPx", 0.8f);
    m_param.lowConfidence = config.GetValueOrDefault<float>("lowConfidence", 0.3f);
    m_param.lowConfidenceAlpha = config.GetValueOrDefault<float>("lowConfidenceAlpha", 0.12f);
    m_param.staticAlpha   = config.GetValueOrDefault<float>("staticAlpha", 0.25f);
    m_param.movingAlpha   = config.GetValueOrDefault<float>("movingAlpha", 0.65f);
    m_param.skeletonAnchorAlpha = config.GetValueOrDefault<float>("skeletonAnchorAlpha", 0.35f);
    m_param.skeletonEndpointAlpha = config.GetValueOrDefault<float>("skeletonEndpointAlpha", 0.12f);
    m_param.skeletonMovingEndpointAlpha = config.GetValueOrDefault<float>("skeletonMovingEndpointAlpha", 0.45f);
    m_param.skeletonFastEndpointAlpha = config.GetValueOrDefault<float>("skeletonFastEndpointAlpha", 0.22f);
    m_param.skeletonFastMovingEndpointAlpha = config.GetValueOrDefault<float>("skeletonFastMovingEndpointAlpha", 0.62f);
    m_param.skeletonFastStaticStepPx = config.GetValueOrDefault<float>("skeletonFastStaticStepPx", 5.0f);
    m_param.skeletonMaxStaticStepPx  = config.GetValueOrDefault<float>("skeletonMaxStaticStepPx", 2.5f);
    m_param.maxLimbStretchRatio = config.GetValueOrDefault<float>("maxLimbStretchRatio", 1.35f);
    LOG_INFO("KeypointSmoother config: enabled={}, method={}", m_param.enabled, m_param.method);
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode KeypointSmoother::Init() {
    LOG_TRACE("KeypointSmoother::Init");
    m_tracks.clear();
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode KeypointSmoother::DeInit() {
    LOG_TRACE("KeypointSmoother::DeInit");
    m_tracks.clear();
    return ns::ErrorCode::SUCCESS;
}

// ---------------------------------------------------------------------------
// Role classification (project-26 indices)
// ---------------------------------------------------------------------------

bool KeypointSmoother::IsAnchorIdx(int i) {
    // left_shoulder(5), right_shoulder(6), left_hip(11), right_hip(12)
    return i == 5 || i == 6 || i == 11 || i == 12;
}
bool KeypointSmoother::IsFastEndpointIdx(int i) {
    // left_wrist(9), right_wrist(10), left_ankle(15), right_ankle(16)
    return i == 9 || i == 10 || i == 15 || i == 16;
}
bool KeypointSmoother::IsEndpointIdx(int i) {
    // fast endpoints + toes/heels (17..22)
    return IsFastEndpointIdx(i) || (i >= 17 && i <= 22);
}

// ---------------------------------------------------------------------------
// Static / moving detection from bbox-center history
// ---------------------------------------------------------------------------

bool KeypointSmoother::IsStatic(TrackHistory& h) {
    if (static_cast<int>(h.bboxCenters.size()) < 2) return true;
    float maxStep = 0.0f;
    for (size_t i = 1; i < h.bboxCenters.size(); ++i) {
        float dx = h.bboxCenters[i].first  - h.bboxCenters[i-1].first;
        float dy = h.bboxCenters[i].second - h.bboxCenters[i-1].second;
        float step = std::sqrt(dx*dx + dy*dy);
        if (step > maxStep) maxStep = step;
    }
    return maxStep <= m_param.staticMotionPx;
}

// ---------------------------------------------------------------------------
// Per-person smoothing
// ---------------------------------------------------------------------------

void KeypointSmoother::SmoothPerson(const PersonPose& in, TrackHistory& h,
                                    PersonPose& out) {
    // Update bbox-center history.
    float cx = 0.5f * (in.x0 + in.x1);
    float cy = 0.5f * (in.y0 + in.y1);
    h.bboxCenters.emplace_back(cx, cy);
    while (static_cast<int>(h.bboxCenters.size()) > m_param.staticWindow) {
        h.bboxCenters.pop_front();
    }
    const bool isStatic = IsStatic(h);

    // Copy detection-level fields.
    out.trackId = in.trackId;
    out.x0 = in.x0; out.y0 = in.y0; out.x1 = in.x1; out.y1 = in.y1;
    out.detectionConfidence = in.detectionConfidence;

    if (!m_param.enabled || !h.hasPrev) {
        // First frame for this track: pass through.
        for (int i = 0; i < kProjectKeypointCount; ++i) out.keypoints[i] = in.keypoints[i];
        std::copy(in.keypoints, in.keypoints + kProjectKeypointCount, h.prev.begin());
        h.hasPrev = true;
        return;
    }

    // Per-keypoint EMA blend with role-dependent alpha.
    for (int i = 0; i < kProjectKeypointCount; ++i) {
        const Keypoint2D& cur = in.keypoints[i];
        const Keypoint2D& prev = h.prev[i];

        if (cur.state == KeypointState::Missing) {
            // Hold previous value (do not re-emit as observed).
            out.keypoints[i] = prev;
            out.keypoints[i].state = (prev.state == KeypointState::Observed)
                                     ? KeypointState::Observed
                                     : KeypointState::Missing;
            continue;
        }
        if (prev.state == KeypointState::Missing || !h.hasPrev) {
            out.keypoints[i] = cur;
            continue;
        }

        float dx = cur.x - prev.x;
        float dy = cur.y - prev.y;
        float dist = std::sqrt(dx*dx + dy*dy);

        float alpha;
        if (cur.confidence < m_param.lowConfidence) {
            alpha = m_param.lowConfidenceAlpha;
        } else if (IsAnchorIdx(i)) {
            alpha = m_param.skeletonAnchorAlpha;
        } else if (IsFastEndpointIdx(i)) {
            if (isStatic && dist <= m_param.skeletonFastStaticStepPx) {
                // Hold previous position (jitter suppression).
                out.keypoints[i] = prev;
                out.keypoints[i].confidence = cur.confidence;
                out.keypoints[i].state = cur.state;
                continue;
            }
            alpha = isStatic ? m_param.skeletonFastEndpointAlpha
                             : m_param.skeletonFastMovingEndpointAlpha;
        } else if (IsEndpointIdx(i)) {
            alpha = isStatic ? m_param.skeletonEndpointAlpha
                             : m_param.skeletonMovingEndpointAlpha;
        } else {
            alpha = isStatic ? m_param.staticAlpha : m_param.movingAlpha;
        }

        // Deadband: if motion is tiny, hold previous to avoid sub-pixel jitter.
        if (dist <= m_param.deadbandPx && !IsFastEndpointIdx(i)) {
            out.keypoints[i] = prev;
            out.keypoints[i].confidence = cur.confidence;
            out.keypoints[i].state = cur.state;
            continue;
        }

        Keypoint2D sm;
        sm.x = prev.x + alpha * dx;
        sm.y = prev.y + alpha * dy;
        sm.confidence = cur.confidence;
        sm.state = cur.state;
        out.keypoints[i] = sm;
    }

    // --- Limb-length constraint ---
    // Limb chains: (shoulder, elbow, wrist) and (hip, knee, ankle).
    static const int kLimbs[4][3] = {
        {5, 7, 9},   // left arm
        {6, 8, 10},  // right arm
        {11, 13, 15},// left leg
        {12, 14, 16} // right leg
    };
    for (auto& limb : kLimbs) {
        int a = limb[0], b = limb[1], c = limb[2];
        if (h.prev[a].state == KeypointState::Missing ||
            h.prev[b].state == KeypointState::Missing ||
            h.prev[c].state == KeypointState::Missing) continue;
        float prevLenAB = std::hypot(h.prev[b].x - h.prev[a].x, h.prev[b].y - h.prev[a].y);
        float prevLenBC = std::hypot(h.prev[c].x - h.prev[b].x, h.prev[c].y - h.prev[b].y);
        float curLenAB  = std::hypot(out.keypoints[b].x - out.keypoints[a].x,
                                     out.keypoints[b].y - out.keypoints[a].y);
        float curLenBC  = std::hypot(out.keypoints[c].x - out.keypoints[b].x,
                                     out.keypoints[c].y - out.keypoints[b].y);
        // If a limb stretched too much, revert the distal endpoint to previous.
        if (prevLenAB > 1e-3f && curLenAB > m_param.maxLimbStretchRatio * prevLenAB) {
            out.keypoints[b] = h.prev[b];
        }
        if (prevLenBC > 1e-3f && curLenBC > m_param.maxLimbStretchRatio * prevLenBC) {
            out.keypoints[c] = h.prev[c];
        }
    }

    // --- Recompute virtual midpoints ---
    auto midpoint = [&](int a, int b, int dst) {
        const Keypoint2D& ka = out.keypoints[a];
        const Keypoint2D& kb = out.keypoints[b];
        bool ok = (ka.state != KeypointState::Missing) && (kb.state != KeypointState::Missing);
        out.keypoints[dst].x = 0.5f * (ka.x + kb.x);
        out.keypoints[dst].y = 0.5f * (ka.y + kb.y);
        out.keypoints[dst].confidence = ok ? 0.5f * (ka.confidence + kb.confidence) : 0.0f;
        out.keypoints[dst].state = ok ? KeypointState::Virtual : KeypointState::Missing;
    };
    midpoint(5, 6, 23);   // neck
    midpoint(11, 12, 24); // pelvis
    midpoint(23, 24, 25); // thorax

    // Save smoothed state as "prev" for next frame.
    for (int i = 0; i < kProjectKeypointCount; ++i) h.prev[i] = out.keypoints[i];
    h.hasPrev = true;
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

void KeypointSmoother::Process(ns::Message& inputMessage) {
    if (!inputMessage.HasData()) {
        LOG_WARN("KeypointSmoother: empty message, ignoring"); return;
    }
    auto* poseMsg = inputMessage.MutPtr<PoseMessage>();
    if (!poseMsg) {
        LOG_WARN("KeypointSmoother: message is not PoseMessage, ignoring"); return;
    }

    SmoothedPoseMessage out;
    out.videoFrame       = poseMsg->videoFrame;
    out.balls            = poseMsg->balls;
    out.rejectedBalls    = poseMsg->rejectedBalls;
    out.rawCounts        = poseMsg->rawCounts;
    out.filteredCounts   = poseMsg->filteredCounts;
    out.activeTrackCount = poseMsg->activeTrackCount;
    out.lostTrackCount   = poseMsg->lostTrackCount;
    out.isEnd            = poseMsg->isEnd;
    out.timestamp        = poseMsg->timestamp;
    out.timestampSec     = poseMsg->timestampSec;

    if (poseMsg->isEnd) { Broadcast(nexusflow::Message(std::move(out))); return; }

    out.persons.reserve(poseMsg->persons.size());
    for (const auto& p : poseMsg->persons) {
        auto& h = m_tracks[p.trackId];
        PersonPose sm;
        SmoothPerson(p, h, sm);
        out.persons.push_back(sm);
    }

    // Evict tracks not seen this frame (simple policy: keep only current IDs).
    // A more sophisticated policy would use a TTL; here we just prune stale entries
    // when the map grows too large.
    if (m_tracks.size() > 4 * out.persons.size() + 64) {
        std::unordered_map<int, TrackHistory> kept;
        for (const auto& p : out.persons) {
            auto it = m_tracks.find(p.trackId);
            if (it != m_tracks.end()) kept.emplace(it->first, std::move(it->second));
        }
        m_tracks.swap(kept);
    }

    LOG_DEBUG("KeypointSmoother: frame={} persons={} tracks={}",
              poseMsg->videoFrame ? poseMsg->videoFrame->frameId : 0,
              out.persons.size(), m_tracks.size());
    Broadcast(nexusflow::Message(std::move(out)));
}
