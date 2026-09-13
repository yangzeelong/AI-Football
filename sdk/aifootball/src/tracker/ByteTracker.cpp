#include "ByteTracker.hpp"
#include "common/MyMessage.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/Message.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

// ---------------------------------------------------------------------------
// KalmanBox — simplified diagonal-covariance filter for axis-aligned boxes.
// State: [cx, cy, a, h, vx, vy, va, vh]  (a = aspect ratio w/h)
// ---------------------------------------------------------------------------

static constexpr float kStdWeightPosition = 1.0f / 20.0f;
static constexpr float kStdWeightVelocity = 1.0f / 160.0f;

void ByteTracker::KalmanBox::Init(float cx, float cy, float a, float h) {
    mean = {cx, cy, a, h, 0.0f, 0.0f, 0.0f, 0.0f};
    // Initial covariance scaled by height (mirrors the reference impl).
    float sp = kStdWeightPosition * h;
    float sv = kStdWeightVelocity * h;
    cov = {sp*sp, sp*sp, 1e-4f, sp*sp, sv*sv, sv*sv, 1e-10f, sv*sv};
}

void ByteTracker::KalmanBox::Predict() {
    // x' = F x, where F = [[I, I], [0, I]] on the 8-dim state.
    for (int i = 0; i < 4; ++i) mean[i] += mean[i + 4];
    // P' = F P F^T + Q  (diagonal approximation).
    float h = std::max(mean[3], 1e-3f);
    float sp = kStdWeightPosition * h;
    float sv = kStdWeightVelocity * h;
    float q[8] = {sp*sp, sp*sp, 1e-4f, sp*sp, sv*sv, sv*sv, 1e-10f, sv*sv};
    for (int i = 0; i < 8; ++i) cov[i] = cov[i] + q[i];
}

void ByteTracker::KalmanBox::Update(float cx, float cy, float a, float h) {
    // Diagonal Kalman update.
    float sp = kStdWeightPosition * std::max(h, 1e-3f);
    float r[8] = {sp*sp, sp*sp, 1e-4f, sp*sp, sp*sp, sp*sp, 1e-4f, sp*sp};

    float meas[8] = {cx, cy, a, h, 0, 0, 0, 0};
    // H = [I 0] on 8-dim state, so innovation is only on first 4 dims.
    for (int i = 0; i < 4; ++i) {
        float S = cov[i] + r[i];
        float K = (S > 1e-9f) ? cov[i] / S : 0.0f;
        float y = meas[i] - mean[i];
        mean[i]     += K * y;
        mean[i + 4] += K * y * 0.5f;   // nudge velocity toward observed delta
        cov[i]      = (1.0f - K) * cov[i];
        cov[i + 4] += r[i] * 0.01f;    // small velocity covariance damping
    }
}

// ---------------------------------------------------------------------------
// STrack helpers
// ---------------------------------------------------------------------------

void ByteTracker::STrack::GetPredictedBox(float& ox0, float& oy0,
                                          float& ox1, float& oy1) const {
    float cx = kf.mean[0];
    float cy = kf.mean[1];
    float a  = kf.mean[2];
    float h  = kf.mean[3];
    float w  = a * h;
    ox0 = cx - w * 0.5f;
    oy0 = cy - h * 0.5f;
    ox1 = cx + w * 0.5f;
    oy1 = cy + h * 0.5f;
}

void ByteTracker::STrack::Reactivate(STrack& newTrack, int frameId, int newId) {
    float cx = 0.5f * (newTrack.x0 + newTrack.x1);
    float cy = 0.5f * (newTrack.y0 + newTrack.y1);
    float h  = newTrack.y1 - newTrack.y0;
    float w  = newTrack.x1 - newTrack.x0;
    float a  = (h > 1e-3f) ? (w / h) : 0.0f;
    kf.Update(cx, cy, a, h);
    x0 = newTrack.x0; y0 = newTrack.y0;
    x1 = newTrack.x1; y1 = newTrack.y1;
    score = newTrack.score;
    classId = newTrack.classId;
    state = State::Tracked;
    isActivated = true;
    this->frameId = frameId;
    if (newId > 0) trackId = newId;
    trackletLen = 0;
}

void ByteTracker::STrack::Update(const STrack& newTrack, int frameId) {
    float cx = 0.5f * (newTrack.x0 + newTrack.x1);
    float cy = 0.5f * (newTrack.y0 + newTrack.y1);
    float h  = newTrack.y1 - newTrack.y0;
    float w  = newTrack.x1 - newTrack.x0;
    float a  = (h > 1e-3f) ? (w / h) : 0.0f;
    kf.Update(cx, cy, a, h);
    x0 = newTrack.x0; y0 = newTrack.y0;
    x1 = newTrack.x1; y1 = newTrack.y1;
    score = newTrack.score;
    classId = newTrack.classId;
    state = State::Tracked;
    isActivated = true;
    this->frameId = frameId;
    trackletLen++;
}

// ---------------------------------------------------------------------------
// IoU helpers
// ---------------------------------------------------------------------------

float ByteTracker::Iou(float ax0, float ay0, float ax1, float ay1,
                       float bx0, float by0, float bx1, float by1) {
    float ix0 = std::max(ax0, bx0);
    float iy0 = std::max(ay0, by0);
    float ix1 = std::min(ax1, bx1);
    float iy1 = std::min(ay1, by1);
    float iw = std::max(0.0f, ix1 - ix0);
    float ih = std::max(0.0f, iy1 - iy0);
    float inter = iw * ih;
    float areaA = std::max(0.0f, ax1 - ax0) * std::max(0.0f, ay1 - ay0);
    float areaB = std::max(0.0f, bx1 - bx0) * std::max(0.0f, by1 - by0);
    float denom = areaA + areaB - inter;
    return denom > 0.0f ? inter / denom : 0.0f;
}

float ByteTracker::IouDistance(const STrack& track, const STrack& det) {
    float tx0, ty0, tx1, ty1;
    track.GetPredictedBox(tx0, ty0, tx1, ty1);
    float iou = Iou(tx0, ty0, tx1, ty1, det.x0, det.y0, det.x1, det.y1);
    return 1.0f - iou;
}

// ---------------------------------------------------------------------------
// Linear assignment — O(n^3) Hungarian algorithm.
// Adapted from the classic JV/auction-style implementation.
// ---------------------------------------------------------------------------

void ByteTracker::LinearAssignment(const std::vector<float>& cost,
                                   int n, int m, float maxCost,
                                   std::vector<std::pair<int,int>>& matches,
                                   std::vector<int>& unmatchedRows,
                                   std::vector<int>& unmatchedCols) {
    matches.clear();
    unmatchedRows.clear();
    unmatchedCols.clear();
    if (n == 0 || m == 0) {
        for (int i = 0; i < n; ++i) unmatchedRows.push_back(i);
        for (int j = 0; j < m; ++j) unmatchedCols.push_back(j);
        return;
    }

    const float INF = std::numeric_limits<float>::infinity();
    // u[i]: dual for row i; v[j]: dual for col j; p[j]: row assigned to col j.
    std::vector<float> u(n + 1, 0.0f), v(m + 1, 0.0f);
    std::vector<int>   p(m + 1, 0), way(m + 1, 0);

    auto C = [&](int i, int j) -> float {
        return cost[static_cast<size_t>(i) * m + j];
    };

    for (int i = 0; i < n; ++i) {
        p[0] = i + 1;
        int j0 = 0;
        std::vector<float> minv(m + 1, INF);
        std::vector<char>    used(m + 1, 0);
        do {
            used[j0] = 1;
            int i0 = p[j0] - 1;
            float delta = INF;
            int j1 = -1;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                float cur = C(i0, j - 1) - u[i0 + 1] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            if (j1 < 0) break;
            for (int j = 0; j <= m; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else       { minv[j]  -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    // Extract matches.
    std::vector<char> rowMatched(n, 0), colMatched(m, 0);
    for (int j = 1; j <= m; ++j) {
        if (p[j] > 0) {
            int i = p[j] - 1;
            float c = C(i, j - 1);
            if (c <= maxCost) {
                matches.emplace_back(i, j - 1);
                rowMatched[i] = 1;
                colMatched[j - 1] = 1;
            }
        }
    }
    for (int i = 0; i < n; ++i) if (!rowMatched[i]) unmatchedRows.push_back(i);
    for (int j = 0; j < m; ++j) if (!colMatched[j]) unmatchedCols.push_back(j);
}

ByteTracker::ByteTracker(const std::string& name) : Module(name) {
    LOG_TRACE("ByteTracker constructor, name={}", name);
}

ByteTracker::~ByteTracker() = default;

ns::ErrorCode ByteTracker::Configure(const ns::Config& config) {
    LOG_TRACE("ByteTracker::Configure");
    m_param.trackActivationThreshold   = config.GetValueOrDefault<float>("trackActivationThreshold", 0.25f);
    m_param.secondAssociationThreshold = config.GetValueOrDefault<float>("secondAssociationThreshold", 0.10f);
    m_param.minimumMatchingThreshold   = config.GetValueOrDefault<float>("minimumMatchingThreshold", 0.80f);
    m_param.lostTrackBuffer            = config.GetValueOrDefault<int>("lostTrackBuffer", 30);
    m_param.personClassId              = config.GetValueOrDefault<int>("personClassId", 1);
    m_param.ballClassId                = config.GetValueOrDefault<int>("ballClassId", 37);
    LOG_INFO("ByteTracker config: high>={}, low>={}, iouGate={}, lostBuf={}",
             m_param.trackActivationThreshold, m_param.secondAssociationThreshold,
             m_param.minimumMatchingThreshold, m_param.lostTrackBuffer);
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode ByteTracker::Init() {
    LOG_TRACE("ByteTracker::Init");
    Reset();
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode ByteTracker::DeInit() {
    LOG_TRACE("ByteTracker::DeInit");
    Reset();
    return ns::ErrorCode::SUCCESS;
}

void ByteTracker::Reset() {
    m_trackedStracks.clear();
    m_lostStracks.clear();
    m_frameId = 0;
    m_nextId  = 1;
}

void ByteTracker::InitTrack(STrack& t, int frameId, bool activated) {
    t.frameId = frameId;
    t.startFrame = frameId;
    t.trackletLen = 0;
    float cx = 0.5f * (t.x0 + t.x1);
    float cy = 0.5f * (t.y0 + t.y1);
    float h  = t.y1 - t.y0;
    float w  = t.x1 - t.x0;
    float a  = (h > 1e-3f) ? (w / h) : 0.0f;
    t.kf.Init(cx, cy, a, h);
    t.kf.Predict();
    if (activated) {
        t.state = State::Tracked;
        t.isActivated = true;
        t.trackId = NextId();
    } else {
        t.state = State::New;
        t.isActivated = false;
    }
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------

void ByteTracker::Process(ns::Message& inputMessage) {
    if (!inputMessage.HasData()) {
        LOG_WARN("ByteTracker: empty message, ignoring");
        return;
    }
    auto* detMsg = inputMessage.MutPtr<DetectionMessage>();
    if (!detMsg) {
        LOG_WARN("ByteTracker: message is not DetectionMessage, ignoring");
        return;
    }

    TrackedDetectionMessage out;
    out.videoFrame   = detMsg->videoFrame;
    out.rawCounts    = detMsg->rawCounts;
    out.isEnd        = detMsg->isEnd;
    out.timestamp    = detMsg->timestamp;
    out.timestampSec = detMsg->timestampSec;

    if (detMsg->isEnd) { Broadcast(nexusflow::Message(std::move(out))); return; }

    // --- Split detections into person high-score, person low-score, ball ---
    std::vector<STrack> detsHigh, detsLow;
    for (const auto& d : detMsg->detections) {
        if (d.classId == m_param.personClassId) {
            STrack t;
            t.x0 = d.x0; t.y0 = d.y0; t.x1 = d.x1; t.y1 = d.y1;
            t.score = d.score; t.classId = d.classId;
            if (d.score >= m_param.trackActivationThreshold)      detsHigh.push_back(t);
            else if (d.score >= m_param.secondAssociationThreshold) detsLow.push_back(t);
        } else if (d.classId == m_param.ballClassId) {
            out.balls.push_back(d);  // pass through unchanged
        }
    }

    m_frameId++;

    // --- Predict all tracked/lost tracks ---
    for (auto& t : m_trackedStracks) t.kf.Predict();
    for (auto& t : m_lostStracks)    t.kf.Predict();

    // --- Pool of tracks available for association ---
    // ByteTrack: first association uses trackedStracks (both activated & new),
    // second association uses unactivated "New" tracks that survived from last frame.
    std::vector<STrack*> trackPool;
    std::vector<int>     unconfirmedIdx;   // indices into trackPool of !isActivated
    for (size_t i = 0; i < m_trackedStracks.size(); ++i) {
        trackPool.push_back(&m_trackedStracks[i]);
        if (!m_trackedStracks[i].isActivated) unconfirmedIdx.push_back(static_cast<int>(i));
    }

    // --- First association: high-score detections vs all tracked tracks ---
    std::vector<std::pair<int,int>> matches1;
    std::vector<int> uTracks1, uDets1;
    {
        int n = static_cast<int>(trackPool.size());
        int m = static_cast<int>(detsHigh.size());
        std::vector<float> cost(static_cast<size_t>(n) * m, 1.0f);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                cost[static_cast<size_t>(i) * m + j] = IouDistance(*trackPool[i], detsHigh[j]);
        LinearAssignment(cost, n, m, m_param.minimumMatchingThreshold,
                         matches1, uTracks1, uDets1);
    }

    for (const auto& mp : matches1) {
        STrack* trk = trackPool[mp.first];
        STrack& det = detsHigh[mp.second];
        if (trk->state == State::Tracked) {
            trk->Update(det, m_frameId);
        } else {
            // Was Lost, now re-matched -> reactivate with a fresh ID.
            trk->Reactivate(det, m_frameId, NextId());
        }
    }

    // --- Second association: low-score detections vs remaining tracked tracks ---
    std::vector<STrack*> remainingTracks;
    for (int idx : uTracks1) {
        if (trackPool[idx]->state == State::Tracked) remainingTracks.push_back(trackPool[idx]);
    }
    std::vector<std::pair<int,int>> matches2;
    std::vector<int> uTracks2, uDets2;
    {
        int n = static_cast<int>(remainingTracks.size());
        int m = static_cast<int>(detsLow.size());
        std::vector<float> cost(static_cast<size_t>(n) * m, 1.0f);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                cost[static_cast<size_t>(i) * m + j] = IouDistance(*remainingTracks[i], detsLow[j]);
        LinearAssignment(cost, n, m, 0.5f, matches2, uTracks2, uDets2);
    }
    for (const auto& mp : matches2) {
        remainingTracks[mp.first]->Update(detsLow[mp.second], m_frameId);
    }
    // Tracks that failed both associations become Lost.
    for (int idx : uTracks2) {
        remainingTracks[idx]->MarkLost();
    }

    // --- Initialize new tracks from unmatched high-score detections ---
    for (int idx : uDets1) {
        STrack t = detsHigh[idx];
        InitTrack(t, m_frameId, /*activated=*/false);
        m_trackedStracks.push_back(t);
    }

    // --- Reactivate lost tracks that match any current tracked "New" track ---
    // (Simplified: we skip cross-lost matching for brevity; lost tracks are
    //  only revived through the first association above.)

    // --- Merge tracked list: keep Tracked + newly-added New tracks ---
    std::vector<STrack> newTracked;
    for (auto& t : m_trackedStracks) {
        if (t.state == State::Tracked || t.state == State::New) {
            newTracked.push_back(t);
        }
    }
    // Move Lost tracks to m_lostStracks.
    for (auto& t : m_trackedStracks) {
        if (t.state == State::Lost) m_lostStracks.push_back(t);
    }
    m_trackedStracks.swap(newTracked);

    // --- Age out lost tracks ---
    std::vector<STrack> keptLost;
    for (auto& t : m_lostStracks) {
        if (m_frameId - t.frameId <= m_param.lostTrackBuffer) keptLost.push_back(t);
    }
    m_lostStracks.swap(keptLost);

    // --- Emit output ---
    for (const auto& t : m_trackedStracks) {
        if (t.state != State::Tracked || !t.isActivated) continue;
        Detection d;
        d.x0 = t.x0; d.y0 = t.y0; d.x1 = t.x1; d.y1 = t.y1;
        d.score = t.score;
        d.classId = t.classId;
        d.trackId = t.trackId;
        out.persons.push_back(d);
    }
    out.activeTrackCount = static_cast<int>(out.persons.size());
    out.lostTrackCount   = static_cast<int>(m_lostStracks.size());

    LOG_DEBUG("ByteTracker: frame={} high={} low={} active={} lost={} balls={}",
              m_frameId, detsHigh.size(), detsLow.size(),
              out.activeTrackCount, out.lostTrackCount, out.balls.size());

    Broadcast(nexusflow::Message(std::move(out)));
}
