#include <touch_eval/TouchDetector.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace touch_eval {
namespace {

float Clamp(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

int NormalizedCandidateCount(const TouchDetectorParams& params) {
    return params.candidateCount > 0 ? params.candidateCount : 1;
}

/// True when the ball carries a tracker id, i.e. it is the ball being
/// followed rather than one of the extra candidates.
bool IsTrackedBall(const TouchFrameInput::Ball& ball) { return ball.trackId > 0; }

bool IsUsableBall(const TouchFrameInput::Ball& ball,
                  const TouchDetectorParams& params) {
    return ball.observed && ball.confidence >= params.ballMinConfidence;
}

} // namespace

float ContactThresholdPx(const TouchDetectorParams& params, int frameWidth,
                         int frameHeight, float personHeightPx) {
    float low = std::max(0.0f, params.minContactPx);
    float high = std::max(low, params.maxContactPx);

    // Frame-level term: a fraction of the shorter side of the source frame.
    float shorterSide = static_cast<float>(
        std::min(frameWidth > 0 ? frameWidth : 0, frameHeight > 0 ? frameHeight : 0));
    float frameTerm = Clamp(shorterSide * params.contactDistanceRatio, low, high);

    // Person-level term: the feet of a distant player are small, so the ball
    // must be closer to them. Never wider than the frame term.
    float personTerm = Clamp(personHeightPx * params.personHeightRatio, low, high);
    return std::min(frameTerm, personTerm);
}

std::string DescribeTouchDetectorParams(const TouchDetectorParams& params) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(4);
    out << "keypointConfidence=" << params.keypointConfidence
        << " ballMinConfidence=" << params.ballMinConfidence
        << " contactDistanceRatio=" << params.contactDistanceRatio
        << " personHeightRatio=" << params.personHeightRatio
        << " minContactPx=" << params.minContactPx
        << " maxContactPx=" << params.maxContactPx
        << " releaseRatio=" << params.releaseRatio
        << " minContactFrames=" << params.minContactFrames
        << " maxGapFrames=" << params.maxGapFrames
        << " minSplitFrames=" << params.minSplitFrames
        << " minRiseFrames=" << params.minRiseFrames
        << " maxClusterFrames=" << params.maxClusterFrames
        << " candidateCount=" << params.candidateCount
        << " preferTrackedBall=" << (params.preferTrackedBall ? "true" : "false")
        << " speedWindowSec=" << params.speedWindowSec
        << " refractoryFrames=" << params.refractoryFrames
        << " minSpeedChangePxPerSec=" << params.minSpeedChangePxPerSec
        << " minSpeedRatio=" << params.minSpeedRatio
        << " requireTrackedBall="
        << (params.requireTrackedBall ? "true" : "false");
    return out.str();
}

std::vector<ContactCandidate> CollectContactCandidates(
    const TouchFrameInput& frame, const TouchDetectorParams& params,
    std::size_t maxCandidates) {
    std::vector<ContactCandidate> candidates;
    if (maxCandidates == 0) return candidates;

    // One candidate per (ball, person, side): the closest of that side's four
    // foot keypoints. Keeping the balls apart matters, because the followed
    // ball has to stay visible to SelectContact() even when an untracked
    // candidate happens to be closer.
    for (std::size_t ballIndex = 0; ballIndex < frame.balls.size(); ++ballIndex) {
        const TouchFrameInput::Ball& ball = frame.balls[ballIndex];
        if (!IsUsableBall(ball, params)) continue;

        for (std::size_t personIndex = 0; personIndex < frame.persons.size();
             ++personIndex) {
            const TouchFrameInput::Person& person = frame.persons[personIndex];
            for (int side = 0; side < 2; ++side) {
                const bool leftSide = (side == 0);
                ContactCandidate best;
                bool found = false;

                for (int slot = 0; slot < kFootKeypointCount; ++slot) {
                    if (IsLeftFootSlot(slot) != leftSide) continue;
                    const TouchFrameInput::FootPoint& foot = person.feet[slot];
                    if (!foot.valid) continue;
                    if (foot.confidence < params.keypointConfidence) continue;

                    const float distance = std::hypot(ball.x - foot.x,
                                                      ball.y - foot.y);
                    if (found && distance >= best.distancePx) continue;

                    best.distancePx = distance;
                    best.personHeightPx = person.heightPx;
                    best.personTrackId = person.trackId;
                    best.leftFoot = leftSide;
                    best.footSlot = slot;
                    best.ballX = ball.x;
                    best.ballY = ball.y;
                    best.ballTrackId = ball.trackId;
                    best.ballConfidence = ball.confidence;
                    best.ballTracked = IsTrackedBall(ball);
                    found = true;
                }

                if (found) candidates.push_back(best);
            }
        }
    }

    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const ContactCandidate& lhs, const ContactCandidate& rhs) {
                         return lhs.distancePx < rhs.distancePx;
                     });
    if (candidates.size() > maxCandidates) candidates.resize(maxCandidates);
    return candidates;
}

FrameEvidence BuildFrameEvidence(const TouchFrameInput& frame,
                                 const TouchDetectorParams& params) {
    FrameEvidence evidence;
    evidence.frameIndex = frame.frameIndex;
    evidence.timestampSec = frame.timestampSec;
    evidence.frameWidth = frame.frameWidth;
    evidence.frameHeight = frame.frameHeight;
    evidence.personCount = static_cast<int>(frame.persons.size());
    evidence.ballCount = static_cast<int>(frame.balls.size());
    for (std::size_t i = 0; i < frame.balls.size(); ++i) {
        if (IsUsableBall(frame.balls[i], params)) ++evidence.observedBallCount;
    }
    for (std::size_t i = 0; i < frame.persons.size(); ++i) {
        const TouchFrameInput::Person& person = frame.persons[i];
        int validPoints = 0;
        for (int slot = 0; slot < kFootKeypointCount; ++slot) {
            if (person.feet[slot].valid) ++validPoints;
        }
        evidence.validFootPointCount += validPoints;
        if (validPoints > 0) ++evidence.personsWithFeet;
    }
    evidence.candidates = CollectContactCandidates(
        frame, params, static_cast<std::size_t>(NormalizedCandidateCount(params)));
    return evidence;
}

FrameContact SelectContact(const FrameEvidence& evidence,
                           const TouchDetectorParams& params) {
    FrameContact out;
    out.frameIndex = evidence.frameIndex;
    out.timestampSec = evidence.timestampSec;
    out.frameWidth = evidence.frameWidth;
    out.frameHeight = evidence.frameHeight;
    out.personCount = evidence.personCount;
    out.observedBallCount = evidence.observedBallCount;
    out.validFootPointCount = evidence.validFootPointCount;
    out.personsWithFeet = evidence.personsWithFeet;

    if (evidence.candidates.empty()) return out;

    // candidates is sorted by distance, so index 0 is the closest pair. A
    // tracked-ball candidate wins whenever one exists: the pipeline follows
    // that ball, so a touch of any other candidate must not masquerade as one
    // of the followed ball.
    std::size_t chosen = 0;
    if (params.preferTrackedBall) {
        for (std::size_t i = 0; i < evidence.candidates.size(); ++i) {
            if (evidence.candidates[i].ballTracked) {
                chosen = i;
                break;
            }
        }
    }

    out.candidate = evidence.candidates[chosen];
    out.hasBall = true;
    out.thresholdPx = ContactThresholdPx(params, evidence.frameWidth,
                                         evidence.frameHeight,
                                         out.candidate.personHeightPx);
    if (out.candidate.distancePx <= out.thresholdPx) {
        out.level = ContactLevel::Contact;
    } else if (out.candidate.distancePx <=
               out.thresholdPx * params.releaseRatio) {
        out.level = ContactLevel::Band;
    } else {
        out.level = ContactLevel::None;
    }
    return out;
}

FrameContact SelectContact(const TouchFrameInput& frame,
                           const std::vector<ContactCandidate>& candidates,
                           const TouchDetectorParams& params) {
    FrameEvidence evidence = BuildFrameEvidence(frame, params);
    evidence.candidates = candidates;
    return SelectContact(evidence, params);
}

// ---------------------------------------------------------------------------
// TouchEventAggregator
// ---------------------------------------------------------------------------

TouchEventAggregator::TouchEventAggregator(const TouchDetectorParams& params)
    : m_params(params) {}

void TouchEventAggregator::Reset() {
    m_rejectedBySpeedGate = 0;
    m_rejectedByBallGate = 0;
    m_cluster.clear();
    m_active = false;
    m_gapFrames = 0;
    m_ballHistory.clear();
}

void TouchEventAggregator::RememberBall(const FrameContact& frame) {
    if (!frame.hasBall) return;
    BallSample sample;
    sample.timestampSec = frame.timestampSec;
    sample.x = frame.candidate.ballX;
    sample.y = frame.candidate.ballY;
    m_ballHistory.push_back(sample);

    // Keep the window bounded: it only feeds the speed diagnostics.
    const double keepSec = std::max(0.05, m_params.speedWindowSec) * 4.0;
    const double newest = m_ballHistory.back().timestampSec;
    while (m_ballHistory.size() > 1 &&
           newest - m_ballHistory.front().timestampSec > keepSec) {
        m_ballHistory.pop_front();
    }
}

float TouchEventAggregator::BallSpeed(double fromSec, double toSec) const {
    if (m_ballHistory.size() < 2) return 0.0f;
    if (toSec <= fromSec) return 0.0f;

    const BallSample* first = nullptr;
    const BallSample* last = nullptr;
    for (std::size_t i = 0; i < m_ballHistory.size(); ++i) {
        const BallSample& sample = m_ballHistory[i];
        if (sample.timestampSec < fromSec || sample.timestampSec > toSec) {
            continue;
        }
        if (!first) first = &sample;
        last = &sample;
    }
    if (!first || !last || first == last) return 0.0f;

    const double dt = last->timestampSec - first->timestampSec;
    if (dt <= 1e-6) return 0.0f;
    const double distance = std::hypot(
        static_cast<double>(last->x - first->x),
        static_cast<double>(last->y - first->y));
    return static_cast<float>(distance / dt);
}

std::vector<TouchEvent> TouchEventAggregator::Push(const FrameContact& frame) {
    RememberBall(frame);

    if (frame.level != ContactLevel::None && !m_active) {
        m_active = true;
        m_gapFrames = 0;
        m_cluster.clear();
    }
    if (!m_active) return std::vector<TouchEvent>();

    const bool inContact = (frame.level == ContactLevel::Contact);
    if (frame.level == ContactLevel::None) {
        // No admissible evidence (ball lost, far away, or no usable foot):
        // tolerate a few frames, then close. The hysteresis band below does
        // not consume this budget, so a ball sitting next to the foot keeps
        // the cluster alive while a lost ball ends it.
        if (m_gapFrames >= m_params.maxGapFrames) return CloseCluster();
        ++m_gapFrames;
    } else {
        m_gapFrames = 0;
    }

    ClusterFrame entry;
    entry.frame = frame;
    entry.inContact = inContact;
    m_cluster.push_back(entry);

    if (static_cast<int>(m_cluster.size()) >= std::max(1, m_params.maxClusterFrames)) {
        return CloseCluster();
    }
    return std::vector<TouchEvent>();
}

std::vector<TouchEvent> TouchEventAggregator::Flush() { return CloseCluster(); }

std::vector<TouchEvent> TouchEventAggregator::CloseCluster() {
    std::vector<TouchEvent> events;
    if (!m_active) return events;

    // Detach the cluster so the state machine is idle before the (const)
    // event construction below.
    std::vector<ClusterFrame> cluster;
    cluster.swap(m_cluster);
    m_active = false;
    m_gapFrames = 0;

    std::vector<int> contactIndices;
    contactIndices.reserve(cluster.size());
    for (std::size_t i = 0; i < cluster.size(); ++i) {
        if (cluster[i].inContact) contactIndices.push_back(static_cast<int>(i));
    }
    if (static_cast<int>(contactIndices.size()) < m_params.minContactFrames) {
        return events;
    }

    // Local minima of the distance over the contact frames of the cluster.
    std::vector<int> minima;
    const std::size_t count = contactIndices.size();
    for (std::size_t k = 0; k < count; ++k) {
        const float distance = cluster[contactIndices[k]].frame.candidate.distancePx;
        const bool belowLeft = (k == 0) ||
            distance <= cluster[contactIndices[k - 1]].frame.candidate.distancePx;
        const bool belowRight = (k + 1 == count) ||
            distance <= cluster[contactIndices[k + 1]].frame.candidate.distancePx;
        if (belowLeft && belowRight) minima.push_back(contactIndices[k]);
    }
    if (minima.empty()) minima.push_back(contactIndices[0]);

    // Merge minima that are too close to be two separate touches, and refuse
    // to split when the distance between them never rises well above the exit
    // threshold: a noisy single-touch valley must stay one event.
    std::vector<int> kept;
    for (std::size_t i = 0; i < minima.size(); ++i) {
        const int candidate = minima[i];
        if (kept.empty()) {
            kept.push_back(candidate);
            continue;
        }
        const int previous = kept.back();
        const bool tooClose =
            (candidate - previous) < std::max(1, m_params.minSplitFrames);
        int riseFrames = 0;
        if (!tooClose) {
            const float release =
                std::min(cluster[previous].frame.thresholdPx,
                         cluster[candidate].frame.thresholdPx) *
                m_params.releaseRatio;
            for (int m = previous + 1; m < candidate; ++m) {
                const FrameContact& between = cluster[m].frame;
                // A frame without an admissible ball counts as a rise: the
                // ball left the neighbourhood of the foot entirely.
                if (!between.hasBall ||
                    between.candidate.distancePx > release) {
                    ++riseFrames;
                }
            }
        }
        const bool separated =
            riseFrames >= std::max(1, m_params.minRiseFrames);
        if (tooClose || !separated) {
            // Keep the closer of the two minima.
            if (cluster[candidate].frame.candidate.distancePx <
                cluster[previous].frame.candidate.distancePx) {
                kept.back() = candidate;
            }
            continue;
        }
        kept.push_back(candidate);
    }

    // Farthest excursion between two minima: the natural cut between two
    // touches. Frames without an admissible ball rank highest, so a lost ball
    // separates the two encounters as well.
    const std::vector<ClusterFrame>& clusterRef = cluster;
    const auto peakIndex = [&clusterRef](int from, int to) -> int {
        int best = from + 1;
        float bestDistance = -1.0f;
        for (int m = from + 1; m < to; ++m) {
            const float distance = clusterRef[m].frame.hasBall
                ? clusterRef[m].frame.candidate.distancePx
                : std::numeric_limits<float>::infinity();
            if (distance > bestDistance) {
                bestDistance = distance;
                best = m;
            }
        }
        return best;
    };

    events.reserve(kept.size());
    for (std::size_t i = 0; i < kept.size(); ++i) {
        const int minimum = kept[i];

        // The event spans the encounter that contains its minimum. Events are
        // separated at the farthest excursion between two minima, so the
        // contact frames of one touch never leak into the next one.
        const int scanFrom =
            (i == 0) ? 0 : peakIndex(kept[i - 1], kept[i]);
        const int scanTo = (i + 1 < kept.size())
            ? peakIndex(kept[i], kept[i + 1])
            : static_cast<int>(cluster.size());

        int first = minimum;
        int contactFrames = 0;
        bool haveFirst = false;
        for (int m = scanFrom; m < scanTo; ++m) {
            if (!cluster[m].inContact) continue;
            if (!haveFirst) {
                first = m;
                haveFirst = true;
            }
            ++contactFrames;
        }
        if (contactFrames == 0) first = minimum;
        // A split segment still has to look like a touch on its own.
        if (contactFrames < std::max(1, m_params.minContactFrames)) continue;

        const FrameContact& frame = cluster[minimum].frame;
        TouchEvent event;
        event.frameIndex = frame.frameIndex;
        event.timestampSec = frame.timestampSec;
        event.firstFrameIndex = cluster[first].frame.frameIndex;
        event.firstTimestampSec = cluster[first].frame.timestampSec;
        event.personTrackId = frame.candidate.personTrackId;
        event.leftFoot = frame.candidate.leftFoot;
        event.distancePx = frame.candidate.distancePx;
        event.thresholdPx = frame.thresholdPx;
        event.ballConfidence = frame.candidate.ballConfidence;
        event.ballTracked = frame.candidate.ballTracked;
        event.ballTrackId = frame.candidate.ballTrackId;
        event.contactFrames = contactFrames;
        event.speedBeforePxPerSec =
            BallSpeed(frame.timestampSec - m_params.speedWindowSec,
                      frame.timestampSec);
        event.speedAfterPxPerSec =
            BallSpeed(frame.timestampSec,
                      frame.timestampSec + m_params.speedWindowSec);
        if (m_params.requireTrackedBall && !event.ballTracked) {
            // A leftover candidate produced this: the pipeline is not following
            // that object, so a touch of it is not something the SDK claims.
            ++m_rejectedByBallGate;
            continue;
        }
        if (!PassesSpeedGate(event)) {
            ++m_rejectedBySpeedGate;
            continue;
        }
        events.push_back(event);
    }
    return events;
}

bool TouchEventAggregator::PassesSpeedGate(const TouchEvent& event) const {
    const bool checkChange = m_params.minSpeedChangePxPerSec > 0.0f;
    const bool checkRatio = m_params.minSpeedRatio > 1.0f;
    if (!checkChange && !checkRatio) return true;

    const float before = event.speedBeforePxPerSec;
    const float after = event.speedAfterPxPerSec;
    if (checkChange &&
        std::fabs(after - before) < m_params.minSpeedChangePxPerSec) {
        return false;
    }
    if (checkRatio && before > 0.0f && after > 0.0f) {
        // Both speeds are known, so the ratio says something. When one of them
        // is unknown (the ball was not reported before or after the touch) the
        // ratio is meaningless and only the change gate applies.
        const float high = std::max(before, after);
        const float low = std::min(before, after);
        if (high / low < m_params.minSpeedRatio) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// TouchDetector
// ---------------------------------------------------------------------------

TouchDetector::TouchDetector(const TouchDetectorParams& params)
    : m_params(params), m_aggregator(params) {}

void TouchDetector::Reset() {
    m_aggregator.Reset();
    m_recentEvents.clear();
    m_suppressedByRefractory = 0;
}

std::vector<TouchEvent> TouchDetector::ApplyRefractory(
    const std::vector<TouchEvent>& events) {
    if (m_params.refractoryFrames <= 0 || events.empty()) {
        for (std::size_t i = 0; i < events.size(); ++i) {
            m_recentEvents.push_back(events[i]);
        }
        return events;
    }

    std::vector<TouchEvent> kept;
    for (std::size_t i = 0; i < events.size(); ++i) {
        const TouchEvent& event = events[i];
        bool repeated = false;
        for (std::size_t r = 0; r < m_recentEvents.size(); ++r) {
            const TouchEvent& recent = m_recentEvents[r];
            if (recent.personTrackId != event.personTrackId) continue;
            if (recent.leftFoot != event.leftFoot) continue;
            const long long gap = static_cast<long long>(event.frameIndex) -
                                  static_cast<long long>(recent.frameIndex);
            if (gap >= 0 && gap < m_params.refractoryFrames) {
                repeated = true;
                break;
            }
        }
        if (repeated) {
            ++m_suppressedByRefractory;
            continue;
        }
        m_recentEvents.push_back(event);
        kept.push_back(event);
    }

    // Only events inside the refractory window can matter; drop the rest so the
    // history stays O(1).
    const uint64_t newest = kept.empty() ? m_recentEvents.back().frameIndex
                                         : kept.back().frameIndex;
    while (!m_recentEvents.empty() &&
           newest - m_recentEvents.front().frameIndex >=
               static_cast<uint64_t>(m_params.refractoryFrames)) {
        m_recentEvents.pop_front();
    }
    return kept;
}

std::vector<TouchEvent> TouchDetector::PushEvidence(
    const FrameEvidence& evidence) {
    m_lastContact = SelectContact(evidence, m_params);
    return ApplyRefractory(m_aggregator.Push(m_lastContact));
}

std::vector<TouchEvent> TouchDetector::Push(const TouchFrameInput& frame) {
    return PushEvidence(BuildFrameEvidence(frame, m_params));
}

std::vector<TouchEvent> TouchDetector::Flush() {
    return ApplyRefractory(m_aggregator.Flush());
}

} // namespace touch_eval
