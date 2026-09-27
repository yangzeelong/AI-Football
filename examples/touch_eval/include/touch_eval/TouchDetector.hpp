#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace touch_eval {

/**
 * @file TouchDetector.hpp
 * @brief Ball-touch detection core shared by the offline evaluator and the
 *        online demo app.
 *
 * The detector works on the same evidence the observations JSONL publishes:
 * ball centers and the 26-keypoint project layout, of which slots 15..22 are
 * the foot joints (left/right ankle, big toe, small toe and heel). A touch is
 * a sequence of frames in which the ball is observed close to a foot; the
 * frames are aggregated into one event and split again when the distance
 * clearly rises between two minima (two touches inside the same second).
 *
 * Everything here is pure computation: no I/O, no third-party dependency, and
 * no randomness, so the same input and parameters always produce the same
 * events. The offline evaluator caches the per-frame candidates once and then
 * replays the aggregation for many threshold settings; the app feeds the raw
 * frames straight in.
 */

/// Number of foot keypoints and their project-26 indices. Slot i of
/// TouchFrameInput::Person::feet corresponds to kFootKeypointIndices[i], so the
/// mapping is declared in exactly one place.
constexpr int kFootKeypointCount = 8;
constexpr int kFootKeypointIndices[kFootKeypointCount] = {
    15, 16, 17, 18, 19, 20, 21, 22};

/// Slot order mirrors kFootKeypointIndices: left ankle, right ankle, left big
/// toe, left small toe, left heel, right big toe, right small toe, right heel.
/// The split matches the left/right grouping used by the annotations.
constexpr bool IsLeftFootSlot(int slot) {
    return slot == 0 || slot == 2 || slot == 3 || slot == 4;
}

/// One frame of contact evidence, reconstructed either from ProcessResult or
/// from a line of observations.jsonl.
struct TouchFrameInput {
    struct Ball {
        float x = 0.0f;
        float y = 0.0f;
        float confidence = 0.0f;
        int trackId = -1;
        /// false means the ball was extrapolated (BallTrackState::Predicted),
        /// which is not admissible evidence of a touch.
        bool observed = false;
    };

    struct FootPoint {
        float x = 0.0f;
        float y = 0.0f;
        float confidence = 0.0f;
        /// false when the keypoint is missing or was interpolated by the
        /// smoother (KeypointState::Virtual).
        bool valid = false;
    };

    struct Person {
        int trackId = -1;
        /// Person box height in pixels, used to scale the distance threshold
        /// for people that are far from the camera.
        float heightPx = 0.0f;
        FootPoint feet[kFootKeypointCount];
    };

    uint64_t frameIndex = 0;
    double timestampSec = 0.0;
    int frameWidth = 0;
    int frameHeight = 0;
    std::vector<Ball> balls;
    std::vector<Person> persons;
};

struct TouchDetectorParams {
    /// Minimum confidence for a foot keypoint to count as evidence.
    float keypointConfidence = 0.2f;
    /// Minimum confidence for a ball to count as evidence. Matches the lower
    /// end of what ByteTracker lets through (and what FootballTracker can
    /// rescue), so the detector never contradicts the ball tracker.
    float ballMinConfidence = 0.15f;
    /// Enter threshold as a fraction of min(frame width, frame height).
    float contactDistanceRatio = 0.045f;
    /// Per-person cap as a fraction of the person box height, for distant
    /// players whose feet are small.
    float personHeightRatio = 0.12f;
    /// Hard floor and ceiling of the enter threshold, in pixels of the source
    /// frame. The floor dominates for players of ordinary size: at 720p a
    /// 200 px tall person only reaches 24 px at 0.12.
    ///
    /// 20 px / 0.12 is the plateau found by the calibration sweep on
    /// data/移动传球.mov (F1 0.63-0.65 over 20-25 px, see the README). It is a
    /// single-video fit: rerun the sweep when the camera or the scene changes.
    float minContactPx = 20.0f;
    float maxContactPx = 64.0f;
    /// Exit threshold = enter threshold * releaseRatio (hysteresis).
    float releaseRatio = 1.4f;
    /// Contact frames needed before a cluster becomes an event.
    int minContactFrames = 2;
    /// Frames without contact tolerated inside a cluster.
    int maxGapFrames = 3;
    /// Minimum frame gap between two events split out of one cluster (~0.2 s
    /// at 60 fps).
    int minSplitFrames = 12;
    /// Frames that must clearly exceed the exit threshold between two minima
    /// before the cluster is split into two events, so that a single noisy
    /// valley does not become two touches.
    int minRiseFrames = 2;
    /// Safety cap on one cluster (~5 s at 60 fps). A ball rolling next to a
    /// standing player would otherwise keep a cluster open forever.
    int maxClusterFrames = 300;
    /// Distinct (ball, person, side) pairs kept per frame. 6 is enough to let
    /// threshold calibration see the runner-up pairs, and caps the memory the
    /// offline cache needs.
    int candidateCount = 6;
    /// Judge the followed ball whenever it produced a candidate, even when an
    /// untracked candidate is closer: the tracked ball is the one the pipeline
    /// is committed to.
    bool preferTrackedBall = true;
    /// Half window (seconds) used for the ball speed diagnostics.
    double speedWindowSec = 0.15;

    // --- gates that remove events once they have been assembled -------------
    // All four were measured on the whole 移动传球 video (133 events) against
    // the count aware metric; see the README for the ablation table.
    //
    // A touch of the same person and of the same foot reported again within
    // this many frames is the same touch: one contact that briefly crossed the
    // exit threshold would otherwise come out twice. 21 frames (0.35 s at
    // 60 fps) is the measured optimum: precision 0.571 -> 0.641 at almost no
    // recall cost (0.768 -> 0.758). 0 disables it.
    int refractoryFrames = 21;
    /// Minimum ball speed change across the touch, px/s. Intended to drop a
    /// ball that simply rolls past a foot. **Off by default because it does not
    /// work here**: on this video the speed change of the events the notation
    /// cannot place (median 499 px/s) is *larger* than that of the accepted
    /// ones (median 164 px/s), so gating on it removes 60-75 % of the real
    /// touches (F1 0.655 -> 0.387 at 500 px/s). Keep it for a scene where the
    /// ball is lost between touches and the notation is finer.
    float minSpeedChangePxPerSec = 0.0f;
    /// Minimum ratio between the larger and the smaller of the two speeds
    /// (1 = no requirement). Only applied when both speeds are known.
    float minSpeedRatio = 1.0f;
    /// Only accept events built from the tracked ball, never from a leftover
    /// candidate (`track_id == -1`). Off by default: 8 % of the touches the
    /// annotations support come from a candidate, and dropping them costs more
    /// recall than it gains precision (F1 0.655 -> 0.646).
    bool requireTrackedBall = false;
};

/// Enter threshold for one person on one frame.
float ContactThresholdPx(const TouchDetectorParams& params, int frameWidth,
                         int frameHeight, float personHeightPx);

/// Human readable parameter dump, written into the evaluation report so the
/// numbers can be audited.
std::string DescribeTouchDetectorParams(const TouchDetectorParams& params);

/// One (ball, person, side) pair reduced to its closest foot keypoint.
struct ContactCandidate {
    float distancePx = 0.0f;
    float personHeightPx = 0.0f;
    int personTrackId = -1;
    bool leftFoot = true;
    /// Slot in TouchFrameInput::Person::feet of the closest keypoint.
    int footSlot = -1;
    float ballX = 0.0f;
    float ballY = 0.0f;
    int ballTrackId = -1;
    float ballConfidence = 0.0f;
    bool ballTracked = false;
};

/// How close the reported pair is on one frame.
enum class ContactLevel {
    /// Farther than the exit threshold, or no admissible evidence at all.
    None = 0,
    /// Inside the hysteresis band: the touch may still be going on, so the
    /// cluster stays open, but the frame is not contact evidence.
    Band = 1,
    /// Inside the enter threshold: a contact frame.
    Contact = 2,
};

/// Per-frame evidence, independent of any threshold.
struct FrameContact {
    uint64_t frameIndex = 0;
    double timestampSec = 0.0;
    int frameWidth = 0;
    int frameHeight = 0;
    int personCount = 0;
    int observedBallCount = 0;
    int validFootPointCount = 0;
    int personsWithFeet = 0;
    /// True when at least one observed ball was matched to a valid foot point.
    bool hasBall = false;
    ContactLevel level = ContactLevel::None;
    /// Reported pair: the tracked ball when it has a candidate, else the
    /// closest pair. Only meaningful when hasBall.
    ContactCandidate candidate;
    /// Enter threshold of the reported pair.
    float thresholdPx = 0.0f;

    bool HasContact() const { return level == ContactLevel::Contact; }
};

/// One candidate per (ball, person, side) combination, reduced to the closest
/// of that side's foot keypoints, sorted by distance and truncated to
/// maxCandidates. Balls below ballMinConfidence and extrapolated balls are
/// ignored. Each ball keeps its own candidate, so the followed ball is still
/// visible to SelectContact() when an untracked candidate is closer.
std::vector<ContactCandidate> CollectContactCandidates(
    const TouchFrameInput& frame, const TouchDetectorParams& params,
    std::size_t maxCandidates);

/// Everything one frame contributes apart from the thresholds: the counts the
/// reports quote plus the candidate list. The offline evaluator caches this
/// once per frame and then replays SelectContact() for as many threshold
/// settings as it likes, without re-parsing the JSONL.
struct FrameEvidence {
    uint64_t frameIndex = 0;
    double timestampSec = 0.0;
    int frameWidth = 0;
    int frameHeight = 0;
    int personCount = 0;
    /// Balls reported on the frame, usable or not.
    int ballCount = 0;
    /// Balls admissible as touch evidence.
    int observedBallCount = 0;
    int validFootPointCount = 0;
    int personsWithFeet = 0;
    std::vector<ContactCandidate> candidates;
};

FrameEvidence BuildFrameEvidence(const TouchFrameInput& frame,
                                 const TouchDetectorParams& params);

/// Apply the thresholds of `params` to one frame of evidence.
FrameContact SelectContact(const FrameEvidence& evidence,
                           const TouchDetectorParams& params);

/// Convenience overload for callers that already have the candidates.
FrameContact SelectContact(const TouchFrameInput& frame,
                           const std::vector<ContactCandidate>& candidates,
                           const TouchDetectorParams& params);

/// One detected touch.
struct TouchEvent {
    /// Frame and time of the closest approach inside the event.
    uint64_t frameIndex = 0;
    double timestampSec = 0.0;
    /// First frame of the event, i.e. when the contact started.
    uint64_t firstFrameIndex = 0;
    double firstTimestampSec = 0.0;
    int personTrackId = -1;
    bool leftFoot = true;
    float distancePx = 0.0f;
    float thresholdPx = 0.0f;
    float ballConfidence = 0.0f;
    bool ballTracked = false;
    int ballTrackId = -1;
    /// Contact frames inside the event (excluding tolerated gaps).
    int contactFrames = 0;
    /// Ball speed before/after the touch. They feed the optional speed gate
    /// (see TouchDetectorParams::minSpeedChangePxPerSec) and are kept in the
    /// output either way.
    float speedBeforePxPerSec = 0.0f;
    float speedAfterPxPerSec = 0.0f;
};

/**
 * Aggregates per-frame contacts into events.
 *
 * Push() returns the events that the current frame closed (none in the common
 * case), so online consumers see an event a few frames after it ended; Flush()
 * closes the last one at end of stream.
 */
class TouchEventAggregator {
public:
    explicit TouchEventAggregator(const TouchDetectorParams& params);

    std::vector<TouchEvent> Push(const FrameContact& frame);
    std::vector<TouchEvent> Flush();
    void Reset();

    /// Frames currently buffered for the open cluster.
    std::size_t OpenClusterSize() const { return m_cluster.size(); }

    /// Events the optional gates refused since Reset(). Reported so a gate that
    /// is too aggressive is visible instead of silently eating recalls.
    int RejectedBySpeedGate() const { return m_rejectedBySpeedGate; }
    int RejectedByBallGate() const { return m_rejectedByBallGate; }

private:
    struct ClusterFrame {
        FrameContact frame;
        bool inContact = false;
    };
    struct BallSample {
        double timestampSec = 0.0;
        float x = 0.0f;
        float y = 0.0f;
    };

    std::vector<TouchEvent> CloseCluster();
    void RememberBall(const FrameContact& frame);
    std::vector<TouchEvent> BuildEvents(std::vector<int> const& minima) const;
    float BallSpeed(double fromSec, double toSec) const;
    /// Speed gate of TouchDetectorParams, evaluated on an assembled event.
    bool PassesSpeedGate(const TouchEvent& event) const;

    TouchDetectorParams m_params;
    std::vector<ClusterFrame> m_cluster;
    bool m_active = false;
    int m_gapFrames = 0;
    std::deque<BallSample> m_ballHistory;
    int m_rejectedBySpeedGate = 0;
    int m_rejectedByBallGate = 0;
};

/**
 * Convenience wrapper: candidates -> per-frame contact -> events. The offline
 * evaluator uses the pieces directly so it can cache candidates and replay the
 * aggregation for many thresholds.
 */
class TouchDetector {
public:
    explicit TouchDetector(const TouchDetectorParams& params);

    const TouchDetectorParams& Params() const { return m_params; }

    /// Per-frame contact of the most recent Push(), for overlays that want to
    /// draw the ball-foot link while the contact is happening.
    const FrameContact& LastContact() const { return m_lastContact; }

    std::vector<TouchEvent> Push(const TouchFrameInput& frame);
    /// Offline path: the caller already cached the per-frame evidence, so the
    /// thresholds can be replayed without going through TouchFrameInput again.
    /// Same gates and same events as Push().
    std::vector<TouchEvent> PushEvidence(const FrameEvidence& evidence);
    std::vector<TouchEvent> Flush();
    void Reset();

    /// Events dropped because the same person and foot had just touched the
    /// ball (refractoryFrames). Online and offline use the same counter.
    int SuppressedByRefractory() const { return m_suppressedByRefractory; }
    const TouchEventAggregator& Aggregator() const { return m_aggregator; }

private:
    /// Drops events that repeat a recently reported touch of the same person
    /// and foot, and keeps the window of recent events bounded.
    std::vector<TouchEvent> ApplyRefractory(
        const std::vector<TouchEvent>& events);

    TouchDetectorParams m_params;
    TouchEventAggregator m_aggregator;
    FrameContact m_lastContact;
    /// Events reported recently, oldest first, for the refractory check. Only
    /// their frame, person and foot matter.
    std::deque<TouchEvent> m_recentEvents;
    int m_suppressedByRefractory = 0;
};

} // namespace touch_eval
