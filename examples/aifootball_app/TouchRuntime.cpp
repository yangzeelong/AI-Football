#include "TouchRuntime.hpp"

#include <nexusflow/Logging.hpp>

namespace aifootball_app {
namespace {

/// Frames the accepted-event badge stays visible after an event closes.
const uint64_t kBadgeFrames = 40;

/// Converts one SDK result into the evidence the touch core consumes. Only
/// `observed` keypoints and balls are marked usable: the smoother's
/// interpolated joints and the tracker's extrapolated ball frames must not
/// create a touch of their own.
touch_eval::TouchFrameInput ToTouchFrame(const aifootball::ProcessResult& result,
                                         int frameWidth, int frameHeight) {
    touch_eval::TouchFrameInput frame;
    frame.frameIndex = result.frameId;
    frame.timestampSec = result.timestampSec;
    frame.frameWidth = frameWidth;
    frame.frameHeight = frameHeight;

    frame.persons.reserve(result.persons.size());
    for (std::size_t i = 0; i < result.persons.size(); ++i) {
        const aifootball::PersonResult& person = result.persons[i];
        touch_eval::TouchFrameInput::Person parsed;
        parsed.trackId = person.trackId;
        const float height = person.bbox[3] - person.bbox[1];
        parsed.heightPx = height > 0.0f ? height : 0.0f;
        for (int slot = 0; slot < touch_eval::kFootKeypointCount; ++slot) {
            const int index = touch_eval::kFootKeypointIndices[slot];
            const aifootball::KeypointResult& keypoint =
                person.keypoints[static_cast<std::size_t>(index)];
            touch_eval::TouchFrameInput::FootPoint& foot = parsed.feet[slot];
            foot.x = keypoint.x;
            foot.y = keypoint.y;
            foot.confidence = keypoint.confidence;
            foot.valid =
                keypoint.state == aifootball::KeypointState::Observed;
        }
        frame.persons.push_back(parsed);
    }

    frame.balls.reserve(result.balls.size());
    for (std::size_t i = 0; i < result.balls.size(); ++i) {
        const aifootball::BallResult& ball = result.balls[i];
        touch_eval::TouchFrameInput::Ball parsed;
        parsed.x = ball.center[0];
        parsed.y = ball.center[1];
        parsed.confidence = ball.confidence;
        parsed.trackId = ball.trackId;
        parsed.observed = ball.state == aifootball::BallTrackState::Observed;
        frame.balls.push_back(parsed);
    }
    return frame;
}

/// Coordinates of the foot keypoint the contact was attributed to.
bool FindFootPoint(const touch_eval::TouchFrameInput& frame,
                   const touch_eval::ContactCandidate& candidate, float* x,
                   float* y) {
    if (candidate.footSlot < 0 ||
        candidate.footSlot >= touch_eval::kFootKeypointCount) {
        return false;
    }
    for (std::size_t i = 0; i < frame.persons.size(); ++i) {
        if (frame.persons[i].trackId != candidate.personTrackId) continue;
        const touch_eval::TouchFrameInput::FootPoint& foot =
            frame.persons[i].feet[candidate.footSlot];
        if (!foot.valid) return false;
        *x = foot.x;
        *y = foot.y;
        return true;
    }
    return false;
}

} // namespace

TouchRuntime::TouchRuntime(const touch_eval::TouchDetectorParams& params,
                           int frameWidth, int frameHeight)
    : m_params(params),
      m_detector(params),
      m_frameWidth(frameWidth),
      m_frameHeight(frameHeight) {}

bool TouchRuntime::Open(const std::string& path, const std::string& videoPath,
                        double fps, const std::string& cameraId) {
    m_outputPath = path;
    if (!m_writer.Open(path, videoPath, fps, cameraId, m_params)) {
        m_lastError = "cannot open touch output '" + path + "'";
        return false;
    }
    LOG_INFO("TouchRuntime: {} -> {}", touch_eval::DescribeTouchDetectorParams(m_params),
             path);
    return true;
}

void TouchRuntime::Close() { m_writer.Close(); }

TouchOverlay TouchRuntime::Push(const aifootball::ProcessResult& result) {
    TouchOverlay overlay;

    if (m_hasLastFrame && result.frameId <= m_lastFrameIndex &&
        !m_orderWarningLogged) {
        // The app consumes futures in submission order, so this should never
        // fire; if it does, the aggregation windows would be meaningless and
        // the report would be silently wrong.
        LOG_WARN("TouchRuntime: frame {} arrived after {}, touches may be "
                 "mis-timed", result.frameId, m_lastFrameIndex);
        m_orderWarningLogged = true;
    }
    m_lastFrameIndex = result.frameId;
    m_hasLastFrame = true;

    const touch_eval::TouchFrameInput frame =
        ToTouchFrame(result, m_frameWidth, m_frameHeight);
    const std::vector<touch_eval::TouchEvent> events = m_detector.Push(frame);

    for (std::size_t i = 0; i < events.size(); ++i) {
        ++m_eventCount;
        m_writer.Write(events[i], m_eventCount);
        m_badgeEvent = events[i];
        m_badgeUntilFrame = result.frameId + kBadgeFrames;
        LOG_DEBUG("TouchRuntime: event {} frame {} person {} {} d={}px",
                  m_eventCount, events[i].frameIndex, events[i].personTrackId,
                  events[i].leftFoot ? "left" : "right", events[i].distancePx);
    }

    // Per-frame contact, for the ball-foot link.
    const touch_eval::FrameContact& contact = m_detector.LastContact();
    if (contact.HasContact()) {
        overlay.hasContact = true;
        overlay.ballX = contact.candidate.ballX;
        overlay.ballY = contact.candidate.ballY;
        overlay.personTrackId = contact.candidate.personTrackId;
        overlay.leftFoot = contact.candidate.leftFoot;
        overlay.distancePx = contact.candidate.distancePx;
        overlay.thresholdPx = contact.thresholdPx;
        FindFootPoint(frame, contact.candidate, &overlay.footX, &overlay.footY);
    }

    overlay.totalEvents = m_eventCount;
    if (m_eventCount > 0 && result.frameId <= m_badgeUntilFrame) {
        overlay.hasBadge = true;
        overlay.badgeFrameIndex = m_badgeEvent.frameIndex;
        overlay.badgeTimestampSec = m_badgeEvent.timestampSec;
        overlay.badgeDistancePx = m_badgeEvent.distancePx;
        overlay.badgePersonTrackId = m_badgeEvent.personTrackId;
        overlay.badgeLeftFoot = m_badgeEvent.leftFoot;
    }
    return overlay;
}

void TouchRuntime::Flush() {
    const std::vector<touch_eval::TouchEvent> events = m_detector.Flush();
    for (std::size_t i = 0; i < events.size(); ++i) {
        ++m_eventCount;
        m_writer.Write(events[i], m_eventCount);
        LOG_INFO("TouchRuntime: event {} frame {} person {} {} d={}px (at end of stream)",
                 m_eventCount, events[i].frameIndex, events[i].personTrackId,
                 events[i].leftFoot ? "left" : "right", events[i].distancePx);
    }
}

} // namespace aifootball_app
