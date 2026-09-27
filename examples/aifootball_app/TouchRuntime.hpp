#pragma once

#include "TouchJsonlWriter.hpp"
#include "VideoRenderer.hpp"

#include <aifootball/AIFootball.hpp>
#include <touch_eval/TouchDetector.hpp>

#include <cstdint>
#include <string>

namespace aifootball_app {

/**
 * @brief Online ball-touch detection on top of the SDK results.
 *
 * Converts every ProcessResult into the touch core's input, writes the accepted
 * events to touches.jsonl and hands the renderer the per-frame overlay. The
 * detection itself is the same code the offline evaluator uses, so what the
 * report measures is exactly what the app shows.
 */
class TouchRuntime {
public:
    TouchRuntime(const touch_eval::TouchDetectorParams& params, int frameWidth,
                 int frameHeight);

    bool Open(const std::string& path, const std::string& videoPath, double fps,
              const std::string& cameraId);
    void Close();

    /// Feeds one frame result and returns the overlay state for that frame.
    TouchOverlay Push(const aifootball::ProcessResult& result);

    /// Closes the last event at end of stream (its overlay is not drawn).
    void Flush();

    int EventCount() const { return m_eventCount; }
    const std::string& OutputPath() const { return m_outputPath; }
    const std::string& LastError() const { return m_lastError; }

    /// Events the optional gates refused, so an over-eager gate is visible in
    /// the run log instead of only in a lower recall.
    int SuppressedByRefractory() const {
        return m_detector.SuppressedByRefractory();
    }
    int RejectedBySpeedGate() const {
        return m_detector.Aggregator().RejectedBySpeedGate();
    }
    int RejectedByBallGate() const {
        return m_detector.Aggregator().RejectedByBallGate();
    }

private:
    touch_eval::TouchDetectorParams m_params;
    touch_eval::TouchDetector m_detector;
    TouchJsonlWriter m_writer;
    std::string m_outputPath;
    std::string m_lastError;
    int m_frameWidth = 0;
    int m_frameHeight = 0;
    int m_eventCount = 0;
    /// Last frame in which an event closed; the badge stays visible for a
    /// handful of frames so a touch that ends between frames is still seen.
    uint64_t m_badgeUntilFrame = 0;
    touch_eval::TouchEvent m_badgeEvent;
    uint64_t m_lastFrameIndex = 0;
    bool m_hasLastFrame = false;
    bool m_orderWarningLogged = false;
};

} // namespace aifootball_app
