#pragma once

#include <touch_eval/TouchDetector.hpp>

#include <fstream>
#include <string>

namespace aifootball_app {

/**
 * @brief JSONL sidecar for the accepted touch events.
 *
 * One metadata line records the video and the thresholds that produced the
 * events, so a file can be read (and audited) without the run log; then one
 * line per event. The schema is intentionally close to the offline evaluator's
 * `touch_events.jsonl`, minus the ground-truth fields that only exist offline.
 */
class TouchJsonlWriter {
public:
    bool Open(const std::string& path, const std::string& videoPath, double fps,
              const std::string& cameraId,
              const touch_eval::TouchDetectorParams& params);
    void Write(const touch_eval::TouchEvent& event, int eventIndex);
    void Close();

private:
    std::ofstream m_file;
};
}
