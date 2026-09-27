#pragma once

#include <touch_eval/TouchDetector.hpp>
#include <touch_eval/TouchMetrics.hpp>

#include <cstdint>
#include <fstream>
#include <string>

namespace touch_eval {

/**
 * Streams observations.jsonl, one frame at a time.
 *
 * The evaluator only needs the ball centers and the foot keypoints of every
 * frame, so the reader converts each line into a TouchFrameInput and throws
 * the JSON away: memory stays O(1) no matter how long the video is.
 */
class ObservationJsonlReader {
public:
    bool Open(const std::string& path, std::string* error);

    const ObservationMetadata& Metadata() const { return m_metadata; }

    /// Reads the next frame line. Returns false at end of file. Malformed lines
    /// are counted in SkippedLines() and reported through LastSkipReason().
    bool Next(TouchFrameInput* frame);

    uint64_t FramesRead() const { return m_framesRead; }
    uint64_t MetadataLines() const { return m_metadataLines; }
    uint64_t SkippedLines() const { return m_skippedLines; }
    const std::string& LastSkipReason() const { return m_lastSkipReason; }

private:
    bool ParseMetadataLine(const std::string& line);
    bool ParseFrameLine(const std::string& line, TouchFrameInput* frame);

    std::ifstream m_file;
    std::string m_path;
    ObservationMetadata m_metadata;
    uint64_t m_lineNumber = 0;
    uint64_t m_framesRead = 0;
    uint64_t m_metadataLines = 0;
    uint64_t m_skippedLines = 0;
    std::string m_lastSkipReason;
};

} // namespace touch_eval
