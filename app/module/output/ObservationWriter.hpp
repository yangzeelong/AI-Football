#pragma once

#include "nexusflow/ErrorCode.hpp"
#include <nexusflow/Nexusflow.hpp>
#include "common/MyMessage.hpp"

#include <fstream>
#include <map>
#include <string>

namespace ns = nexusflow;

/**
 * @brief ObservationWriter - JSONL observation output.
 *
 * Mirrors AI-Football observations.py JsonlObservationWriter:
 *   Line 1: {"type":"metadata","schema_version":1,"video":{...}}
 *   Line N: {"type":"frame","frame_index":...,"timestamp_sec":...,"camera_id":"C1",
 *            "persons":[...],"balls":[...],"raw_detection_counts":{...}}
 *
 * Input : BallTrackMessage
 * Output: (file I/O; no downstream broadcast)
 */
class ObservationWriter : public ns::Module {
public:
    ObservationWriter(const std::string& name);
    ~ObservationWriter() override;

    ns::ErrorCode Configure(const ns::Config& config) override;
    ns::ErrorCode Init() override;
    ns::ErrorCode DeInit() override;

protected:
    void Process(ns::Message& inputMessage) override;

private:
    struct Param {
        std::string outputPath = "output/observations.jsonl";
        std::string cameraId   = "C1";
        std::string videoPath;
        double fps = 25.0;
        int    width = 0;
        int    height = 0;
        int    frameCount = 0;
        int    stride = 1;
    } m_param;

    std::ofstream m_file;
    bool m_metadataWritten = false;
    int  m_framesWritten = 0;

    void WriteMetadata();
    void WriteFrame(const struct BallTrackMessage& msg,
                    const std::map<std::string, double>& moduleTimingsMs);

    // Minimal JSON helpers (no external deps).
    static void JsonEscape(std::ostream& os, const std::string& s);
    static void JsonFloat(std::ostream& os, float v);
    static void JsonDouble(std::ostream& os, double v);
};

NEXUSFLOW_REGISTER_MODULE(ObservationWriter);
