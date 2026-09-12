#pragma once

#include "nexusflow/ErrorCode.hpp"
#include <nexusflow/Nexusflow.hpp>
#include "common/MyMessage.hpp"

#include <string>

namespace ns = nexusflow;

/**
 * @brief FootballTracker - single-ball tracker with linear velocity prediction.
 *
 * Port of AI-Football tracking.py (FootballTracker).
 *
 * Input : SmoothedPoseMessage (balls field contains raw ball detections)
 * Output: BallTrackMessage (persons pass-through, balls contains tracked ball)
 *
 * Algorithm:
 *   - If no active track: pick highest-confidence detection, initialize.
 *   - Else: pick detection closest to predicted center within
 *     maxAssociationDistancePx; if none, fall back to highest confidence.
 *   - If no detection matched: predict center via linear velocity,
 *     decay confidence, increment missedFrames; drop track when
 *     missedFrames > maxMissedFrames.
 */
class FootballTracker : public ns::Module {
public:
    FootballTracker(const std::string& name);
    ~FootballTracker() override;

    ns::ErrorCode Configure(const ns::Config& config) override;
    ns::ErrorCode Init() override;
    ns::ErrorCode DeInit() override;

protected:
    void Process(ns::Message& inputMessage) override;

private:
    struct Param {
        int   maxMissedFrames = 12;
        float maxAssociationDistancePx = 180.0f;
        float confidenceDecay = 0.75f;
        int   ballClassId = 37;
    } m_param;

    struct TrackState {
        bool  active = false;
        int   trackId = -1;
        float cx = 0, cy = 0;      // center (px)
        float vx = 0, vy = 0;      // velocity (px/s)
        float w = 0, h = 0;        // bbox size
        float confidence = 0.0f;
        double timestampSec = 0.0;
        int   missedFrames = 0;
    } m_track;

    int m_nextTrackId = 1;

    void UpdateFromDetection(const Detection& d, double ts);
    void Predict(double ts);
    void Emit(BallTrackMessage& out) const;
};

NEXUSFLOW_REGISTER_MODULE(FootballTracker);
