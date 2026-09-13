#pragma once

#include "common/MyMessage.hpp"
#include "nexusflow/ErrorCode.hpp"
#include <nexusflow/Nexusflow.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ns = nexusflow;

/**
 * @brief ByteTracker - multi-object tracker using Kalman filter + IoU Hungarian
 *        matching with a two-stage high/low-score association strategy.
 *
 * Reference: ByteTrack (Zhang et al., 2021). Simplified port aligned with the
 * Python implementation used by AI-Football.
 *
 * Input : DetectionMessage (person class 0, ball class 32)
 * Output: TrackedDetectionMessage
 *         - person detections with trackId assigned
 *         - ball detections passed through untouched (FootballTracker handles them)
 */
class ByteTracker : public ns::Module {
public:
    ByteTracker(const std::string& name);
    ~ByteTracker() override;

    ns::ErrorCode Configure(const ns::Config& config) override;
    ns::ErrorCode Init() override;
    ns::ErrorCode DeInit() override;

protected:
    void Process(ns::Message& inputMessage) override;

private:
    struct Param {
        float trackActivationThreshold   = 0.25f; // high-score threshold
        float secondAssociationThreshold = 0.10f; // low-score threshold
        float minimumMatchingThreshold   = 0.80f; // IoU gate for 1st assoc
        int   lostTrackBuffer            = 30;    // frames before removing lost track
        int   personClassId              = 1;
        int   ballClassId                = 37;
        bool  roiEnabled                 = false;
        int   roiWidth                   = 0;
        int   roiHeight                  = 0;
        std::vector<std::pair<float, float>> roiPolygon;
    } m_param;

    enum class State : int { New = 0, Tracked = 1, Lost = 2, Removed = 3 };

    // --- Kalman state: x = [cx, cy, a, h, vx, vy, va, vh]^T ---
    // Simplified diagonal-covariance Kalman filter for axis-aligned boxes.
    struct KalmanBox {
        std::array<float, 8> mean{};
        std::array<float, 8> cov{};   // diagonal only
        void Init(float cx, float cy, float a, float h);
        void Predict();
        void Update(float cx, float cy, float a, float h);
    };

    struct STrack {
        int   trackId = -1;
        KalmanBox kf;
        // Latest measurement (xyxy)
        float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        float score   = 0.0f;
        int   classId = 0;
        int   frameId = 0;      // last frame this track was updated
        int   startFrame = 0;
        int   trackletLen = 0;
        bool  isActivated = false;
        State state = State::New;

        void GetPredictedBox(float& ox0, float& oy0, float& ox1, float& oy1) const;
        void Reactivate(STrack& newTrack, int frameId, int newId);
        void Update(const STrack& newTrack, int frameId);
        void MarkLost()   { state = State::Lost; }
        void MarkRemoved(){ state = State::Removed; }
    };

    // --- Association helpers ---
    static float Iou(float ax0, float ay0, float ax1, float ay1,
                     float bx0, float by0, float bx1, float by1);
    static float IouDistance(const STrack& track, const STrack& det);

    // O(n^3) Hungarian / Jonker-Volgenant linear assignment.
    // costMatrix: n rows x m cols (row-major, cost, not distance-to-maximize).
    // Pairs with cost > maxCost are rejected as unmatched.
    static void LinearAssignment(const std::vector<float>& costMatrix,
                                 int n, int m, float maxCost,
                                 std::vector<std::pair<int,int>>& matches,
                                 std::vector<int>& unmatchedRows,
                                 std::vector<int>& unmatchedCols);

    void InitTrack(STrack& t, int frameId, bool activated);

    // --- State ---
    std::vector<STrack> m_trackedStracks;
    std::vector<STrack> m_lostStracks;
    int m_frameId = 0;
    int m_nextId  = 1;

    int NextId() { return m_nextId++; }
    void Reset();

    bool IsInsideRoi(const Detection& detection,
                    const VideoFramePtr& videoFrame) const;
    static bool PointInPolygon(float x, float y,
                               const std::vector<std::pair<float, float>>& polygon);
};

NEXUSFLOW_REGISTER_MODULE(ByteTracker);
