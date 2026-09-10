#pragma once

#include "nexusflow/ErrorCode.hpp"
#include <nexusflow/Nexusflow.hpp>
#include "common/MyMessage.hpp"

#include <array>
#include <deque>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ns = nexusflow;

/**
 * @brief KeypointSmoother - skeleton-aware temporal smoothing of 26 keypoints.
 *
 * Port of AI-Football keypoint_smoothing.py (method="skeleton").
 *
 * Input : PoseMessage
 * Output: SmoothedPoseMessage
 *
 * Algorithm (per track):
 *   1. Estimate static/moving state from bbox-center motion over a sliding window.
 *   2. Per-keypoint EMA blend with role-dependent alpha:
 *        - anchor (shoulders/hips):        skeleton_anchor_alpha
 *        - endpoint (toes/heels):          skeleton_endpoint_alpha / skeleton_moving_endpoint_alpha
 *        - fast endpoint (wrists/ankles):  skeleton_fast_endpoint_alpha / skeleton_fast_moving_endpoint_alpha
 *   3. Limb-length constraint: reject smoothed positions that stretch a limb
 *      beyond max_limb_stretch_ratio * previous limb length.
 *   4. Recompute virtual midpoints (neck / pelvis / thorax).
 */
class KeypointSmoother : public ns::Module {
public:
    KeypointSmoother(const std::string& name);
    ~KeypointSmoother() override;

    ns::ErrorCode Configure(const ns::Config& config) override;
    ns::ErrorCode Init() override;
    ns::ErrorCode DeInit() override;

protected:
    void Process(ns::Message& inputMessage) override;

private:
    struct Param {
        bool  enabled = true;
        std::string method = "skeleton";
        int   staticWindow = 8;
        float staticMotionPx = 2.5f;
        float deadbandPx = 0.8f;
        float lowConfidence = 0.3f;
        float lowConfidenceAlpha = 0.12f;
        float staticAlpha = 0.25f;
        float movingAlpha = 0.65f;
        float skeletonAnchorAlpha = 0.35f;
        float skeletonEndpointAlpha = 0.12f;
        float skeletonMovingEndpointAlpha = 0.45f;
        float skeletonFastEndpointAlpha = 0.22f;
        float skeletonFastMovingEndpointAlpha = 0.62f;
        float skeletonFastStaticStepPx = 5.0f;
        float skeletonMaxStaticStepPx = 2.5f;
        float maxLimbStretchRatio = 1.35f;
    } m_param;

    struct TrackHistory {
        std::deque<std::pair<float,float>> bboxCenters;
        std::array<Keypoint2D, kProjectKeypointCount> prev{};
        bool hasPrev = false;
    };

    static bool IsAnchorIdx(int i);
    static bool IsEndpointIdx(int i);
    static bool IsFastEndpointIdx(int i);

    bool IsStatic(TrackHistory& h);
    void SmoothPerson(const PersonPose& in, TrackHistory& h, PersonPose& out);

    std::unordered_map<int, TrackHistory> m_tracks;
};

NEXUSFLOW_REGISTER_MODULE(KeypointSmoother);
