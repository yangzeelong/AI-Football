#pragma once

#include "common/MyMessage.hpp"
#include "inference/IInferenceEngine.hpp"

#include <memory>
#include <string>
#include <vector>

namespace pose {

/// Pre/post-processing + inference for HRNet-W48-DARK, decoupled from Module.
class HRNetPoseEstimatorInfer {
public:
    struct Param {
        std::string enginePath;
        std::string inputBindingName  = "images";
        std::string outputBindingName = "heatmaps";
        int   inputWidth       = 288;
        int   inputHeight      = 384;
        int   heatmapWidth     = 72;
        int   heatmapHeight    = 96;
        int   numKeypoints     = 133;
        int   stride           = 4;
        bool  useDark          = true;
        bool  poseBoxExpansion = true;
        float xPadRatio        = 0.15f;
        float yPadRatio        = 0.25f;
        float minPadPx         = 12.0f;
        int   maxBatch         = 16;
        float keypointConfThr  = 0.05f;
        float meanR = 123.675f, meanG = 116.28f, meanB = 103.53f;
        float stdR  = 58.395f,  stdG  = 57.12f,  stdB  = 57.375f;
    };

    /// Input: one person crop request.
    struct PersonInput {
        const uint8_t* frameRgb;
        int   frameW, frameH;
        float x0, y0, x1, y1;   // detection box (expanded internally)
        int   trackId;
        float score;
    };

    HRNetPoseEstimatorInfer() = default;
    ~HRNetPoseEstimatorInfer();

    bool Init(const Param& param);
    void Release();
    bool IsReady() const { return m_ready; }

    /**
     * @brief Run pose estimation on a batch of person crops.
     * @param persons  Input person boxes + frame data.
     * @param results  Output: PersonPose per input (original-frame coords).
     * @return true on success.
     */
    bool InferBatch(const std::vector<PersonInput>& persons,
                    std::vector<PersonPose>& results);

private:
    void ExpandPoseBox(float& x0, float& y0, float& x1, float& y1,
                       int imgW, int imgH) const;
    void PreprocessCrop(const uint8_t* frameRgb, int frameW, int frameH,
                        float cx0, float cy0, float cx1, float cy1,
                        float* dstChw) const;
    static void DarkRefine(const float* heatmap, int H, int W,
                           int px, int py, float& outX, float& outY);
    void DecodeHeatmaps(const float* heatmaps, int K, int H, int W,
                        Keypoint2D out[kProjectKeypointCount]) const;

    Param m_param;
    std::unique_ptr<inference::IInferenceEngine> m_engine;
    bool m_ready = false;

    std::vector<float> m_inputHost;
    std::vector<float> m_outputHost;
};

} // namespace pose
