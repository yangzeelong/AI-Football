#pragma once

#include "inference/IInferenceEngine.hpp"

#include <memory>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace detector {

/// Single detection result in original frame coordinates.
struct Detection {
    float x0, y0, x1, y1;
    float score;
    int   classId;
};

/// Pre/post-processing + inference for RF-DETR, decoupled from Nexusflow Module.
class RFDetrDetectorInfer {
public:
    struct Param {
        std::string enginePath;
        std::string inputBindingName   = "image";
        std::string outputBindingName  = "detections";
        std::string logitsBindingName  = "pred_logits";
        std::string boxesBindingName   = "pred_boxes";
        int   inputSize              = 560;
        float confidenceThreshold    = 0.25f;
        std::unordered_set<int> targetClasses;
        int   numClasses             = 80;
        int   numQueries             = 300;
        float meanR = 0.485f, meanG = 0.456f, meanB = 0.406f;
        float stdR  = 0.229f, stdG  = 0.224f, stdB  = 0.225f;
        int   maxBatchSize = 1;  // Requested batch; static engines may clamp it.
    };

    RFDetrDetectorInfer() = default;
    ~RFDetrDetectorInfer();

    /// Initialize engine + detect output format + allocate host buffers.
    bool Init(const Param& param);

    /// Release engine and buffers.
    void Release();

    bool IsReady() const { return m_ready; }
    int  InputSize() const { return m_param.inputSize; }
    int  MaxBatch() const { return m_effectiveMaxBatch; }

    /**
     * @brief Run inference on a batch of RGB frames.
     *
     * @param frames     Vector of (rgb_ptr, width, height) tuples.
     * @param results    Output: per-frame detection vectors in original coords.
     * @return true on success.
     */
    struct FrameInput {
        const uint8_t* rgb;
        int width;
        int height;
        uint64_t frameId = 0;
    };

    bool InferBatch(const std::vector<FrameInput>& frames,
                    std::vector<std::vector<Detection>>& results);

private:
    enum class OutputFormat { Baked, Raw };

    struct ResizeInfo {
        float scaleX = 1.0f;
        float scaleY = 1.0f;
        int   dx = 0, dy = 0;
    };

    struct RawBox {
        float x0, y0, x1, y1;
        float score;
        int   classId;
    };

    static ResizeInfo ComputeResize(int srcW, int srcH, int dstW, int dstH);
    bool PreprocessToHost(const uint8_t* rgb, int srcW, int srcH,
                          ResizeInfo& resizeOut, float* dstChw) const;
    void DecodeBaked(const float* out, int numQueries, std::vector<RawBox>& boxesOut) const;
    void DecodeRaw(const float* logits, const float* boxes,
                   int numQueries, int numClassesPlus1, std::vector<RawBox>& boxesOut) const;
    void UndoResize(std::vector<RawBox>& boxes, const ResizeInfo& resize,
                     int origW, int origH) const;

    Param m_param;
    std::unique_ptr<inference::IInferenceEngine> m_engine;
    bool m_ready = false;
    int m_effectiveMaxBatch = 1;
    OutputFormat m_outputFormat = OutputFormat::Baked;

    // Host buffers
    std::vector<float> m_inputHost;
    std::vector<float> m_outputHost;   // baked
    std::vector<float> m_logitsHost;   // raw
    std::vector<float> m_boxesHost;    // raw
};

} // namespace detector
