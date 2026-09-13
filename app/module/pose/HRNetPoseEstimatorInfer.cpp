#include "HRNetPoseEstimatorInfer.hpp"
#include "inference/TensorRTEngine.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/TimerRegistry.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace pose {

// ---------------------------------------------------------------------------

HRNetPoseEstimatorInfer::~HRNetPoseEstimatorInfer() {
    Release();
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

bool HRNetPoseEstimatorInfer::Init(const Param& param) {
    m_param = param;
    m_param.maxBatch = std::max(1, m_param.maxBatch);

    if (m_param.enginePath.empty()) {
        LOG_WARN("HRNetPoseEstimatorInfer: enginePath is empty");
        return false;
    }

    m_engine = std::make_unique<inference::TensorRTEngine>();
    if (!m_engine->Load(m_param.enginePath)) {
        LOG_ERROR("HRNetPoseEstimatorInfer: failed to load engine {}", m_param.enginePath);
        return false;
    }

    const auto inputInfo = m_engine->GetInputInfo(m_param.inputBindingName);
    if (inputInfo.dims.count <= 0) {
        LOG_ERROR("HRNetPoseEstimatorInfer: input tensor '{}' has no shape",
                  m_param.inputBindingName);
        return false;
    }
    const int engineBatch = inputInfo.dims.d[0];
    m_effectiveMaxBatch = engineBatch > 0 ? engineBatch : m_param.maxBatch;
    m_effectiveMaxBatch = std::max(1, m_effectiveMaxBatch);
    if (engineBatch > 0 && m_param.maxBatch > engineBatch) {
        LOG_WARN("HRNetPoseEstimatorInfer: static engine batch={} clamps requested batch={} to {}",
                 engineBatch, m_param.maxBatch, m_effectiveMaxBatch);
    }

    // --- Inspect output tensor shape ---
    auto outNames = m_engine->GetOutputNames();
    if (outNames.empty()) {
        LOG_ERROR("HRNetPoseEstimatorInfer: engine has no output tensors");
        return false;
    }

    std::string outName = m_param.outputBindingName;
    bool found = false;
    for (const auto& n : outNames) {
        if (n == outName) { found = true; break; }
    }
    if (!found) {
        outName = outNames[0];
        m_param.outputBindingName = outName;
        LOG_WARN("HRNetPoseEstimatorInfer: output '{}' not found, using '{}'",
                 m_param.outputBindingName, outName);
    }

    auto outInfo = m_engine->GetOutputInfo(outName);
    int K = 0, H = 0, W = 0;
    if (outInfo.dims.count == 4) {
        K = outInfo.dims.d[1]; H = outInfo.dims.d[2]; W = outInfo.dims.d[3];
    } else if (outInfo.dims.count == 3) {
        K = outInfo.dims.d[0]; H = outInfo.dims.d[1]; W = outInfo.dims.d[2];
    } else {
        LOG_ERROR("HRNetPoseEstimatorInfer: unexpected output dims count={}", outInfo.dims.count);
        return false;
    }

    m_param.numKeypoints  = K;
    m_param.heatmapHeight = H;
    m_param.heatmapWidth  = W;

    if (m_param.heatmapWidth > 0 && m_param.inputWidth > 0) {
        int inferredStride = m_param.inputWidth / m_param.heatmapWidth;
        if (inferredStride > 0 && inferredStride != m_param.stride) {
            LOG_INFO("HRNetPoseEstimatorInfer: inferred stride={} (config had {})",
                     inferredStride, m_param.stride);
            m_param.stride = inferredStride;
        }
    }

    LOG_INFO("HRNetPoseEstimatorInfer: K={}, heatmap={}x{}, stride={}", K, W, H, m_param.stride);

    // Pre-allocate host buffers for max batch.
    const size_t perPersonInput  = static_cast<size_t>(3) * m_param.inputHeight * m_param.inputWidth;
    const size_t perPersonOutput = static_cast<size_t>(K) * H * W;
    m_inputHost.assign(perPersonInput * m_effectiveMaxBatch, 0.0f);
    m_outputHost.assign(perPersonOutput * m_effectiveMaxBatch, 0.0f);

    LOG_INFO("HRNetPoseEstimatorInfer: requestedBatch={}, effectiveBatch={}, engineBatch={}",
             m_param.maxBatch, m_effectiveMaxBatch, engineBatch > 0 ? engineBatch : -1);

    m_ready = true;
    return true;
}

void HRNetPoseEstimatorInfer::Release() {
    if (m_engine) { m_engine->Release(); m_engine.reset(); }
    m_ready = false;
    m_inputHost.clear();
    m_outputHost.clear();
    m_effectiveMaxBatch = 1;
}

// ---------------------------------------------------------------------------
// InferBatch
// ---------------------------------------------------------------------------

bool HRNetPoseEstimatorInfer::InferBatch(const std::vector<PersonInput>& persons,
                                         std::vector<PersonPose>& results) {
    results.clear();
    if (!m_ready || persons.empty()) return false;

    const int B = static_cast<int>(persons.size());
    if (B > m_effectiveMaxBatch) {
        LOG_ERROR("HRNetPoseEstimatorInfer: batch size {} exceeds effective engine batch {}",
                  B, m_effectiveMaxBatch);
        return false;
    }
    TIMER_SCOPE_AVERAGE_MS("PoseEstimator.Batch", static_cast<uint64_t>(B), 5000);
    const size_t perPersonInput  = static_cast<size_t>(3) * m_param.inputHeight * m_param.inputWidth;
    const size_t perPersonOutput = static_cast<size_t>(m_param.numKeypoints) *
                                   m_param.heatmapHeight * m_param.heatmapWidth;

    // --- Preprocess: expand box + bilinear crop + normalize -> CHW ---
    struct CropBox { float x0, y0, x1, y1; };
    std::vector<CropBox> crops(B);

    {
        TIMER_SCOPE_AVERAGE_MS("PoseEstimator.Preprocess", static_cast<uint64_t>(B), 5000);
        for (int i = 0; i < B; ++i) {
            const auto& p = persons[i];
            float x0 = p.x0, y0 = p.y0, x1 = p.x1, y1 = p.y1;
            ExpandPoseBox(x0, y0, x1, y1, p.frameW, p.frameH);
            crops[i] = {x0, y0, x1, y1};
            PreprocessCrop(p.frameRgb, p.frameW, p.frameH, x0, y0, x1, y1,
                           m_inputHost.data() + i * perPersonInput);
        }
    }

    // --- Engine inference ---
    inference::Dims batchDims(B, 3, m_param.inputHeight, m_param.inputWidth);
    size_t batchInputBytes = B * perPersonInput * sizeof(float);

    {
        TIMER_SCOPE_AVERAGE_MS("PoseEstimator.TensorRT", static_cast<uint64_t>(B), 5000);
        if (!m_engine->SetInputFromHost(m_param.inputBindingName,
                                        m_inputHost.data(), batchInputBytes, batchDims)) {
            LOG_ERROR("HRNetPoseEstimatorInfer: SetInputFromHost failed");
            return false;
        }
        if (!m_engine->Infer()) {
            LOG_ERROR("HRNetPoseEstimatorInfer: Infer failed");
            return false;
        }
    }

    size_t outBytes = B * perPersonOutput * sizeof(float);
    {
        TIMER_SCOPE_AVERAGE_MS("PoseEstimator.CopyOutput", static_cast<uint64_t>(B), 5000);
        if (!m_engine->CopyOutputToHost(m_param.outputBindingName,
                                        m_outputHost.data(), outBytes)) {
            LOG_ERROR("HRNetPoseEstimatorInfer: CopyOutputToHost failed");
            return false;
        }
    }

    // --- Decode heatmaps + inverse map to original frame coords ---
    {
        TIMER_SCOPE_AVERAGE_MS("PoseEstimator.Postprocess", static_cast<uint64_t>(B), 5000);
        results.resize(B);
        for (int i = 0; i < B; ++i) {
        const float* hm = m_outputHost.data() + i * perPersonOutput;
        PersonPose& pp = results[i];

        pp.trackId = persons[i].trackId;
        pp.x0 = persons[i].x0; pp.y0 = persons[i].y0;
        pp.x1 = persons[i].x1; pp.y1 = persons[i].y1;
        pp.detectionConfidence = persons[i].score;

        DecodeHeatmaps(hm, m_param.numKeypoints,
                       m_param.heatmapHeight, m_param.heatmapWidth,
                       pp.keypoints);

        // Crop-space -> original frame coords.
        const CropBox& cb = crops[i];
        float cropW = cb.x1 - cb.x0;
        float cropH = cb.y1 - cb.y0;
        float scaleX = cropW / static_cast<float>(m_param.inputWidth);
        float scaleY = cropH / static_cast<float>(m_param.inputHeight);
        for (int k = 0; k < kProjectKeypointCount; ++k) {
            if (pp.keypoints[k].state == KeypointState::Missing) continue;
            pp.keypoints[k].x = cb.x0 + pp.keypoints[k].x * scaleX;
            pp.keypoints[k].y = cb.y0 + pp.keypoints[k].y * scaleY;
        }
    }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

void HRNetPoseEstimatorInfer::ExpandPoseBox(float& x0, float& y0, float& x1, float& y1,
                                            int imgW, int imgH) const {
    if (!m_param.poseBoxExpansion) return;
    float w = x1 - x0;
    float h = y1 - y0;
    float xPad = std::max(w * m_param.xPadRatio, m_param.minPadPx);
    float yPad = std::max(h * m_param.yPadRatio, m_param.minPadPx);
    x0 = std::max(0.0f, x0 - xPad);
    y0 = std::max(0.0f, y0 - yPad);
    x1 = std::min(static_cast<float>(imgW - 1), x1 + xPad);
    y1 = std::min(static_cast<float>(imgH - 1), y1 + yPad);
}

void HRNetPoseEstimatorInfer::PreprocessCrop(const uint8_t* frameRgb, int frameW, int frameH,
                                             float cx0, float cy0, float cx1, float cy1,
                                             float* dstChw) const {
    const int dstW = m_param.inputWidth;
    const int dstH = m_param.inputHeight;
    const float cropW = cx1 - cx0;
    const float cropH = cy1 - cy0;
    if (cropW <= 0 || cropH <= 0) {
        std::fill(dstChw, dstChw + 3 * dstW * dstH, 0.0f);
        return;
    }
    const float sx = cropW / dstW;
    const float sy = cropH / dstH;

    float* planeR = dstChw;
    float* planeG = dstChw + static_cast<size_t>(dstW) * dstH;
    float* planeB = dstChw + 2 * static_cast<size_t>(dstW) * dstH;

    const float invStdR = 1.0f / m_param.stdR;
    const float invStdG = 1.0f / m_param.stdG;
    const float invStdB = 1.0f / m_param.stdB;

    for (int y = 0; y < dstH; ++y) {
        float fy = cy0 + (y + 0.5f) * sy - 0.5f;
        fy = std::max(0.0f, std::min(fy, static_cast<float>(frameH - 1)));
        int y0 = static_cast<int>(std::floor(fy));
        int y1 = std::min(y0 + 1, frameH - 1);
        float wy = fy - y0;
        for (int x = 0; x < dstW; ++x) {
            float fx = cx0 + (x + 0.5f) * sx - 0.5f;
            fx = std::max(0.0f, std::min(fx, static_cast<float>(frameW - 1)));
            int x0 = static_cast<int>(std::floor(fx));
            int x1 = std::min(x0 + 1, frameW - 1);
            float wx = fx - x0;

            const uint8_t* p00 = frameRgb + (static_cast<size_t>(y0) * frameW + x0) * 3;
            const uint8_t* p01 = frameRgb + (static_cast<size_t>(y0) * frameW + x1) * 3;
            const uint8_t* p10 = frameRgb + (static_cast<size_t>(y1) * frameW + x0) * 3;
            const uint8_t* p11 = frameRgb + (static_cast<size_t>(y1) * frameW + x1) * 3;
            float w00 = (1 - wx) * (1 - wy), w01 = wx * (1 - wy);
            float w10 = (1 - wx) * wy,       w11 = wx * wy;

            float r = p00[0]*w00 + p01[0]*w01 + p10[0]*w10 + p11[0]*w11;
            float g = p00[1]*w00 + p01[1]*w01 + p10[1]*w10 + p11[1]*w11;
            float b = p00[2]*w00 + p01[2]*w01 + p10[2]*w10 + p11[2]*w11;

            size_t idx = static_cast<size_t>(y) * dstW + x;
            planeR[idx] = (r - m_param.meanR) * invStdR;
            planeG[idx] = (g - m_param.meanG) * invStdG;
            planeB[idx] = (b - m_param.meanB) * invStdB;
        }
    }
}

// ---------------------------------------------------------------------------
// DARK sub-pixel refinement
// ---------------------------------------------------------------------------

void HRNetPoseEstimatorInfer::DarkRefine(const float* heatmap, int H, int W,
                                         int px, int py, float& outX, float& outY) {
    outX = static_cast<float>(px);
    outY = static_cast<float>(py);
    if (px < 1 || px >= W - 1 || py < 1 || py >= H - 1) return;

    const float eps = 1e-10f;
    float L[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            L[i][j] = std::log(std::max(heatmap[static_cast<size_t>(py - 1 + i) * W + (px - 1 + j)], eps));

    float gx  = (L[1][2] - L[1][0]) * 0.5f;
    float gy  = (L[2][1] - L[0][1]) * 0.5f;
    float hxx = L[1][2] - 2.0f * L[1][1] + L[1][0];
    float hyy = L[2][1] - 2.0f * L[1][1] + L[0][1];
    float hxy = (L[2][2] - L[2][0] - L[0][2] + L[0][0]) * 0.25f;

    float det = hxx * hyy - hxy * hxy;
    if (std::abs(det) < eps) return;

    float invDet = 1.0f / det;
    float ox = -invDet * (hyy * gx - hxy * gy);
    float oy = -invDet * (-hxy * gx + hxx * gy);
    ox = std::max(-0.5f, std::min(0.5f, ox));
    oy = std::max(-0.5f, std::min(0.5f, oy));

    outX = px + ox;
    outY = py + oy;
}

// ---------------------------------------------------------------------------
// Heatmap decode
// ---------------------------------------------------------------------------

void HRNetPoseEstimatorInfer::DecodeHeatmaps(const float* heatmaps, int K, int H, int W,
                                             Keypoint2D out[kProjectKeypointCount]) const {
    const int* wbMap = GetWholebodyToProject26();
    const size_t planeSize = static_cast<size_t>(H) * W;

    for (int p = 0; p < kProjectKeypointCount; ++p) {
        int k = wbMap[p];
        if (k < 0 || k >= K || k >= 23) {
            out[p] = Keypoint2D{};
            out[p].state = KeypointState::Missing;
            continue;
        }

        const float* hm = heatmaps + static_cast<size_t>(k) * planeSize;
        int bestIdx = 0;
        float bestVal = hm[0];
        for (int i = 1; i < H * W; ++i) {
            if (hm[i] > bestVal) { bestVal = hm[i]; bestIdx = i; }
        }
        int py = bestIdx / W;
        int px = bestIdx % W;

        float refX = static_cast<float>(px);
        float refY = static_cast<float>(py);
        if (m_param.useDark) DarkRefine(hm, H, W, px, py, refX, refY);

        out[p].x = refX * m_param.stride;
        out[p].y = refY * m_param.stride;
        out[p].confidence = bestVal;
        out[p].state = (bestVal >= m_param.keypointConfThr) ? KeypointState::Observed
                                                            : KeypointState::Missing;
    }

    // Virtual midpoints.
    auto midpoint = [&](int a, int b, int dst) {
        const Keypoint2D& ka = out[a];
        const Keypoint2D& kb = out[b];
        bool ok = (ka.state == KeypointState::Observed) && (kb.state == KeypointState::Observed);
        out[dst].x = 0.5f * (ka.x + kb.x);
        out[dst].y = 0.5f * (ka.y + kb.y);
        out[dst].confidence = ok ? 0.5f * (ka.confidence + kb.confidence) : 0.0f;
        out[dst].state = ok ? KeypointState::Virtual : KeypointState::Missing;
    };
    midpoint(5, 6, 23);   // neck
    midpoint(11, 12, 24); // pelvis
    midpoint(23, 24, 25); // thorax
}

} // namespace pose
