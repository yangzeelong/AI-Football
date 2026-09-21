#include "HRNetPoseEstimatorInfer.hpp"
#include "inference/TensorRTEngine.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/TimerRegistry.hpp>

#ifdef WITH_CUDA
#include <cuda_runtime_api.h>
#endif

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
    if (!EnsureHostBuffer(m_inputHost, perPersonInput * m_effectiveMaxBatch,
                          "PoseEstimator.InputHost") ||
        !EnsureHostBuffer(m_outputHost, perPersonOutput * m_effectiveMaxBatch,
                          "PoseEstimator.OutputHost") ||
        !EnsureHostBuffer(m_flipOutputHost,
                          perPersonOutput * m_effectiveMaxBatch,
                          "PoseEstimator.FlipOutputHost")) {
        Release();
        return false;
    }
    m_sampleX0.resize(m_param.inputWidth);
    m_sampleWx.resize(m_param.inputWidth);
    m_sampleY0.resize(m_param.inputHeight);
    m_sampleWy.resize(m_param.inputHeight);

    if (m_param.useDark) {
        const int kernel = (m_param.darkBlurKernel % 2 == 1 && m_param.darkBlurKernel >= 3)
                           ? m_param.darkBlurKernel : 11;
        const int radius = kernel / 2;
        const float sigma = 0.3f * (0.5f * (kernel - 1) - 1.0f) + 0.8f;
        m_darkGaussian.resize(kernel);
        float gaussianSum = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            const float value = std::exp(-0.5f * (i * i) / (sigma * sigma));
            m_darkGaussian[i + radius] = value;
            gaussianSum += value;
        }
        for (float& value : m_darkGaussian) value /= gaussianSum;
    }

    LOG_INFO("HRNetPoseEstimatorInfer: requestedBatch={}, effectiveBatch={}, engineBatch={}",
             m_param.maxBatch, m_effectiveMaxBatch, engineBatch > 0 ? engineBatch : -1);

    m_ready = true;
    return true;
}

void HRNetPoseEstimatorInfer::Release() {
    if (m_engine) { m_engine->Release(); m_engine.reset(); }
    m_ready = false;
    ReleaseHostBuffer(m_inputHost);
    ReleaseHostBuffer(m_outputHost);
    ReleaseHostBuffer(m_flipOutputHost);
    m_darkGaussian.clear();
    m_sampleX0.clear();
    m_sampleWx.clear();
    m_sampleY0.clear();
    m_sampleWy.clear();
    m_effectiveMaxBatch = 1;
}

bool HRNetPoseEstimatorInfer::EnsureHostBuffer(HostFloatBuffer& buffer,
                                               size_t floats,
                                               const char* name) {
    if (floats == 0) return false;
    if (buffer.size >= floats && buffer.Data() != nullptr) return true;

    ReleaseHostBuffer(buffer);
#ifdef WITH_CUDA
    void* allocated = nullptr;
    const size_t bytes = floats * sizeof(float);
    if (cudaHostAlloc(&allocated, bytes, cudaHostAllocDefault) == cudaSuccess) {
        buffer.pinned = static_cast<float*>(allocated);
        buffer.size = floats;
        buffer.usingPinned = true;
        LOG_DEBUG("HRNetPoseEstimatorInfer: allocated pinned host buffer '{}' ({} bytes)",
                  name, bytes);
        return true;
    }
    LOG_WARN("HRNetPoseEstimatorInfer: pinned host allocation failed for '{}'; using pageable memory",
             name);
#endif
    try {
        buffer.fallback.assign(floats, 0.0f);
    } catch (const std::exception& error) {
        LOG_ERROR("HRNetPoseEstimatorInfer: host allocation failed for '{}': {}",
                  name, error.what());
        return false;
    }
    buffer.size = floats;
    buffer.usingPinned = false;
    return true;
}

void HRNetPoseEstimatorInfer::ReleaseHostBuffer(HostFloatBuffer& buffer) {
#ifdef WITH_CUDA
    if (buffer.usingPinned && buffer.pinned != nullptr) {
        cudaFreeHost(buffer.pinned);
    }
#endif
    buffer.pinned = nullptr;
    buffer.size = 0;
    buffer.usingPinned = false;
    buffer.fallback.clear();
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

    // --- Preprocess: Python's explicit expansion + MMPose center/scale affine ---
    std::vector<CropTransform> crops(B);

    {
        TIMER_SCOPE_AVERAGE_MS("PoseEstimator.Preprocess", static_cast<uint64_t>(B), 5000);
        for (int i = 0; i < B; ++i) {
            const auto& p = persons[i];
            float x0 = p.x0, y0 = p.y0, x1 = p.x1, y1 = p.y1;
            ExpandPoseBox(x0, y0, x1, y1, p.frameW, p.frameH);
            crops[i] = MakeCropTransform(x0, y0, x1, y1);
            PreprocessCrop(p.frameRgb, p.frameW, p.frameH, crops[i],
                           m_inputHost.Data() + i * perPersonInput);
        }
    }

    // Run MMPose flip-test in one TensorRT batch when there is enough engine
    // capacity. This preserves the two input tensors and only avoids the
    // second enqueue/copy synchronization.
    const bool combineFlipBatch = m_param.flipTest && (2 * B <= m_effectiveMaxBatch);
    const int engineBatch = combineFlipBatch ? 2 * B : B;
    if (combineFlipBatch) {
        for (int i = 0; i < B; ++i) {
            float* flippedInput = m_inputHost.Data() + static_cast<size_t>(B + i) * perPersonInput;
            std::memcpy(flippedInput,
                        m_inputHost.Data() + static_cast<size_t>(i) * perPersonInput,
                        perPersonInput * sizeof(float));
            FlipInput(flippedInput);
        }
    }

    // --- Engine inference ---
    inference::Dims batchDims(engineBatch, 3, m_param.inputHeight, m_param.inputWidth);
    size_t batchInputBytes = static_cast<size_t>(engineBatch) * perPersonInput * sizeof(float);

    {
        TIMER_SCOPE_AVERAGE_MS("PoseEstimator.TensorRT", static_cast<uint64_t>(engineBatch), 5000);
        if (!m_engine->SetInputFromHost(m_param.inputBindingName,
                                        m_inputHost.Data(), batchInputBytes, batchDims)) {
            LOG_ERROR("HRNetPoseEstimatorInfer: SetInputFromHost failed");
            return false;
        }
        if (!m_engine->Infer()) {
            LOG_ERROR("HRNetPoseEstimatorInfer: Infer failed");
            return false;
        }
        if (!m_engine->CopyOutputToHost(m_param.outputBindingName,
                                        m_outputHost.Data(),
                                        static_cast<size_t>(engineBatch) * perPersonOutput * sizeof(float))) {
            LOG_ERROR("HRNetPoseEstimatorInfer: CopyOutputToHost failed");
            return false;
        }
    }

    // MMPose's test_cfg enables heatmap flip-test. The flipped pass operates
    // on the already warped tensor, so only the CHW image needs reversing.
    if (m_param.flipTest) {
        if (!combineFlipBatch) {
            inference::Dims flipBatchDims(B, 3, m_param.inputHeight, m_param.inputWidth);
            const size_t flipBatchInputBytes = static_cast<size_t>(B) * perPersonInput * sizeof(float);
            for (int i = 0; i < B; ++i) {
                FlipInput(m_inputHost.Data() + static_cast<size_t>(i) * perPersonInput);
            }
            TIMER_SCOPE_AVERAGE_MS("PoseEstimator.TensorRTFlip", static_cast<uint64_t>(B), 5000);
            if (!m_engine->SetInputFromHost(m_param.inputBindingName,
                                            m_inputHost.Data(), flipBatchInputBytes, flipBatchDims) ||
                !m_engine->Infer()) {
                LOG_ERROR("HRNetPoseEstimatorInfer: flipped inference failed");
                return false;
            }
            if (!m_engine->CopyOutputToHost(m_param.outputBindingName,
                                            m_flipOutputHost.Data(),
                                            B * perPersonOutput * sizeof(float))) {
                LOG_ERROR("HRNetPoseEstimatorInfer: flipped output copy failed");
                return false;
            }
        }

        // Equivalent to mmpose.models.utils.tta.flip_heatmaps(...): reverse
        // x, swap symmetric channels, shift one heatmap pixel right, average.
        static const int kFlipIndices[133] = {
             0,  2,  1,  4,  3,  6,  5,  8,  7, 10,  9, 12, 11, 14, 13, 16,
            15, 20, 21, 22, 17, 18, 19, 39, 38, 37, 36, 35, 34, 33, 32, 31,
            30, 29, 28, 27, 26, 25, 24, 23, 49, 48, 47, 46, 45, 44, 43, 42,
            41, 40, 50, 51, 52, 53, 58, 57, 56, 55, 54, 68, 67, 66, 65, 70,
            69, 62, 61, 60, 59, 64, 63, 77, 76, 75, 74, 73, 72, 71, 82, 81,
            80, 79, 78, 87, 86, 85, 84, 83, 90, 89, 88, 112, 113, 114, 115,
            116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
            129, 130, 131, 132, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
            102, 103, 104, 105, 106, 107, 108, 109, 110, 111
        };
        for (int i = 0; i < B; ++i) {
            float* orig = m_outputHost.Data() + i * perPersonOutput;
            const float* flipped = combineFlipBatch
                ? m_outputHost.Data() + static_cast<size_t>(B + i) * perPersonOutput
                : m_flipOutputHost.Data() + static_cast<size_t>(i) * perPersonOutput;
            const int H = m_param.heatmapHeight;
            const int W = m_param.heatmapWidth;
            if (m_param.numKeypoints > 133) {
                LOG_ERROR("HRNetPoseEstimatorInfer: flip-test supports at most 133 keypoints, got {}",
                          m_param.numKeypoints);
                return false;
            }
            for (int k = 0; k < m_param.numKeypoints; ++k) {
                const float* src = flipped + static_cast<size_t>(kFlipIndices[k]) * H * W;
                float* dst = orig + static_cast<size_t>(k) * H * W;
                for (int y = 0; y < H; ++y) {
                    for (int x = 0; x < W; ++x) {
                        int sourceX = (x == 0) ? (W - 1) : (W - x);
                        dst[static_cast<size_t>(y) * W + x] =
                            0.5f * (dst[static_cast<size_t>(y) * W + x] +
                                     src[static_cast<size_t>(y) * W + sourceX]);
                    }
                }
            }
        }
    }

    // --- Decode heatmaps + inverse map to original frame coords ---
    {
        TIMER_SCOPE_AVERAGE_MS("PoseEstimator.Postprocess", static_cast<uint64_t>(B), 5000);
        results.resize(B);
        for (int i = 0; i < B; ++i) {
        const float* hm = m_outputHost.Data() + i * perPersonOutput;
        PersonPose& pp = results[i];

        pp.trackId = persons[i].trackId;
        // Keep the detector/tracker box in the public result. The expanded
        // crop is only an internal pose-inference input.
        pp.x0 = persons[i].x0; pp.y0 = persons[i].y0;
        pp.x1 = persons[i].x1; pp.y1 = persons[i].y1;
        pp.detectionConfidence = persons[i].score;

        DecodeHeatmaps(hm, m_param.numKeypoints,
                       m_param.heatmapHeight, m_param.heatmapWidth,
                       pp.keypoints);

        // Crop-space -> original frame coords.
        const CropTransform& crop = crops[i];
        float scaleX = crop.scaleW / static_cast<float>(m_param.inputWidth);
        float scaleY = crop.scaleH / static_cast<float>(m_param.inputHeight);
        for (int k = 0; k < kProjectKeypointCount; ++k) {
            if (pp.keypoints[k].state == KeypointState::Missing) continue;
            pp.keypoints[k].x = crop.centerX - 0.5f * crop.scaleW +
                                pp.keypoints[k].x * scaleX;
            pp.keypoints[k].y = crop.centerY - 0.5f * crop.scaleH +
                                pp.keypoints[k].y * scaleY;
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
    // Match Python's _expand_bbox: coordinates are clamped to image width /
    // height, not to the last pixel index.
    x1 = std::min(static_cast<float>(imgW), x1 + xPad);
    y1 = std::min(static_cast<float>(imgH), y1 + yPad);
}

HRNetPoseEstimatorInfer::CropTransform
HRNetPoseEstimatorInfer::MakeCropTransform(float x0, float y0, float x1, float y1) const {
    CropTransform transform;
    transform.centerX = 0.5f * (x0 + x1);
    transform.centerY = 0.5f * (y0 + y1);
    transform.scaleW = (x1 - x0) * m_param.bboxPadding;
    transform.scaleH = (y1 - y0) * m_param.bboxPadding;
    const float aspect = static_cast<float>(m_param.inputWidth) /
                         static_cast<float>(m_param.inputHeight);
    if (transform.scaleW > transform.scaleH * aspect) {
        transform.scaleH = transform.scaleW / aspect;
    } else {
        transform.scaleW = transform.scaleH * aspect;
    }
    return transform;
}

void HRNetPoseEstimatorInfer::PreprocessCrop(const uint8_t* frameRgb, int frameW, int frameH,
                                             const CropTransform& transform,
                                             float* dstChw) {
    const int dstW = m_param.inputWidth;
    const int dstH = m_param.inputHeight;
    const float cropW = transform.scaleW;
    const float cropH = transform.scaleH;
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

    for (int x = 0; x < dstW; ++x) {
        const float fx = transform.centerX + (static_cast<float>(x) - 0.5f * dstW) * sx;
        m_sampleX0[x] = static_cast<int>(std::floor(fx));
        m_sampleWx[x] = fx - m_sampleX0[x];
    }
    for (int y = 0; y < dstH; ++y) {
        const float fy = transform.centerY + (static_cast<float>(y) - 0.5f * dstH) * sy;
        m_sampleY0[y] = static_cast<int>(std::floor(fy));
        m_sampleWy[y] = fy - m_sampleY0[y];
    }

    for (int y = 0; y < dstH; ++y) {
        const int y0 = m_sampleY0[y];
        int y1 = y0 + 1;
        const float wy = m_sampleWy[y];
        for (int x = 0; x < dstW; ++x) {
            const int x0 = m_sampleX0[x];
            const int x1 = x0 + 1;
            const float wx = m_sampleWx[x];
            float w00 = (1 - wx) * (1 - wy), w01 = wx * (1 - wy);
            float w10 = (1 - wx) * wy,       w11 = wx * wy;
            const uint8_t* p00 = (x0 >= 0 && x0 < frameW && y0 >= 0 && y0 < frameH)
                ? frameRgb + (static_cast<size_t>(y0) * frameW + x0) * 3 : nullptr;
            const uint8_t* p01 = (x1 >= 0 && x1 < frameW && y0 >= 0 && y0 < frameH)
                ? frameRgb + (static_cast<size_t>(y0) * frameW + x1) * 3 : nullptr;
            const uint8_t* p10 = (x0 >= 0 && x0 < frameW && y1 >= 0 && y1 < frameH)
                ? frameRgb + (static_cast<size_t>(y1) * frameW + x0) * 3 : nullptr;
            const uint8_t* p11 = (x1 >= 0 && x1 < frameW && y1 >= 0 && y1 < frameH)
                ? frameRgb + (static_cast<size_t>(y1) * frameW + x1) * 3 : nullptr;
            float r = (p00 ? p00[0] : 0.0f)*w00 + (p01 ? p01[0] : 0.0f)*w01 +
                      (p10 ? p10[0] : 0.0f)*w10 + (p11 ? p11[0] : 0.0f)*w11;
            float g = (p00 ? p00[1] : 0.0f)*w00 + (p01 ? p01[1] : 0.0f)*w01 +
                      (p10 ? p10[1] : 0.0f)*w10 + (p11 ? p11[1] : 0.0f)*w11;
            float b = (p00 ? p00[2] : 0.0f)*w00 + (p01 ? p01[2] : 0.0f)*w01 +
                      (p10 ? p10[2] : 0.0f)*w10 + (p11 ? p11[2] : 0.0f)*w11;

            size_t idx = static_cast<size_t>(y) * dstW + x;
            planeR[idx] = (r - m_param.meanR) * invStdR;
            planeG[idx] = (g - m_param.meanG) * invStdG;
            planeB[idx] = (b - m_param.meanB) * invStdB;
        }
    }
}

void HRNetPoseEstimatorInfer::FlipInput(float* chw) const {
    const size_t plane = static_cast<size_t>(m_param.inputWidth) * m_param.inputHeight;
    for (int c = 0; c < 3; ++c) {
        float* row = chw + static_cast<size_t>(c) * plane;
        for (int y = 0; y < m_param.inputHeight; ++y) {
            float* begin = row + static_cast<size_t>(y) * m_param.inputWidth;
            std::reverse(begin, begin + m_param.inputWidth);
        }
    }
}

// ---------------------------------------------------------------------------
// DARK sub-pixel refinement
// ---------------------------------------------------------------------------

void HRNetPoseEstimatorInfer::DarkRefine(const float* heatmap, int H, int W,
                                         int px, int py, float& outX, float& outY) const {
    outX = static_cast<float>(px);
    outY = static_cast<float>(py);
    if (px <= 1 || px >= W - 2 || py <= 1 || py >= H - 2) return;

    const int kernel = (m_param.darkBlurKernel % 2 == 1 && m_param.darkBlurKernel >= 3)
                       ? m_param.darkBlurKernel : 11;
    const int radius = kernel / 2;
    const float* gaussian = m_darkGaussian.data();

    // DARK only uses a 5x5 neighborhood around the heatmap peak. The positive
    // scale restoration in MMPose becomes an additive constant inside log(),
    // so it cancels from the gradients and Hessian used for the offset.
    auto blurredAt = [&](int x, int y) {
        float value = 0.0f;
        for (int dy = -radius; dy <= radius; ++dy) {
            const int sourceY = y + dy;
            if (sourceY < 0 || sourceY >= H) continue;
            float horizontal = 0.0f;
            for (int dx = -radius; dx <= radius; ++dx) {
                const int sourceX = x + dx;
                if (sourceX < 0 || sourceX >= W) continue;
                horizontal += heatmap[static_cast<size_t>(sourceY) * W + sourceX] *
                              gaussian[dx + radius];
            }
            value += horizontal * gaussian[dy + radius];
        }
        return value;
    };

    auto logBlurred = [&](int x, int y) {
        return std::log(std::max(blurredAt(x, y), 1e-10f));
    };

    float gx  = (logBlurred(px + 1, py) - logBlurred(px - 1, py)) * 0.5f;
    float gy  = (logBlurred(px, py + 1) - logBlurred(px, py - 1)) * 0.5f;
    float hxx = (logBlurred(px + 2, py) - 2.0f * logBlurred(px, py) +
                 logBlurred(px - 2, py)) * 0.25f;
    float hyy = (logBlurred(px, py + 2) - 2.0f * logBlurred(px, py) +
                 logBlurred(px, py - 2)) * 0.25f;
    float hxy = (logBlurred(px + 1, py + 1) - logBlurred(px + 1, py - 1) -
                 logBlurred(px - 1, py + 1) + logBlurred(px - 1, py - 1)) * 0.25f;

    float det = hxx * hyy - hxy * hxy;
    if (det == 0.0f) return;

    float invDet = 1.0f / det;
    float ox = -invDet * (hyy * gx - hxy * gy);
    float oy = -invDet * (-hxy * gx + hxx * gy);

    outX = px + ox;
    outY = py + oy;
}

// ---------------------------------------------------------------------------
// Heatmap decode
// ---------------------------------------------------------------------------

void HRNetPoseEstimatorInfer::DecodeHeatmaps(const float* heatmaps, int K, int H, int W,
                                             Keypoint2D out[kProjectKeypointCount]) {
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
        out[dst].confidence = ok ? std::min(ka.confidence, kb.confidence) : 0.0f;
        out[dst].state = ok ? KeypointState::Virtual : KeypointState::Missing;
    };
    midpoint(5, 6, 23);   // neck
    midpoint(11, 12, 24); // pelvis
    midpoint(23, 24, 25); // thorax
}

} // namespace pose
