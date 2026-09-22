#include "RFDetrDetectorInfer.hpp"
#include "RFDetrCudaPreprocess.hpp"
#include "inference/TensorRTEngine.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/TimerRegistry.hpp>

#ifdef WITH_CUDA_KERNELS
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>

namespace {

float Sigmoid(float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-value));
    }
    const float e = std::exp(value);
    return e / (1.0f + e);
}

bool ShouldLogDetectorFrame(uint64_t frameId) {
    return frameId < 5 || (frameId % 60) == 0;
}

} // namespace

namespace detector {

// ---------------------------------------------------------------------------

RFDetrDetectorInfer::~RFDetrDetectorInfer() {
    Release();
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

bool RFDetrDetectorInfer::Init(const Param& param) {
    m_param = param;
    m_param.maxBatchSize = std::max(1, m_param.maxBatchSize);

    if (m_param.enginePath.empty()) {
        LOG_WARN("RFDetrDetectorInfer: enginePath is empty");
        return false;
    }

    m_engine = std::make_unique<inference::TensorRTEngine>();
    if (!m_engine->Load(m_param.enginePath)) {
        LOG_ERROR("RFDetrDetectorInfer: failed to load engine {}", m_param.enginePath);
        return false;
    }

    const auto inputInfo = m_engine->GetInputInfo(m_param.inputBindingName);
    if (inputInfo.dims.count <= 0) {
        LOG_ERROR("RFDetrDetectorInfer: input tensor '{}' has no shape",
                  m_param.inputBindingName);
        return false;
    }

#ifdef WITH_CUDA_KERNELS
    const auto* trtEngine = dynamic_cast<const inference::TensorRTEngine*>(
        m_engine.get());
    m_gpuPreprocessAvailable =
        m_param.useGpuPreprocess && inputInfo.dtype == inference::DataType::kFloat32 &&
        trtEngine != nullptr && trtEngine->GetCudaStream() != nullptr;
    if (m_param.useGpuPreprocess && !m_gpuPreprocessAvailable) {
        LOG_WARN("RFDetrDetectorInfer: GPU preprocessing unavailable; using CPU preprocessing");
    }
#else
    m_gpuPreprocessAvailable = false;
    if (m_param.useGpuPreprocess) {
        LOG_WARN("RFDetrDetectorInfer: built without CUDA; using CPU preprocessing");
    }
#endif
    const int engineBatch = inputInfo.dims.d[0];
    m_effectiveMaxBatch = engineBatch > 0 ? engineBatch : m_param.maxBatchSize;
    m_effectiveMaxBatch = std::max(1, m_effectiveMaxBatch);
    if (engineBatch > 0 && m_param.maxBatchSize > engineBatch) {
        LOG_WARN("RFDetrDetectorInfer: static engine batch={} clamps requested batch={} to {}",
                 engineBatch, m_param.maxBatchSize, m_effectiveMaxBatch);
    }

    // --- Auto-detect output format ---
    auto outNames = m_engine->GetOutputNames();
    LOG_INFO("RFDetrDetectorInfer: {} output tensor(s)", outNames.size());

    bool haveBaked = false;
    if (outNames.size() == 1) {
        m_param.outputBindingName = outNames[0];
        auto dims = m_engine->GetOutputInfo(m_param.outputBindingName).dims;
        if (dims.count == 3 && dims.d[2] == 6) {
            haveBaked = true;
            m_param.numQueries = dims.d[1];
        } else if (dims.count == 2 && dims.d[1] == 6) {
            haveBaked = true;
            m_param.numQueries = dims.d[0];
        }
    }

    bool haveRaw = false;
    if (!haveBaked) {
        std::string logitsName = m_param.logitsBindingName;
        std::string boxesName  = m_param.boxesBindingName;
        bool foundLogits = false, foundBoxes = false;
        inference::Dims logitsDims, boxesDims;

        for (const auto& n : outNames) {
            auto d = m_engine->GetOutputInfo(n).dims;
            if (d.count == 3 && d.d[2] == 4) {
                boxesName = n; boxesDims = d; foundBoxes = true;
            } else if (d.count == 3 && d.d[2] > 4) {
                logitsName = n; logitsDims = d; foundLogits = true;
            }
        }
        if (!foundLogits || !foundBoxes) {
            logitsDims = m_engine->GetOutputInfo(logitsName).dims;
            boxesDims  = m_engine->GetOutputInfo(boxesName).dims;
            foundLogits = (logitsDims.count >= 2);
            foundBoxes  = (boxesDims.count >= 2);
        }
        if (foundLogits && foundBoxes) {
            haveRaw = true;
            m_param.logitsBindingName = logitsName;
            m_param.boxesBindingName  = boxesName;
            m_param.numQueries = (logitsDims.count == 3) ? logitsDims.d[1] : logitsDims.d[0];
            int numClsPlus1 = (logitsDims.count == 3) ? logitsDims.d[2] : logitsDims.d[1];
            // RF-DETR exports all COCO category indices, including the
            // sparse COCO ids such as person=1 and sports ball=37.
            m_param.numClasses = numClsPlus1;
        }
    }

    if (haveBaked) {
        m_outputFormat = OutputFormat::Baked;
        LOG_INFO("RFDetrDetectorInfer: baked [N={}, 6]", m_param.numQueries);
    } else if (haveRaw) {
        m_outputFormat = OutputFormat::Raw;
        LOG_INFO("RFDetrDetectorInfer: raw (logits=[N,{}], boxes=[N,4])", m_param.numClasses);
    } else {
        LOG_ERROR("RFDetrDetectorInfer: cannot determine output format");
        return false;
    }

    LOG_INFO("RFDetrDetectorInfer: requestedBatch={}, effectiveBatch={}, engineBatch={}",
             m_param.maxBatchSize, m_effectiveMaxBatch, engineBatch > 0 ? engineBatch : -1);

    // --- Allocate host buffers (single frame; batch caller provides count) ---
    // We'll resize dynamically in InferBatch based on actual batch size.
    m_ready = true;
    return true;
}

void RFDetrDetectorInfer::Release() {
    ReleaseHostBuffers();
    ReleaseGpuBuffers();
    if (m_engine) { m_engine->Release(); m_engine.reset(); }
    m_ready = false;
    m_gpuPreprocessAvailable = false;
    m_effectiveMaxBatch = 1;
}

// ---------------------------------------------------------------------------
// InferBatch: preprocess → engine → postprocess
// ---------------------------------------------------------------------------

bool RFDetrDetectorInfer::InferBatch(const std::vector<FrameInput>& frames,
                                     std::vector<std::vector<Detection>>& results) {
    results.clear();
    if (!m_ready || frames.empty()) return false;

    const int B = static_cast<int>(frames.size());
    if (B > m_effectiveMaxBatch) {
        LOG_ERROR("RFDetrDetectorInfer: batch size {} exceeds effective engine batch {}",
                  B, m_effectiveMaxBatch);
        return false;
    }
    TIMER_SCOPE_AVERAGE_MS("Detector.Batch", static_cast<uint64_t>(B), 5000);
    const size_t perFrameFloats = static_cast<size_t>(m_param.inputSize) * m_param.inputSize * 3;
    inference::Dims batchDims(B, 3, m_param.inputSize, m_param.inputSize);
    const size_t batchInputBytes = B * perFrameFloats * sizeof(float);

    // --- Preprocess ---
    std::vector<ResizeInfo> resizeInfos(B);
    bool usedGpuPreprocess = false;
    {
        TIMER_SCOPE_AVERAGE_MS("Detector.Preprocess", static_cast<uint64_t>(B), 5000);
        if (m_gpuPreprocessAvailable) {
#ifdef WITH_CUDA_KERNELS
            auto* trtEngine = dynamic_cast<inference::TensorRTEngine*>(m_engine.get());
            void* stream = trtEngine ? trtEngine->GetCudaStream() : nullptr;
            void* inputDevice = nullptr;
            {
                TIMER_SCOPE_AVERAGE_MS("Detector.PrepareInput", static_cast<uint64_t>(B), 5000);
                inputDevice = trtEngine
                    ? trtEngine->PrepareInputDevice(m_param.inputBindingName,
                                                     batchInputBytes, batchDims)
                    : nullptr;
            }
            usedGpuPreprocess = inputDevice != nullptr && stream != nullptr &&
                                PreprocessToGpu(frames, resizeInfos, inputDevice, stream);
            if (!usedGpuPreprocess) {
                LOG_WARN("RFDetrDetectorInfer: GPU preprocessing failed; falling back to CPU");
                m_gpuPreprocessAvailable = false;
            }
#endif
        }

        if (!usedGpuPreprocess) {
            // CPU fallback. Keep this path available for unsupported input
            // dtypes, non-CUDA builds, and runtime CUDA failures.
            if (!EnsureHostBuffer(m_inputHost, perFrameFloats * B,
                                  "Detector.InputHost")) {
                return false;
            }
            for (int i = 0; i < B; ++i) {
                float* dst = m_inputHost.Data() + i * perFrameFloats;
                if (!frames[i].rgb || frames[i].width <= 0 || frames[i].height <= 0) {
                    std::fill(dst, dst + perFrameFloats, 0.0f);
                    resizeInfos[i] = ResizeInfo{};
                } else {
                    PreprocessToHost(frames[i].rgb, frames[i].width, frames[i].height,
                                     resizeInfos[i], dst);
                }
            }
        }
    }

    // --- Engine inference ---
    {
        TIMER_SCOPE_AVERAGE_MS("Detector.TensorRT", static_cast<uint64_t>(B), 5000);
        if (!usedGpuPreprocess) {
            TIMER_SCOPE_AVERAGE_MS("Detector.InputH2D", static_cast<uint64_t>(B), 5000);
            if (!m_engine->SetInputFromHost(m_param.inputBindingName,
                                            m_inputHost.Data(), batchInputBytes, batchDims)) {
                LOG_ERROR("RFDetrDetectorInfer: SetInputFromHost failed");
                return false;
            }
        }
        if (!m_engine->Infer()) {
            LOG_ERROR("RFDetrDetectorInfer: Infer failed");
            return false;
        }
    }

    // --- Postprocess per frame ---
    {
        TIMER_SCOPE_AVERAGE_MS("Detector.Postprocess", static_cast<uint64_t>(B), 5000);
        results.resize(B);
        for (int i = 0; i < B; ++i) {
        std::vector<RawBox> rawBoxes;

        if (m_outputFormat == OutputFormat::Baked) {
            size_t perFrameOut = static_cast<size_t>(m_param.numQueries) * 6;
            size_t totalOut = perFrameOut * static_cast<size_t>(B);
            if (!EnsureHostBuffer(m_outputHost, totalOut,
                                  "Detector.OutputHost")) {
                return false;
            }
            if (i == 0) {
                TIMER_SCOPE_AVERAGE_MS("Detector.CopyOutput", static_cast<uint64_t>(B), 5000);
                if (!m_engine->CopyOutputToHost(m_param.outputBindingName,
                                                m_outputHost.Data(),
                                                totalOut * sizeof(float))) {
                    LOG_ERROR("RFDetrDetectorInfer: CopyOutputToHost failed for '{}'",
                              m_param.outputBindingName);
                    return false;
                }
            }
            size_t offset = static_cast<size_t>(i) * perFrameOut;
            DecodeBaked(m_outputHost.Data() + offset, m_param.numQueries, rawBoxes);
        } else {
            size_t perFrameLogits = static_cast<size_t>(m_param.numQueries) * m_param.numClasses;
            size_t perFrameBoxes  = static_cast<size_t>(m_param.numQueries) * 4;
            size_t logitsOff = static_cast<size_t>(i) * perFrameLogits;
            size_t boxesOff  = static_cast<size_t>(i) * perFrameBoxes;

            const size_t totalLogits = perFrameLogits * static_cast<size_t>(B);
            const size_t totalBoxes = perFrameBoxes * static_cast<size_t>(B);
            if (!EnsureHostBuffer(m_logitsHost, totalLogits,
                                  "Detector.LogitsHost") ||
                !EnsureHostBuffer(m_boxesHost, totalBoxes,
                                  "Detector.BoxesHost")) {
                return false;
            }

            if (i == 0) {
                TIMER_SCOPE_AVERAGE_MS("Detector.CopyOutput", static_cast<uint64_t>(B), 5000);
                // Both tensors travel on one stream sync; RF-DETR has no reason
                // to drain the device once per output tensor.
                if (!m_engine->CopyOutputsToHost(
                        {{m_param.logitsBindingName, m_logitsHost.Data(),
                          totalLogits * sizeof(float)},
                         {m_param.boxesBindingName, m_boxesHost.Data(),
                          totalBoxes * sizeof(float)}})) {
                    LOG_ERROR("RFDetrDetectorInfer: failed to copy raw batch outputs");
                    return false;
                }
            }

            {
                const auto& frame = frames[i];
                if (ShouldLogDetectorFrame(frame.frameId)) {
                    float inputMin = 0.0f;
                    float inputMax = 0.0f;
                    if (!usedGpuPreprocess) {
                        inputMin = std::numeric_limits<float>::max();
                        inputMax = std::numeric_limits<float>::lowest();
                        const float* input = m_inputHost.Data() +
                            static_cast<size_t>(i) * perFrameFloats;
                        for (size_t k = 0; k < perFrameFloats; ++k) {
                            inputMin = std::min(inputMin, input[k]);
                            inputMax = std::max(inputMax, input[k]);
                        }
                    }

                    float logitMin = std::numeric_limits<float>::max();
                    float logitMax = std::numeric_limits<float>::lowest();
                    float boxMin = std::numeric_limits<float>::max();
                    float boxMax = std::numeric_limits<float>::lowest();
                    float bestAnyScore = 0.0f;
                    int bestAnyClass = -1;
                    int bestAnyQuery = -1;
                    float bestTargetScore = 0.0f;
                    int bestTargetClass = -1;
                    int bestTargetQuery = -1;
                    int anyPairsAbove = 0;
                    int targetPairsAbove = 0;

                    const float* logits = m_logitsHost.Data() + logitsOff;
                    const float* boxes = m_boxesHost.Data() + boxesOff;
                    for (size_t k = 0; k < perFrameLogits; ++k) {
                        logitMin = std::min(logitMin, logits[k]);
                        logitMax = std::max(logitMax, logits[k]);
                    }
                    for (size_t k = 0; k < perFrameBoxes; ++k) {
                        boxMin = std::min(boxMin, boxes[k]);
                        boxMax = std::max(boxMax, boxes[k]);
                    }
                    for (int q = 0; q < m_param.numQueries; ++q) {
                        for (int c = 0; c < m_param.numClasses; ++c) {
                            const float score = Sigmoid(
                                logits[static_cast<size_t>(q) * m_param.numClasses + c]);
                            if (score > bestAnyScore) {
                                bestAnyScore = score;
                                bestAnyClass = c;
                                bestAnyQuery = q;
                            }
                            if (score >= m_param.confidenceThreshold) {
                                ++anyPairsAbove;
                            }
                            if (!m_param.targetClasses.empty() &&
                                m_param.targetClasses.count(c) != 0) {
                                if (score > bestTargetScore) {
                                    bestTargetScore = score;
                                    bestTargetClass = c;
                                    bestTargetQuery = q;
                                }
                                if (score >= m_param.confidenceThreshold) {
                                    ++targetPairsAbove;
                                }
                            }
                        }
                    }

                    LOG_INFO(
                        "RFDetrDetector debug: frame={} input={}x{} normalized=[{:.3f},{:.3f}] "
                        "input_mode={} "
                        "logits=[{:.3f},{:.3f}] boxes=[{:.3f},{:.3f}] "
                        "bestAny=(q={},class={},score={:.3f}) "
                        "bestTarget=(q={},class={},score={:.3f}) "
                        "pairsAbove={} targetPairsAbove={} threshold={:.2f}",
                        frame.frameId, frame.width, frame.height, inputMin, inputMax,
                        usedGpuPreprocess ? "gpu" : "cpu",
                        logitMin, logitMax, boxMin, boxMax,
                        bestAnyQuery, bestAnyClass, bestAnyScore,
                        bestTargetQuery, bestTargetClass, bestTargetScore,
                        anyPairsAbove, targetPairsAbove, m_param.confidenceThreshold);
                }

                DecodeRaw(m_logitsHost.Data() + logitsOff,
                          m_boxesHost.Data() + boxesOff,
                          m_param.numQueries, m_param.numClasses, rawBoxes);
            }
        }

        UndoResize(rawBoxes, resizeInfos[i], frames[i].width, frames[i].height);

        if (m_outputFormat == OutputFormat::Raw &&
            ShouldLogDetectorFrame(frames[i].frameId)) {
            LOG_INFO("RFDetrDetector debug: frame={} decoded_target_detections={}",
                     frames[i].frameId, rawBoxes.size());
        }

        results[i].reserve(rawBoxes.size());
        for (const auto& rb : rawBoxes) {
                results[i].push_back(Detection{rb.x0, rb.y0, rb.x1, rb.y1, rb.score, rb.classId});
        }
    }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Preprocessing (pure CPU)
// ---------------------------------------------------------------------------

RFDetrDetectorInfer::ResizeInfo
RFDetrDetectorInfer::ComputeResize(int srcW, int srcH, int dstW, int dstH) {
    ResizeInfo resize;
    if (srcW <= 0 || srcH <= 0) return resize;
    // Match AI-Football's Python detector: preserve aspect ratio and pad the
    // resized RGB image with neutral gray (114) to a square.
    const float scale = std::min(static_cast<float>(dstW) / srcW,
                                 static_cast<float>(dstH) / srcH);
    resize.scaleX = scale;
    resize.scaleY = scale;
    resize.resizedWidth = std::max(1, static_cast<int>(std::nearbyint(srcW * scale)));
    resize.resizedHeight = std::max(1, static_cast<int>(std::nearbyint(srcH * scale)));
    resize.sampleScaleX = static_cast<float>(resize.resizedWidth) / srcW;
    resize.sampleScaleY = static_cast<float>(resize.resizedHeight) / srcH;
    resize.dx = (dstW - resize.resizedWidth) / 2;
    resize.dy = (dstH - resize.resizedHeight) / 2;
    return resize;
}

bool RFDetrDetectorInfer::PreprocessToHost(const uint8_t* rgb, int srcW, int srcH,
                                           ResizeInfo& resizeOut, float* dstChw) const {
    if (!rgb || srcW <= 0 || srcH <= 0 || !dstChw) return false;
    const int dstW = m_param.inputSize;
    const int dstH = m_param.inputSize;
    resizeOut = ComputeResize(srcW, srcH, dstW, dstH);

    const int newW = resizeOut.resizedWidth;
    const int newH = resizeOut.resizedHeight;
    const int dx = resizeOut.dx;
    const int dy = resizeOut.dy;

    float* planeR = dstChw;
    float* planeG = dstChw + static_cast<size_t>(dstW) * dstH;
    float* planeB = dstChw + 2 * static_cast<size_t>(dstW) * dstH;

    const float grayR = (114.0f / 255.0f - m_param.meanR) / m_param.stdR;
    const float grayG = (114.0f / 255.0f - m_param.meanG) / m_param.stdG;
    const float grayB = (114.0f / 255.0f - m_param.meanB) / m_param.stdB;
    std::fill(planeR, planeR + static_cast<size_t>(dstW) * dstH, grayR);
    std::fill(planeG, planeG + static_cast<size_t>(dstW) * dstH, grayG);
    std::fill(planeB, planeB + static_cast<size_t>(dstW) * dstH, grayB);

    const float inv255 = 1.0f / 255.0f;
    const float invStdR = 1.0f / m_param.stdR;
    const float invStdG = 1.0f / m_param.stdG;
    const float invStdB = 1.0f / m_param.stdB;

    for (int y = 0; y < newH; ++y) {
        float sy = (y + 0.5f) / resizeOut.sampleScaleY - 0.5f;
        if (sy < 0) sy = 0;
        if (sy > srcH - 1) sy = static_cast<float>(srcH - 1);
        int y0 = static_cast<int>(std::floor(sy));
        int y1 = std::min(y0 + 1, srcH - 1);
        float wy = sy - y0;
        int dstY = y + dy;
        if (dstY < 0 || dstY >= dstH) continue;

        for (int x = 0; x < newW; ++x) {
            float sx = (x + 0.5f) / resizeOut.sampleScaleX - 0.5f;
            if (sx < 0) sx = 0;
            if (sx > srcW - 1) sx = static_cast<float>(srcW - 1);
            int x0 = static_cast<int>(std::floor(sx));
            int x1 = std::min(x0 + 1, srcW - 1);
            float wx = sx - x0;

            const uint8_t* p00 = rgb + (static_cast<size_t>(y0) * srcW + x0) * 3;
            const uint8_t* p01 = rgb + (static_cast<size_t>(y0) * srcW + x1) * 3;
            const uint8_t* p10 = rgb + (static_cast<size_t>(y1) * srcW + x0) * 3;
            const uint8_t* p11 = rgb + (static_cast<size_t>(y1) * srcW + x1) * 3;

            float w00 = (1 - wx) * (1 - wy), w01 = wx * (1 - wy);
            float w10 = (1 - wx) * wy,       w11 = wx * wy;

            float r = (p00[0]*w00 + p01[0]*w01 + p10[0]*w10 + p11[0]*w11) * inv255;
            float g = (p00[1]*w00 + p01[1]*w01 + p10[1]*w10 + p11[1]*w11) * inv255;
            float b = (p00[2]*w00 + p01[2]*w01 + p10[2]*w10 + p11[2]*w11) * inv255;

            size_t dstIdx = static_cast<size_t>(dstY) * dstW + (x + dx);
            planeR[dstIdx] = (r - m_param.meanR) * invStdR;
            planeG[dstIdx] = (g - m_param.meanG) * invStdG;
            planeB[dstIdx] = (b - m_param.meanB) * invStdB;
        }
    }
    return true;
}

bool RFDetrDetectorInfer::PreprocessToGpu(
    const std::vector<FrameInput>& frames,
    std::vector<ResizeInfo>& resizeInfos,
    void* inputDevice,
    void* stream) {
#ifdef WITH_CUDA_KERNELS
    if (!inputDevice || !stream || frames.empty()) return false;

    const size_t batch = frames.size();
    size_t requiredStrideBytes = 1;
    for (const auto& frame : frames) {
        if (frame.width > 0 && frame.height > 0) {
            requiredStrideBytes = std::max(
                requiredStrideBytes,
                static_cast<size_t>(frame.width) * frame.height * 3);
        }
    }

    if (m_rgbStrideBytes < requiredStrideBytes) {
        if (m_rgbDevice) cudaFree(m_rgbDevice);
        m_rgbDevice = nullptr;
        m_rgbDeviceBytes = 0;
        m_rgbStrideBytes = requiredStrideBytes;
    }

    const size_t requiredDeviceBytes = m_rgbStrideBytes * batch;
    if (m_rgbDeviceBytes < requiredDeviceBytes || !m_rgbDevice) {
        if (m_rgbDevice) cudaFree(m_rgbDevice);
        m_rgbDevice = nullptr;
        m_rgbDeviceBytes = 0;
        if (cudaMalloc(&m_rgbDevice, requiredDeviceBytes) != cudaSuccess) {
            LOG_ERROR("RFDetrDetectorInfer: cudaMalloc RGB staging buffer failed ({} bytes)",
                      requiredDeviceBytes);
            return false;
        }
        m_rgbDeviceBytes = requiredDeviceBytes;
    }

    const size_t infoBytes = batch * sizeof(RfdetrGpuFrameInfo);
    if (m_frameInfoDeviceBytes < infoBytes || !m_frameInfoDevice) {
        if (m_frameInfoDevice) cudaFree(m_frameInfoDevice);
        m_frameInfoDevice = nullptr;
        m_frameInfoDeviceBytes = 0;
        if (cudaMalloc(&m_frameInfoDevice, infoBytes) != cudaSuccess) {
            LOG_ERROR("RFDetrDetectorInfer: cudaMalloc frame metadata buffer failed ({} bytes)",
                      infoBytes);
            return false;
        }
        m_frameInfoDeviceBytes = infoBytes;
    }

    std::vector<RfdetrGpuFrameInfo> frameInfo(batch);
    for (size_t i = 0; i < batch; ++i) {
        const FrameInput& frame = frames[i];
        const size_t expectedBytes = frame.width > 0 && frame.height > 0
            ? static_cast<size_t>(frame.width) * frame.height * 3 : 0;
        const bool valid = frame.rgb != nullptr && frame.width > 0 &&
                           frame.height > 0 && frame.channels == 3 &&
                           (frame.dataBytes == 0 || frame.dataBytes >= expectedBytes);
        resizeInfos[i] = valid
            ? ComputeResize(frame.width, frame.height,
                            m_param.inputSize, m_param.inputSize)
            : ResizeInfo{};
        frameInfo[i].width = frame.width;
        frameInfo[i].height = frame.height;
        frameInfo[i].resizedWidth = resizeInfos[i].resizedWidth;
        frameInfo[i].resizedHeight = resizeInfos[i].resizedHeight;
        frameInfo[i].padX = resizeInfos[i].dx;
        frameInfo[i].padY = resizeInfos[i].dy;
        frameInfo[i].valid = valid ? 1 : 0;

        if (valid) {
            TIMER_SCOPE_AVERAGE_MS("Detector.RawH2D", 1, 5000);
            if (cudaMemcpyAsync(
                    static_cast<uint8_t*>(m_rgbDevice) + i * m_rgbStrideBytes,
                    frame.rgb, expectedBytes, cudaMemcpyHostToDevice,
                    static_cast<cudaStream_t>(stream)) != cudaSuccess) {
                LOG_ERROR("RFDetrDetectorInfer: RGB H2D upload failed for batch item {}", i);
                return false;
            }
        }
    }

    {
        TIMER_SCOPE_AVERAGE_MS("Detector.FrameInfoH2D", 1, 5000);
        if (cudaMemcpyAsync(m_frameInfoDevice, frameInfo.data(), infoBytes,
                            cudaMemcpyHostToDevice,
                            static_cast<cudaStream_t>(stream)) != cudaSuccess) {
            LOG_ERROR("RFDetrDetectorInfer: frame metadata H2D upload failed");
            return false;
        }
    }

    {
        TIMER_SCOPE_AVERAGE_MS("Detector.PreprocessKernel", batch, 5000);
        if (!LaunchRfdetrGpuPreprocess(
                static_cast<const uint8_t*>(m_rgbDevice), m_rgbStrideBytes,
                static_cast<const RfdetrGpuFrameInfo*>(m_frameInfoDevice),
                static_cast<float*>(inputDevice), static_cast<int>(batch),
                m_param.inputSize, m_param.inputSize,
                m_param.meanR, m_param.meanG, m_param.meanB,
                1.0f / m_param.stdR, 1.0f / m_param.stdG, 1.0f / m_param.stdB,
                stream)) {
            LOG_ERROR("RFDetrDetectorInfer: GPU preprocessing kernel launch failed");
            return false;
        }
    }
    return true;
#else
    (void)frames;
    (void)resizeInfos;
    (void)inputDevice;
    (void)stream;
    return false;
#endif
}

void RFDetrDetectorInfer::ReleaseGpuBuffers() {
#ifdef WITH_CUDA_KERNELS
    if (m_rgbDevice) cudaFree(m_rgbDevice);
    if (m_frameInfoDevice) cudaFree(m_frameInfoDevice);
#endif
    m_rgbDevice = nullptr;
    m_frameInfoDevice = nullptr;
    m_rgbStrideBytes = 0;
    m_rgbDeviceBytes = 0;
    m_frameInfoDeviceBytes = 0;
}

bool RFDetrDetectorInfer::EnsureHostBuffer(HostFloatBuffer& buffer,
                                           size_t floats,
                                           const char* name) {
    if (floats == 0) return false;
    if (buffer.capacity >= floats && buffer.Data() != nullptr) {
        buffer.size = floats;
        return true;
    }

    ReleaseHostBuffer(buffer);

#ifdef WITH_TENSORRT
    void* allocated = nullptr;
    const size_t bytes = floats * sizeof(float);
    if (cudaHostAlloc(&allocated, bytes, cudaHostAllocDefault) == cudaSuccess) {
        buffer.pinned = static_cast<float*>(allocated);
        buffer.capacity = floats;
        buffer.size = floats;
        buffer.usingPinned = true;
        LOG_DEBUG("RFDetrDetectorInfer: allocated pinned host buffer '{}' ({} bytes)",
                  name, bytes);
        return true;
    }
    LOG_WARN("RFDetrDetectorInfer: pinned host allocation failed for '{}'; using pageable memory",
             name);
#endif

    try {
        buffer.fallback.resize(floats);
    } catch (const std::exception& exc) {
        LOG_ERROR("RFDetrDetectorInfer: host buffer '{}' allocation failed: {}",
                  name, exc.what());
        return false;
    }
    buffer.capacity = floats;
    buffer.size = floats;
    buffer.usingPinned = false;
    return true;
}

void RFDetrDetectorInfer::ReleaseHostBuffer(HostFloatBuffer& buffer) {
#ifdef WITH_TENSORRT
    if (buffer.usingPinned && buffer.pinned != nullptr) {
        cudaFreeHost(buffer.pinned);
    }
#endif
    buffer.pinned = nullptr;
    buffer.size = 0;
    buffer.capacity = 0;
    buffer.usingPinned = false;
    buffer.fallback.clear();
    buffer.fallback.shrink_to_fit();
}

void RFDetrDetectorInfer::ReleaseHostBuffers() {
    ReleaseHostBuffer(m_inputHost);
    ReleaseHostBuffer(m_outputHost);
    ReleaseHostBuffer(m_logitsHost);
    ReleaseHostBuffer(m_boxesHost);
}

// ---------------------------------------------------------------------------
// Postprocessing (pure CPU)
// ---------------------------------------------------------------------------

void RFDetrDetectorInfer::DecodeBaked(const float* out, int numQueries,
                                      std::vector<RawBox>& boxesOut) const {
    boxesOut.clear();
    for (int q = 0; q < numQueries; ++q) {
        const float* row = out + static_cast<size_t>(q) * 6;
        float score = row[4];
        int classId = static_cast<int>(row[5]);
        if (score < m_param.confidenceThreshold) continue;
        if (!m_param.targetClasses.empty() &&
            m_param.targetClasses.find(classId) == m_param.targetClasses.end()) continue;
        boxesOut.push_back(RawBox{row[0], row[1], row[2], row[3], score, classId});
    }
}

void RFDetrDetectorInfer::DecodeRaw(const float* logits, const float* boxes,
                                    int numQueries, int numClasses,
                                    std::vector<RawBox>& boxesOut) const {
    boxesOut.clear();
    const float inputSize = static_cast<float>(m_param.inputSize);

    // RF-DETR's Python postprocess ranks all query/class pairs globally and
    // keeps the top 300 before applying the confidence and class filters.
    // Selecting only one target class per query changes both the candidate
    // set and the tracker input, especially when multiple target classes are
    // confident for the same query.
    struct Candidate {
        float score;
        int query;
        int classId;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(numQueries) * numClasses);
    for (int q = 0; q < numQueries; ++q) {
        const float* logitRow = logits + static_cast<size_t>(q) * numClasses;
        for (int c = 0; c < numClasses; ++c) {
            candidates.push_back(Candidate{Sigmoid(logitRow[c]), q, c});
        }
    }

    const size_t topK = std::min<size_t>(300, candidates.size());
    std::partial_sort(
        candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(topK),
        candidates.end(),
        [](const Candidate& lhs, const Candidate& rhs) {
            return lhs.score > rhs.score;
        });

    for (size_t i = 0; i < topK; ++i) {
        const auto& candidate = candidates[i];
        if (candidate.score <= m_param.confidenceThreshold) continue;
        if (!m_param.targetClasses.empty() &&
            m_param.targetClasses.find(candidate.classId) == m_param.targetClasses.end()) {
            continue;
        }

        const float* boxRow = boxes + static_cast<size_t>(candidate.query) * 4;

        float cx = boxRow[0] * inputSize;
        float cy = boxRow[1] * inputSize;
        float w  = boxRow[2] * inputSize;
        float h  = boxRow[3] * inputSize;
        boxesOut.push_back(RawBox{cx - w*0.5f, cy - h*0.5f,
                                  cx + w*0.5f, cy + h*0.5f,
                                  candidate.score, candidate.classId});
    }
}

void RFDetrDetectorInfer::UndoResize(std::vector<RawBox>& boxes,
                                      const ResizeInfo& resize,
                                      int origW, int origH) const {
    if (resize.scaleX <= 0.0f || resize.scaleY <= 0.0f) return;
    const float invScaleX = 1.0f / resize.scaleX;
    const float invScaleY = 1.0f / resize.scaleY;
    for (auto& b : boxes) {
        b.x0 = std::max(0.0f, std::min((b.x0 - resize.dx) * invScaleX, static_cast<float>(origW)));
        b.y0 = std::max(0.0f, std::min((b.y0 - resize.dy) * invScaleY, static_cast<float>(origH)));
        b.x1 = std::max(0.0f, std::min((b.x1 - resize.dx) * invScaleX, static_cast<float>(origW)));
        b.y1 = std::max(0.0f, std::min((b.y1 - resize.dy) * invScaleY, static_cast<float>(origH)));
    }
}

} // namespace detector
