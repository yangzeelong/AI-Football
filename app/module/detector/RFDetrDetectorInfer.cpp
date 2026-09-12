#include "RFDetrDetectorInfer.hpp"
#include "inference/TensorRTEngine.hpp"

#include <nexusflow/Logging.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
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

    if (m_param.enginePath.empty()) {
        LOG_WARN("RFDetrDetectorInfer: enginePath is empty");
        return false;
    }

    m_engine = std::make_unique<inference::TensorRTEngine>();
    if (!m_engine->Load(m_param.enginePath)) {
        LOG_ERROR("RFDetrDetectorInfer: failed to load engine {}", m_param.enginePath);
        return false;
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

    // --- Allocate host buffers (single frame; batch caller provides count) ---
    // We'll resize dynamically in InferBatch based on actual batch size.
    m_ready = true;
    return true;
}

void RFDetrDetectorInfer::Release() {
    if (m_engine) { m_engine->Release(); m_engine.reset(); }
    m_ready = false;
    m_inputHost.clear();
    m_outputHost.clear();
    m_logitsHost.clear();
    m_boxesHost.clear();
}

// ---------------------------------------------------------------------------
// InferBatch: preprocess → engine → postprocess
// ---------------------------------------------------------------------------

bool RFDetrDetectorInfer::InferBatch(const std::vector<FrameInput>& frames,
                                     std::vector<std::vector<Detection>>& results) {
    results.clear();
    if (!m_ready || frames.empty()) return false;

    const int B = static_cast<int>(frames.size());
    const size_t perFrameFloats = static_cast<size_t>(m_param.inputSize) * m_param.inputSize * 3;

    // Ensure host buffers are large enough.
    if (m_inputHost.size() < perFrameFloats * B)
        m_inputHost.resize(perFrameFloats * B, 0.0f);

    // --- Preprocess ---
    std::vector<ResizeInfo> resizeInfos(B);
    for (int i = 0; i < B; ++i) {
        float* dst = m_inputHost.data() + i * perFrameFloats;
        if (!frames[i].rgb || frames[i].width <= 0 || frames[i].height <= 0) {
            std::fill(dst, dst + perFrameFloats, 0.0f);
            resizeInfos[i] = ResizeInfo{};
        } else {
            PreprocessToHost(frames[i].rgb, frames[i].width, frames[i].height,
                             resizeInfos[i], dst);
        }
    }

    // --- Engine inference ---
    inference::Dims batchDims(B, 3, m_param.inputSize, m_param.inputSize);
    size_t batchInputBytes = B * perFrameFloats * sizeof(float);

    if (!m_engine->SetInputFromHost(m_param.inputBindingName,
                                    m_inputHost.data(), batchInputBytes, batchDims)) {
        LOG_ERROR("RFDetrDetectorInfer: SetInputFromHost failed");
        return false;
    }
    if (!m_engine->Infer()) {
        LOG_ERROR("RFDetrDetectorInfer: Infer failed");
        return false;
    }

    // --- Postprocess per frame ---
    results.resize(B);
    for (int i = 0; i < B; ++i) {
        std::vector<RawBox> rawBoxes;

        if (m_outputFormat == OutputFormat::Baked) {
            size_t perFrameOut = static_cast<size_t>(m_param.numQueries) * 6;
            size_t offset = static_cast<size_t>(i) * perFrameOut;
            if (m_outputHost.size() < offset + perFrameOut)
                m_outputHost.resize(offset + perFrameOut, 0.0f);

            if (m_engine->CopyOutputToHost(m_param.outputBindingName,
                                           m_outputHost.data() + offset,
                                           perFrameOut * sizeof(float))) {
                DecodeBaked(m_outputHost.data() + offset, m_param.numQueries, rawBoxes);
            }
        } else {
            size_t perFrameLogits = static_cast<size_t>(m_param.numQueries) * m_param.numClasses;
            size_t perFrameBoxes  = static_cast<size_t>(m_param.numQueries) * 4;
            size_t logitsOff = static_cast<size_t>(i) * perFrameLogits;
            size_t boxesOff  = static_cast<size_t>(i) * perFrameBoxes;

            if (m_logitsHost.size() < logitsOff + perFrameLogits)
                m_logitsHost.resize(logitsOff + perFrameLogits, 0.0f);
            if (m_boxesHost.size() < boxesOff + perFrameBoxes)
                m_boxesHost.resize(boxesOff + perFrameBoxes, 0.0f);

            bool okL = m_engine->CopyOutputToHost(m_param.logitsBindingName,
                                                  m_logitsHost.data() + logitsOff,
                                                  perFrameLogits * sizeof(float));
            bool okB = m_engine->CopyOutputToHost(m_param.boxesBindingName,
                                                  m_boxesHost.data() + boxesOff,
                                                  perFrameBoxes * sizeof(float));
            if (okL && okB) {
                const auto& frame = frames[i];
                if (ShouldLogDetectorFrame(frame.frameId)) {
                    float inputMin = std::numeric_limits<float>::max();
                    float inputMax = std::numeric_limits<float>::lowest();
                    const float* input = m_inputHost.data() +
                        static_cast<size_t>(i) * perFrameFloats;
                    for (size_t k = 0; k < perFrameFloats; ++k) {
                        inputMin = std::min(inputMin, input[k]);
                        inputMax = std::max(inputMax, input[k]);
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

                    const float* logits = m_logitsHost.data() + logitsOff;
                    const float* boxes = m_boxesHost.data() + boxesOff;
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
                        "logits=[{:.3f},{:.3f}] boxes=[{:.3f},{:.3f}] "
                        "bestAny=(q={},class={},score={:.3f}) "
                        "bestTarget=(q={},class={},score={:.3f}) "
                        "pairsAbove={} targetPairsAbove={} threshold={:.2f}",
                        frame.frameId, frame.width, frame.height, inputMin, inputMax,
                        logitMin, logitMax, boxMin, boxMax,
                        bestAnyQuery, bestAnyClass, bestAnyScore,
                        bestTargetQuery, bestTargetClass, bestTargetScore,
                        anyPairsAbove, targetPairsAbove, m_param.confidenceThreshold);
                }

                DecodeRaw(m_logitsHost.data() + logitsOff,
                          m_boxesHost.data() + boxesOff,
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
    return true;
}

// ---------------------------------------------------------------------------
// Preprocessing (pure CPU)
// ---------------------------------------------------------------------------

RFDetrDetectorInfer::ResizeInfo
RFDetrDetectorInfer::ComputeResize(int srcW, int srcH, int dstW, int dstH) {
    ResizeInfo resize;
    if (srcW <= 0 || srcH <= 0) return resize;
    // Match rfdetr.predict(): direct resize to a square, without letterbox.
    resize.scaleX = static_cast<float>(dstW) / srcW;
    resize.scaleY = static_cast<float>(dstH) / srcH;
    return resize;
}

bool RFDetrDetectorInfer::PreprocessToHost(const uint8_t* rgb, int srcW, int srcH,
                                           ResizeInfo& resizeOut, float* dstChw) const {
    if (!rgb || srcW <= 0 || srcH <= 0 || !dstChw) return false;
    const int dstW = m_param.inputSize;
    const int dstH = m_param.inputSize;
    resizeOut = ComputeResize(srcW, srcH, dstW, dstH);

    const int newW = dstW;
    const int newH = dstH;
    const int dx = 0;
    const int dy = 0;

    float* planeR = dstChw;
    float* planeG = dstChw + static_cast<size_t>(dstW) * dstH;
    float* planeB = dstChw + 2 * static_cast<size_t>(dstW) * dstH;

    std::fill(planeR, planeR + static_cast<size_t>(dstW) * dstH, 0.0f);
    std::fill(planeG, planeG + static_cast<size_t>(dstW) * dstH, 0.0f);
    std::fill(planeB, planeB + static_cast<size_t>(dstW) * dstH, 0.0f);

    const float inv255 = 1.0f / 255.0f;
    const float invStdR = 1.0f / m_param.stdR;
    const float invStdG = 1.0f / m_param.stdG;
    const float invStdB = 1.0f / m_param.stdB;

    for (int y = 0; y < newH; ++y) {
        float sy = (y + 0.5f) / resizeOut.scaleY - 0.5f;
        if (sy < 0) sy = 0;
        if (sy > srcH - 1) sy = static_cast<float>(srcH - 1);
        int y0 = static_cast<int>(std::floor(sy));
        int y1 = std::min(y0 + 1, srcH - 1);
        float wy = sy - y0;
        int dstY = y + dy;
        if (dstY < 0 || dstY >= dstH) continue;

        for (int x = 0; x < newW; ++x) {
            float sx = (x + 0.5f) / resizeOut.scaleX - 0.5f;
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

    for (int q = 0; q < numQueries; ++q) {
        const float* logitRow = logits + static_cast<size_t>(q) * numClasses;
        const float* boxRow   = boxes  + static_cast<size_t>(q) * 4;

        int bestCls = -1;
        float bestScore = 0.0f;
        for (int c = 0; c < numClasses; ++c) {
            if (!m_param.targetClasses.empty() &&
                m_param.targetClasses.find(c) == m_param.targetClasses.end()) {
                continue;
            }
            float s = Sigmoid(logitRow[c]);
            if (s > bestScore) { bestScore = s; bestCls = c; }
        }
        if (bestCls < 0) continue;
        if (bestScore < m_param.confidenceThreshold) continue;

        float cx = boxRow[0] * inputSize;
        float cy = boxRow[1] * inputSize;
        float w  = boxRow[2] * inputSize;
        float h  = boxRow[3] * inputSize;
        boxesOut.push_back(RawBox{cx - w*0.5f, cy - h*0.5f, cx + w*0.5f, cy + h*0.5f, bestScore, bestCls});
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
