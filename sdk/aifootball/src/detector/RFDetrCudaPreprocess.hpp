#pragma once

#include <cstddef>
#include <cstdint>

namespace detector {

struct RfdetrGpuFrameInfo {
    int width = 0;
    int height = 0;
    int resizedWidth = 0;
    int resizedHeight = 0;
    int padX = 0;
    int padY = 0;
    int valid = 0;
};

/// Upload-independent CUDA implementation of RF-DETR preprocessing.
///
/// The source batch is laid out as fixed-size RGB24 slots and the destination
/// is NCHW float32. The caller owns both device buffers and the CUDA stream.
bool LaunchRfdetrGpuPreprocess(const uint8_t* srcBatch,
                               size_t srcStrideBytes,
                               const RfdetrGpuFrameInfo* frameInfo,
                               float* dstBatch,
                               int batch,
                               int dstWidth,
                               int dstHeight,
                               float meanR,
                               float meanG,
                               float meanB,
                               float invStdR,
                               float invStdG,
                               float invStdB,
                               void* stream);

} // namespace detector
