#include "RFDetrCudaPreprocess.hpp"

#include <cuda_runtime.h>

namespace detector {
namespace {

__device__ inline float Clamp(float value, float low, float high) {
    return value < low ? low : (value > high ? high : value);
}

__device__ inline float SampleChannel(const uint8_t* source,
                                      int width,
                                      int height,
                                      float sx,
                                      float sy,
                                      int channel) {
    sx = Clamp(sx, 0.0f, static_cast<float>(width - 1));
    sy = Clamp(sy, 0.0f, static_cast<float>(height - 1));

    const int x0 = static_cast<int>(floorf(sx));
    const int y0 = static_cast<int>(floorf(sy));
    const int x1 = x0 + 1 < width ? x0 + 1 : x0;
    const int y1 = y0 + 1 < height ? y0 + 1 : y0;
    const float wx = sx - static_cast<float>(x0);
    const float wy = sy - static_cast<float>(y0);

    const size_t p00 = (static_cast<size_t>(y0) * width + x0) * 3 + channel;
    const size_t p01 = (static_cast<size_t>(y0) * width + x1) * 3 + channel;
    const size_t p10 = (static_cast<size_t>(y1) * width + x0) * 3 + channel;
    const size_t p11 = (static_cast<size_t>(y1) * width + x1) * 3 + channel;

    const float top = static_cast<float>(source[p00]) * (1.0f - wx) +
                      static_cast<float>(source[p01]) * wx;
    const float bottom = static_cast<float>(source[p10]) * (1.0f - wx) +
                         static_cast<float>(source[p11]) * wx;
    return (top * (1.0f - wy) + bottom * wy) * (1.0f / 255.0f);
}

__device__ inline float Normalize(float value, float mean, float invStd) {
    return (value * (1.0f / 255.0f) - mean) * invStd;
}

__global__ void RfdetrPreprocessKernel(
    const uint8_t* srcBatch,
    size_t srcStrideBytes,
    const RfdetrGpuFrameInfo* frameInfo,
    float* dstBatch,
    int dstWidth,
    int dstHeight,
    float meanR,
    float meanG,
    float meanB,
    float invStdR,
    float invStdG,
    float invStdB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batchIndex = blockIdx.z;
    if (x >= dstWidth || y >= dstHeight) return;

    const size_t planeSize = static_cast<size_t>(dstWidth) * dstHeight;
    float* output = dstBatch + static_cast<size_t>(batchIndex) * 3 * planeSize;
    const RfdetrGpuFrameInfo info = frameInfo[batchIndex];
    const size_t index = static_cast<size_t>(y) * dstWidth + x;

    if (!info.valid || info.width <= 0 || info.height <= 0 ||
        info.resizedWidth <= 0 || info.resizedHeight <= 0) {
        const float grayR = Normalize(114.0f, meanR, invStdR);
        const float grayG = Normalize(114.0f, meanG, invStdG);
        const float grayB = Normalize(114.0f, meanB, invStdB);
        output[index] = grayR;
        output[planeSize + index] = grayG;
        output[2 * planeSize + index] = grayB;
        return;
    }

    const bool inImage = x >= info.padX && x < info.padX + info.resizedWidth &&
                         y >= info.padY && y < info.padY + info.resizedHeight;
    if (!inImage) {
        output[index] = Normalize(114.0f, meanR, invStdR);
        output[planeSize + index] = Normalize(114.0f, meanG, invStdG);
        output[2 * planeSize + index] = Normalize(114.0f, meanB, invStdB);
        return;
    }

    const uint8_t* source = srcBatch + static_cast<size_t>(batchIndex) *
                            srcStrideBytes;
    const float scaleX = static_cast<float>(info.resizedWidth) / info.width;
    const float scaleY = static_cast<float>(info.resizedHeight) / info.height;
    const float sx = (static_cast<float>(x - info.padX) + 0.5f) / scaleX - 0.5f;
    const float sy = (static_cast<float>(y - info.padY) + 0.5f) / scaleY - 0.5f;

    const float r = SampleChannel(source, info.width, info.height, sx, sy, 0);
    const float g = SampleChannel(source, info.width, info.height, sx, sy, 1);
    const float b = SampleChannel(source, info.width, info.height, sx, sy, 2);
    output[index] = (r - meanR) * invStdR;
    output[planeSize + index] = (g - meanG) * invStdG;
    output[2 * planeSize + index] = (b - meanB) * invStdB;
}

} // namespace

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
                               void* stream) {
    if (!srcBatch || srcStrideBytes == 0 || !frameInfo || !dstBatch ||
        batch <= 0 || dstWidth <= 0 || dstHeight <= 0 || !stream) {
        return false;
    }

    const dim3 block(16, 16, 1);
    const dim3 grid((dstWidth + block.x - 1) / block.x,
                    (dstHeight + block.y - 1) / block.y,
                    static_cast<unsigned int>(batch));
    RfdetrPreprocessKernel<<<grid, block, 0,
                             static_cast<cudaStream_t>(stream)>>>(
        srcBatch, srcStrideBytes, frameInfo, dstBatch, dstWidth, dstHeight,
        meanR, meanG, meanB, invStdR, invStdG, invStdB);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace detector
