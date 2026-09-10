#pragma once

#include "IInferenceEngine.hpp"

#ifdef WITH_TENSORRT
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#endif

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace inference {

/**
 * @brief TensorRT 10.x backend for IInferenceEngine.
 *
 * Owns ALL GPU resources: runtime, engine, context, stream, input/output
 * device buffers. Callers interact exclusively through host pointers.
 *
 * Typical usage:
 *   TensorRTEngine engine;
 *   engine.Load("model.engine");
 *   engine.SetInputFromHost("images", hostBuf, bytes, Dims{1,3,640,640});
 *   engine.Infer();
 *   engine.CopyOutputToHost("output0", hostOut, outBytes);
 */
class TensorRTEngine : public IInferenceEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine() override;

    // Non-copyable, non-movable (owns GPU resources).
    TensorRTEngine(const TensorRTEngine&) = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;

    // --- IInferenceEngine interface ---
    bool Load(const std::string& enginePath) override;
    void Release() override;
    bool IsReady() const override;

    int GetInputCount() const override;
    int GetOutputCount() const override;
    std::vector<std::string> GetInputNames() const override;
    std::vector<std::string> GetOutputNames() const override;
    TensorInfo GetInputInfo(const std::string& name) const override;
    TensorInfo GetOutputInfo(const std::string& name) const override;

    bool SetInputFromHost(const std::string& name,
                          const void* hostPtr,
                          size_t bytes,
                          const Dims& dims) override;
    bool Infer() override;
    bool CopyOutputToHost(const std::string& name,
                          void* hostPtr,
                          size_t bytes) override;

private:
#ifdef WITH_TENSORRT
    struct BufferInfo {
        void*  devicePtr = nullptr;
        size_t sizeBytes = 0;
    };

    bool CreateContext();
    bool EnsureInputBuffer(const std::string& name, size_t bytes);
    bool EnsureOutputBuffer(const std::string& name, size_t bytes);

    static Dims     ConvertDims(const nvinfer1::Dims& trtDims);
    static DataType ConvertDType(nvinfer1::DataType trtType);
    static nvinfer1::Dims ConvertToTrtDims(const Dims& dims);

    std::unique_ptr<nvinfer1::IRuntime>           m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine>        m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext>  m_context;
    cudaStream_t m_stream = nullptr;

    std::unordered_map<std::string, BufferInfo> m_inputBuffers;
    std::unordered_map<std::string, BufferInfo> m_outputBuffers;

    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    };
    static Logger s_logger;
#endif
};

} // namespace inference
