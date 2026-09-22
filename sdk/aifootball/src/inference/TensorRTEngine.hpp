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
 * device buffers. Host callers normally use host pointers, while CUDA-aware
 * callers can prepare an input device buffer on the engine's stream.
 *
 * Typical usage:
 *   TensorRTEngine engine;
 *   engine.Load("model.engine");
 *   engine.SetInputFromHost("images", hostBuf, bytes, Dims{1,3,640,640});
 *   engine.Infer();                  // enqueue on the engine stream
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

    /// Prepare and return the engine-owned device buffer for an input tensor.
    /// This sets the dynamic shape but does not copy data.
    void* PrepareInputDevice(const std::string& name,
                             size_t bytes,
                             const Dims& dims);

    /// Return the CUDA stream used by SetInput/Infer, or nullptr when CUDA is
    /// unavailable. The returned handle is exposed as void* to keep this
    /// concrete header usable without CUDA in stub builds.
    void* GetCudaStream() const;

    bool Infer() override;
    bool CopyOutputToHost(const std::string& name,
                          void* hostPtr,
                          size_t bytes) override;
    bool CopyOutputsToHost(const std::vector<HostCopy>& copies) override;

private:
#ifdef WITH_TENSORRT
    struct BufferInfo {
        void*  devicePtr = nullptr;
        size_t sizeBytes = 0;
    };

    bool CreateContext();
    bool EnsureInputBuffer(const std::string& name, size_t bytes);
    bool EnsureOutputBuffer(const std::string& name, size_t bytes);
    /// Validate and enqueue one device-to-host copy without synchronizing.
    bool EnqueueOutputCopy(const std::string& name, void* hostPtr, size_t bytes);

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
