#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace inference {

// ---------------------------------------------------------------------------
// Backend-agnostic tensor metadata
// ---------------------------------------------------------------------------

/// Maximum number of dimensions supported.
static constexpr int kMaxDims = 8;

/// Simple N-dimensional shape (backend-agnostic replacement for nvinfer1::Dims).
struct Dims {
    int count = 0;
    int d[kMaxDims] = {};

    Dims() = default;
    Dims(int d0) : count(1), d{d0} {}
    Dims(int d0, int d1) : count(2), d{d0, d1} {}
    Dims(int d0, int d1, int d2) : count(3), d{d0, d1, d2} {}
    Dims(int d0, int d1, int d2, int d3) : count(4), d{d0, d1, d2, d3} {}

    int operator[](int i) const { return d[i]; }
    int& operator[](int i) { return d[i]; }

    /// Total element count (product of all dims). Returns 0 if count == 0.
    size_t NumElements() const {
        if (count <= 0) return 0;
        size_t n = 1;
        for (int i = 0; i < count; ++i) n *= static_cast<size_t>(d[i]);
        return n;
    }
};

/// Element data type.
enum class DataType {
    kFloat32,
    kFloat16,
    kInt32,
    kInt8,
    kUnknown,
};

/// Size in bytes of a single element.
inline size_t DataTypeSize(DataType dt) {
    switch (dt) {
        case DataType::kFloat32: return 4;
        case DataType::kFloat16: return 2;
        case DataType::kInt32:   return 4;
        case DataType::kInt8:    return 1;
        default:                 return 0;
    }
}

/// Metadata describing an input or output tensor.
struct TensorInfo {
    std::string name;
    Dims        dims;
    DataType    dtype = DataType::kUnknown;

    size_t SizeBytes() const { return dims.NumElements() * DataTypeSize(dtype); }
};

// ---------------------------------------------------------------------------
// Abstract inference engine interface
// ---------------------------------------------------------------------------

/**
 * @brief Backend-agnostic inference engine interface.
 *
 * Implementations (TensorRT, ONNX Runtime, OpenVINO, ...) own ALL device
 * resources internally. Callers only interact with host memory:
 *
 *   1. Load(enginePath)
 *   2. SetInputFromHost(name, hostPtr, bytes, dims)  // H2D handled internally
 *   3. Infer()
 *   4. CopyOutputToHost(name, hostPtr, bytes)         // D2H handled internally
 *
 * This keeps task modules (detectors, pose estimators) free of CUDA/TRT
 * headers, #ifdef guards, and manual buffer management.
 */
class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;

    // --- Lifecycle ---

    /// Load a serialized engine/model from disk. Returns true on success.
    virtual bool Load(const std::string& enginePath) = 0;

    /// Release all resources. Safe to call multiple times.
    virtual void Release() = 0;

    /// True if the engine is loaded and ready for inference.
    virtual bool IsReady() const = 0;

    // --- Tensor introspection ---

    /// Number of input tensors.
    virtual int GetInputCount() const = 0;

    /// Number of output tensors.
    virtual int GetOutputCount() const = 0;

    /// Get input tensor names (ordered by binding index).
    virtual std::vector<std::string> GetInputNames() const = 0;

    /// Get output tensor names (ordered by binding index).
    virtual std::vector<std::string> GetOutputNames() const = 0;

    /// Query metadata for an input tensor by name.
    virtual TensorInfo GetInputInfo(const std::string& name) const = 0;

    /// Query metadata for an output tensor by name.
    virtual TensorInfo GetOutputInfo(const std::string& name) const = 0;

    // --- Inference ---

    /**
     * @brief Upload host data to an input tensor and set its shape.
     *
     * The engine handles:
     *   - Allocating/resizing the device buffer if needed
     *   - cudaMemcpy H2D (async on internal stream)
     *   - Setting the dynamic input shape
     *
     * @param name     Input tensor name.
     * @param hostPtr  Host buffer containing the data.
     * @param bytes    Size of host data in bytes.
     * @param dims     Shape to set for this inference call.
     * @return true on success.
     */
    virtual bool SetInputFromHost(const std::string& name,
                                  const void* hostPtr,
                                  size_t bytes,
                                  const Dims& dims) = 0;

    /**
     * @brief Enqueue inference on the backend stream.
     *
     * All input tensors must have been set via SetInputFromHost() beforehand.
     * Output buffers are populated on device; use CopyOutputToHost() to read.
     * Backends may return before device execution completes. The host-output
     * copy is responsible for waiting until the requested output is ready.
     */
    virtual bool Infer() = 0;

    /**
     * @brief Copy an output tensor from device to host.
     *
     * @param name     Output tensor name.
     * @param hostPtr  Destination host buffer (must be >= bytes).
     * @param bytes    Number of bytes to copy.
     * @return true on success.
     */
    virtual bool CopyOutputToHost(const std::string& name,
                                  void* hostPtr,
                                  size_t bytes) = 0;
};

} // namespace inference
