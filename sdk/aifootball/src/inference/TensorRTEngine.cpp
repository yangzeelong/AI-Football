#include "TensorRTEngine.hpp"

#include <nexusflow/Logging.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>

namespace inference {

#ifdef WITH_TENSORRT

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

TensorRTEngine::Logger TensorRTEngine::s_logger;

void TensorRTEngine::Logger::log(Severity severity, const char* msg) noexcept {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:   LOG_ERROR("[TensorRT] {}", msg); break;
        case Severity::kWARNING: LOG_WARN("[TensorRT] {}", msg);  break;
        case Severity::kINFO:    LOG_INFO("[TensorRT] {}", msg);  break;
        case Severity::kVERBOSE: LOG_DEBUG("[TensorRT] {}", msg); break;
    }
}

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

TensorRTEngine::TensorRTEngine() = default;

TensorRTEngine::~TensorRTEngine() {
    Release();
}

// ---------------------------------------------------------------------------
// Load engine from file + create context
// ---------------------------------------------------------------------------

bool TensorRTEngine::Load(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("TensorRTEngine: failed to open engine file: {}", enginePath);
        return false;
    }

    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    if (!file.read(engineData.data(), static_cast<std::streamsize>(size))) {
        LOG_ERROR("TensorRTEngine: failed to read engine file: {}", enginePath);
        return false;
    }
    file.close();

    m_runtime.reset(nvinfer1::createInferRuntime(s_logger));
    if (!m_runtime) {
        LOG_ERROR("TensorRTEngine: failed to create runtime");
        return false;
    }

    m_engine.reset(m_runtime->deserializeCudaEngine(engineData.data(), size));
    if (!m_engine) {
        LOG_ERROR("TensorRTEngine: failed to deserialize engine");
        return false;
    }

    LOG_INFO("TensorRTEngine: loaded {}, ioTensors={}", enginePath, m_engine->getNbIOTensors());

    return CreateContext();
}

bool TensorRTEngine::CreateContext() {
    if (!m_engine) return false;

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        LOG_ERROR("TensorRTEngine: failed to create execution context");
        return false;
    }

    if (cudaStreamCreate(&m_stream) != cudaSuccess) {
        LOG_ERROR("TensorRTEngine: failed to create CUDA stream");
        return false;
    }

    // Enumerate IO tensors and classify as input/output.
    int nbTensors = m_engine->getNbIOTensors();
    for (int i = 0; i < nbTensors; ++i) {
        const char* name = m_engine->getIOTensorName(i);
        auto mode = m_engine->getTensorIOMode(name);
        BufferInfo info;
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            m_inputBuffers[name] = info;
        } else {
            m_outputBuffers[name] = info;
        }
    }

    LOG_INFO("TensorRTEngine: context created, {} inputs, {} outputs",
             m_inputBuffers.size(), m_outputBuffers.size());
    return true;
}

// ---------------------------------------------------------------------------
// Release
// ---------------------------------------------------------------------------

void TensorRTEngine::Release() {
    for (auto& kv : m_inputBuffers) {
        if (kv.second.devicePtr) { cudaFree(kv.second.devicePtr); kv.second.devicePtr = nullptr; }
    }
    for (auto& kv : m_outputBuffers) {
        if (kv.second.devicePtr) { cudaFree(kv.second.devicePtr); kv.second.devicePtr = nullptr; }
    }
    m_inputBuffers.clear();
    m_outputBuffers.clear();

    m_context.reset();
    if (m_stream) { cudaStreamDestroy(m_stream); m_stream = nullptr; }
    m_engine.reset();
    m_runtime.reset();
}

bool TensorRTEngine::IsReady() const {
    return m_context != nullptr;
}

// ---------------------------------------------------------------------------
// Tensor introspection
// ---------------------------------------------------------------------------

int TensorRTEngine::GetInputCount() const {
    return static_cast<int>(m_inputBuffers.size());
}

int TensorRTEngine::GetOutputCount() const {
    return static_cast<int>(m_outputBuffers.size());
}

std::vector<std::string> TensorRTEngine::GetInputNames() const {
    std::vector<std::string> names;
    names.reserve(m_inputBuffers.size());
    for (const auto& kv : m_inputBuffers) names.push_back(kv.first);
    return names;
}

std::vector<std::string> TensorRTEngine::GetOutputNames() const {
    std::vector<std::string> names;
    names.reserve(m_outputBuffers.size());
    for (const auto& kv : m_outputBuffers) names.push_back(kv.first);
    return names;
}

TensorInfo TensorRTEngine::GetInputInfo(const std::string& name) const {
    TensorInfo info;
    if (!m_engine) return info;
    info.name  = name;
    info.dims  = ConvertDims(m_engine->getTensorShape(name.c_str()));
    info.dtype = ConvertDType(m_engine->getTensorDataType(name.c_str()));
    return info;
}

TensorInfo TensorRTEngine::GetOutputInfo(const std::string& name) const {
    TensorInfo info;
    if (!m_engine) return info;
    info.name  = name;
    info.dims  = ConvertDims(m_engine->getTensorShape(name.c_str()));
    info.dtype = ConvertDType(m_engine->getTensorDataType(name.c_str()));
    return info;
}

// ---------------------------------------------------------------------------
// PrepareInputDevice: validate shape + allocate device input + SetInputShape
// ---------------------------------------------------------------------------

void* TensorRTEngine::PrepareInputDevice(const std::string& name,
                                         size_t bytes,
                                         const Dims& dims) {
    if (!m_context || bytes == 0) return nullptr;

    auto inputIt = m_inputBuffers.find(name);
    if (inputIt == m_inputBuffers.end()) {
        LOG_ERROR("TensorRTEngine: input tensor '{}' not found", name);
        return nullptr;
    }

    const nvinfer1::Dims engineDims = m_engine->getTensorShape(name.c_str());
    if (engineDims.nbDims != dims.count) {
        LOG_ERROR("TensorRTEngine: input '{}' rank mismatch, engine={} request={}",
                  name, engineDims.nbDims, dims.count);
        return nullptr;
    }

    bool dynamic = false;
    for (int i = 0; i < engineDims.nbDims; ++i) {
        if (engineDims.d[i] < 0) {
            dynamic = true;
            break;
        }
    }

    // Static engines cannot accept a larger batch merely because the caller
    // supplied a larger host buffer. Reject shape mismatches here, before a
    // potentially confusing enqueue failure.
    if (!dynamic) {
        for (int i = 0; i < engineDims.nbDims; ++i) {
            if (engineDims.d[i] != dims.d[i]) {
                LOG_ERROR("TensorRTEngine: input '{}' shape mismatch at dim {}, engine={} request={}",
                          name, i, engineDims.d[i], dims.d[i]);
                return nullptr;
            }
        }
    } else {
        for (int i = 0; i < dims.count; ++i) {
            if (dims.d[i] <= 0) {
                LOG_ERROR("TensorRTEngine: input '{}' has invalid dynamic dim {}={}",
                          name, i, dims.d[i]);
                return nullptr;
            }
            if (engineDims.d[i] > 0 && engineDims.d[i] != dims.d[i]) {
                LOG_ERROR("TensorRTEngine: input '{}' fixed dim {} mismatch, engine={} request={}",
                          name, i, engineDims.d[i], dims.d[i]);
                return nullptr;
            }
        }
    }

    const size_t expectedBytes = dims.NumElements() *
        DataTypeSize(ConvertDType(m_engine->getTensorDataType(name.c_str())));
    if (expectedBytes == 0 || bytes != expectedBytes) {
        LOG_ERROR("TensorRTEngine: input '{}' byte size mismatch, expected={} request={}",
                  name, expectedBytes, bytes);
        return nullptr;
    }

    // Ensure device buffer is large enough.
    if (!EnsureInputBuffer(name, bytes)) return nullptr;

    // Static ONNX inputs already carry their shape in the engine. Dynamic
    // inputs need an execution-context shape before enqueueV3.
    if (dynamic) {
        nvinfer1::Dims trtDims = ConvertToTrtDims(dims);
        if (!m_context->setInputShape(name.c_str(), trtDims)) {
            LOG_ERROR("TensorRTEngine: setInputShape failed for '{}'", name);
            return nullptr;
        }
    }

    return inputIt->second.devicePtr;
}

void* TensorRTEngine::GetCudaStream() const {
    return reinterpret_cast<void*>(m_stream);
}

bool TensorRTEngine::SetInputFromHost(const std::string& name,
                                      const void* hostPtr,
                                      size_t bytes,
                                      const Dims& dims) {
    if (!hostPtr || bytes == 0) return false;
    void* devicePtr = PrepareInputDevice(name, bytes, dims);
    if (!devicePtr) return false;

    // H2D copy.
    if (cudaMemcpyAsync(devicePtr, hostPtr, bytes,
                        cudaMemcpyHostToDevice, m_stream) != cudaSuccess) {
        LOG_ERROR("TensorRTEngine: cudaMemcpy H2D failed for '{}'", name);
        return false;
    }

    return true;
}

bool TensorRTEngine::EnsureInputBuffer(const std::string& name, size_t bytes) {
    auto it = m_inputBuffers.find(name);
    if (it == m_inputBuffers.end()) {
        LOG_ERROR("TensorRTEngine: input tensor '{}' not found", name);
        return false;
    }
    if (it->second.sizeBytes >= bytes && it->second.devicePtr) return true;

    // Reallocate.
    if (it->second.devicePtr) cudaFree(it->second.devicePtr);
    if (cudaMalloc(&it->second.devicePtr, bytes) != cudaSuccess) {
        LOG_ERROR("TensorRTEngine: cudaMalloc input '{}' failed ({}bytes)", name, bytes);
        it->second.devicePtr = nullptr;
        it->second.sizeBytes = 0;
        return false;
    }
    it->second.sizeBytes = bytes;
    m_context->setTensorAddress(name.c_str(), it->second.devicePtr);
    return true;
}

// ---------------------------------------------------------------------------
// Infer: enqueueV3 + synchronize
// ---------------------------------------------------------------------------

bool TensorRTEngine::Infer() {
    if (!m_context) return false;

    // Allocate output buffers based on current context shapes (lazy).
    int nbTensors = m_engine->getNbIOTensors();
    for (int i = 0; i < nbTensors; ++i) {
        const char* name = m_engine->getIOTensorName(i);
        if (m_engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kOUTPUT) continue;

        auto dims = m_context->getTensorShape(name);
        size_t elemSize = DataTypeSize(ConvertDType(m_engine->getTensorDataType(name)));
        size_t totalBytes = 1;
        for (int d = 0; d < dims.nbDims; ++d) totalBytes *= static_cast<size_t>(dims.d[d]);
        totalBytes *= elemSize;

        if (!EnsureOutputBuffer(name, totalBytes)) return false;
    }

    if (!m_context->enqueueV3(m_stream)) {
        LOG_ERROR("TensorRTEngine: enqueueV3 failed");
        return false;
    }
    cudaStreamSynchronize(m_stream);
    return true;
}

bool TensorRTEngine::EnsureOutputBuffer(const std::string& name, size_t bytes) {
    auto it = m_outputBuffers.find(name);
    if (it == m_outputBuffers.end()) {
        LOG_ERROR("TensorRTEngine: output tensor '{}' not found", name);
        return false;
    }
    if (it->second.sizeBytes >= bytes && it->second.devicePtr) return true;

    if (it->second.devicePtr) cudaFree(it->second.devicePtr);
    if (cudaMalloc(&it->second.devicePtr, bytes) != cudaSuccess) {
        LOG_ERROR("TensorRTEngine: cudaMalloc output '{}' failed ({}bytes)", name, bytes);
        it->second.devicePtr = nullptr;
        it->second.sizeBytes = 0;
        return false;
    }
    it->second.sizeBytes = bytes;
    m_context->setTensorAddress(name.c_str(), it->second.devicePtr);
    return true;
}

// ---------------------------------------------------------------------------
// CopyOutputToHost: D2H
// ---------------------------------------------------------------------------

bool TensorRTEngine::CopyOutputToHost(const std::string& name,
                                      void* hostPtr,
                                      size_t bytes) {
    auto it = m_outputBuffers.find(name);
    if (it == m_outputBuffers.end() || !it->second.devicePtr) return false;
    if (bytes > it->second.sizeBytes) {
        LOG_ERROR("TensorRTEngine: CopyOutputToHost '{}' requested {} > allocated {}",
                  name, bytes, it->second.sizeBytes);
        return false;
    }
    if (cudaMemcpyAsync(hostPtr, it->second.devicePtr, bytes,
                        cudaMemcpyDeviceToHost, m_stream) != cudaSuccess) {
        return false;
    }
    cudaStreamSynchronize(m_stream);
    return true;
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

Dims TensorRTEngine::ConvertDims(const nvinfer1::Dims& trtDims) {
    Dims out;
    out.count = std::min(trtDims.nbDims, kMaxDims);
    for (int i = 0; i < out.count; ++i) out.d[i] = trtDims.d[i];
    return out;
}

DataType TensorRTEngine::ConvertDType(nvinfer1::DataType trtType) {
    switch (trtType) {
        case nvinfer1::DataType::kFLOAT: return DataType::kFloat32;
        case nvinfer1::DataType::kHALF:  return DataType::kFloat16;
        case nvinfer1::DataType::kINT32: return DataType::kInt32;
        case nvinfer1::DataType::kINT8:  return DataType::kInt8;
        default:                         return DataType::kUnknown;
    }
}

nvinfer1::Dims TensorRTEngine::ConvertToTrtDims(const Dims& dims) {
    nvinfer1::Dims trtDims{};
    trtDims.nbDims = std::min(dims.count, static_cast<int>(nvinfer1::Dims::MAX_DIMS));
    for (int i = 0; i < trtDims.nbDims; ++i) trtDims.d[i] = dims.d[i];
    return trtDims;
}

#else // !WITH_TENSORRT

// ---------------------------------------------------------------------------
// Stub implementation when TensorRT is unavailable
// ---------------------------------------------------------------------------

TensorRTEngine::TensorRTEngine() = default;
TensorRTEngine::~TensorRTEngine() = default;

bool TensorRTEngine::Load(const std::string&) { return false; }
void TensorRTEngine::Release() {}
bool TensorRTEngine::IsReady() const { return false; }

int TensorRTEngine::GetInputCount() const { return 0; }
int TensorRTEngine::GetOutputCount() const { return 0; }
std::vector<std::string> TensorRTEngine::GetInputNames() const { return {}; }
std::vector<std::string> TensorRTEngine::GetOutputNames() const { return {}; }
TensorInfo TensorRTEngine::GetInputInfo(const std::string&) const { return {}; }
TensorInfo TensorRTEngine::GetOutputInfo(const std::string&) const { return {}; }

bool TensorRTEngine::SetInputFromHost(const std::string&, const void*, size_t, const Dims&) { return false; }
void* TensorRTEngine::PrepareInputDevice(const std::string&, size_t, const Dims&) { return nullptr; }
void* TensorRTEngine::GetCudaStream() const { return nullptr; }
bool TensorRTEngine::Infer() { return false; }
bool TensorRTEngine::CopyOutputToHost(const std::string&, void*, size_t) { return false; }

#endif // WITH_TENSORRT

} // namespace inference
