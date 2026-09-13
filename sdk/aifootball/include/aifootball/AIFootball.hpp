#pragma once

#include <nexusflow/ErrorCode.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace aifootball {

/** A point in the source-frame coordinate system used by an ROI polygon. */
struct RoiPoint {
    float x = 0.0f;
    float y = 0.0f;
};

/** Runtime ROI configuration parsed by the embedding application. */
struct RoiConfig {
    bool enabled = false;
    int width = 0;
    int height = 0;
    std::vector<RoiPoint> points;
};

/** Algorithm-level configuration passed into the SDK runtime. */
struct AlgoConfig {
    RoiConfig roi;
};

/** Pixel layout accepted by the SDK input API. */
enum class PixelFormat {
    RGB24 = 0,
};

/**
 * Non-owning view of one decoded video frame.
 *
 * Process is asynchronous. When dataOwner is set, the SDK retains it until
 * the asynchronous result has been delivered, enabling zero-copy integration
 * with caller-owned frame buffers. When dataOwner is empty, the SDK copies the
 * bytes before Process returns.
 */
struct DecodedFrameView {
    uint64_t frameId = 0;
    double timestampSec = 0.0;
    int width = 0;
    int height = 0;
    int strideBytes = 0;
    const uint8_t* data = nullptr;
    std::size_t dataBytes = 0;
    std::shared_ptr<const void> dataOwner;
    PixelFormat pixelFormat = PixelFormat::RGB24;
};

enum class KeypointState {
    Missing = 0,
    Observed = 1,
    Virtual = 2,
};

struct KeypointResult {
    float x = 0.0f;
    float y = 0.0f;
    float confidence = 0.0f;
    KeypointState state = KeypointState::Missing;
};

struct PersonResult {
    int trackId = -1;
    std::array<float, 4> bbox{{0.0f, 0.0f, 0.0f, 0.0f}};
    float confidence = 0.0f;
    std::array<KeypointResult, 26> keypoints{};
};

enum class BallTrackState {
    Lost = 0,
    Predicted = 1,
    Observed = 2,
};

struct BallResult {
    int trackId = -1;
    std::array<float, 4> bbox{{0.0f, 0.0f, 0.0f, 0.0f}};
    std::array<float, 2> center{{0.0f, 0.0f}};
    float confidence = 0.0f;
    BallTrackState state = BallTrackState::Lost;
};

struct DetectionCounts {
    int total = 0;
    int person = 0;
    int ball = 0;
};

/** Algorithm output corresponding to one input frame. */
struct ProcessResult {
    uint64_t frameId = 0;
    double timestampSec = 0.0;
    std::vector<PersonResult> persons;
    std::vector<BallResult> balls;
    DetectionCounts rawCounts;
    std::map<std::string, double> moduleTimingsMs;
};

/** Runtime options for the algorithm-only SDK pipeline. */
struct RuntimeOptions {
    /// YAML graph configuration containing model and algorithm settings.
    std::string configPath;
    /// CUDA device selected before model initialization.
    int deviceId = 0;
    /// Maximum number of decoded frames buffered before Process applies
    /// backpressure and returns FAILURE.
    std::size_t maxPendingFrames = 64;
};

/**
 * Stable algorithm facade for embedding AI-Football in another program.
 *
 * The caller owns demuxing, decoding, rendering, and result persistence. The
 * runtime owns only the algorithm modules and their asynchronous pipeline.
 */
class Runtime {
public:
    using ResultCallback = std::function<void(ProcessResult)>;

    static std::unique_ptr<Runtime> Create(
        const RuntimeOptions& options,
        const AlgoConfig& algoConfig = AlgoConfig());

    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    nexusflow::ErrorCode Init();
    /// Enqueue one decoded frame and return without waiting for inference.
    nexusflow::ErrorCode Process(const DecodedFrameView& frame);
    /// Drain all frames submitted before this call, including a partial model
    /// batch. This is a synchronization point, not an inference result API.
    nexusflow::ErrorCode Flush();
    /// Retrieve one asynchronous result. timeoutMs=0 performs a non-blocking
    /// poll; a positive value waits up to that many milliseconds.
    nexusflow::ErrorCode PollResult(ProcessResult& result,
                                    uint32_t timeoutMs = 0);
    /// Select callback delivery instead of queue polling for future results.
    nexusflow::ErrorCode SetResultCallback(ResultCallback callback);
    nexusflow::ErrorCode DeInit();

private:
    explicit Runtime(RuntimeOptions options, AlgoConfig algoConfig);

    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace aifootball
