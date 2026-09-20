#pragma once

#include <aifootball/Future.hpp>
#include <nexusflow/ErrorCode.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
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

/** ROI context parsed by the embedding application. */
struct RoiContext {
    bool enabled = false;
    int width = 0;
    int height = 0;
    std::vector<RoiPoint> points;
};

/** Pixel layout accepted by the SDK input API. */
enum class PixelFormat {
    RGB24 = 0,
};

/** Policy applied when the asynchronous SDK input queue reaches capacity. */
enum class QueuePolicy {
    /// Wait until the pipeline worker consumes at least one queued frame.
    Block = 0,
    /// Drop the oldest queued frame and enqueue the current frame.
    DropOldest = 1,
    /// Drop the current frame; ProcessAsync returns a failure result future.
    DropNew = 2,
};

/**
 * Non-owning view of one decoded video frame.
 *
 * ProcessAsync is asynchronous. When dataOwner is set, the SDK retains it until
 * the asynchronous result has been delivered, enabling zero-copy integration
 * with caller-owned frame buffers. When dataOwner is empty, the SDK copies the
 * bytes before ProcessAsync returns.
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

/** Future value returned for one submitted frame. */
struct ProcessFutureResult {
    nexusflow::ErrorCode status = nexusflow::FAILURE;
    ProcessResult result;
};

using ProcessFuture = Future<ProcessFutureResult>;

/** Configuration and execution context for one algorithm pipeline. */
struct AIFootballContext {
    /// YAML graph configuration containing model and algorithm settings.
    std::string configPath;
    /// CUDA device selected before model initialization.
    int deviceId = 0;
    /// Maximum number of decoded frames buffered before ProcessAsync applies
    /// inputQueuePolicy. 0 means unbounded.
    std::size_t maxPendingFrames = 0;
    /// Behavior when maxPendingFrames has been reached.
    QueuePolicy inputQueuePolicy = QueuePolicy::Block;
    /// Algorithm ROI in source-frame coordinates.
    RoiContext roi;
};

/**
 * Stable algorithm facade for embedding AI-Football in another program.
 *
 * The caller owns demuxing, decoding, rendering, and result persistence. The
 * runtime owns only the algorithm modules and their asynchronous pipeline.
 */
class AIFootballPipeline {
public:
    static std::unique_ptr<AIFootballPipeline> Create(
        const AIFootballContext& context);

    ~AIFootballPipeline();

    AIFootballPipeline(const AIFootballPipeline&) = delete;
    AIFootballPipeline& operator=(const AIFootballPipeline&) = delete;

    nexusflow::ErrorCode Init();
    /// Enqueue one decoded frame without waiting for inference. May block when
    /// the bounded input queue is full and inputQueuePolicy is Block. The
    /// returned future becomes ready when this frame's result is available, or
    /// with a failure status when the frame cannot be submitted or is dropped.
    ProcessFuture ProcessAsync(const DecodedFrameView& frame);
    /// Drain all frames submitted before this call, including a partial model
    /// batch. Results are delivered through their ProcessAsync futures.
    nexusflow::ErrorCode Drain();
    nexusflow::ErrorCode DeInit();

private:
    explicit AIFootballPipeline(AIFootballContext context);

    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace aifootball
