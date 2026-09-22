#pragma once

#include <aifootball/AIFootball.hpp>

#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>

struct AVCodecContext;
struct AVFormatContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace aifootball_app {

/** Optional offline renderer for frames and results produced by the SDK. */
class VideoRenderer {
public:
    ~VideoRenderer();

    bool Open(const std::string& outputPath, double fps, int width, int height,
              bool enabled);
    bool SubmitFrame(const aifootball::DecodedFrameView& frame);
    bool Write(const aifootball::ProcessResult& result);
    void Close();

    /// Outline the algorithm ROI on every frame. Points are in ROI context
    /// coordinates (roi.width x roi.height) and are mapped onto the frame, the
    /// same way the tracker maps detections into ROI space.
    void SetRoi(const aifootball::RoiContext& roi);

    /// Mean wall time per rendered frame spent in this renderer: the frame
    /// copy, the overlay and the encoder. Zero when rendering is disabled or
    /// nothing was written yet. The app subtracts it from its own per-frame
    /// figure so the pipeline rate can be read on its own.
    double RenderMsPerFrame() const;
    uint64_t RenderedFrames() const { return m_renderFrames; }

private:
    bool InitializeEncoder();
    bool EncodeFrame(std::string& rgb);
    void DrainPackets();
    void FinalizeEncoder();
    void CleanupEncoder();
    void DrawOverlay(std::string& rgb,
                     const aifootball::ProcessResult& result) const;

    struct CachedFrame {
        const uint8_t* data = nullptr;
        std::size_t bytes = 0;
        std::shared_ptr<const void> owner;
        std::string ownedCopy;
    };

    bool m_enabled = false;
    bool m_opened = false;
    bool m_headerWritten = false;
    bool m_encoderFailed = false;
    std::string m_outputPath;
    aifootball::RoiContext m_roi;
    double m_fps = 25.0;
    int m_width = 0;
    int m_height = 0;
    int64_t m_outputFrameIndex = 0;
    std::map<uint64_t, CachedFrame> m_frames;
    /// Pipeline throughput over a trailing time window. The overlay shows this
    /// rather than the output frame rate, which is a constant taken from the
    /// config and says nothing about the run. A time window rather than a
    /// frame count or the last interval: results arrive in bursts whenever the
    /// consumer drains a backlog, and a burst short enough would report the
    /// renderer's own speed instead of the pipeline's.
    std::deque<std::chrono::steady_clock::time_point> m_writeTimes;
    double m_processingFps = 0.0;
    /// Accumulated time in the overlay plus encoder, so the app can report the
    /// pipeline rate without them.
    double m_renderMs = 0.0;
    uint64_t m_renderFrames = 0;

    AVFormatContext* m_formatCtx = nullptr;
    AVCodecContext* m_codecCtx = nullptr;
    AVFrame* m_yuvFrame = nullptr;
    AVFrame* m_rgbFrame = nullptr;
    AVPacket* m_packet = nullptr;
    SwsContext* m_swsCtx = nullptr;
};

} // namespace aifootball_app
