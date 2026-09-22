#pragma once

#include <aifootball/AIFootball.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct AVCodecContext;
struct AVFormatContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace aifootball_app {

class VideoReader {
public:
    using FrameCallback =
        std::function<bool(const aifootball::DecodedFrameView&)>;

    ~VideoReader();

    bool Open(const std::string& path);
    bool Decode(const FrameCallback& callback);

    /// Number of decoded RGB frames kept for reuse (config:
    /// `graph.modules[VideoReader].config.rgbRingSize`).
    ///
    /// A slot is only recycled once the pipeline released its last reference to
    /// the frame, so the ring should be at least as large as the number of
    /// frames that can be in flight (`algorithm.maxPendingFrames` plus the
    /// pipeline depth). A smaller ring still works: the reader then allocates
    /// per frame instead of reusing.
    void SetRgbRingSize(std::size_t slots);

    int width() const { return m_width; }
    int height() const { return m_height; }
    double fps() const { return m_fps; }

private:
    /// An RGB24 frame handed to the pipeline. Downstream stages keep it alive
    /// through DecodedFrameView::dataOwner, so it is a shared buffer.
    using RgbBuffer = std::shared_ptr<uint8_t>;

    /// Used when the configuration does not set rgbRingSize.
    static constexpr std::size_t kDefaultRgbRingSize = 12;

    bool EmitFrame(const FrameCallback& callback);
    /// Reuse the first ring slot that no consumer references any more, or
    /// allocate one. Avoids a per-frame multi-megabyte allocation.
    RgbBuffer AcquireRgbBuffer(std::size_t bytes);
    void ReleaseRgbRing();
    void Cleanup();

    AVFormatContext* m_format = nullptr;
    AVCodecContext* m_codec = nullptr;
    AVFrame* m_frame = nullptr;
    AVPacket* m_packet = nullptr;
    SwsContext* m_sws = nullptr;
    int m_streamIndex = -1;
    int m_width = 0;
    int m_height = 0;
    double m_fps = 25.0;
    int m_timeBaseNum = 0;
    int m_timeBaseDen = 1;
    uint64_t m_frameIndex = 0;
    std::vector<RgbBuffer> m_rgbRing;
    std::size_t m_rgbRingCursor = 0;
    std::size_t m_rgbBytes = 0;
    /// Frames allocated because every ring slot was still in flight.
    std::size_t m_rgbAllocations = 0;
};

} // namespace aifootball_app
