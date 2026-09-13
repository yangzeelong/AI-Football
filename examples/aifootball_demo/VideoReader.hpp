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

namespace aifootball_demo {

class VideoReader {
public:
    using FrameCallback =
        std::function<bool(const aifootball::DecodedFrameView&)>;

    ~VideoReader();

    bool Open(const std::string& path);
    bool Decode(const FrameCallback& callback);

    int width() const { return m_width; }
    int height() const { return m_height; }
    double fps() const { return m_fps; }

private:
    bool EmitFrame(const FrameCallback& callback);
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
    std::shared_ptr<std::vector<uint8_t>> m_rgb;
};

} // namespace aifootball_demo
