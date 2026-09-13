#pragma once

#include <aifootball/AIFootball.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>

struct AVCodecContext;
struct AVFormatContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace aifootball_demo {

/** Optional offline renderer for frames and results produced by the SDK. */
class VideoRenderer {
public:
    ~VideoRenderer();

    bool Open(const std::string& outputPath, double fps, int width, int height,
              bool enabled);
    bool SubmitFrame(const aifootball::DecodedFrameView& frame);
    bool Write(const aifootball::ProcessResult& result);
    void Close();

private:
    bool InitializeEncoder();
    bool EncodeFrame(std::string& rgb);
    void DrainPackets();
    void FinalizeEncoder();
    void CleanupEncoder();
    static void DrawOverlay(std::string& rgb, int width, int height,
                            const aifootball::ProcessResult& result);

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
    double m_fps = 25.0;
    int m_width = 0;
    int m_height = 0;
    int64_t m_outputFrameIndex = 0;
    std::map<uint64_t, CachedFrame> m_frames;

    AVFormatContext* m_formatCtx = nullptr;
    AVCodecContext* m_codecCtx = nullptr;
    AVFrame* m_yuvFrame = nullptr;
    AVFrame* m_rgbFrame = nullptr;
    AVPacket* m_packet = nullptr;
    SwsContext* m_swsCtx = nullptr;
};

} // namespace aifootball_demo
