#pragma once

#include "nexusflow/ErrorCode.hpp"
#include <nexusflow/Nexusflow.hpp>

// Forward-declare FFmpeg types to avoid leaking C headers.
struct AVFormatContext;
struct AVPacket;

namespace ns = nexusflow;

/**
 * @brief VideoReader — pure demux module.
 *
 * Opens a video file, reads compressed packets (no decoding),
 * and broadcasts PacketMessage to downstream (VideoDecoder).
 */
class VideoReader : public ns::Module {
public:
    VideoReader(const std::string& name);

    ~VideoReader() override;

    // --- Lifecycle ---
    ns::ErrorCode Configure(const ns::Config& config) override;

    ns::ErrorCode Init() override;

    ns::ErrorCode DeInit() override;

protected:
    void Process(ns::Message& inputMessage) override;

private:
    void Cleanup();
    void SendStreamInfo();

    struct {
        std::string videoPath;
        int stride = 1; // Send every N-th packet (1 = all)
    } m_param;

    // FFmpeg demux state
    AVFormatContext* m_formatCtx = nullptr;
    AVPacket* m_packet = nullptr;
    int m_videoStreamIdx = -1;
    double m_fps = 25.0;
    uint32_t m_packetIdx = 0;
    uint32_t m_sentIdx = 0;
    bool m_endOfFile = false;
    bool m_eofSent = false;
    bool m_streamInfoSent = false;
};

NEXUSFLOW_REGISTER_MODULE(VideoReader);
