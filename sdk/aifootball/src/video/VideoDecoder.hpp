#pragma once

#include "nexusflow/ErrorCode.hpp"
#include <nexusflow/Nexusflow.hpp>

// Forward-declare FFmpeg types
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace ns = nexusflow;

/**
 * @brief VideoDecoder — decode compressed packets to raw RGB frames.
 *
 * Receives PacketMessage from VideoReader/StreamPuller,
 * decodes via FFmpeg (software), converts to RGB24,
 * and broadcasts FrameMessage downstream.
 */
class VideoDecoder : public ns::Module {
public:
    VideoDecoder(const std::string& name);

    ~VideoDecoder() override;

    // --- Lifecycle ---
    ns::ErrorCode Configure(const ns::Config& config) override;

    ns::ErrorCode Init() override;

    ns::ErrorCode DeInit() override;

protected:
    void Process(ns::Message& inputMessage) override;

private:
    bool InitDecoderFromStreamInfo(const struct StreamInfo& info);
    void Cleanup();

    struct {
        std::string pixelFormat = "rgb24";
    } m_param;

    // FFmpeg decode state
    AVCodecContext* m_codecCtx = nullptr;
    AVFrame* m_frame = nullptr;
    AVFrame* m_rgbFrame = nullptr;
    AVPacket* m_avPacket = nullptr;
    SwsContext* m_swsCtx = nullptr;
    uint8_t* m_rgbBuffer = nullptr;

    int m_width = 0;
    int m_height = 0;
    double m_fps = 25.0;
    int m_timeBaseNum = 0;
    int m_timeBaseDen = 1;
    uint32_t m_frameIdx = 0;
    bool m_decoderReady = false;
};

NEXUSFLOW_REGISTER_MODULE(VideoDecoder);
