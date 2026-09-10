#include "VideoDecoder.hpp"
#include "common/MyMessage.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/Message.hpp>

// FFmpeg C headers
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <chrono>
#include <cstring>

// ---------------------------------------------------------------------------

VideoDecoder::VideoDecoder(const std::string& name) : Module(name) {
    LOG_TRACE("VideoDecoder constructor, name={}", name);
}

VideoDecoder::~VideoDecoder() {
    LOG_TRACE("VideoDecoder destructor, name={}", GetModuleName());
    Cleanup();
}

ns::ErrorCode VideoDecoder::Configure(const ns::Config& config) {
    LOG_TRACE("VideoDecoder::Configure");
    m_param.pixelFormat = config.GetValueOrDefault<std::string>("pixelFormat", "rgb24");
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode VideoDecoder::Init() {
    LOG_TRACE("VideoDecoder::Init");
    // Decoder initialization is deferred until we receive the first PacketMessage
    // with stream info (codec parameters).
    m_frame = av_frame_alloc();
    m_rgbFrame = av_frame_alloc();
    m_avPacket = av_packet_alloc();
    LOG_INFO("VideoDecoder: waiting for stream info from upstream");
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode VideoDecoder::DeInit() {
    LOG_TRACE("VideoDecoder::DeInit");
    Cleanup();
    return ns::ErrorCode::SUCCESS;
}

// ---------------------------------------------------------------------------
// Process — receive PacketMessage, decode to FrameMessage
// ---------------------------------------------------------------------------

void VideoDecoder::Process(nexusflow::Message& inputMessage) {
    if (!inputMessage.HasData()) {
        LOG_WARN("VideoDecoder: received empty message, ignoring");
        return;
    }

    auto* pktMsg = inputMessage.MutPtr<PacketMessage>();
    if (!pktMsg) {
        LOG_WARN("VideoDecoder: message is not PacketMessage, ignoring");
        return;
    }

    // --- Initialize decoder on first packet with stream info ---
    if (pktMsg->hasStreamInfo && !m_decoderReady) {
        if (!InitDecoderFromStreamInfo(pktMsg->streamInfo)) {
            LOG_ERROR("VideoDecoder: failed to initialize decoder");
            return;
        }
    }

    // --- Handle EOF ---
    if (pktMsg->isEnd) {
        FrameMessage endMsg;
        endMsg.isEnd = true;
        endMsg.timestamp = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        LOG_INFO("VideoDecoder: received EOF after {} frames", m_frameIdx);
        Broadcast(nexusflow::Message(std::move(endMsg)));
        return;
    }

    if (!m_decoderReady) {
        LOG_WARN("VideoDecoder: decoder not ready, dropping packet #{}", pktMsg->packetIndex);
        return;
    }

    // --- Feed compressed data to FFmpeg decoder ---
    m_avPacket->data = reinterpret_cast<uint8_t*>(const_cast<char*>(pktMsg->packetData.data()));
    m_avPacket->size = static_cast<int>(pktMsg->packetData.size());
    m_avPacket->pts = pktMsg->pts;
    m_avPacket->dts = pktMsg->dts;

    int ret = avcodec_send_packet(m_codecCtx, m_avPacket);
    if (ret < 0) {
        LOG_WARN("VideoDecoder: avcodec_send_packet failed, av_err={}", ret);
        return;
    }

    // --- Receive decoded frame(s) ---
    while (true) {
        ret = avcodec_receive_frame(m_codecCtx, m_frame);
        if (ret == AVERROR(EAGAIN)) {
            break; // Need more packets
        }
        if (ret < 0) {
            LOG_WARN("VideoDecoder: avcodec_receive_frame failed, av_err={}", ret);
            break;
        }

        // --- Convert to RGB24 ---
        sws_scale(m_swsCtx, m_frame->data, m_frame->linesize, 0, m_height,
                  m_rgbFrame->data, m_rgbFrame->linesize);

        const int rgbLineSize = m_rgbFrame->linesize[0];
        const size_t frameBytes = static_cast<size_t>(m_width) * m_height * 3;

        // Pack into contiguous buffer (linesize may have padding)
        std::string rawPixels(frameBytes, '\0');
        if (rgbLineSize == m_width * 3) {
            std::memcpy(&rawPixels[0], m_rgbFrame->data[0], frameBytes);
        } else {
            for (int row = 0; row < m_height; ++row) {
                std::memcpy(&rawPixels[row * m_width * 3],
                            m_rgbFrame->data[0] + row * rgbLineSize,
                            m_width * 3);
            }
        }

        // --- Compute timestamp ---
        double timestampSec = 0.0;
        if (m_frame->pts != AV_NOPTS_VALUE && m_timeBaseDen > 0) {
            timestampSec = static_cast<double>(m_frame->pts) * m_timeBaseNum / m_timeBaseDen;
        }

        // --- Build FrameMessage ---
        FrameMessage msg;
        msg.videoFrame.frameId = m_frameIdx;
        msg.videoFrame.frameData = std::move(rawPixels);
        msg.videoFrame.width = m_width;
        msg.videoFrame.height = m_height;
        msg.videoFrame.channels = 3;
        msg.isKeyFrame = (m_frame->flags & AV_FRAME_FLAG_KEY) != 0;
        msg.isEnd = false;
        msg.timestamp = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        msg.timestampSec = timestampSec;

        LOG_DEBUG("VideoDecoder: {}", msg.toString());
        Broadcast(nexusflow::Message(std::move(msg)));
        ++m_frameIdx;
    }
}

// ---------------------------------------------------------------------------
// Decoder initialization from stream info
// ---------------------------------------------------------------------------

bool VideoDecoder::InitDecoderFromStreamInfo(const StreamInfo& info) {
    LOG_INFO("VideoDecoder: initializing decoder  codecId={}  {}x{}  fps={:.2f}",
             info.codecId, info.width, info.height, info.fps);

    const AVCodec* codec = avcodec_find_decoder(static_cast<AVCodecID>(info.codecId));
    if (!codec) {
        LOG_ERROR("VideoDecoder: unsupported codec id={}", info.codecId);
        return false;
    }

    m_codecCtx = avcodec_alloc_context3(codec);
    if (!m_codecCtx) {
        LOG_ERROR("VideoDecoder: failed to alloc codec context");
        return false;
    }

    m_codecCtx->width = info.width;
    m_codecCtx->height = info.height;
    m_codecCtx->pix_fmt = static_cast<AVPixelFormat>(info.pixFmt);
    m_codecCtx->time_base = {info.timeBaseNum, info.timeBaseDen};

    // Set extradata (SPS/PPS for H.264, etc.)
    if (!info.extradata.empty()) {
        m_codecCtx->extradata = static_cast<uint8_t*>(
            av_mallocz(info.extradata.size() + AV_INPUT_BUFFER_PADDING_SIZE));
        if (m_codecCtx->extradata) {
            std::memcpy(m_codecCtx->extradata, info.extradata.data(), info.extradata.size());
            m_codecCtx->extradata_size = static_cast<int>(info.extradata.size());
        }
    }

    int ret = avcodec_open2(m_codecCtx, codec, nullptr);
    if (ret < 0) {
        LOG_ERROR("VideoDecoder: failed to open codec, av_err={}", ret);
        return false;
    }

    m_width = info.width;
    m_height = info.height;
    m_fps = info.fps;
    m_timeBaseNum = info.timeBaseNum;
    m_timeBaseDen = info.timeBaseDen;

    // --- Allocate RGB24 buffer ---
    int bufSize = av_image_get_buffer_size(AV_PIX_FMT_RGB24, m_width, m_height, 1);
    m_rgbBuffer = static_cast<uint8_t*>(av_malloc(bufSize));
    av_image_fill_arrays(m_rgbFrame->data, m_rgbFrame->linesize, m_rgbBuffer,
                         AV_PIX_FMT_RGB24, m_width, m_height, 1);

    // --- Create color-conversion context ---
    m_swsCtx = sws_getContext(m_width, m_height, m_codecCtx->pix_fmt,
                              m_width, m_height, AV_PIX_FMT_RGB24,
                              SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!m_swsCtx) {
        LOG_ERROR("VideoDecoder: failed to create SwsContext");
        return false;
    }

    m_decoderReady = true;
    LOG_INFO("VideoDecoder: decoder ready  {}x{}  fps={:.2f}  codec={}",
             m_width, m_height, m_fps, codec->name);
    return true;
}

// ---------------------------------------------------------------------------

void VideoDecoder::Cleanup() {
    if (m_swsCtx) {
        sws_freeContext(m_swsCtx);
        m_swsCtx = nullptr;
    }
    if (m_rgbBuffer) {
        av_freep(&m_rgbBuffer);
    }
    if (m_rgbFrame) {
        av_frame_free(&m_rgbFrame);
    }
    if (m_frame) {
        av_frame_free(&m_frame);
    }
    if (m_avPacket) {
        av_packet_free(&m_avPacket);
    }
    if (m_codecCtx) {
        avcodec_free_context(&m_codecCtx);
    }
    m_width = 0;
    m_height = 0;
    m_frameIdx = 0;
    m_decoderReady = false;
}
