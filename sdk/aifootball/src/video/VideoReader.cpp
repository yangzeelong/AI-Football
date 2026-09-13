#include "VideoReader.hpp"
#include "common/MyMessage.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/Message.hpp>

// FFmpeg C headers (demux only — no codec/swscale needed)
extern "C" {
#include <libavformat/avformat.h>
}

#include <chrono>
#include <cstring>
#include <thread>

// ---------------------------------------------------------------------------

VideoReader::VideoReader(const std::string& name) : Module(name) {
    LOG_TRACE("VideoReader constructor, name={}", name);
}

VideoReader::~VideoReader() {
    LOG_TRACE("VideoReader destructor, name={}", GetModuleName());
    Cleanup();
}

ns::ErrorCode VideoReader::Configure(const ns::Config& config) {
    LOG_TRACE("VideoReader::Configure");
    m_param.videoPath = config.GetValueOrDefault<std::string>("videoPath", "");
    m_param.stride = config.GetValueOrDefault<int>("stride", 1);
    if (m_param.stride < 1) {
        m_param.stride = 1;
    }
    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode VideoReader::Init() {
    LOG_TRACE("VideoReader::Init");

    if (m_param.videoPath.empty()) {
        LOG_ERROR("VideoReader: videoPath is empty");
        return ns::ErrorCode::FAILURE;
    }

    // --- Open input (demux only, no codec opening) ---
    int ret = avformat_open_input(&m_formatCtx, m_param.videoPath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        LOG_ERROR("VideoReader: failed to open '{}', av_err={}", m_param.videoPath, ret);
        return ns::ErrorCode::FAILURE;
    }

    ret = avformat_find_stream_info(m_formatCtx, nullptr);
    if (ret < 0) {
        LOG_ERROR("VideoReader: failed to find stream info, av_err={}", ret);
        Cleanup();
        return ns::ErrorCode::FAILURE;
    }

    // --- Find the first video stream ---
    m_videoStreamIdx = -1;
    for (unsigned i = 0; i < m_formatCtx->nb_streams; ++i) {
        if (m_formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            m_videoStreamIdx = static_cast<int>(i);
            break;
        }
    }
    if (m_videoStreamIdx < 0) {
        LOG_ERROR("VideoReader: no video stream found in '{}'", m_param.videoPath);
        Cleanup();
        return ns::ErrorCode::FAILURE;
    }

    // --- Compute FPS ---
    AVStream* videoStream = m_formatCtx->streams[m_videoStreamIdx];
    AVRational frameRate = videoStream->avg_frame_rate;
    if (frameRate.den > 0 && frameRate.num > 0) {
        m_fps = av_q2d(frameRate);
    } else {
        AVRational timeBase = videoStream->time_base;
        m_fps = (timeBase.num > 0 && timeBase.den > 0) ? (1.0 / av_q2d(timeBase)) : 25.0;
    }

    m_packet = av_packet_alloc();

    LOG_INFO("VideoReader: opened '{}'  {}x{}  fps={:.2f}  stride={}",
             m_param.videoPath,
             videoStream->codecpar->width,
             videoStream->codecpar->height,
             m_fps,
             m_param.stride);

    return ns::ErrorCode::SUCCESS;
}

ns::ErrorCode VideoReader::DeInit() {
    LOG_TRACE("VideoReader::DeInit");
    Cleanup();
    return ns::ErrorCode::SUCCESS;
}

// ---------------------------------------------------------------------------
// Process — read one compressed packet per call
// ---------------------------------------------------------------------------

void VideoReader::Process(nexusflow::Message& inputMessage) {
    if (inputMessage.HasData()) {
        LOG_WARN("VideoReader: unexpected input message, ignoring");
        return;
    }

    // EOF: send end marker once
    if (m_endOfFile) {
        if (m_eofSent) {
            // Source workers continue polling until Pipeline::Stop(). Avoid a
            // hot loop after the single end marker has been delivered.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return;
        }
        PacketMessage endMsg;
        endMsg.isEnd = true;
        endMsg.packetIndex = m_sentIdx;
        Broadcast(nexusflow::Message(std::move(endMsg)));
        LOG_INFO("VideoReader: EOF, sent end marker after {} packets", m_sentIdx);
        m_eofSent = true;
        return;
    }

    // Send stream info with the first packet
    if (!m_streamInfoSent) {
        SendStreamInfo();
    }

    // --- Read packets, skip non-video streams ---
    while (true) {
        int ret = av_read_frame(m_formatCtx, m_packet);
        if (ret < 0) {
            m_endOfFile = true;
            LOG_INFO("VideoReader: reached end of file after {} packets (sent {})", m_packetIdx, m_sentIdx);
            return;
        }

        if (m_packet->stream_index != m_videoStreamIdx) {
            av_packet_unref(m_packet);
            continue;
        }

        // Apply stride: skip packets that don't match
        uint32_t currentIdx = m_packetIdx++;
        if (m_param.stride > 1 && (currentIdx % m_param.stride) != 0) {
            av_packet_unref(m_packet);
            continue;
        }

        // --- Build PacketMessage ---
        PacketMessage msg;
        msg.packetData.assign(reinterpret_cast<const char*>(m_packet->data), m_packet->size);
        msg.pts = m_packet->pts;
        msg.dts = m_packet->dts;
        msg.isKeyFrame = (m_packet->flags & AV_PKT_FLAG_KEY) != 0;
        msg.packetIndex = m_sentIdx++;

        // Attach stream info on first packet
        if (!m_streamInfoSent) {
            msg.hasStreamInfo = true;
            // streamInfo is already filled by SendStreamInfo()
            AVStream* vs = m_formatCtx->streams[m_videoStreamIdx];
            msg.streamInfo.codecId = static_cast<int>(vs->codecpar->codec_id);
            msg.streamInfo.width = vs->codecpar->width;
            msg.streamInfo.height = vs->codecpar->height;
            msg.streamInfo.pixFmt = static_cast<int>(vs->codecpar->format);
            msg.streamInfo.fps = m_fps;
            msg.streamInfo.timeBaseNum = vs->time_base.num;
            msg.streamInfo.timeBaseDen = vs->time_base.den;
            if (vs->codecpar->extradata && vs->codecpar->extradata_size > 0) {
                msg.streamInfo.extradata.assign(
                    reinterpret_cast<const char*>(vs->codecpar->extradata),
                    vs->codecpar->extradata_size);
            }
            m_streamInfoSent = true;
        }

        av_packet_unref(m_packet);

        LOG_DEBUG("VideoReader: {}", msg.toString());
        Broadcast(nexusflow::Message(std::move(msg)));
        return;
    }
}

// ---------------------------------------------------------------------------

void VideoReader::SendStreamInfo() {
    // Stream info is attached to the first PacketMessage in Process().
    // This method exists as a hook for future use (e.g., sending a separate init message).
    LOG_TRACE("VideoReader: will attach stream info to first packet");
}

void VideoReader::Cleanup() {
    if (m_packet) {
        av_packet_free(&m_packet);
    }
    if (m_formatCtx) {
        avformat_close_input(&m_formatCtx);
    }
    m_videoStreamIdx = -1;
    m_packetIdx = 0;
    m_sentIdx = 0;
    m_endOfFile = false;
    m_eofSent = false;
    m_streamInfoSent = false;
}
