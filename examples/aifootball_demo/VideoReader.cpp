#include "VideoReader.hpp"

#include <nexusflow/Logging.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <memory>
#include <vector>

namespace aifootball_demo {

VideoReader::~VideoReader() { Cleanup(); }

bool VideoReader::Open(const std::string& path) {
    if (avformat_open_input(&m_format, path.c_str(), nullptr, nullptr) < 0 ||
        avformat_find_stream_info(m_format, nullptr) < 0) {
        Cleanup();
        return false;
    }
    for (unsigned i = 0; i < m_format->nb_streams; ++i) {
        if (m_format->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            m_streamIndex = static_cast<int>(i);
            break;
        }
    }
    if (m_streamIndex < 0) return false;
    AVStream* stream = m_format->streams[m_streamIndex];
    const AVCodec* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!decoder) return false;
    m_codec = avcodec_alloc_context3(decoder);
    if (!m_codec || avcodec_parameters_to_context(m_codec, stream->codecpar) < 0 ||
        avcodec_open2(m_codec, decoder, nullptr) < 0) {
        Cleanup();
        return false;
    }
    m_frame = av_frame_alloc();
    m_packet = av_packet_alloc();
    if (!m_frame || !m_packet) { Cleanup(); return false; }
    m_width = m_codec->width;
    m_height = m_codec->height;
    m_timeBaseNum = stream->time_base.num;
    m_timeBaseDen = stream->time_base.den;
    if (stream->avg_frame_rate.den > 0 && stream->avg_frame_rate.num > 0) {
        m_fps = av_q2d(stream->avg_frame_rate);
    }
    return true;
}

bool VideoReader::EmitFrame(const FrameCallback& callback) {
    m_sws = sws_getCachedContext(
        m_sws, m_width, m_height, static_cast<AVPixelFormat>(m_frame->format),
        m_width, m_height, AV_PIX_FMT_RGB24, SWS_BILINEAR,
        nullptr, nullptr, nullptr);
    if (!m_sws) return false;
    const int bytes = av_image_get_buffer_size(
        AV_PIX_FMT_RGB24, m_width, m_height, 1);
    m_rgb = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(bytes));
    uint8_t* dstData[4] = {nullptr, nullptr, nullptr, nullptr};
    int dstLinesize[4] = {0, 0, 0, 0};
    av_image_fill_arrays(dstData, dstLinesize, m_rgb->data(),
                         AV_PIX_FMT_RGB24, m_width, m_height, 1);
    sws_scale(m_sws, m_frame->data, m_frame->linesize, 0, m_height,
              dstData, dstLinesize);

    double timestampSec = static_cast<double>(m_frameIndex) / m_fps;
    if (m_frame->best_effort_timestamp != AV_NOPTS_VALUE && m_timeBaseDen > 0) {
        timestampSec = static_cast<double>(m_frame->best_effort_timestamp) *
                       m_timeBaseNum / m_timeBaseDen;
    }
    aifootball::DecodedFrameView view;
    view.frameId = m_frameIndex++;
    view.timestampSec = timestampSec;
    view.width = m_width;
    view.height = m_height;
    view.strideBytes = m_width * 3;
    view.data = m_rgb->data();
    view.dataBytes = m_rgb->size();
    view.dataOwner = m_rgb;
    return callback(view);
}

bool VideoReader::Decode(const FrameCallback& callback) {
    if (!m_format || !m_codec || !callback) return false;
    bool keepRunning = true;
    while (keepRunning && av_read_frame(m_format, m_packet) >= 0) {
        if (m_packet->stream_index == m_streamIndex) {
            if (avcodec_send_packet(m_codec, m_packet) < 0) {
                av_packet_unref(m_packet);
                return false;
            }
            while (keepRunning && avcodec_receive_frame(m_codec, m_frame) == 0) {
                keepRunning = EmitFrame(callback);
            }
        }
        av_packet_unref(m_packet);
    }
    if (!keepRunning) return true;
    if (avcodec_send_packet(m_codec, nullptr) < 0) return false;
    while (avcodec_receive_frame(m_codec, m_frame) == 0) {
        if (!EmitFrame(callback)) break;
    }
    return true;
}

void VideoReader::Cleanup() {
    if (m_sws) sws_freeContext(m_sws);
    if (m_packet) av_packet_free(&m_packet);
    if (m_frame) av_frame_free(&m_frame);
    if (m_codec) avcodec_free_context(&m_codec);
    if (m_format) avformat_close_input(&m_format);
    m_sws = nullptr;
    m_streamIndex = -1;
    m_width = 0;
    m_height = 0;
    m_frameIndex = 0;
    m_rgb.reset();
}

} // namespace aifootball_demo
