#include "VideoReader.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/TimerRegistry.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <memory>

namespace aifootball_app {

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
    const int bytes = av_image_get_buffer_size(
        AV_PIX_FMT_RGB24, m_width, m_height, 1);
    if (bytes <= 0) return false;

    RgbBuffer rgb;
    {
        // Only the conversion is timed here: the callback below submits the
        // frame to the pipeline and can block on queue backpressure.
        auto timer = nexusflow::TimerRegistry::Instance().ScopeAverageMs(
            "Video.ColorConvert", 1, 5000);
        m_sws = sws_getCachedContext(
            m_sws, m_width, m_height, static_cast<AVPixelFormat>(m_frame->format),
            m_width, m_height, AV_PIX_FMT_RGB24, SWS_BILINEAR,
            nullptr, nullptr, nullptr);
        if (!m_sws) return false;
        rgb = AcquireRgbBuffer(static_cast<std::size_t>(bytes));
        if (!rgb) return false;
        uint8_t* dstData[4] = {nullptr, nullptr, nullptr, nullptr};
        int dstLinesize[4] = {0, 0, 0, 0};
        av_image_fill_arrays(dstData, dstLinesize, rgb.get(),
                             AV_PIX_FMT_RGB24, m_width, m_height, 1);
        sws_scale(m_sws, m_frame->data, m_frame->linesize, 0, m_height,
                  dstData, dstLinesize);
    }

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
    view.data = rgb.get();
    view.dataBytes = static_cast<std::size_t>(bytes);
    view.dataOwner = rgb;
    return callback(view);
}

VideoReader::RgbBuffer VideoReader::AcquireRgbBuffer(std::size_t bytes) {
    if (bytes == 0) return nullptr;

    // A different frame size invalidates every recycled slot.
    if (m_rgbBytes != bytes) {
        ReleaseRgbRing();
        m_rgbBytes = bytes;
    }
    if (m_rgbRing.empty()) m_rgbRing.resize(kDefaultRgbRingSize);

    for (std::size_t i = 0; i < m_rgbRing.size(); ++i) {
        const std::size_t index = (m_rgbRingCursor + i) % m_rgbRing.size();
        const RgbBuffer& candidate = m_rgbRing[index];
        // use_count() == 1 means no consumer holds this frame any more. Only
        // this thread touches the ring, so the check cannot race.
        if (candidate && candidate.use_count() == 1) {
            m_rgbRingCursor = (index + 1) % m_rgbRing.size();
            return candidate;
        }
    }

    // The pipeline holds more frames in flight than the ring has slots. The
    // outgoing frames keep their own reference, so the slot can be replaced.
    RgbBuffer buffer(new uint8_t[bytes], [](uint8_t* ptr) { delete[] ptr; });
    m_rgbRing[m_rgbRingCursor] = buffer;
    m_rgbRingCursor = (m_rgbRingCursor + 1) % m_rgbRing.size();
    ++m_rgbAllocations;
    return buffer;
}

void VideoReader::ReleaseRgbRing() {
    for (RgbBuffer& buffer : m_rgbRing) buffer.reset();
    m_rgbRingCursor = 0;
}

void VideoReader::SetRgbRingSize(std::size_t slots) {
    if (slots == 0) {
        LOG_WARN("VideoReader: rgbRingSize=0 is invalid, keeping {} slots",
                 kDefaultRgbRingSize);
        slots = kDefaultRgbRingSize;
    }
    if (slots == m_rgbRing.size()) return;
    ReleaseRgbRing();
    m_rgbRing.assign(slots, nullptr);
    LOG_INFO("VideoReader: RGB ring size set to {} frames", slots);
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
    if (m_rgbAllocations > 0) {
        // Every extra allocation is a frame that could not be served by the
        // ring: the pipeline held more frames in flight than there were slots,
        // so the reader allocated instead of reusing. Raise rgbRingSize (or
        // lower algorithm.maxPendingFrames) to remove the churn.
        LOG_WARN("VideoReader: rgbRingSize={} was too small; allocated {} extra "
                 "frames ({:.1f} MB of malloc/free churn)",
                 m_rgbRing.size(), m_rgbAllocations,
                 m_rgbAllocations * static_cast<double>(m_rgbBytes) / (1024.0 * 1024.0));
        m_rgbAllocations = 0;
    }
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
    ReleaseRgbRing();
    m_rgbBytes = 0;
}

} // namespace aifootball_app
