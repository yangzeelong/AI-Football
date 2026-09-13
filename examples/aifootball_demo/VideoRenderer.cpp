#include "VideoRenderer.hpp"

#include "DemoConfig.hpp"

#include <nexusflow/Logging.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace aifootball_demo {
namespace {

std::string AvError(int error) {
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(error, buffer, sizeof(buffer));
    return buffer;
}

void SetPixel(std::string& image, int width, int height, int x, int y,
              uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    const std::size_t offset = (static_cast<std::size_t>(y) * width + x) * 3;
    image[offset] = static_cast<char>(r);
    image[offset + 1] = static_cast<char>(g);
    image[offset + 2] = static_cast<char>(b);
}

void DrawPoint(std::string& image, int width, int height, int x, int y,
               int radius, uint8_t r, uint8_t g, uint8_t b) {
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy <= radius * radius) {
                SetPixel(image, width, height, x + dx, y + dy, r, g, b);
            }
        }
    }
}

void DrawLine(std::string& image, int width, int height, int x0, int y0,
              int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;
    while (true) {
        DrawPoint(image, width, height, x0, y0, 1, r, g, b);
        if (x0 == x1 && y0 == y1) break;
        const int twice = 2 * error;
        if (twice >= dy) { error += dy; x0 += sx; }
        if (twice <= dx) { error += dx; y0 += sy; }
    }
}

void DrawRect(std::string& image, int width, int height, float x0, float y0,
              float x1, float y1, uint8_t r, uint8_t g, uint8_t b) {
    const int left = static_cast<int>(std::lround(x0));
    const int top = static_cast<int>(std::lround(y0));
    const int right = static_cast<int>(std::lround(x1));
    const int bottom = static_cast<int>(std::lround(y1));
    DrawLine(image, width, height, left, top, right, top, r, g, b);
    DrawLine(image, width, height, right, top, right, bottom, r, g, b);
    DrawLine(image, width, height, right, bottom, left, bottom, r, g, b);
    DrawLine(image, width, height, left, bottom, left, top, r, g, b);
}

void DrawPose(std::string& image, int width, int height,
              const aifootball::PersonResult& person) {
    static const int kEdges[][2] = {
        {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8},
        {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14},
        {14, 16}, {15, 17}, {15, 18}, {15, 19}, {16, 20}, {16, 21}, {16, 22},
    };
    for (const auto& edge : kEdges) {
        const auto& a = person.keypoints[edge[0]];
        const auto& b = person.keypoints[edge[1]];
        if (a.state == aifootball::KeypointState::Missing ||
            b.state == aifootball::KeypointState::Missing) continue;
        DrawLine(image, width, height, static_cast<int>(std::lround(a.x)),
                 static_cast<int>(std::lround(a.y)),
                 static_cast<int>(std::lround(b.x)),
                 static_cast<int>(std::lround(b.y)), 40, 200, 255);
    }
    for (const auto& point : person.keypoints) {
        if (point.state == aifootball::KeypointState::Missing) continue;
        DrawPoint(image, width, height, static_cast<int>(std::lround(point.x)),
                  static_cast<int>(std::lround(point.y)), 3, 255, 220, 40);
    }
}

} // namespace

VideoRenderer::~VideoRenderer() { Close(); }

bool VideoRenderer::Open(const std::string& outputPath, double fps, int width,
                         int height, bool enabled) {
    Close();
    m_enabled = enabled;
    m_opened = true;
    m_outputPath = outputPath;
    m_fps = fps > 0.0 ? fps : 25.0;
    m_width = width;
    m_height = height;
    if (!m_enabled) return true;
    if (width <= 0 || height <= 0 || (width & 1) != 0 || (height & 1) != 0) {
        LOG_ERROR("VideoRenderer: invalid dimensions {}x{}", width, height);
        return false;
    }
    const std::size_t slash = outputPath.find_last_of('/');
    if (slash != std::string::npos &&
        !EnsureDirectory(outputPath.substr(0, slash))) {
        LOG_ERROR("VideoRenderer: failed to create output directory for '{}'",
                  outputPath);
        return false;
    }
    if (!InitializeEncoder()) {
        CleanupEncoder();
        return false;
    }
    LOG_INFO("VideoRenderer: enabled output='{}' size={}x{} fps={:.3f}",
             outputPath, width, height, m_fps);
    return true;
}

bool VideoRenderer::SubmitFrame(const aifootball::DecodedFrameView& frame) {
    if (!m_enabled) return true;
    const std::size_t expected = static_cast<std::size_t>(frame.width) *
                                 frame.height * 3;
    if (!m_opened || frame.data == nullptr || frame.width != m_width ||
        frame.height != m_height || frame.strideBytes != frame.width * 3 ||
        frame.dataBytes < expected) {
        LOG_ERROR("VideoRenderer: invalid frame {}", frame.frameId);
        return false;
    }
    CachedFrame cached;
    cached.data = frame.data;
    cached.bytes = expected;
    cached.owner = frame.dataOwner;
    if (!cached.owner) {
        cached.ownedCopy.assign(reinterpret_cast<const char*>(frame.data), expected);
    }
    m_frames[frame.frameId] = std::move(cached);
    return true;
}

bool VideoRenderer::Write(const aifootball::ProcessResult& result) {
    if (!m_enabled) return true;
    auto it = m_frames.find(result.frameId);
    if (it == m_frames.end()) {
        LOG_ERROR("VideoRenderer: no cached RGB frame for result {}", result.frameId);
        return false;
    }
    CachedFrame cached = std::move(it->second);
    m_frames.erase(it);
    if (m_encoderFailed) return false;
    std::string rendered = cached.owner
        ? std::string(reinterpret_cast<const char*>(cached.data), cached.bytes)
        : std::move(cached.ownedCopy);
    DrawOverlay(rendered, m_width, m_height, result);
    if (!EncodeFrame(rendered)) {
        m_encoderFailed = true;
        LOG_ERROR("VideoRenderer: rendering disabled after encoding failure");
        return false;
    }
    return true;
}

void VideoRenderer::Close() {
    if (!m_opened) return;
    if (m_enabled) FinalizeEncoder();
    CleanupEncoder();
    m_frames.clear();
    m_opened = false;
}

bool VideoRenderer::InitializeEncoder() {
    int ret = avformat_alloc_output_context2(&m_formatCtx, nullptr, nullptr,
                                              m_outputPath.c_str());
    if (ret < 0 || !m_formatCtx) {
        LOG_ERROR("VideoRenderer: failed to allocate output context: {}", AvError(ret));
        return false;
    }
    const AVCodec* codec = avcodec_find_encoder_by_name("libx264");
    if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
    if (!codec) {
        LOG_ERROR("VideoRenderer: no H.264 or MPEG-4 encoder is available");
        return false;
    }
    AVStream* stream = avformat_new_stream(m_formatCtx, nullptr);
    if (!stream) return false;
    m_codecCtx = avcodec_alloc_context3(codec);
    if (!m_codecCtx) return false;
    const AVRational rate = av_d2q(m_fps, 100000);
    m_codecCtx->codec_id = codec->id;
    m_codecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
    m_codecCtx->width = m_width;
    m_codecCtx->height = m_height;
    m_codecCtx->pix_fmt = AV_PIX_FMT_YUV420P;
    m_codecCtx->time_base = AVRational{rate.den, rate.num};
    m_codecCtx->framerate = rate;
    m_codecCtx->gop_size = std::max(1, static_cast<int>(std::lround(m_fps)));
    m_codecCtx->max_b_frames = 0;
    m_codecCtx->bit_rate = static_cast<int64_t>(m_width) * m_height * 4;
    if (m_formatCtx->oformat->flags & AVFMT_GLOBALHEADER) {
        m_codecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    if (std::strcmp(codec->name, "libx264") == 0) {
        av_opt_set(m_codecCtx->priv_data, "preset", "veryfast", 0);
        av_opt_set(m_codecCtx->priv_data, "crf", "23", 0);
    }
    if ((ret = avcodec_open2(m_codecCtx, codec, nullptr)) < 0) return false;
    if ((ret = avcodec_parameters_from_context(stream->codecpar, m_codecCtx)) < 0) {
        LOG_ERROR("VideoRenderer: failed to copy encoder parameters: {}", AvError(ret));
        return false;
    }
    stream->time_base = m_codecCtx->time_base;
    if (!(m_formatCtx->oformat->flags & AVFMT_NOFILE) &&
        (ret = avio_open(&m_formatCtx->pb, m_outputPath.c_str(), AVIO_FLAG_WRITE)) < 0) {
        LOG_ERROR("VideoRenderer: failed to open output: {}", AvError(ret));
        return false;
    }
    if ((ret = avformat_write_header(m_formatCtx, nullptr)) < 0) return false;
    m_yuvFrame = av_frame_alloc();
    m_rgbFrame = av_frame_alloc();
    m_packet = av_packet_alloc();
    if (!m_yuvFrame || !m_rgbFrame || !m_packet) return false;
    m_yuvFrame->format = m_codecCtx->pix_fmt;
    m_yuvFrame->width = m_width;
    m_yuvFrame->height = m_height;
    if ((ret = av_frame_get_buffer(m_yuvFrame, 32)) < 0) return false;
    m_swsCtx = sws_getContext(m_width, m_height, AV_PIX_FMT_RGB24,
                              m_width, m_height, m_codecCtx->pix_fmt,
                              SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (!m_swsCtx) return false;
    m_headerWritten = true;
    return true;
}

void VideoRenderer::DrawOverlay(
    std::string& rgb, int width, int height,
    const aifootball::ProcessResult& result) {
    for (const auto& person : result.persons) {
        DrawRect(rgb, width, height, person.bbox[0], person.bbox[1],
                 person.bbox[2], person.bbox[3], 50, 230, 80);
        DrawPose(rgb, width, height, person);
    }
    for (const auto& ball : result.balls) {
        DrawRect(rgb, width, height, ball.bbox[0], ball.bbox[1],
                 ball.bbox[2], ball.bbox[3], 255, 80, 40);
        DrawPoint(rgb, width, height, static_cast<int>(std::lround(ball.center[0])),
                  static_cast<int>(std::lround(ball.center[1])), 5, 255, 80, 40);
    }
}

bool VideoRenderer::EncodeFrame(std::string& rgb) {
    m_rgbFrame->data[0] = reinterpret_cast<uint8_t*>(&rgb[0]);
    m_rgbFrame->linesize[0] = m_width * 3;
    if (av_frame_make_writable(m_yuvFrame) < 0) return false;
    if (sws_scale(m_swsCtx, m_rgbFrame->data, m_rgbFrame->linesize, 0,
                  m_height, m_yuvFrame->data, m_yuvFrame->linesize) <= 0) {
        return false;
    }
    m_yuvFrame->pts = m_outputFrameIndex++;
    if (avcodec_send_frame(m_codecCtx, m_yuvFrame) < 0) return false;
    DrainPackets();
    return true;
}

void VideoRenderer::DrainPackets() {
    while (true) {
        const int ret = avcodec_receive_packet(m_codecCtx, m_packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return;
        if (ret < 0) {
            LOG_ERROR("VideoRenderer: receive packet failed: {}", AvError(ret));
            return;
        }
        av_packet_rescale_ts(m_packet, m_codecCtx->time_base,
                             m_formatCtx->streams[0]->time_base);
        m_packet->stream_index = 0;
        av_interleaved_write_frame(m_formatCtx, m_packet);
        av_packet_unref(m_packet);
    }
}

void VideoRenderer::FinalizeEncoder() {
    if (!m_headerWritten || !m_codecCtx || !m_formatCtx) return;
    const int ret = avcodec_send_frame(m_codecCtx, nullptr);
    if (ret >= 0 || ret == AVERROR_EOF) DrainPackets();
    av_write_trailer(m_formatCtx);
    m_headerWritten = false;
}

void VideoRenderer::CleanupEncoder() {
    if (m_swsCtx) sws_freeContext(m_swsCtx);
    if (m_packet) av_packet_free(&m_packet);
    if (m_rgbFrame) av_frame_free(&m_rgbFrame);
    if (m_yuvFrame) av_frame_free(&m_yuvFrame);
    if (m_codecCtx) avcodec_free_context(&m_codecCtx);
    if (m_formatCtx) {
        if (!(m_formatCtx->oformat->flags & AVFMT_NOFILE) && m_formatCtx->pb) {
            avio_closep(&m_formatCtx->pb);
        }
        avformat_free_context(m_formatCtx);
    }
    m_swsCtx = nullptr;
    m_width = 0;
    m_height = 0;
    m_outputFrameIndex = 0;
}

} // namespace aifootball_demo
