#include "VideoRenderer.hpp"

#include "AppConfig.hpp"

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
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace aifootball_app {
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

/// 5x7 dot matrix glyph, one byte per column with bit 0 on the top row. Only
/// the characters the overlay labels use are listed; DrawGlyph skips the rest.
struct Glyph {
    char character;
    uint8_t columns[5];
};

const Glyph kFont[] = {
    {' ', {0x00, 0x00, 0x00, 0x00, 0x00}},
    {'.', {0x00, 0x60, 0x60, 0x00, 0x00}},
    {':', {0x00, 0x36, 0x36, 0x00, 0x00}},
    {'+', {0x08, 0x08, 0x3E, 0x08, 0x08}},
    {'=', {0x14, 0x14, 0x14, 0x14, 0x14}},
    {'0', {0x3E, 0x51, 0x49, 0x45, 0x3E}},
    {'1', {0x00, 0x42, 0x7F, 0x40, 0x00}},
    {'2', {0x42, 0x61, 0x51, 0x49, 0x46}},
    {'3', {0x21, 0x41, 0x45, 0x4B, 0x31}},
    {'4', {0x18, 0x14, 0x12, 0x7F, 0x10}},
    {'5', {0x27, 0x45, 0x45, 0x45, 0x39}},
    {'6', {0x3C, 0x4A, 0x49, 0x49, 0x30}},
    {'7', {0x01, 0x71, 0x09, 0x05, 0x03}},
    {'8', {0x36, 0x49, 0x49, 0x49, 0x36}},
    {'9', {0x06, 0x49, 0x49, 0x29, 0x1E}},
    {'a', {0x20, 0x54, 0x54, 0x54, 0x78}},
    {'b', {0x7F, 0x48, 0x44, 0x44, 0x38}},
    {'c', {0x38, 0x44, 0x44, 0x44, 0x20}},
    {'d', {0x38, 0x44, 0x44, 0x48, 0x7F}},
    {'e', {0x38, 0x54, 0x54, 0x54, 0x18}},
    {'f', {0x00, 0x08, 0x7E, 0x09, 0x02}},
    {'g', {0x0C, 0x52, 0x52, 0x52, 0x3E}},
    {'i', {0x00, 0x44, 0x7D, 0x40, 0x00}},
    {'k', {0x7F, 0x08, 0x14, 0x22, 0x41}},
    {'l', {0x00, 0x41, 0x7F, 0x40, 0x00}},
    {'m', {0x7C, 0x04, 0x18, 0x04, 0x78}},
    {'n', {0x7C, 0x08, 0x04, 0x04, 0x78}},
    {'o', {0x38, 0x44, 0x44, 0x44, 0x38}},
    {'p', {0x7C, 0x14, 0x14, 0x14, 0x08}},
    {'r', {0x7C, 0x08, 0x04, 0x04, 0x08}},
    {'s', {0x48, 0x54, 0x54, 0x54, 0x20}},
    {'t', {0x04, 0x3F, 0x44, 0x40, 0x20}},
    {'u', {0x3C, 0x40, 0x40, 0x40, 0x3C}},
    {'v', {0x1C, 0x20, 0x40, 0x20, 0x1C}},
    {'w', {0x1C, 0x20, 0x18, 0x20, 0x1C}},
    {'z', {0x44, 0x64, 0x54, 0x4C, 0x44}},
};

void DrawText(std::string& image, int width, int height, const char* text,
              int x, int y, int scale, uint8_t r, uint8_t g, uint8_t b) {
    int cursor = x;
    for (const char* ch = text; ch != nullptr && *ch != '\0'; ++ch) {
        for (const auto& glyph : kFont) {
            if (glyph.character != *ch) continue;
            for (int column = 0; column < 5; ++column) {
                const uint8_t bits = glyph.columns[column];
                for (int row = 0; row < 7; ++row) {
                    if ((bits & (1u << row)) == 0) continue;
                    for (int dy = 0; dy < scale; ++dy) {
                        for (int dx = 0; dx < scale; ++dx) {
                            SetPixel(image, width, height,
                                     cursor + column * scale + dx,
                                     y + row * scale + dy, r, g, b);
                        }
                    }
                }
            }
            break;
        }
        cursor += 6 * scale;  // Five columns plus one of spacing.
    }
}

/// Seconds the reported processing rate averages over. Long enough that the
/// batch rhythm of the pipeline (detector and pose run in waves) averages out,
/// short enough to show a slowdown while it happens.
const double kProcessingWindowSeconds = 5.0;

/// Label size used by every text in the overlay.
const int kLabelScale = 2;
/// One glyph cell: five columns plus one of spacing, times the scale.
const int kGlyphAdvance = 6;

int TextWidth(const std::string& text, int scale) {
    return static_cast<int>(text.size()) * kGlyphAdvance * scale;
}

void FillRect(std::string& image, int width, int height, int x0, int y0,
              int x1, int y1, uint8_t r, uint8_t g, uint8_t b, float alpha) {
    const float keep = 1.0f - alpha;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            if (x < 0 || y < 0 || x >= width || y >= height) continue;
            const std::size_t offset =
                (static_cast<std::size_t>(y) * width + x) * 3;
            image[offset] = static_cast<char>(static_cast<uint8_t>(
                static_cast<unsigned char>(image[offset]) * keep + r * alpha));
            image[offset + 1] = static_cast<char>(static_cast<uint8_t>(
                static_cast<unsigned char>(image[offset + 1]) * keep + g * alpha));
            image[offset + 2] = static_cast<char>(static_cast<uint8_t>(
                static_cast<unsigned char>(image[offset + 2]) * keep + b * alpha));
        }
    }
}

/// Text block on a dimmed background, mirroring the Python debug panel.
void DrawPanel(std::string& image, int width, int height,
               const std::vector<std::string>& lines, int x, int y) {
    if (lines.empty()) return;
    int textWidth = 0;
    for (const auto& line : lines) {
        textWidth = std::max(textWidth, TextWidth(line, kLabelScale));
    }
    const int padding = 10;
    const int lineHeight = 7 * kLabelScale + 8;
    const int bottom = y + static_cast<int>(lines.size() - 1) * lineHeight +
                       7 * kLabelScale;
    FillRect(image, width, height, x - padding, y - padding,
             x + textWidth + padding, bottom + padding, 0, 0, 0, 0.6f);
    int cursorY = y;
    for (const auto& line : lines) {
        DrawText(image, width, height, line.c_str(), x, cursorY, kLabelScale,
                 255, 255, 255);
        cursorY += lineHeight;
    }
}

void DrawRoi(std::string& image, int width, int height,
             const aifootball::RoiContext& roi) {
    if (!roi.enabled || roi.points.size() < 3) return;
    const float scaleX = roi.width > 0
        ? static_cast<float>(width) / static_cast<float>(roi.width) : 1.0f;
    const float scaleY = roi.height > 0
        ? static_cast<float>(height) / static_cast<float>(roi.height) : 1.0f;
    std::vector<std::pair<int, int>> polygon;
    polygon.reserve(roi.points.size());
    for (const auto& point : roi.points) {
        polygon.push_back(std::make_pair(
            static_cast<int>(std::lround(point.x * scaleX)),
            static_cast<int>(std::lround(point.y * scaleY))));
    }
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const std::pair<int, int>& from = polygon[i];
        const std::pair<int, int>& to = polygon[(i + 1) % polygon.size()];
        DrawLine(image, width, height, from.first, from.second, to.first,
                 to.second, 0, 255, 255);
    }
    for (const auto& vertex : polygon) {
        DrawPoint(image, width, height, vertex.first, vertex.second, 5, 0, 255,
                  255);
    }
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
    // Throughput over the trailing window: this is what the overlay reports
    // as fps.
    const std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    m_writeTimes.push_back(now);
    while (m_writeTimes.size() > 1 &&
           std::chrono::duration<double>(now - m_writeTimes.front()).count() >
               kProcessingWindowSeconds) {
        m_writeTimes.pop_front();
    }
    if (m_writeTimes.size() > 1) {
        const double seconds = std::chrono::duration<double>(
            now - m_writeTimes.front()).count();
        if (seconds > 0.0) {
            m_processingFps =
                static_cast<double>(m_writeTimes.size() - 1) / seconds;
        }
    }

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
    DrawOverlay(rendered, result);
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

void VideoRenderer::SetRoi(const aifootball::RoiContext& roi) {
    m_roi = roi;
    LOG_INFO("VideoRenderer: ROI {} with {} point(s)",
             roi.enabled ? "enabled" : "disabled", roi.points.size());
}

void VideoRenderer::DrawOverlay(
    std::string& rgb, const aifootball::ProcessResult& result) const {
    DrawRoi(rgb, m_width, m_height, m_roi);
    for (const auto& person : result.persons) {
        DrawRect(rgb, m_width, m_height, person.bbox[0], person.bbox[1],
                 person.bbox[2], person.bbox[3], 50, 230, 80);
        DrawPose(rgb, m_width, m_height, person);
    }
    // Balls the box filters dropped, coloured by reason like Python does:
    // red = confidence, magenta = size, purple = both.
    int filteredByConfidence = 0;
    int filteredBySize = 0;
    for (const auto& rejected : result.rejectedBalls) {
        uint8_t red = 255, green = 0, blue = 0;
        const char* reason = "conf";
        if (rejected.failedConfidence && rejected.failedSize) {
            red = 160; green = 32; blue = 240; reason = "conf+area";
        } else if (!rejected.failedConfidence) {
            red = 255; green = 0; blue = 255; reason = "area";
        }
        if (rejected.failedConfidence) ++filteredByConfidence;
        if (rejected.failedSize) ++filteredBySize;
        DrawRect(rgb, m_width, m_height, rejected.bbox[0], rejected.bbox[1],
                 rejected.bbox[2], rejected.bbox[3], red, green, blue);
        DrawText(rgb, m_width, m_height, reason,
                 static_cast<int>(std::lround(rejected.bbox[0])),
                 std::max(0, static_cast<int>(std::lround(rejected.bbox[1])) - 18),
                 kLabelScale, red, green, blue);
    }
    // Every ball detection the frame was processed from. Drawn as boxes with
    // their confidence, the layer Python labels "sports ball 0.62".
    for (const auto& detection : result.ballDetections) {
        DrawRect(rgb, m_width, m_height, detection.bbox[0], detection.bbox[1],
                 detection.bbox[2], detection.bbox[3], 255, 80, 40);
        char label[24] = {};
        std::snprintf(label, sizeof(label), "%.2f", detection.confidence);
        DrawText(rgb, m_width, m_height, label,
                 static_cast<int>(std::lround(detection.bbox[0])),
                 std::max(0, static_cast<int>(std::lround(detection.bbox[1])) - 18),
                 kLabelScale, 255, 80, 40);
    }

    int trackedBalls = 0;
    const char* ballState = "missing";
    for (const auto& ball : result.balls) {
        if (ball.trackId < 0) continue;  // Already drawn as a detection above.
        // The tracked ball carries the marker the Python renderer uses: a
        // filled circle labelled with its ID and state, orange while observed
        // and blue once it is only predicted.
        const int cx = static_cast<int>(std::lround(ball.center[0]));
        const int cy = static_cast<int>(std::lround(ball.center[1]));
        const bool observed = ball.state == aifootball::BallTrackState::Observed;
        const uint8_t blue = observed ? 40 : 255;
        ++trackedBalls;
        ballState = observed ? "observed" : "predicted";
        DrawPoint(rgb, m_width, m_height, cx, cy, 8, 255, 80, blue);
        char label[48] = {};
        std::snprintf(label, sizeof(label), "ball id=%d %s", ball.trackId,
                      observed ? "observed" : "predicted");
        DrawText(rgb, m_width, m_height, label, cx + 12,
                 std::max(0, cy - 20), kLabelScale, 255, 80, blue);
    }

    char header[96] = {};
    std::snprintf(header, sizeof(header), "frame=%llu time=%.2fs det=%d",
                  static_cast<unsigned long long>(result.frameId),
                  result.timestampSec,
                  result.filteredCounts.total + result.rescuedBallCount);
    DrawText(rgb, m_width, m_height, header, 24, 12, kLabelScale, 50, 230, 80);
    char fps[48] = {};
    std::snprintf(fps, sizeof(fps), "fps=%.2f processing", m_processingFps);
    DrawText(rgb, m_width, m_height, fps, 24, 40, kLabelScale, 50, 230, 80);

    // Same three lines as the Python debug panel: what the detector found,
    // what survived the ROI and the filters, and what is tracked now.
    char ballLine[160] = {};
    std::snprintf(ballLine, sizeof(ballLine),
                  "ball: raw=%d roi=%d kept=%d rescued=%d tracked=%d state=%s",
                  result.detectorCounts.ball,
                  result.filteredCounts.ball +
                      static_cast<int>(result.rejectedBalls.size()),
                  result.filteredCounts.ball, result.rescuedBallCount,
                  trackedBalls, ballState);
    char filterLine[96] = {};
    std::snprintf(filterLine, sizeof(filterLine),
                  "filter: conf=%d size=%d total=%d", filteredByConfidence,
                  filteredBySize,
                  static_cast<int>(result.rejectedBalls.size()));
    char personLine[96] = {};
    std::snprintf(personLine, sizeof(personLine), "persons: det=%d track=%d",
                  result.filteredCounts.person,
                  static_cast<int>(result.persons.size()));
    const std::vector<std::string> lines = {ballLine, filterLine, personLine};
    DrawPanel(rgb, m_width, m_height, lines, 24, 90);
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

} // namespace aifootball_app
