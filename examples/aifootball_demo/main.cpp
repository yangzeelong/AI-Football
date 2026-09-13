#include "CommandParser.hpp"

#include <aifootball/AIFootball.hpp>
#include <nexusflow/Logging.hpp>
#include <yaml-cpp/yaml.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

std::atomic<bool> g_stop{false};

void HandleSignal(int) { g_stop.store(true, std::memory_order_relaxed); }

bool EnsureDirectory(const std::string& path) {
    if (path.empty()) return true;
    std::string current;
    size_t begin = 0;
    if (path[0] == '/') {
        current = "/";
        begin = 1;
    }
    while (begin < path.size()) {
        const size_t slash = path.find('/', begin);
        const size_t end = slash == std::string::npos ? path.size() : slash;
        if (end > begin) {
            if (!current.empty() && current.back() != '/') current.push_back('/');
            current.append(path, begin, end - begin);
            if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
        if (slash == std::string::npos) break;
        begin = slash + 1;
    }
    return true;
}

std::string JoinPath(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    return dir.back() == '/' ? dir + name : dir + "/" + name;
}

aifootball::AlgoConfig LoadAlgoConfig(const std::string& configPath) {
    aifootball::AlgoConfig algoConfig;
    const YAML::Node root = YAML::LoadFile(configPath);
    const YAML::Node roi = root["algorithm"]["roi"];
    if (!roi) return algoConfig;

    if (roi["enabled"]) algoConfig.roi.enabled = roi["enabled"].as<bool>();
    if (roi["width"]) algoConfig.roi.width = roi["width"].as<int>();
    if (roi["height"]) algoConfig.roi.height = roi["height"].as<int>();
    const YAML::Node points = roi["points"];
    if (points) {
        if (!points.IsSequence()) {
            throw std::runtime_error("algorithm.roi.points must be a sequence");
        }
        for (const auto& point : points) {
            if (!point.IsSequence() || point.size() < 2) {
                throw std::runtime_error("each algorithm.roi point must be [x, y]");
            }
            algoConfig.roi.points.push_back({point[0].as<float>(), point[1].as<float>()});
        }
    }
    if (algoConfig.roi.enabled && algoConfig.roi.points.size() < 3) {
        throw std::runtime_error("algorithm.roi.enabled requires at least 3 points");
    }
    return algoConfig;
}

std::string LoadVideoPath(const std::string& configPath) {
    const YAML::Node root = YAML::LoadFile(configPath);
    const YAML::Node modules = root["graph"]["modules"];
    if (!modules || !modules.IsSequence()) return {};
    for (const auto& module : modules) {
        if (module["class"] && module["class"].as<std::string>() == "VideoReader") {
            return module["config"]["videoPath"].as<std::string>("");
        }
    }
    return {};
}

std::string LoadCameraId(const std::string& configPath) {
    const YAML::Node root = YAML::LoadFile(configPath);
    const YAML::Node modules = root["graph"]["modules"];
    if (!modules || !modules.IsSequence()) return "C1";
    for (const auto& module : modules) {
        if (module["class"] && module["class"].as<std::string>() == "ObservationWriter") {
            return module["config"]["cameraId"].as<std::string>("C1");
        }
    }
    return "C1";
}

void JsonEscape(std::ostream& os, const std::string& value) {
    os << '"';
    for (char c : value) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default: os << c; break;
        }
    }
    os << '"';
}

void JsonFloat(std::ostream& os, float value) {
    if (!std::isfinite(value)) { os << "null"; return; }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.4f", value);
    os << buffer;
}

void JsonDouble(std::ostream& os, double value) {
    if (!std::isfinite(value)) { os << "null"; return; }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6f", value);
    os << buffer;
}

const char* KeypointStateName(aifootball::KeypointState state) {
    switch (state) {
        case aifootball::KeypointState::Observed: return "observed";
        case aifootball::KeypointState::Virtual: return "virtual";
        case aifootball::KeypointState::Missing: return "missing";
    }
    return "missing";
}

const char* BallStateName(aifootball::BallTrackState state) {
    switch (state) {
        case aifootball::BallTrackState::Observed: return "observed";
        case aifootball::BallTrackState::Predicted: return "predicted";
        case aifootball::BallTrackState::Lost: return "lost";
    }
    return "lost";
}

void WriteMetadata(std::ofstream& file, const std::string& videoPath,
                   double fps, int width, int height, int stride,
                   const std::string& cameraId) {
    file << "{\"type\":\"metadata\",\"schema_version\":1,\"video\":{";
    file << "\"video_path\":"; JsonEscape(file, videoPath); file << ",";
    file << "\"fps\":"; JsonDouble(file, fps); file << ",";
    file << "\"width\":" << width << ",\"height\":" << height << ",";
    file << "\"frame_count\":0,\"stride\":" << stride << ",";
    file << "\"camera_id\":"; JsonEscape(file, cameraId);
    file << "}}\n";
}

void WriteResult(std::ofstream& file, const aifootball::ProcessResult& result,
                 const std::string& cameraId) {
    static const char* const names[26] = {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
        "right_small_toe", "right_heel", "neck", "pelvis", "thorax",
    };
    file << "{\"type\":\"frame\",\"frame_index\":" << result.frameId << ",";
    file << "\"timestamp_sec\":"; JsonDouble(file, result.timestampSec); file << ",";
    file << "\"camera_id\":"; JsonEscape(file, cameraId); file << ",";
    file << "\"persons\":[";
    for (size_t i = 0; i < result.persons.size(); ++i) {
        if (i) file << ',';
        const auto& person = result.persons[i];
        file << "{\"track_id\":" << person.trackId << ",\"bbox\":[";
        for (size_t j = 0; j < person.bbox.size(); ++j) {
            if (j) file << ',';
            JsonFloat(file, person.bbox[j]);
        }
        file << "],\"confidence\":"; JsonFloat(file, person.confidence);
        file << ",\"keypoints\":[";
        for (size_t j = 0; j < person.keypoints.size(); ++j) {
            if (j) file << ',';
            const auto& keypoint = person.keypoints[j];
            file << "{\"name\":"; JsonEscape(file, names[j]);
            file << ",\"x\":";
            if (keypoint.state == aifootball::KeypointState::Missing) file << "null";
            else JsonFloat(file, keypoint.x);
            file << ",\"y\":";
            if (keypoint.state == aifootball::KeypointState::Missing) file << "null";
            else JsonFloat(file, keypoint.y);
            file << ",\"confidence\":"; JsonFloat(file, keypoint.confidence);
            file << ",\"state\":\"" << KeypointStateName(keypoint.state) << "\"}";
        }
        file << "],\"state\":\"observed\"}";
    }
    file << "],\"balls\":[";
    for (size_t i = 0; i < result.balls.size(); ++i) {
        if (i) file << ',';
        const auto& ball = result.balls[i];
        file << "{\"track_id\":" << ball.trackId << ",\"bbox\":[";
        for (size_t j = 0; j < ball.bbox.size(); ++j) {
            if (j) file << ',';
            JsonFloat(file, ball.bbox[j]);
        }
        file << "],\"center\":["; JsonFloat(file, ball.center[0]); file << ',';
        JsonFloat(file, ball.center[1]); file << "],\"confidence\":";
        JsonFloat(file, ball.confidence);
        file << ",\"state\":\"" << BallStateName(ball.state) << "\"}";
    }
    file << "],\"raw_detection_counts\":{";
    file << "\"total\":" << result.rawCounts.total << ",\"person\":"
         << result.rawCounts.person << ",\"ball\":" << result.rawCounts.ball << "},";
    file << "\"module_timings_ms\":{";
    bool first = true;
    for (const auto& timing : result.moduleTimingsMs) {
        if (!first) file << ',';
        first = false;
        JsonEscape(file, timing.first); file << ':'; JsonDouble(file, timing.second);
    }
    file << "}}\n";
}

struct Decoder {
    AVFormatContext* format = nullptr;
    AVCodecContext* codec = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* sws = nullptr;
    int streamIndex = -1;
    AVRational timeBase{0, 1};
    double fps = 25.0;
    int width = 0;
    int height = 0;

    ~Decoder() {
        if (sws) sws_freeContext(sws);
        if (packet) av_packet_free(&packet);
        if (frame) av_frame_free(&frame);
        if (codec) avcodec_free_context(&codec);
        if (format) avformat_close_input(&format);
    }

    bool Open(const std::string& path) {
        if (avformat_open_input(&format, path.c_str(), nullptr, nullptr) < 0 ||
            avformat_find_stream_info(format, nullptr) < 0) return false;
        for (unsigned i = 0; i < format->nb_streams; ++i) {
            if (format->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                streamIndex = static_cast<int>(i);
                break;
            }
        }
        if (streamIndex < 0) return false;
        AVStream* stream = format->streams[streamIndex];
        const AVCodec* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
        if (!decoder) return false;
        codec = avcodec_alloc_context3(decoder);
        if (!codec || avcodec_parameters_to_context(codec, stream->codecpar) < 0 ||
            avcodec_open2(codec, decoder, nullptr) < 0) return false;
        frame = av_frame_alloc();
        packet = av_packet_alloc();
        if (!frame || !packet) return false;
        timeBase = stream->time_base;
        if (stream->avg_frame_rate.den > 0 && stream->avg_frame_rate.num > 0) {
            fps = av_q2d(stream->avg_frame_rate);
        }
        width = codec->width;
        height = codec->height;
        return true;
    }
};

} // namespace

int main(int argc, char* argv[]) {
    app::CommandParser cli(argc, argv);
    cli.AddPositional("config", "Path to config.yaml", true);
    cli.AddArgument("--video_path", "-v", "Override video path from config");
    cli.AddArgument("--output_dir", "-o", "Output directory", false, "output/sdk_demo");
    cli.AddArgument("--max_frames", "-n", "Stop after N decoded frames (0 = all)", false, "0");
    cli.AddArgument("--stride", "-S", "Process every Nth decoded frame", false, "1");
    cli.AddArgument("--device", "-d", "CUDA device id", false, "0");
    cli.AddArgument("--verbose", "-V", "Enable DEBUG-level logging", false,
                    app::CommandParser::FlagMarker());
    cli.AddArgument("--quiet", "-q", "Suppress INFO-level logging", false,
                    app::CommandParser::FlagMarker());
    if (!cli.Parse()) { cli.PrintHelp(); return 1; }
    if (cli.IsHelpRequested()) { cli.PrintHelp(); return 0; }

    nexusflow::logger::LoggerParam logParams;
    logParams.logLevel = cli.IsFlagSet("verbose") ? nexusflow::logger::LogLevel::DEBUG :
                         cli.IsFlagSet("quiet") ? nexusflow::logger::LogLevel::WARN :
                                                    nexusflow::logger::LogLevel::INFO;
    nexusflow::logger::InitializeGlobalLogger(logParams);
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    const std::string configPath = cli.Get("config");
    const std::string videoPath = cli.Get("video_path").empty()
        ? LoadVideoPath(configPath) : cli.Get("video_path");
    const std::string outputDir = cli.Get("output_dir");
    const int maxFrames = cli.GetInt("max_frames");
    const int stride = std::max(1, cli.GetInt("stride", 1));
    const int deviceId = cli.GetInt("device", 0);
    if (videoPath.empty()) {
        LOG_ERROR("No video path provided and graph.modules has no VideoReader path");
        return 2;
    }
    if (!EnsureDirectory(outputDir)) {
        LOG_ERROR("Failed to create output directory '{}'");
        return 2;
    }

    const std::string observationsPath = JoinPath(outputDir, "observations.jsonl");
    std::ofstream observations(observationsPath, std::ios::out | std::ios::trunc);
    if (!observations.is_open()) {
        LOG_ERROR("Failed to open '{}'");
        return 2;
    }

    try {
        Decoder decoder;
        if (!decoder.Open(videoPath)) {
            LOG_ERROR("Failed to open/decode video '{}'");
            return 2;
        }
        const auto algoConfig = LoadAlgoConfig(configPath);
        const std::string cameraId = LoadCameraId(configPath);
        WriteMetadata(observations, videoPath, decoder.fps, decoder.width,
                      decoder.height, stride, cameraId);

        aifootball::RuntimeOptions options;
        options.configPath = configPath;
        options.deviceId = deviceId;
        auto runtime = aifootball::Runtime::Create(options, algoConfig);
        if (runtime->Init() != nexusflow::SUCCESS) {
            LOG_ERROR("AI-Football SDK initialization failed");
            return 2;
        }

        std::vector<uint8_t> rgb;
        uint64_t decodedIndex = 0;
        int processedFrames = 0;
        int pendingFrames = 0;
        auto pollReadyResults = [&]() -> bool {
            while (pendingFrames > 0) {
                aifootball::ProcessResult result;
                if (runtime->PollResult(result, 0) != nexusflow::SUCCESS) break;
                WriteResult(observations, result, cameraId);
                --pendingFrames;
            }
            return true;
        };
        auto processFrame = [&](AVFrame* decoded) -> bool {
            const uint64_t frameId = decodedIndex++;
            if ((frameId % static_cast<uint64_t>(stride)) != 0) return true;
            if (maxFrames > 0 && processedFrames >= maxFrames) return false;

            decoder.sws = sws_getCachedContext(
                decoder.sws, decoder.width, decoder.height,
                static_cast<AVPixelFormat>(decoded->format), decoder.width,
                decoder.height, AV_PIX_FMT_RGB24, SWS_BILINEAR,
                nullptr, nullptr, nullptr);
            if (!decoder.sws) return false;
            const int size = av_image_get_buffer_size(
                AV_PIX_FMT_RGB24, decoder.width, decoder.height, 1);
            rgb.resize(static_cast<size_t>(size));
            uint8_t* dstData[4] = {nullptr, nullptr, nullptr, nullptr};
            int dstLinesize[4] = {0, 0, 0, 0};
            av_image_fill_arrays(dstData, dstLinesize, rgb.data(), AV_PIX_FMT_RGB24,
                                 decoder.width, decoder.height, 1);
            sws_scale(decoder.sws, decoded->data, decoded->linesize, 0,
                      decoder.height, dstData, dstLinesize);

            double timestampSec = static_cast<double>(frameId) / decoder.fps;
            const int64_t pts = decoded->best_effort_timestamp;
            if (pts != AV_NOPTS_VALUE && decoder.timeBase.den > 0) {
                timestampSec = pts * av_q2d(decoder.timeBase);
            }
            aifootball::DecodedFrameView view;
            view.frameId = frameId;
            view.timestampSec = timestampSec;
            view.width = decoder.width;
            view.height = decoder.height;
            view.strideBytes = decoder.width * 3;
            view.data = rgb.data();
            view.dataBytes = rgb.size();
            if (runtime->Process(view) != nexusflow::SUCCESS) return false;
            ++pendingFrames;
            ++processedFrames;
            return pollReadyResults() &&
                   !g_stop.load(std::memory_order_relaxed);
        };

        bool keepRunning = true;
        while (keepRunning && !g_stop.load(std::memory_order_relaxed) &&
               av_read_frame(decoder.format, decoder.packet) >= 0) {
            if (decoder.packet->stream_index == decoder.streamIndex) {
                if (avcodec_send_packet(decoder.codec, decoder.packet) < 0) {
                    keepRunning = false;
                } else {
                    while (keepRunning && avcodec_receive_frame(decoder.codec, decoder.frame) == 0) {
                        keepRunning = processFrame(decoder.frame);
                    }
                }
            }
            av_packet_unref(decoder.packet);
        }
        avcodec_send_packet(decoder.codec, nullptr);
        while (keepRunning && avcodec_receive_frame(decoder.codec, decoder.frame) == 0) {
            keepRunning = processFrame(decoder.frame);
        }
        if (runtime->Flush() != nexusflow::SUCCESS) {
            LOG_ERROR("Failed to drain pending SDK results");
            runtime->DeInit();
            return 2;
        }
        while (pendingFrames > 0) {
            aifootball::ProcessResult result;
            if (runtime->PollResult(result, 300000) != nexusflow::SUCCESS) {
                LOG_ERROR("Failed to poll pending SDK result");
                runtime->DeInit();
                return 2;
            }
            WriteResult(observations, result, cameraId);
            --pendingFrames;
        }
        runtime->DeInit();
        LOG_INFO("AI-Football SDK demo processed {} frames, observations='{}'",
                 processedFrames, observationsPath);
    } catch (const std::exception& error) {
        LOG_ERROR("Fatal: {}", error.what());
        return 2;
    }
    return 0;
}
