#include "pose/HRNetPoseEstimatorInfer.hpp"

#include "VideoReader.hpp"

#include <nexusflow/Logging.hpp>
#include <nexusflow/TimerRegistry.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string enginePath =
        "/home/hx1/yzl/Work/AI-Football/models/mmpose/hrnet/hrnet-w48-dark.engine";
    std::string videoPath = "data/射门1-1080p60.mov";
    int iterations = 100;
    bool flipTest = true;
    bool requireDynamicBatch = true;
    std::vector<int> batches{1, 4, 8, 16};
};

void PrintUsage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --engine PATH       HRNet TensorRT engine\n"
        << "  --video PATH        RGB source video used for one real frame\n"
        << "  --iterations N      Timed iterations per batch (default: 100)\n"
        << "  --flip 0|1          Enable MMPose flip-test (default: 1)\n"
        << "  --batches LIST      Comma-separated batch sizes (default: 1,4,8,16)\n"
        << "  --allow-static      Allow a static batch-1 engine\n"
        << "  --help              Show this help\n";
}

bool ParseInt(const std::string& value, int& output) {
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed <= 0 || parsed > 1000000) {
        return false;
    }
    output = static_cast<int>(parsed);
    return true;
}

bool ParseBatches(const std::string& value, std::vector<int>& batches) {
    batches.clear();
    size_t begin = 0;
    while (begin < value.size()) {
        const size_t comma = value.find(',', begin);
        const size_t end = comma == std::string::npos ? value.size() : comma;
        int batch = 0;
        if (!ParseInt(value.substr(begin, end - begin), batch)) return false;
        batches.push_back(batch);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return !batches.empty();
}

bool ParseOptions(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            PrintUsage(argv[0]);
            return false;
        }
        if (argument == "--allow-static") {
            options.requireDynamicBatch = false;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << argument << '\n';
            return false;
        }
        const std::string value = argv[++i];
        if (argument == "--engine") {
            options.enginePath = value;
        } else if (argument == "--video") {
            options.videoPath = value;
        } else if (argument == "--iterations") {
            if (!ParseInt(value, options.iterations)) return false;
        } else if (argument == "--flip") {
            if (value != "0" && value != "1") return false;
            const int flip = value == "1" ? 1 : 0;
            options.flipTest = flip != 0;
        } else if (argument == "--batches") {
            if (!ParseBatches(value, options.batches)) return false;
        } else {
            std::cerr << "Unknown option: " << argument << '\n';
            PrintUsage(argv[0]);
            return false;
        }
    }
    return true;
}

bool LoadFirstFrame(const std::string& path,
                   aifootball::DecodedFrameView& frame) {
    aifootball_demo::VideoReader reader;
    if (!reader.Open(path)) {
        std::cerr << "Failed to open video: " << path << '\n';
        return false;
    }
    bool received = false;
    const bool decoded = reader.Decode([&](const aifootball::DecodedFrameView& view) {
        frame = view;
        received = true;
        return false;
    });
    if (!decoded || !received || frame.data == nullptr || !frame.dataOwner) {
        std::cerr << "Failed to decode first frame: " << path << '\n';
        return false;
    }
    return true;
}

std::vector<pose::HRNetPoseEstimatorInfer::PersonInput> MakeInputs(
    const aifootball::DecodedFrameView& frame, int batch) {
    const float width = static_cast<float>(frame.width);
    const float height = static_cast<float>(frame.height);
    const std::vector<std::array<float, 4>> boxes = {
        {width * 0.43f, height * 0.25f, width * 0.53f, height * 0.72f},
        {width * 0.52f, height * 0.20f, width * 0.64f, height * 0.69f},
        {width * 0.30f, height * 0.32f, width * 0.40f, height * 0.78f},
        {width * 0.64f, height * 0.28f, width * 0.74f, height * 0.76f},
    };

    std::vector<pose::HRNetPoseEstimatorInfer::PersonInput> inputs;
    inputs.reserve(static_cast<size_t>(batch));
    for (int i = 0; i < batch; ++i) {
        const auto& box = boxes[static_cast<size_t>(i) % boxes.size()];
        inputs.push_back({frame.data, frame.width, frame.height,
                          box[0], box[1], box[2], box[3], i + 1, 0.9f});
    }
    return inputs;
}

void PrintPoseTimers() {
    auto stats = nexusflow::TimerRegistry::Instance().Snapshot();
    std::sort(stats.begin(), stats.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
    });
    for (const auto& item : stats) {
        if (item.name.find("PoseEstimator.") != 0) continue;
        std::cout << "    " << std::left << std::setw(30) << item.name
                  << "samples=" << std::setw(6) << item.samples
                  << "work=" << std::setw(6) << item.workUnits
                  << "avg_batch_ms=" << std::setw(10) << item.AvgBatchMs()
                  << "avg_item_ms=" << item.AvgItemMs() << '\n';
    }
}

bool RunBatch(const Options& options,
              const aifootball::DecodedFrameView& frame,
              pose::HRNetPoseEstimatorInfer& infer,
              int batch) {
    if (batch > infer.MaxBatch()) {
        std::cout << "batch=" << batch << " skipped (engine max batch="
                  << infer.MaxBatch() << ")\n";
        return true;
    }

    const auto inputs = MakeInputs(frame, batch);
    std::vector<PersonPose> results;
    for (int i = 0; i < 10; ++i) {
        if (!infer.InferBatch(inputs, results)) {
            std::cerr << "Warmup inference failed for batch=" << batch << '\n';
            return false;
        }
    }

    nexusflow::TimerRegistry::Instance().Reset();
    results.clear();
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < options.iterations; ++i) {
        if (!infer.InferBatch(inputs, results)) {
            std::cerr << "Timed inference failed for batch=" << batch << '\n';
            return false;
        }
        if (results.size() != inputs.size()) {
            std::cerr << "Unexpected result count for batch=" << batch << '\n';
            return false;
        }
    }
    const double elapsedMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    const double batchMs = elapsedMs / options.iterations;
    const double itemMs = batchMs / batch;
    const double personsPerSecond =
        static_cast<double>(options.iterations * batch) / (elapsedMs / 1000.0);

    std::cout << std::fixed << std::setprecision(3)
              << "batch=" << batch
              << " iterations=" << options.iterations
              << " total_ms=" << elapsedMs
              << " avg_batch_ms=" << batchMs
              << " avg_person_ms=" << itemMs
              << " persons_per_sec=" << personsPerSecond << '\n';
    PrintPoseTimers();
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!ParseOptions(argc, argv, options)) return argc > 1 ? 1 : 0;

    nexusflow::logger::InitializeGlobalLogger({nexusflow::logger::LogLevel::WARN});

    aifootball::DecodedFrameView frame;
    if (!LoadFirstFrame(options.videoPath, frame)) return 2;

    pose::HRNetPoseEstimatorInfer::Param param;
    param.enginePath = options.enginePath;
    param.flipTest = options.flipTest;
    param.maxBatch = *std::max_element(options.batches.begin(), options.batches.end());

    pose::HRNetPoseEstimatorInfer infer;
    if (!infer.Init(param)) return 2;
    if (options.requireDynamicBatch && infer.MaxBatch() <= 1) {
        std::cerr << "The pose benchmark requires a dynamic/multi-batch engine; "
                  << "effective max batch is " << infer.MaxBatch()
                  << ". Use --allow-static only for diagnostics.\n";
        infer.Release();
        return 2;
    }

    std::cout << "Pose benchmark\n"
              << "  engine: " << options.enginePath << '\n'
              << "  video: " << options.videoPath << '\n'
              << "  frame: " << frame.width << 'x' << frame.height << '\n'
              << "  flip_test: " << (options.flipTest ? "true" : "false") << '\n'
              << "  batch_policy: "
              << (options.requireDynamicBatch ? "dynamic_required" : "static_allowed")
              << '\n'
              << "  effective_max_batch: " << infer.MaxBatch() << '\n';

    for (const int batch : options.batches) {
        if (!RunBatch(options, frame, infer, batch)) return 3;
    }
    infer.Release();
    return 0;
}
