#include "AppConfig.hpp"

#include <yaml-cpp/yaml.h>

#include <cerrno>
#include <cstddef>
#include <stdexcept>
#include <sys/stat.h>

namespace aifootball_app {

namespace {

/// `graph.modules` entry whose `class` matches, or an empty node.
YAML::Node FindModule(const YAML::Node& modules, const std::string& className) {
    if (!modules || !modules.IsSequence()) return {};
    for (const auto& module : modules) {
        if (module["class"] && module["class"].as<std::string>() == className) {
            return module;
        }
    }
    return {};
}

} // namespace

AppConfig::AppConfig(const std::string& path) : m_path(path) {
    const YAML::Node root = YAML::LoadFile(path);
    const YAML::Node modules = root["graph"]["modules"];

    // --- graph.modules ---
    if (const YAML::Node reader = FindModule(modules, "VideoReader")) {
        m_videoPath = reader["config"]["videoPath"].as<std::string>("");

        const int ringSize = reader["config"]["rgbRingSize"].as<int>(0);
        if (ringSize < 0) {
            throw std::runtime_error(
                "graph.modules[VideoReader].config.rgbRingSize must be >= 1");
        }
        m_rgbRingSize = ringSize;
    }
    if (const YAML::Node writer = FindModule(modules, "ObservationWriter")) {
        m_cameraId = writer["config"]["cameraId"].as<std::string>("C1");
    }

    // --- algorithm ---
    const YAML::Node algorithm = root["algorithm"];
    if (!algorithm) return;

    // Bound the asynchronous input queue. Each frame submitted but not yet
    // consumed pins a decoded frame (~6 MB at 1080p), so leaving this at 0
    // (unbounded) lets the decoder run arbitrarily far ahead of the slowest
    // pipeline stage. The default QueuePolicy::Block turns the bound into
    // backpressure by blocking ProcessAsync().
    if (algorithm["maxPendingFrames"]) {
        const int maxPendingFrames = algorithm["maxPendingFrames"].as<int>();
        if (maxPendingFrames < 0) {
            throw std::runtime_error(
                "algorithm.maxPendingFrames must be >= 0 (0 means unbounded)");
        }
        m_context.maxPendingFrames = static_cast<std::size_t>(maxPendingFrames);
    }
    if (algorithm["inputQueuePolicy"]) {
        const std::string policy =
            algorithm["inputQueuePolicy"].as<std::string>();
        if (policy == "block") {
            m_context.inputQueuePolicy = aifootball::QueuePolicy::Block;
        } else if (policy == "dropOldest") {
            m_context.inputQueuePolicy = aifootball::QueuePolicy::DropOldest;
        } else if (policy == "dropNew") {
            m_context.inputQueuePolicy = aifootball::QueuePolicy::DropNew;
        } else {
            throw std::runtime_error(
                "unknown algorithm.inputQueuePolicy: " + policy);
        }
    }

    const YAML::Node roi = algorithm["roi"];
    if (!roi) return;

    if (roi["enabled"]) m_context.roi.enabled = roi["enabled"].as<bool>();
    if (roi["width"]) m_context.roi.width = roi["width"].as<int>();
    if (roi["height"]) m_context.roi.height = roi["height"].as<int>();
    const YAML::Node points = roi["points"];
    if (points) {
        if (!points.IsSequence()) {
            throw std::runtime_error("algorithm.roi.points must be a sequence");
        }
        for (const auto& point : points) {
            if (!point.IsSequence() || point.size() < 2) {
                throw std::runtime_error("each algorithm.roi point must be [x, y]");
            }
            m_context.roi.points.push_back(
                {point[0].as<float>(), point[1].as<float>()});
        }
    }
    if (m_context.roi.enabled && m_context.roi.points.size() < 3) {
        throw std::runtime_error("algorithm.roi.enabled requires at least 3 points");
    }
}

bool EnsureDirectory(const std::string& path) {
    if (path.empty()) return true;
    std::string current;
    size_t begin = 0;
    if (path[0] == '/') { current = "/"; begin = 1; }
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

} // namespace aifootball_app
