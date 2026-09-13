#include "DemoConfig.hpp"

#include <yaml-cpp/yaml.h>

#include <cerrno>
#include <stdexcept>
#include <sys/stat.h>

namespace aifootball_demo {

aifootball::AIFootballContext LoadContext(const std::string& configPath) {
    aifootball::AIFootballContext context;
    const YAML::Node root = YAML::LoadFile(configPath);
    const YAML::Node roi = root["algorithm"]["roi"];
    if (!roi) return context;

    if (roi["enabled"]) context.roi.enabled = roi["enabled"].as<bool>();
    if (roi["width"]) context.roi.width = roi["width"].as<int>();
    if (roi["height"]) context.roi.height = roi["height"].as<int>();
    const YAML::Node points = roi["points"];
    if (points) {
        if (!points.IsSequence()) {
            throw std::runtime_error("algorithm.roi.points must be a sequence");
        }
        for (const auto& point : points) {
            if (!point.IsSequence() || point.size() < 2) {
                throw std::runtime_error("each algorithm.roi point must be [x, y]");
            }
            context.roi.points.push_back(
                {point[0].as<float>(), point[1].as<float>()});
        }
    }
    if (context.roi.enabled && context.roi.points.size() < 3) {
        throw std::runtime_error("algorithm.roi.enabled requires at least 3 points");
    }
    return context;
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

} // namespace aifootball_demo
