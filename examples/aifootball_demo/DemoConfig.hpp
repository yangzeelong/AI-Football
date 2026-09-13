#pragma once

#include <aifootball/AIFootball.hpp>

#include <string>

namespace aifootball_demo {

aifootball::AIFootballContext LoadContext(const std::string& configPath);
std::string LoadVideoPath(const std::string& configPath);
std::string LoadCameraId(const std::string& configPath);
bool EnsureDirectory(const std::string& path);
std::string JoinPath(const std::string& dir, const std::string& name);

} // namespace aifootball_demo
