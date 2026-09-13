#pragma once

#include <aifootball/AIFootball.hpp>

#include <fstream>
#include <string>

namespace aifootball_demo {

class JsonlWriter {
public:
    bool Open(const std::string& path, const std::string& videoPath,
              double fps, int width, int height, int stride,
              const std::string& cameraId);
    void Write(const aifootball::ProcessResult& result);
    void Close();

private:
    std::ofstream m_file;
    std::string m_cameraId;
};

} // namespace aifootball_demo
