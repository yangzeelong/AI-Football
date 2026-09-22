#pragma once

#include <aifootball/AIFootball.hpp>

#include <string>

namespace aifootball_app {

/**
 * @brief One parsed view of `examples/aifootball_app/config.yaml`.
 *
 * The app needs several unrelated parts of the same file: the algorithm
 * context, the video source, the decoder ring size and the output metadata.
 * Parsing it once in the constructor keeps those readers independent, makes
 * every consumer see the same snapshot, and leaves a single place to report a
 * malformed entry.
 *
 * The class throws std::runtime_error / YAML exceptions on invalid input.
 */
class AppConfig {
public:
    explicit AppConfig(const std::string& path);

    const std::string& Path() const { return m_path; }

    /// `algorithm` section: ROI, backpressure and queue policy.
    const aifootball::AIFootballContext& Context() const { return m_context; }

    /// `graph.modules[VideoReader].config.videoPath`.
    const std::string& VideoPath() const { return m_videoPath; }

    /// `graph.modules[VideoReader].config.rgbRingSize`, 0 when unset.
    int RgbRingSize() const { return m_rgbRingSize; }

    /// `graph.modules[ObservationWriter].config.cameraId`, defaults to "C1".
    const std::string& CameraId() const { return m_cameraId; }

private:
    std::string m_path;
    aifootball::AIFootballContext m_context;
    std::string m_videoPath;
    int m_rgbRingSize = 0;
    std::string m_cameraId = "C1";
};

// Filesystem helpers. They do not depend on the parsed configuration, so they
// stay free functions.

bool EnsureDirectory(const std::string& path);
std::string JoinPath(const std::string& dir, const std::string& name);

} // namespace aifootball_app
