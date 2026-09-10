#ifndef NEXUSFLOW_PIPELINE_HPP
#define NEXUSFLOW_PIPELINE_HPP

#include <nexusflow/ErrorCode.hpp>
#include <nexusflow/Module.hpp>

#include <memory>
#include <string>
#include <unordered_map>

// TODO 能否做不依赖Graph
// Forward declaration
class Graph;
namespace nexusflow {
class PipelineBuilder;
}

namespace nexusflow {

/**
 * @brief Pipeline
 *
 */
class Pipeline {
public:
    static std::unique_ptr<Pipeline> CreateFromYaml(const std::string& configPath);

    ErrorCode Init();

    ErrorCode Start();

    ErrorCode Stop();

    ErrorCode DeInit();

    ~Pipeline();

private:
    friend PipelineBuilder;

    Pipeline();

    void InitWithGraph(std::unique_ptr<Graph> graph);

    class Impl;
    std::unique_ptr<Impl> m_pImpl;
};

} // namespace nexusflow

#endif