#ifndef NEXUSFLOW_PIPELINE_BUILDER_HPP
#define NEXUSFLOW_PIPELINE_BUILDER_HPP

#include <nexusflow/Module.hpp>

#include <memory>
#include <string>
#include <vector>

namespace nexusflow {

// Forward declaration
class Pipeline;

/**
 * @class PipelineBuilder
 * @brief A utility class to programmatically construct a Pipeline.
 *
 * This class provides a fluent interface to define the topology of a pipeline
 * by adding modules and defining the connections between them, without exposing
 * the internal Graph data structure.
 */
class PipelineBuilder {
public:
    PipelineBuilder();
    ~PipelineBuilder();

    // PipelineBuilder
    PipelineBuilder(const PipelineBuilder&) = delete;
    PipelineBuilder& operator=(const PipelineBuilder&) = delete;

    PipelineBuilder(PipelineBuilder&&) noexcept;
    PipelineBuilder& operator=(PipelineBuilder&&) noexcept;

    /**
     * @brief Adds a module instance to the pipeline definition.
     * @param module A shared_ptr to the module instance.
     * @return A reference to this builder, for method chaining (fluent interface).
     */
    PipelineBuilder& AddModule(const std::shared_ptr<Module>& module);

    /**
     * @brief Defines a connection from one module to another.
     * @param srcModuleName The name of the source module.
     * @param dstModuleName The name of the destination module.
     * @return A reference to this builder for chaining.
     */
    PipelineBuilder& Connect(const std::string& srcModuleName, const std::string& dstModuleName);

    /**
     * @brief Builds the Pipeline instance from the defined configuration.
     * This method consumes the builder. After calling build(), the builder
     * will be in an invalid state.
     * @return A unique_ptr to the constructed Pipeline, or nullptr on failure.
     */
    std::unique_ptr<Pipeline> Build();

private:
    class Impl;
    std::unique_ptr<Impl> m_pImpl;
};

} // namespace nexusflow

#endif