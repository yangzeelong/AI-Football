#ifndef NEXUSFLOW_MODULE_HPP
#define NEXUSFLOW_MODULE_HPP

#include <nexusflow/Config.hpp>
#include <nexusflow/ErrorCode.hpp>
#include <nexusflow/Message.hpp>
#include <nexusflow/TypeTraits.hpp>

#include <memory>
#include <unordered_map>

// --- Forward Declarations ---
// Forward-declare internal and framework classes to keep this header clean.
namespace nexusflow { namespace dispatcher {
class Dispatcher;
}} // namespace nexusflow::dispatcher

namespace nexusflow {
class Pipeline;
}

namespace nexusflow {

/**
 * @class Module
 * @brief An abstract base class for a processing unit within a data pipeline.
 *
 * A Module focuses exclusively on the business logic of "what to do with data."
 * It passively receives data through its `process` methods and sends results
 * via protected APIs. All threading, data I/O, and lifecycle management are
 * handled прозрачно by the framework.
 */
class Module {
public:
    /**
     * @brief Constructs a Module with a given name.
     * @param name A unique identifier for this module instance.
     */
    explicit Module(std::string name);

    /**
     * @brief Virtual destructor to ensure proper cleanup of derived classes.
     */
    virtual ~Module();

    // Modules are unique components and should not be copied or moved.
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    // --- Lifecycle ---
    virtual ErrorCode Configure(const Config& config);

    /**
     * @brief User-defined initialization logic.
     * Called by the framework once before the pipeline starts.
     * @return An ErrorCode indicating success or failure.
     */
    virtual ErrorCode Init();

    /**
     * @brief User-defined resource cleanup logic.
     * Called by the framework once after the pipeline has stopped.
     * @return An ErrorCode indicating success or failure.
     */
    virtual ErrorCode DeInit();

    // --- Processing ---

    /**
     * @brief The core processing logic for a single message.
     * @note Derived classes MUST implement this method.
     * @param inputMessage The message to be processed, passed in by the framework.
     */
    virtual void Process(Message& inputMessage) = 0;

    /**
     * @brief The core processing logic for a batch of messages.
     * The framework calls this method by default. The base implementation
     * simply iterates through the batch and calls `process()` for each message.
     * Override this for more efficient batch-oriented processing.
     * @param inputBatchMessages A batch of messages to be processed.
     */
    virtual void ProcessBatch(std::vector<Message>& inputBatchMessages);

    // --- Getter and Setter ---

    /**
     * @brief Gets the unique name of the module.
     * @return A const reference to the module's name.
     */
    const std::string& GetModuleName() const;

protected:
    // --- Protected API for Derived Classes ---

    /**
     * @brief Broadcasts a message to all connected downstream outputs.
     * @param msg The message to be sent.
     */
    void Broadcast(const Message& msg);

    /**
     * @brief Sends a message to a specific downstream output.
     * @param outputName The name of the output port to send the message to.
     * @param msg The message to be sent.
     */
    void SendTo(const std::string& outputName, const Message& msg);

private:
    friend class ModuleActor;

    // A private setter for the internal handle, callable only by the Pipeline.
    void SetDispatcher(const std::shared_ptr<dispatcher::Dispatcher>& dispatcher);

    std::string m_moduleName;

    // The internal dispatcher handle.
    std::shared_ptr<dispatcher::Dispatcher> m_dispatcherPtr;
};

} // namespace nexusflow

#endif // NEXUSFLOW_MODULE_HPP