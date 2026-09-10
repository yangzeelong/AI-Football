#ifndef NEXUSFLOW_WORKER_HPP
#define NEXUSFLOW_WORKER_HPP

#include "base/Define.hpp"
#include "common/ViewPtr.hpp"
#include "nexusflow/ErrorCode.hpp"
#include "nexusflow/Module.hpp"
#include "utils/logging.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <unordered_map>

namespace nexusflow { namespace core {

/**
 * @class Worker
 * @brief An internal worker class responsible for driving the execution of a single Module instance.
 *
 * The Worker manages a dedicated thread and implements all "framework logic," such as
 * pulling data from an input queue, invoking the Module's processing methods,
 * and responding to lifecycle controls (start/stop) from the Pipeline.
 */
class Worker {
public:
    Worker(const std::shared_ptr<Module>& modulePtr, const ViewPtr<Config>& configPtr);

    /**
     * @brief Destructor. Ensures the worker thread is properly joined.
     */
    ~Worker();

    // Workers manage a thread and are unique; they should not be copied or moved.
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    /**
     * @brief Starts the worker's thread.
     * @return An ErrorCode indicating the result of the operation.
     */
    ErrorCode Start();

    /**
     * @brief Signals the worker thread to stop and waits for it to join.
     * @return An ErrorCode indicating the result of the operation.
     */
    ErrorCode Stop();

    // Setter and getter for the input queue map.
    void AddQueue(const std::string& name, ViewPtr<MessageQueue> queue) {
        if (m_inputQueueMap.find(name) != m_inputQueueMap.end()) {
            LOG_ERROR("Output queue with name {} already exists", name);
            throw std::invalid_argument("Output queue with name " + name + " already exists");
        }
        m_inputQueueMap[name] = std::move(queue);
    }
    // ViewPtr<MessageQueue> GetQueue(const std::string& name) { return m_inputQueueMap[name]; }
    // void RemoveQueue(const std::string& name) { m_inputQueueMap.erase(name); }
    // void ClearQueues() { m_inputQueueMap.clear(); }

    void WorkLoop();

private:
    void RunFusion();

    /**
     * @brief Efficiently pulls a batch of messages from the input queue.
     * @details This function implements an efficient, two-phase strategy to gather messages
     * without causing busy-waiting on the CPU.
     *
     * 1.  **Greedy Phase:** It first performs a quick, non-blocking poll (`tryPop`) across all
     *     input queues to immediately collect any readily available messages.
     * 2.  **Blocking Poll Phase:** If the batch is not yet full, it enters a loop that
     *     iterates through the input queues. For each queue, it performs a short-duration
     *     blocking wait (`waitAndPopFor`). This allows the thread to sleep efficiently
     *     if no messages are available, yielding the CPU, yet remaining highly responsive.
     *     The loop continues until the batch is full or the timeout is reached.
     *
     * @param maxBatchSize The maximum number of messages to pull.
     * @param batchTimeout The maximum time to wait for messages to become available.
     */
    std::vector<Message> PullBatchMessage(size_t maxBatchSize, std::chrono::milliseconds batchTimeout);

private:
    std::shared_ptr<Module> m_modulePtr = nullptr;
    ViewPtr<Config> m_configPtr;
    std::unordered_map<std::string, ViewPtr<MessageQueue>> m_inputQueueMap;

    std::atomic<bool> m_stopFlag{false};
};

}} // namespace nexusflow::core

#endif // NEXUSFLOW_WORKER_HPP