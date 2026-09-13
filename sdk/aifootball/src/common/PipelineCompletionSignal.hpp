#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>

/**
 * @brief Process-wide completion signal for the AI-Football pipeline.
 *
 * Terminal modules (e.g. ObservationWriter) call NotifyComplete() when they
 * observe an isEnd=true message.  main() calls WaitForCompletion() instead of
 * sleeping for a fixed duration, so the process exits as soon as the last
 * frame has been flushed.
 *
 * A user callback can be registered via SetOnComplete() for side-effects such
 * as logging, metrics, or signalling another thread.  The callback is invoked
 * exactly once per NotifyComplete() (the first one that flips the flag).
 *
 * Thread-safe.  Header-only singleton - no .cpp needed.
 */
class PipelineCompletionSignal {
public:
    static PipelineCompletionSignal& Instance() {
        static PipelineCompletionSignal s_instance;
        return s_instance;
    }

    // --- Registration ------------------------------------------------------

    /// Register a callback invoked once when NotifyComplete() first fires.
    /// Pass nullptr to clear.  Safe to call from any thread before Notify().
    void SetOnComplete(std::function<void()> cb) {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_callback = std::move(cb);
    }

    // --- Notification ------------------------------------------------------

    /// Called by a terminal module when it sees isEnd=true.  Only the first
    /// invocation flips the flag and wakes waiters; subsequent calls are no-ops.
    void NotifyComplete() {
        std::function<void()> cb;
        {
            std::lock_guard<std::mutex> lk(m_mutex);
            if (m_completed) return;
            m_completed = true;
            cb = m_callback;
        }
        m_cv.notify_all();
        if (cb) cb();
    }

    // --- Waiting -----------------------------------------------------------

    /// Block until NotifyComplete() has been called, or until the timeout
    /// expires.  Returns true if completion was observed, false on timeout.
    /// A zero or negative timeout means "wait forever".
    template <typename Rep, typename Period>
    bool WaitForCompletion(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lk(m_mutex);
        if (timeout <= std::chrono::duration<Rep, Period>::zero()) {
            m_cv.wait(lk, [this] { return m_completed.load(); });
            return true;
        }
        return m_cv.wait_for(lk, timeout, [this] { return m_completed.load(); });
    }

    /// Non-blocking query.
    bool IsCompleted() const { return m_completed.load(); }

    // --- Reset (for tests / multi-run processes) ---------------------------

    void Reset() {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_completed = false;
    }

private:
    PipelineCompletionSignal() = default;
    ~PipelineCompletionSignal() = default;
    PipelineCompletionSignal(const PipelineCompletionSignal&) = delete;
    PipelineCompletionSignal& operator=(const PipelineCompletionSignal&) = delete;

    mutable std::mutex      m_mutex;
    std::condition_variable m_cv;
    std::atomic<bool>       m_completed{false};
    std::function<void()>   m_callback;
};
