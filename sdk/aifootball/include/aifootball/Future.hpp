#pragma once

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace aifootball {

template <typename T>
class Promise;

/**
 * A single-consumer asynchronous value.
 *
 * The object is intentionally small and move-only. It has no executor and
 * does not create a thread; completion is performed by the producer and the
 * consumer only waits on the shared state.
 */
template <typename T>
class Future {
public:
    Future() = default;
    Future(const Future&) = delete;
    Future& operator=(const Future&) = delete;
    Future(Future&&) noexcept = default;
    Future& operator=(Future&&) noexcept = default;

    bool Valid() const { return static_cast<bool>(m_state); }

    bool IsReady() const {
        if (!m_state) return false;
        std::lock_guard<std::mutex> lock(m_state->mutex);
        return m_state->ready || m_state->abandoned;
    }

    void Wait() const {
        if (!m_state) return;
        std::unique_lock<std::mutex> lock(m_state->mutex);
        m_state->condition.wait(lock, [this] {
            return m_state->ready || m_state->abandoned;
        });
    }

    template <typename Rep, typename Period>
    bool WaitFor(const std::chrono::duration<Rep, Period>& timeout) const {
        if (!m_state) return false;
        std::unique_lock<std::mutex> lock(m_state->mutex);
        return m_state->condition.wait_for(lock, timeout, [this] {
            return m_state->ready || m_state->abandoned;
        });
    }

    T Get() {
        if (!m_state) throw std::runtime_error("Future has no state");
        std::unique_lock<std::mutex> lock(m_state->mutex);
        m_state->condition.wait(lock, [this] {
            return m_state->ready || m_state->abandoned;
        });
        if (m_state->consumed) {
            throw std::runtime_error("Future value has already been consumed");
        }
        if (m_state->abandoned && !m_state->ready) {
            throw std::runtime_error("Promise was abandoned before setting a value");
        }
        m_state->consumed = true;
        T value = std::move(*m_state->value);
        m_state->value.reset();
        return value;
    }

private:
    struct State {
        mutable std::mutex mutex;
        std::condition_variable condition;
        bool ready = false;
        bool abandoned = false;
        bool consumed = false;
        bool futureCreated = false;
        std::unique_ptr<T> value;
    };

    explicit Future(std::shared_ptr<State> state) : m_state(std::move(state)) {}

    std::shared_ptr<State> m_state;

    friend class Promise<T>;
};

/** Producer side of a Future shared state. */
template <typename T>
class Promise {
public:
    Promise() : m_state(std::make_shared<typename Future<T>::State>()) {}
    ~Promise() { Abandon(); }

    Promise(const Promise&) = delete;
    Promise& operator=(const Promise&) = delete;
    Promise(Promise&&) noexcept = default;
    Promise& operator=(Promise&& other) noexcept {
        if (this != &other) {
            Abandon();
            m_state = std::move(other.m_state);
        }
        return *this;
    }

    Future<T> GetFuture() {
        if (!m_state) throw std::runtime_error("Promise has no state");
        std::lock_guard<std::mutex> lock(m_state->mutex);
        if (m_state->futureCreated) {
            throw std::runtime_error("Promise future has already been created");
        }
        m_state->futureCreated = true;
        return Future<T>(m_state);
    }

    bool TrySetValue(T value) {
        if (!m_state) return false;
        {
            std::lock_guard<std::mutex> lock(m_state->mutex);
            if (m_state->ready || m_state->abandoned) return false;
            m_state->value.reset(new T(std::move(value)));
            m_state->ready = true;
        }
        m_state->condition.notify_all();
        return true;
    }

    void SetValue(T value) {
        if (!TrySetValue(std::move(value))) {
            throw std::runtime_error("Promise value has already been set");
        }
    }

private:
    void Abandon() {
        if (!m_state) return;
        {
            std::lock_guard<std::mutex> lock(m_state->mutex);
            if (m_state->ready || m_state->abandoned) return;
            m_state->abandoned = true;
        }
        m_state->condition.notify_all();
    }

    std::shared_ptr<typename Future<T>::State> m_state;
};

} // namespace aifootball
