#ifndef NEXUSFLOW_TIMER_REGISTRY_HPP
#define NEXUSFLOW_TIMER_REGISTRY_HPP

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace nexusflow {

struct TimerStats {
    std::string name;
    uint64_t samples = 0;
    uint64_t workUnits = 0;
    double totalMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;

    double AvgBatchMs() const;
    double AvgItemMs() const;
    double BatchQps() const;
    double Qps() const;
};

class TimerRegistry {
public:
    class ScopedTimer;

    static TimerRegistry& Instance();

    // Start and end one sample on the calling thread. Nested samples with
    // the same name are supported by a thread-local stack. The start call
    // only configures the periodic print interval; work units are counted by
    // IncrementAverage or by Scope.
    void StartAverage(const std::string& name, uint64_t printIntervalMs = 1000);
    void IncrementAverage(const std::string& name, uint64_t workUnits = 1);
    double EndAverage(const std::string& name);

    // Add a completed sample when the elapsed time was measured elsewhere.
    void AddSample(const std::string& name, double elapsedMs,
                   uint64_t workUnits = 1);

    bool GetStats(const std::string& name, TimerStats& stats) const;
    std::vector<TimerStats> Snapshot() const;

    void Print() const;
    bool PrintIfDue(uint64_t intervalMs = 1000);
    void Reset();

    ScopedTimer Scope(const std::string& name, uint64_t workUnits = 1,
                      uint64_t printIntervalMs = 1000);

private:
    struct Aggregate {
        uint64_t samples = 0;
        uint64_t workUnits = 0;
        double totalMs = 0.0;
        double minMs = 0.0;
        double maxMs = 0.0;
    };

    mutable std::mutex m_mutex;
    std::map<std::string, Aggregate> m_stats;
    mutable std::mutex m_reportMutex;
    std::chrono::steady_clock::time_point m_lastReport;
};

class TimerRegistry::ScopedTimer {
public:
    ScopedTimer(TimerRegistry& registry, std::string name,
                uint64_t workUnits, uint64_t printIntervalMs);
    ~ScopedTimer();

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    ScopedTimer(ScopedTimer&& other) noexcept;
    ScopedTimer& operator=(ScopedTimer&& other) noexcept;

private:
    TimerRegistry* m_registry = nullptr;
    std::string m_name;
    bool m_active = false;
};

} // namespace nexusflow

#define NEXUSFLOW_TIMER_CONCAT_INNER(a, b) a##b
#define NEXUSFLOW_TIMER_CONCAT(a, b) NEXUSFLOW_TIMER_CONCAT_INNER(a, b)

#define TIME_START_AVERAGE(name, printIntervalMs) \
    ::nexusflow::TimerRegistry::Instance().StartAverage((name), (printIntervalMs))

#define TIMER_START_AVERAGE(name, printIntervalMs) \
    ::nexusflow::TimerRegistry::Instance().StartAverage((name), (printIntervalMs))

#define TIMER_INCREMENT_AVERAGE(name, workUnits) \
    ::nexusflow::TimerRegistry::Instance().IncrementAverage((name), (workUnits))

#define TIMER_END_AVERAGE(name) \
    ::nexusflow::TimerRegistry::Instance().EndAverage((name))

#define TIMER_SCOPE_AVERAGE(name, workUnits) \
    auto NEXUSFLOW_TIMER_CONCAT(nexusflowTimer_, __LINE__) = \
        ::nexusflow::TimerRegistry::Instance().Scope((name), (workUnits))

#define TIMER_SCOPE_AVERAGE_EVERY(name, workUnits, printIntervalMs) \
    auto NEXUSFLOW_TIMER_CONCAT(nexusflowTimer_, __LINE__) = \
        ::nexusflow::TimerRegistry::Instance().Scope((name), (workUnits), (printIntervalMs))

#endif
