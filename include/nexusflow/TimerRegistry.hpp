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

struct TimerPrintPolicy {
    bool printEverySample = false;
    uint64_t everySamples = 0;
    uint64_t everyMs = 0;

    static TimerPrintPolicy EverySample();
    static TimerPrintPolicy EveryN(uint64_t samples);
    static TimerPrintPolicy EveryMs(uint64_t ms);
    static TimerPrintPolicy Every(uint64_t samples, uint64_t ms);
};

class TimerRegistry {
public:
    class ScopedTimer;

    static TimerRegistry& Instance();

    // Start and end one sample on the calling thread. Nested samples with
    // the same name are supported by a thread-local stack. The start call
    // configures the print policy; work units are counted by IncrementAverage
    // or by Scope.
    void StartAverage(const std::string& name, TimerPrintPolicy policy);
    void StartAverageMs(const std::string& name, uint64_t ms);
    void StartAverageN(const std::string& name, uint64_t samples);
    void StartAverageEvery(const std::string& name, uint64_t samples,
                           uint64_t ms);
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

    ScopedTimer Scope(const std::string& name, uint64_t workUnits = 1);
    ScopedTimer ScopeAverageMs(const std::string& name, uint64_t workUnits,
                               uint64_t ms);
    ScopedTimer ScopeAverageN(const std::string& name, uint64_t workUnits,
                              uint64_t samples);
    ScopedTimer ScopeAverage(const std::string& name, uint64_t workUnits,
                             uint64_t samples, uint64_t ms);

private:
    struct Aggregate {
        uint64_t samples = 0;
        uint64_t workUnits = 0;
        double totalMs = 0.0;
        double minMs = 0.0;
        double maxMs = 0.0;
        uint64_t samplesSincePrint = 0;
        std::chrono::steady_clock::time_point lastPrint;
    };

    struct CompletedSample {
        std::string name;
        double elapsedMs = 0.0;
        uint64_t workUnits = 0;
        bool printSample = false;
        bool printAverage = false;
        TimerStats stats;
    };

    CompletedSample AddSampleWithPolicy(
        const std::string& name, double elapsedMs, uint64_t workUnits,
        const TimerPrintPolicy& policy,
        std::chrono::steady_clock::time_point started,
        std::chrono::steady_clock::time_point ended);
    void PrintStats(const TimerStats& stats) const;
    void PrintSample(const CompletedSample& sample) const;

    mutable std::mutex m_mutex;
    std::map<std::string, Aggregate> m_stats;
    mutable std::mutex m_reportMutex;
    std::chrono::steady_clock::time_point m_lastReport;
};

class TimerRegistry::ScopedTimer {
public:
    ScopedTimer(TimerRegistry& registry, std::string name,
                uint64_t workUnits, TimerPrintPolicy policy);
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

#define TIMER_SCOPE(name) \
    auto NEXUSFLOW_TIMER_CONCAT(nexusflowTimer_, __LINE__) = \
        ::nexusflow::TimerRegistry::Instance().Scope((name))

#define TIMER_SCOPE_UNITS(name, workUnits) \
    auto NEXUSFLOW_TIMER_CONCAT(nexusflowTimer_, __LINE__) = \
        ::nexusflow::TimerRegistry::Instance().Scope((name), (workUnits))

#define TIMER_START_AVERAGE_MS(name, ms) \
    ::nexusflow::TimerRegistry::Instance().StartAverageMs((name), (ms))

#define TIMER_START_AVERAGE_N(name, samples) \
    ::nexusflow::TimerRegistry::Instance().StartAverageN((name), (samples))

#define TIMER_START_AVERAGE(name, samples, ms) \
    ::nexusflow::TimerRegistry::Instance().StartAverageEvery((name), (samples), (ms))

#define TIME_START_AVERAGE(name, ms) \
    ::nexusflow::TimerRegistry::Instance().StartAverageMs((name), (ms))

#define TIME_START_AVERAGE_MS(name, ms) \
    ::nexusflow::TimerRegistry::Instance().StartAverageMs((name), (ms))

#define TIMER_INCREMENT_AVERAGE(name, workUnits) \
    ::nexusflow::TimerRegistry::Instance().IncrementAverage((name), (workUnits))

#define TIMER_END_AVERAGE(name) \
    ::nexusflow::TimerRegistry::Instance().EndAverage((name))

#define TIMER_SCOPE_AVERAGE_MS(name, workUnits, ms) \
    auto NEXUSFLOW_TIMER_CONCAT(nexusflowTimer_, __LINE__) = \
        ::nexusflow::TimerRegistry::Instance().ScopeAverageMs((name), (workUnits), (ms))

#define TIMER_SCOPE_AVERAGE_N(name, workUnits, samples) \
    auto NEXUSFLOW_TIMER_CONCAT(nexusflowTimer_, __LINE__) = \
        ::nexusflow::TimerRegistry::Instance().ScopeAverageN((name), (workUnits), (samples))

#define TIMER_SCOPE_AVERAGE(name, workUnits, samples, ms) \
    auto NEXUSFLOW_TIMER_CONCAT(nexusflowTimer_, __LINE__) = \
        ::nexusflow::TimerRegistry::Instance().ScopeAverage((name), (workUnits), (samples), (ms))

#define TIMER_SCOPE_AVERAGE_EVERY(name, workUnits, ms) \
    auto NEXUSFLOW_TIMER_CONCAT(nexusflowTimer_, __LINE__) = \
        ::nexusflow::TimerRegistry::Instance().ScopeAverageMs((name), (workUnits), (ms))

#endif
