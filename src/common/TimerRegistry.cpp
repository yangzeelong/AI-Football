#include "nexusflow/TimerRegistry.hpp"

#include <algorithm>

#include "nexusflow/Logging.hpp"

namespace nexusflow {
namespace {

struct ActiveTimer {
    std::chrono::steady_clock::time_point started;
    TimerPrintPolicy policy;
    uint64_t workUnits = 0;
};

thread_local std::map<std::string, std::vector<ActiveTimer>> g_activeTimers;

} // namespace

double TimerStats::AvgBatchMs() const {
    return samples == 0 ? 0.0 : totalMs / static_cast<double>(samples);
}

double TimerStats::AvgItemMs() const {
    return workUnits == 0 ? 0.0 : totalMs / static_cast<double>(workUnits);
}

double TimerStats::BatchQps() const {
    const double seconds = totalMs / 1000.0;
    return seconds <= 0.0 ? 0.0 : static_cast<double>(samples) / seconds;
}

double TimerStats::Qps() const {
    const double seconds = totalMs / 1000.0;
    return seconds <= 0.0 ? 0.0 : static_cast<double>(workUnits) / seconds;
}

TimerPrintPolicy TimerPrintPolicy::EverySample() {
    TimerPrintPolicy policy;
    policy.printEverySample = true;
    return policy;
}

TimerPrintPolicy TimerPrintPolicy::EveryN(uint64_t samples) {
    TimerPrintPolicy policy;
    policy.everySamples = samples;
    return policy;
}

TimerPrintPolicy TimerPrintPolicy::EveryMs(uint64_t ms) {
    TimerPrintPolicy policy;
    policy.everyMs = ms;
    return policy;
}

TimerPrintPolicy TimerPrintPolicy::Every(uint64_t samples, uint64_t ms) {
    TimerPrintPolicy policy;
    policy.everySamples = samples;
    policy.everyMs = ms;
    return policy;
}

TimerRegistry& TimerRegistry::Instance() {
    static TimerRegistry registry;
    return registry;
}

void TimerRegistry::StartAverage(const std::string& name,
                                 TimerPrintPolicy policy) {
    g_activeTimers[name].push_back(
        ActiveTimer{std::chrono::steady_clock::now(), policy, 0});
}

void TimerRegistry::StartAverageMs(const std::string& name, uint64_t ms) {
    StartAverage(name, TimerPrintPolicy::EveryMs(ms));
}

void TimerRegistry::StartAverageN(const std::string& name, uint64_t samples) {
    StartAverage(name, TimerPrintPolicy::EveryN(samples));
}

void TimerRegistry::StartAverageEvery(const std::string& name,
                                      uint64_t samples, uint64_t ms) {
    StartAverage(name, TimerPrintPolicy::Every(samples, ms));
}

void TimerRegistry::IncrementAverage(const std::string& name,
                                     uint64_t workUnits) {
    auto it = g_activeTimers.find(name);
    if (it == g_activeTimers.end() || it->second.empty()) return;
    it->second.back().workUnits += workUnits;
}

double TimerRegistry::EndAverage(const std::string& name) {
    auto it = g_activeTimers.find(name);
    if (it == g_activeTimers.end() || it->second.empty()) return 0.0;

    const ActiveTimer active = it->second.back();
    it->second.pop_back();
    if (it->second.empty()) g_activeTimers.erase(it);

    const auto ended = std::chrono::steady_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(
        ended - active.started).count();
    const CompletedSample completed = AddSampleWithPolicy(
        name, elapsedMs, active.workUnits, active.policy, active.started,
        ended);
    if (completed.printSample) PrintSample(completed);
    if (completed.printAverage) PrintStats(completed.stats);
    return elapsedMs;
}

void TimerRegistry::AddSample(const std::string& name, double elapsedMs,
                              uint64_t workUnits) {
    elapsedMs = std::max(0.0, elapsedMs);
    std::lock_guard<std::mutex> lock(m_mutex);
    Aggregate& aggregate = m_stats[name];
    ++aggregate.samples;
    aggregate.workUnits += workUnits;
    aggregate.totalMs += elapsedMs;
    if (aggregate.samples == 1) {
        aggregate.minMs = elapsedMs;
        aggregate.maxMs = elapsedMs;
    } else {
        aggregate.minMs = std::min(aggregate.minMs, elapsedMs);
        aggregate.maxMs = std::max(aggregate.maxMs, elapsedMs);
    }
}

TimerRegistry::CompletedSample TimerRegistry::AddSampleWithPolicy(
    const std::string& name, double elapsedMs, uint64_t workUnits,
    const TimerPrintPolicy& policy,
    std::chrono::steady_clock::time_point started,
    std::chrono::steady_clock::time_point ended) {
    elapsedMs = std::max(0.0, elapsedMs);

    CompletedSample completed;
    completed.name = name;
    completed.elapsedMs = elapsedMs;
    completed.workUnits = workUnits;
    completed.printSample = policy.printEverySample;

    std::lock_guard<std::mutex> lock(m_mutex);
    Aggregate& aggregate = m_stats[name];
    ++aggregate.samples;
    ++aggregate.samplesSincePrint;
    aggregate.workUnits += workUnits;
    aggregate.totalMs += elapsedMs;
    if (aggregate.samples == 1) {
        aggregate.minMs = elapsedMs;
        aggregate.maxMs = elapsedMs;
    } else {
        aggregate.minMs = std::min(aggregate.minMs, elapsedMs);
        aggregate.maxMs = std::max(aggregate.maxMs, elapsedMs);
    }

    TimerStats stats;
    stats.name = name;
    stats.samples = aggregate.samples;
    stats.workUnits = aggregate.workUnits;
    stats.totalMs = aggregate.totalMs;
    stats.minMs = aggregate.minMs;
    stats.maxMs = aggregate.maxMs;

    const bool dueBySamples = policy.everySamples > 0 &&
        aggregate.samplesSincePrint >= policy.everySamples;
    bool dueByTime = false;
    if (policy.everyMs > 0) {
        if (aggregate.lastPrint.time_since_epoch().count() == 0) {
            aggregate.lastPrint = started;
        }
        dueByTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            ended - aggregate.lastPrint).count() >=
            static_cast<int64_t>(policy.everyMs);
    }

    if (dueBySamples || dueByTime) {
        aggregate.samplesSincePrint = 0;
        aggregate.lastPrint = ended;
        completed.printAverage = true;
        completed.stats = stats;
    }

    return completed;
}

bool TimerRegistry::GetStats(const std::string& name, TimerStats& stats) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto it = m_stats.find(name);
    if (it == m_stats.end()) return false;

    stats.name = name;
    stats.samples = it->second.samples;
    stats.workUnits = it->second.workUnits;
    stats.totalMs = it->second.totalMs;
    stats.minMs = it->second.minMs;
    stats.maxMs = it->second.maxMs;
    return true;
}

std::vector<TimerStats> TimerRegistry::Snapshot() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<TimerStats> snapshot;
    snapshot.reserve(m_stats.size());
    for (const auto& entry : m_stats) {
        TimerStats stats;
        stats.name = entry.first;
        stats.samples = entry.second.samples;
        stats.workUnits = entry.second.workUnits;
        stats.totalMs = entry.second.totalMs;
        stats.minMs = entry.second.minMs;
        stats.maxMs = entry.second.maxMs;
        snapshot.push_back(std::move(stats));
    }
    return snapshot;
}

void TimerRegistry::Print() const {
    for (const auto& stats : Snapshot()) {
        PrintStats(stats);
    }
}

void TimerRegistry::PrintStats(const TimerStats& stats) const {
    LOG_INFO(
        "Timer: name='{}' samples={} work_units={} "
        "avg_batch_ms={:.3f} avg_item_ms={:.3f} "
        "batch_qps={:.3f} qps={:.3f} min_ms={:.3f} max_ms={:.3f}",
        stats.name, stats.samples, stats.workUnits,
        stats.AvgBatchMs(), stats.AvgItemMs(), stats.BatchQps(),
        stats.Qps(), stats.minMs, stats.maxMs);
}

void TimerRegistry::PrintSample(const CompletedSample& sample) const {
    const double itemMs = sample.workUnits == 0 ? 0.0 :
        sample.elapsedMs / static_cast<double>(sample.workUnits);
    LOG_INFO(
        "TimerScope: name='{}' elapsed_ms={:.3f} work_units={} item_ms={:.3f}",
        sample.name, sample.elapsedMs, sample.workUnits, itemMs);
}

bool TimerRegistry::PrintIfDue(uint64_t intervalMs) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(m_reportMutex);
    if (m_lastReport.time_since_epoch().count() != 0 &&
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - m_lastReport).count() < static_cast<int64_t>(intervalMs)) {
        return false;
    }
    m_lastReport = now;
    Print();
    return true;
}

void TimerRegistry::Reset() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stats.clear();
    }
    std::lock_guard<std::mutex> lock(m_reportMutex);
    m_lastReport = std::chrono::steady_clock::time_point();
}

TimerRegistry::ScopedTimer TimerRegistry::Scope(const std::string& name,
                                                uint64_t workUnits) {
    return ScopedTimer(*this, name, workUnits,
                       TimerPrintPolicy::EverySample());
}

TimerRegistry::ScopedTimer TimerRegistry::ScopeAverageMs(
    const std::string& name, uint64_t workUnits, uint64_t ms) {
    return ScopedTimer(*this, name, workUnits, TimerPrintPolicy::EveryMs(ms));
}

TimerRegistry::ScopedTimer TimerRegistry::ScopeAverageN(
    const std::string& name, uint64_t workUnits, uint64_t samples) {
    return ScopedTimer(*this, name, workUnits,
                       TimerPrintPolicy::EveryN(samples));
}

TimerRegistry::ScopedTimer TimerRegistry::ScopeAverage(
    const std::string& name, uint64_t workUnits, uint64_t samples,
    uint64_t ms) {
    return ScopedTimer(*this, name, workUnits,
                       TimerPrintPolicy::Every(samples, ms));
}

TimerRegistry::ScopedTimer::ScopedTimer(TimerRegistry& registry,
                                        std::string name,
                                        uint64_t workUnits,
                                        TimerPrintPolicy policy)
    : m_registry(&registry), m_name(std::move(name)), m_active(true) {
    m_registry->StartAverage(m_name, policy);
    m_registry->IncrementAverage(m_name, workUnits);
}

TimerRegistry::ScopedTimer::~ScopedTimer() {
    if (m_active && m_registry != nullptr) {
        m_registry->EndAverage(m_name);
    }
}

TimerRegistry::ScopedTimer::ScopedTimer(ScopedTimer&& other) noexcept
    : m_registry(other.m_registry),
      m_name(std::move(other.m_name)),
      m_active(other.m_active) {
    other.m_registry = nullptr;
    other.m_active = false;
}

TimerRegistry::ScopedTimer& TimerRegistry::ScopedTimer::operator=(
    ScopedTimer&& other) noexcept {
    if (this == &other) return *this;
    if (m_active && m_registry != nullptr) {
        m_registry->EndAverage(m_name);
    }
    m_registry = other.m_registry;
    m_name = std::move(other.m_name);
    m_active = other.m_active;
    other.m_registry = nullptr;
    other.m_active = false;
    return *this;
}

} // namespace nexusflow
