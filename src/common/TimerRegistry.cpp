#include "nexusflow/TimerRegistry.hpp"

#include <algorithm>

#include "nexusflow/Logging.hpp"

namespace nexusflow {
namespace {

struct ActiveTimer {
    std::chrono::steady_clock::time_point started;
    uint64_t printIntervalMs = 1000;
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

TimerRegistry& TimerRegistry::Instance() {
    static TimerRegistry registry;
    return registry;
}

void TimerRegistry::StartAverage(const std::string& name,
                                 uint64_t printIntervalMs) {
    const auto now = std::chrono::steady_clock::now();
    g_activeTimers[name].push_back(
        ActiveTimer{now, printIntervalMs, 0});

    if (printIntervalMs > 0) {
        std::lock_guard<std::mutex> lock(m_reportMutex);
        if (m_lastReport.time_since_epoch().count() == 0) {
            m_lastReport = now;
        }
    }
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

    const double elapsedMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - active.started).count();
    AddSample(name, elapsedMs, active.workUnits);
    PrintIfDue(active.printIntervalMs);
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
        LOG_INFO(
            "Timer: name='{}' samples={} work_units={} "
            "avg_batch_ms={:.3f} avg_item_ms={:.3f} "
            "batch_qps={:.3f} qps={:.3f} min_ms={:.3f} max_ms={:.3f}",
            stats.name, stats.samples, stats.workUnits,
            stats.AvgBatchMs(), stats.AvgItemMs(), stats.BatchQps(),
            stats.Qps(), stats.minMs, stats.maxMs);
    }
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
                                                uint64_t workUnits,
                                                uint64_t printIntervalMs) {
    return ScopedTimer(*this, name, workUnits, printIntervalMs);
}

TimerRegistry::ScopedTimer::ScopedTimer(TimerRegistry& registry,
                                        std::string name,
                                        uint64_t workUnits,
                                        uint64_t printIntervalMs)
    : m_registry(&registry), m_name(std::move(name)), m_active(true) {
    m_registry->StartAverage(m_name, printIntervalMs);
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
