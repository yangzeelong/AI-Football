#include "nexusflow/TimerRegistry.hpp"

#include "gtest/gtest.h"

#include <chrono>
#include <thread>
#include <vector>

using nexusflow::TimerRegistry;
using nexusflow::TimerStats;

TEST(TimerRegistryTest, AggregatesBatchAndItemMetrics) {
    auto& registry = TimerRegistry::Instance();
    registry.Reset();

    registry.AddSample("test.aggregate", 10.0, 2);
    registry.AddSample("test.aggregate", 20.0, 3);

    TimerStats stats;
    ASSERT_TRUE(registry.GetStats("test.aggregate", stats));
    EXPECT_EQ(stats.samples, 2U);
    EXPECT_EQ(stats.workUnits, 5U);
    EXPECT_DOUBLE_EQ(stats.totalMs, 30.0);
    EXPECT_DOUBLE_EQ(stats.AvgBatchMs(), 15.0);
    EXPECT_DOUBLE_EQ(stats.AvgItemMs(), 6.0);
    EXPECT_DOUBLE_EQ(stats.BatchQps(), 200.0 / 3.0);
    EXPECT_DOUBLE_EQ(stats.Qps(), 500.0 / 3.0);
}

TEST(TimerRegistryTest, SupportsScopedAndIncrementedSamples) {
    auto& registry = TimerRegistry::Instance();
    registry.Reset();

    registry.StartAverageMs("test.scoped", 1000);
    registry.IncrementAverage("test.scoped", 5);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const double elapsedMs = registry.EndAverage("test.scoped");

    EXPECT_GT(elapsedMs, 0.0);
    TimerStats stats;
    ASSERT_TRUE(registry.GetStats("test.scoped", stats));
    EXPECT_EQ(stats.samples, 1U);
    EXPECT_EQ(stats.workUnits, 5U);
    EXPECT_DOUBLE_EQ(stats.totalMs, elapsedMs);

    {
        auto timer = registry.ScopeAverageMs("test.raii", 4, 1000);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(registry.GetStats("test.raii", stats));
    EXPECT_EQ(stats.samples, 1U);
    EXPECT_EQ(stats.workUnits, 4U);
    EXPECT_GT(stats.totalMs, 0.0);

    {
        auto timer = registry.Scope("test.every-sample");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(registry.GetStats("test.every-sample", stats));
    EXPECT_EQ(stats.samples, 1U);
    EXPECT_EQ(stats.workUnits, 1U);

    registry.StartAverageN("test.every-n", 2);
    registry.IncrementAverage("test.every-n", 1);
    registry.EndAverage("test.every-n");
    registry.StartAverageN("test.every-n", 2);
    registry.IncrementAverage("test.every-n", 1);
    registry.EndAverage("test.every-n");
    ASSERT_TRUE(registry.GetStats("test.every-n", stats));
    EXPECT_EQ(stats.samples, 2U);
    EXPECT_EQ(stats.workUnits, 2U);
}

TEST(TimerRegistryTest, AggregationIsThreadSafe) {
    auto& registry = TimerRegistry::Instance();
    registry.Reset();

    constexpr int kThreadCount = 4;
    constexpr int kSamplesPerThread = 100;
    std::vector<std::thread> workers;
    workers.reserve(kThreadCount);
    for (int thread = 0; thread < kThreadCount; ++thread) {
        workers.emplace_back([&registry]() {
            for (int sample = 0; sample < kSamplesPerThread; ++sample) {
                registry.AddSample("test.concurrent", 2.0, 3);
            }
        });
    }
    for (auto& worker : workers) worker.join();

    TimerStats stats;
    ASSERT_TRUE(registry.GetStats("test.concurrent", stats));
    EXPECT_EQ(stats.samples, static_cast<uint64_t>(kThreadCount * kSamplesPerThread));
    EXPECT_EQ(stats.workUnits, static_cast<uint64_t>(
        kThreadCount * kSamplesPerThread * 3));
    EXPECT_DOUBLE_EQ(stats.totalMs,
                     static_cast<double>(kThreadCount * kSamplesPerThread * 2));
}
