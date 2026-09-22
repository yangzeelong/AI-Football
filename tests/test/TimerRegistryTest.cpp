#include "nexusflow/TimerRegistry.hpp"

#include "gtest/gtest.h"

#include <chrono>
#include <thread>
#include <vector>

using nexusflow::TimerPrintPolicy;
using nexusflow::TimerRegistry;
using nexusflow::TimerStats;

TEST(TimerRegistryTest, CountsOneItemPerMeasurementByDefault) {
    auto& registry = TimerRegistry::Instance();
    registry.Reset();

    registry.AddItems("test.aggregate", 10.0);
    registry.AddItems("test.aggregate", 20.0);

    TimerStats stats;
    ASSERT_TRUE(registry.GetStats("test.aggregate", stats));
    EXPECT_EQ(stats.items, 2U);
    EXPECT_DOUBLE_EQ(stats.totalMs, 30.0);
    // Every duration is per item, so a two-item timer reports both samples.
    EXPECT_DOUBLE_EQ(stats.AvgItemMs(), 15.0);
    EXPECT_DOUBLE_EQ(stats.ItemQps(), 200.0 / 3.0);
    EXPECT_DOUBLE_EQ(stats.minMs, 10.0);
    EXPECT_DOUBLE_EQ(stats.maxMs, 20.0);
    // Nearest rank over both items: 0.95 * (2 - 1) rounds to the maximum.
    EXPECT_DOUBLE_EQ(stats.p95Ms, 20.0);
}

TEST(TimerRegistryTest, DeclaredItemsNormalizeEveryDuration) {
    auto& registry = TimerRegistry::Instance();
    registry.Reset();

    // One measurement covering four items: the reported numbers describe a
    // single item, both in the average and in the extremes.
    registry.AddItems("test.batch", 40.0, 4);
    registry.AddItems("test.batch", 80.0, 8);

    TimerStats stats;
    ASSERT_TRUE(registry.GetStats("test.batch", stats));
    EXPECT_EQ(stats.items, 12U);
    EXPECT_DOUBLE_EQ(stats.totalMs, 120.0);
    EXPECT_DOUBLE_EQ(stats.AvgItemMs(), 10.0);
    EXPECT_DOUBLE_EQ(stats.minMs, 10.0);
    EXPECT_DOUBLE_EQ(stats.maxMs, 10.0);
    EXPECT_DOUBLE_EQ(stats.p95Ms, 10.0);
}

TEST(TimerRegistryTest, SupportsScopedAndIncrementedItems) {
    auto& registry = TimerRegistry::Instance();
    registry.Reset();

    registry.StartAverage("test.scoped", TimerPrintPolicy::EveryMs(1000));
    registry.IncrementAverage("test.scoped", 5);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const double elapsedMs = registry.EndAverage("test.scoped");

    EXPECT_GT(elapsedMs, 0.0);
    TimerStats stats;
    ASSERT_TRUE(registry.GetStats("test.scoped", stats));
    EXPECT_EQ(stats.items, 5U);
    EXPECT_DOUBLE_EQ(stats.totalMs, elapsedMs);
    EXPECT_DOUBLE_EQ(stats.AvgItemMs(), elapsedMs / 5.0);
    // The single call is normalized by the declared count.
    EXPECT_DOUBLE_EQ(stats.maxMs, elapsedMs / 5.0);

    {
        auto timer = registry.Scope("test.raii", TimerPrintPolicy::EveryMs(1000));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(registry.GetStats("test.raii", stats));
    EXPECT_EQ(stats.items, 1U);
    EXPECT_GT(stats.totalMs, 0.0);

    {
        auto timer =
            registry.Scope("test.every-call", TimerPrintPolicy::EachCall());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(registry.GetStats("test.every-call", stats));
    EXPECT_EQ(stats.items, 1U);

    // Two measurements that declare nothing count two items.
    registry.StartAverage("test.two-calls", TimerPrintPolicy::EveryMs(1000));
    registry.EndAverage("test.two-calls");
    registry.StartAverage("test.two-calls", TimerPrintPolicy::EveryMs(1000));
    registry.EndAverage("test.two-calls");
    ASSERT_TRUE(registry.GetStats("test.two-calls", stats));
    EXPECT_EQ(stats.items, 2U);
}

TEST(TimerRegistryTest, ReportingPoliciesDoNotAffectAggregation) {
    auto& registry = TimerRegistry::Instance();
    registry.Reset();

    // Windowed reporting resets its window every N measurements, and the
    // cumulative numbers have to survive that.
    for (int call = 0; call < 2; ++call) {
        registry.StartAverage("test.interval", TimerPrintPolicy::Interval(2));
        registry.IncrementAverage("test.interval", 4);
        registry.EndAverage("test.interval");
    }
    registry.StartAverage("test.interval", TimerPrintPolicy::EveryCalls(2));
    registry.EndAverage("test.interval");

    TimerStats stats;
    ASSERT_TRUE(registry.GetStats("test.interval", stats));
    EXPECT_EQ(stats.items, 9U);
    EXPECT_GT(stats.totalMs, 0.0);
}

TEST(TimerRegistryTest, AggregationIsThreadSafe) {
    auto& registry = TimerRegistry::Instance();
    registry.Reset();

    constexpr int kThreadCount = 4;
    constexpr int kItemsPerThread = 100;
    std::vector<std::thread> workers;
    workers.reserve(kThreadCount);
    for (int thread = 0; thread < kThreadCount; ++thread) {
        workers.emplace_back([&registry]() {
            for (int item = 0; item < kItemsPerThread; ++item) {
                registry.AddItems("test.concurrent", 2.0, 3);
            }
        });
    }
    for (auto& worker : workers) worker.join();

    TimerStats stats;
    ASSERT_TRUE(registry.GetStats("test.concurrent", stats));
    EXPECT_EQ(stats.items, static_cast<uint64_t>(kThreadCount * kItemsPerThread * 3));
    EXPECT_DOUBLE_EQ(stats.totalMs,
                     static_cast<double>(kThreadCount * kItemsPerThread * 2));
    // 2 ms per measurement spread over 3 items.
    EXPECT_DOUBLE_EQ(stats.AvgItemMs(), 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(stats.maxMs, 2.0 / 3.0);
}
