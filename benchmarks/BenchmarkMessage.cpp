#include <benchmark/benchmark.h>
#include <memory>
#include <vector>

// --- Assume these headers are available ---
#include <nexusflow/Message.hpp> // Our final, optimized Message

// --- Scenario 1: Inheritance-based approach ---
namespace inheritance {
struct MessageBase {
    virtual ~MessageBase() = default;
};
struct IntMessage : MessageBase {
    explicit IntMessage(int v) : value(v) {}
    int value;
};
struct DoubleMessage : MessageBase {
    explicit DoubleMessage(double v) : value(v) {}
    double value;
};
} // namespace inheritance

// --- Scenario 2: Type-erasure approach ---
namespace type_erasure {
using Message = nexusflow::Message;
} // namespace type_erasure

// ========================================================================
// Benchmark 1: Message Creation
// ========================================================================

static void BM_Inheritance_Create(benchmark::State& state) {
    for (auto _ : state) {
        // Repeatedly create a message on the heap.
        auto msg = std::make_shared<inheritance::IntMessage>(42);
        benchmark::DoNotOptimize(msg);
    }
}
BENCHMARK(BM_Inheritance_Create);

static void BM_TypeErasure_Create(benchmark::State& state) {
    for (auto _ : state) {
        // Create a message holding an int. Internally, this also heap-allocates.
        auto msg = type_erasure::Message(42);
        benchmark::DoNotOptimize(msg);
    }
}
BENCHMARK(BM_TypeErasure_Create);

// ========================================================================
// Benchmark 2: Message Broadcasting (Copying)
// ========================================================================

const int NUM_SUBSCRIBERS = 10;

static void BM_Inheritance_Broadcast(benchmark::State& state) {
    auto original_msg = std::make_shared<inheritance::IntMessage>(42);
    std::vector<std::shared_ptr<inheritance::MessageBase>> subscribers(NUM_SUBSCRIBERS);

    for (auto _ : state) {
        for (int i = 0; i < NUM_SUBSCRIBERS; ++i) {
            subscribers[i] = original_msg; // Copy shared_ptr
            benchmark::DoNotOptimize(subscribers[i]);
        }
        benchmark::ClobberMemory(); // Prevent compiler from optimizing away the loop
    }
}
BENCHMARK(BM_Inheritance_Broadcast);

static void BM_TypeErasure_Broadcast(benchmark::State& state) {
    auto original_msg = type_erasure::Message(42);
    std::vector<type_erasure::Message> subscribers(NUM_SUBSCRIBERS);

    for (auto _ : state) {
        for (int i = 0; i < NUM_SUBSCRIBERS; ++i) {
            subscribers[i] = original_msg; // Copy Message (which copies its internal shared_ptr)
            benchmark::DoNotOptimize(subscribers[i]);
        }
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_TypeErasure_Broadcast);

// ========================================================================
// Benchmark 3: Message Processing (Data Access) - THE KEY DIFFERENCE
// ========================================================================

static void BM_Inheritance_Process(benchmark::State& state) {
    // Create a vector of different message types
    std::vector<std::shared_ptr<inheritance::MessageBase>> messages;
    messages.push_back(std::make_shared<inheritance::IntMessage>(1));
    messages.push_back(std::make_shared<inheritance::DoubleMessage>(2.0));

    long processed_sum = 0;
    for (auto _ : state) {
        for (const auto& msg_base : messages) {
            // The costly dynamic_cast chain
            if (auto int_msg = std::dynamic_pointer_cast<inheritance::IntMessage>(msg_base)) {
                processed_sum += int_msg->value;
            } else if (auto dbl_msg = std::dynamic_pointer_cast<inheritance::DoubleMessage>(msg_base)) {
                processed_sum += static_cast<long>(dbl_msg->value);
            }
        }
    }
    benchmark::DoNotOptimize(processed_sum);
}
BENCHMARK(BM_Inheritance_Process);

static void BM_TypeErasure_Process(benchmark::State& state) {
    // Create a vector of different message types
    std::vector<type_erasure::Message> messages;
    messages.emplace_back(1);
    messages.emplace_back(2.0);

    long processed_sum = 0;
    for (auto _ : state) {
        for (const auto& msg : messages) {
            // The fast, type-safe GetData<T>()
            if (auto* val = msg.BorrowPtr<int>()) {
                processed_sum += *val;
            } else if (auto* val = msg.BorrowPtr<double>()) {
                processed_sum += static_cast<long>(*val);
            }
        }
    }
    benchmark::DoNotOptimize(processed_sum);
}
BENCHMARK(BM_TypeErasure_Process);
