#include "nexusflow/Message.hpp" // Assumes Message.hpp is in this include path
#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

// Use the nexusflow namespace for convenience.
using namespace nexusflow;

// --- Test Fixture ---
// A test fixture can be used to share setup and teardown code between tests.
class MessageTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

// --- Basic Functionality Tests ---
TEST_F(MessageTest, BasicFunctionality) {
    // 1. Test a default-constructed (empty) message.
    Message empty_msg;
    EXPECT_FALSE(empty_msg.HasData());
    EXPECT_FALSE(empty_msg.HasType<int>());

    // 2. Test construction with data.
    std::string test_string = "hello world";
    Message str_msg = MakeMessage(test_string, "TestSender");

    EXPECT_TRUE(str_msg.HasData());
    EXPECT_TRUE(str_msg.HasType<std::string>());
    EXPECT_FALSE(str_msg.HasType<int>());

    // 3. Test metadata.
    auto meta = str_msg.GetMetaData();
    EXPECT_EQ(meta.sourceName, "TestSender");
    // Message ID and Timestamp will vary, so just check for non-default values.
    EXPECT_NE(meta.messageId, (uint64_t)-1); // A simple check.
    EXPECT_GT(meta.timestamp, 0);
}

// --- Data Access Tests ---
TEST_F(MessageTest, DataAccess) {
    auto msg = MakeMessage(std::vector<int>{10, 20});

    // 1. Test successful access with Borrow<T>.
    const auto& vec_ref = msg.Borrow<std::vector<int>>();
    ASSERT_EQ(vec_ref.size(), 2);
    EXPECT_EQ(vec_ref[0], 10);

    // 2. Test that Borrow<T> throws an exception on type mismatch.
    EXPECT_THROW(msg.Borrow<std::string>(), std::runtime_error);

    // 3. Test successful access with BorrowPtr<T>.
    const auto* vec_ptr = msg.BorrowPtr<std::vector<int>>();
    ASSERT_NE(vec_ptr, nullptr);
    EXPECT_EQ((*vec_ptr)[1], 20);

    // 4. Test that BorrowPtr<T> returns nullptr on type mismatch (no-throw).
    const auto* str_ptr = msg.BorrowPtr<std::string>();
    EXPECT_EQ(str_ptr, nullptr);

    // 5. Test successful mutable access with Mut<T>.
    msg.Mut<std::vector<int>>()[0] = 11;
    EXPECT_EQ(msg.Borrow<std::vector<int>>()[0], 11);

    // 6. Test that Mut<T> throws an exception on type mismatch.
    EXPECT_THROW(msg.Mut<double>(), std::runtime_error);

    // 7. Test successful mutable access with MutPtr<T>.
    auto* mut_vec_ptr = msg.MutPtr<std::vector<int>>();
    ASSERT_NE(mut_vec_ptr, nullptr);
    mut_vec_ptr->push_back(30);
    EXPECT_EQ(msg.Borrow<std::vector<int>>().size(), 3);

    // 8. Test that MutPtr<T> returns nullptr on type mismatch (no-throw).
    auto* mut_double_ptr = msg.MutPtr<double>();
    EXPECT_EQ(mut_double_ptr, nullptr);
}

// --- Core Copy-On-Write (COW) Test ---
TEST_F(MessageTest, CopyOnWrite) {
    // 1. Create an original message and a copy.
    auto original_msg = MakeMessage(std::vector<int>{1, 2, 3});
    auto shared_copy = original_msg; // Cheap copy, data is shared.

    // Verify that data is identical initially.
    EXPECT_EQ(original_msg.Borrow<std::vector<int>>()[1], 2);
    EXPECT_EQ(shared_copy.Borrow<std::vector<int>>()[1], 2);

    // 2. Perform a mutable operation on the copy, which should trigger COW.
    shared_copy.Mut<std::vector<int>>()[1] = 99;

    // 3. Verify that the copy's data has been modified.
    EXPECT_EQ(shared_copy.Borrow<std::vector<int>>()[1], 99);

    // 4. Verify that the original message's data has *not* been modified.
    EXPECT_EQ(original_msg.Borrow<std::vector<int>>()[1], 2);
}

TEST_F(MessageTest, CopyOnWriteWithPtr) {
    // Test COW semantics using the pointer-based MutPtr.
    auto original_msg = MakeMessage(std::string("original"));
    auto shared_copy = original_msg;

    auto* str_ptr = shared_copy.MutPtr<std::string>();
    ASSERT_NE(str_ptr, nullptr);
    *str_ptr += " (modified)";

    EXPECT_EQ(shared_copy.Borrow<std::string>(), "original (modified)");
    EXPECT_EQ(original_msg.Borrow<std::string>(), "original");
}

// --- Clone Test ---
TEST_F(MessageTest, Clone) {
    auto original_msg = MakeMessage(42, "SourceA");
    auto cloned_msg = original_msg.Clone();

    // 1. Verify that the cloned message is not empty and has the correct type.
    ASSERT_TRUE(cloned_msg.HasData());
    ASSERT_TRUE(cloned_msg.HasType<int>());

    // 2. Modify the cloned message.
    cloned_msg.Mut<int>() = 99;

    // 3. Verify the cloned message is modified, but the original is unaffected.
    EXPECT_EQ(cloned_msg.Borrow<int>(), 99);
    EXPECT_EQ(original_msg.Borrow<int>(), 42);

    // 4. Verify that the metadata was also copied.
    EXPECT_EQ(original_msg.GetMetaData().sourceName, cloned_msg.GetMetaData().sourceName);
    EXPECT_EQ(original_msg.GetMetaData().messageId, cloned_msg.GetMetaData().messageId);
}

// --- Ownership and Lifecycle Test ---
TEST_F(MessageTest, OwnershipAndLifecycle) {
    auto original_ptr = std::make_shared<int>(10);
    // Use a weak_ptr to track the lifetime of the underlying data.
    std::weak_ptr<int> tracker = original_ptr;

    {
        Message msg1 = MakeMessage(original_ptr);
        original_ptr.reset(); // Release the external shared_ptr.
        EXPECT_FALSE(tracker.expired()); // The Message should still hold the data.

        {
            Message msg2 = msg1; // Copy construction.
            Message msg3;
            msg3 = msg2; // Copy assignment.
            EXPECT_FALSE(tracker.expired());
        } // msg2 and msg3 are destroyed.

        EXPECT_FALSE(tracker.expired());
    } // msg1 is destroyed.

    // All Message objects are gone, so the data should be released.
    EXPECT_TRUE(tracker.expired());
}

// --- Multithreading Test ---
TEST_F(MessageTest, Multithreading) {
    auto shared_msg = MakeMessage(std::vector<int>{0});
    std::atomic<int> read_sum{0};
    const int num_readers = 10;

    std::vector<std::thread> threads;

    // Create multiple reader threads.
    for (int i = 0; i < num_readers; ++i) {
        threads.emplace_back([&shared_msg, &read_sum]() {
            // Readers continuously read the data for a short period.
            for (int j = 0; j < 100; ++j) {
                // Use BorrowPtr for safe, read-only access.
                if (const auto* vec = shared_msg.BorrowPtr<std::vector<int>>()) {
                    if (!vec->empty()) {
                        read_sum += (*vec)[0];
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    // Create one writer thread.
    threads.emplace_back([&shared_msg]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); // Wait for readers to start.
        // The writer modifies the data, which should trigger COW.
        auto* vec = shared_msg.MutPtr<std::vector<int>>();
        if (vec) {
            (*vec)[0] = 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        // Modify it again.
        vec = shared_msg.MutPtr<std::vector<int>>();
        if (vec) {
            (*vec)[0] = 2;
        }
    });

    for (auto& t : threads) {
        t.join();
    }

    // Verification:
    // The main purpose of this test is to ensure that concurrent access
    // does not lead to data races or crashes (is memory-safe).
    // The exact value of `read_sum` is non-deterministic because readers
    // might access the value before, during, or after modifications.
    // However, the final state of `shared_msg` should be consistent.

    EXPECT_EQ(shared_msg.Borrow<std::vector<int>>()[0], 2);

    // If the test completes without crashing, it's a good indication of thread safety.
    SUCCEED();
}