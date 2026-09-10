#include <gtest/gtest.h>

/**
 * @brief The main entry point for the test runner executable.
 *
 * This file's purpose is to initialize the Google Test framework and run all
 * tests that have been discovered and compiled into this executable.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @return 0 if all tests pass, 1 otherwise. This return value is crucial
 *         for continuous integration (CI) systems.
 */
int main(int argc, char** argv) {
    // 1. Initializes the Google Test framework. This must be called before RUN_ALL_TESTS().
    //    It parses command-line arguments for Google Test and removes them.
    //    This allows you to control test execution with flags like --gtest_filter=*.
    ::testing::InitGoogleTest(&argc, argv);

    // 2. Runs all tests in the current project.
    //    It's a macro that automatically discovers and runs all tests defined
    //    with TEST() and TEST_F() macros.
    return RUN_ALL_TESTS();
}