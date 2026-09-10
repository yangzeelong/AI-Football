include(FetchContent)
FetchContent_Declare(
    GoogleBenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.0
)
FetchContent_MakeAvailable(GoogleBenchmark)
