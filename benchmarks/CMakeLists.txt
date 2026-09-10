add_subdirectory(pipeline)

# Collect all benchmark source files
file(GLOB_RECURSE ALL_BENCHMARK_SOURCES "Benchmark*.cpp")
add_executable(nexusflow_benchmarks benchmark_main.cpp ${ALL_BENCHMARK_SOURCES})
target_link_libraries(nexusflow_benchmarks PRIVATE nexusflow benchmark::benchmark)

add_test(NAME nexusflow_benchmarks COMMAND nexusflow_benchmarks)

