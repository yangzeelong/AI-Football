add_subdirectory(pipeline)

if(TARGET aifootball_sdk)
    find_path(AIFOOTBALL_BENCHMARK_FFMPEG_INCLUDE_DIR libavcodec/avcodec.h
        PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include)
    find_library(AIFOOTBALL_BENCHMARK_AVFORMAT_LIB avformat
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
    find_library(AIFOOTBALL_BENCHMARK_AVCODEC_LIB avcodec
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
    find_library(AIFOOTBALL_BENCHMARK_AVUTIL_LIB avutil
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
    find_library(AIFOOTBALL_BENCHMARK_SWSCALE_LIB swscale
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)

    add_executable(pose_benchmark
        pose_benchmark.cpp
        ${PROJECT_SOURCE_DIR}/examples/aifootball_demo/VideoReader.cpp)
    target_include_directories(pose_benchmark PRIVATE
        "${PROJECT_SOURCE_DIR}/sdk/aifootball/src"
        "${PROJECT_SOURCE_DIR}/examples/aifootball_demo"
        "${AIFOOTBALL_BENCHMARK_FFMPEG_INCLUDE_DIR}")
    target_link_libraries(pose_benchmark PRIVATE
        aifootball_sdk
        ${AIFOOTBALL_BENCHMARK_AVFORMAT_LIB}
        ${AIFOOTBALL_BENCHMARK_AVCODEC_LIB}
        ${AIFOOTBALL_BENCHMARK_AVUTIL_LIB}
        ${AIFOOTBALL_BENCHMARK_SWSCALE_LIB})
endif()

# Collect all benchmark source files
file(GLOB_RECURSE ALL_BENCHMARK_SOURCES "Benchmark*.cpp")
add_executable(nexusflow_benchmarks benchmark_main.cpp ${ALL_BENCHMARK_SOURCES})
target_link_libraries(nexusflow_benchmarks PRIVATE nexusflow benchmark::benchmark)

add_test(NAME nexusflow_benchmarks COMMAND nexusflow_benchmarks)
