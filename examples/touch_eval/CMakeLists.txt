# Ball-touch detection and its accuracy evaluation.
#
# The core is a dependency-free static library so the offline evaluator can be
# rebuilt and rerun in seconds without CUDA, TensorRT or ffmpeg. The demo app
# links the same library, which keeps the online and offline semantics
# identical.
add_library(touch_eval_core STATIC
    src/TouchDetector.cpp
    src/TouchGroundTruth.cpp
    src/TouchMetrics.cpp
    src/ObservationJsonlReader.cpp)

target_include_directories(touch_eval_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(touch_eval_core PUBLIC cxx_std_14)
target_link_libraries(touch_eval_core PUBLIC JsonCpp::JsonCpp)

# Offline evaluation CLI: pure CPU, reads observations.jsonl and the label.
add_executable(touch_eval main.cpp)
target_include_directories(touch_eval PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(touch_eval PRIVATE touch_eval_core)

# Unit tests live next to the code that they cover. examples/ is configured
# after tests/, so the test executable already exists and can be extended here.
if(WITH_TESTING AND TARGET nexusflow_tests)
    target_sources(nexusflow_tests PRIVATE tests/TouchEvalTest.cpp)
    target_link_libraries(nexusflow_tests PRIVATE touch_eval_core)
endif()
