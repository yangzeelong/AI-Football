cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(PipelineDAG VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS ON)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(INSTALL_ROOT ${PROJECT_ROOT}/install)

### Define options
option(WITH_TESTING "Enable testing" ON)
option(WITH_BENCHMARK "Enable benchmark" ON)

if(NOT DEFINED CMAKE_BUILD_TYPE)
  message(STATUS "No build type specified, defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(CMAKE_INSTALL_PREFIX STREQUAL "" OR CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
  set(CMAKE_INSTALL_PREFIX ${INSTALL_ROOT})
endif()

### include custom cmake
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include("cmake/include.cmake")
include("cmake/Utils.cmake")


# Define global variables.
set_property(GLOBAL PROPERTY GLOBAL_TEST_SOURCES "")
message(STATUS "Initialized global test sources list.")

add_subdirectory(src)
add_subdirectory(examples)

# 
if(WITH_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(WITH_BENCHMARK)
  add_subdirectory(benchmarks)
endif()
