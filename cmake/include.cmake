include("cmake/3rdparty/spdlog.cmake")
include("cmake/3rdparty/yaml-cpp.cmake")

if(WITH_TESTING)
    include("cmake/3rdparty/gtest.cmake")
endif()

if(WITH_BENCHMARK)
    include("cmake/3rdparty/gbenchmark.cmake")
endif()

include("cmake/Summary.cmake")
