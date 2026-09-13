add_executable(aifootball_demo main.cpp)
find_path(AIFOOTBALL_DEMO_FFMPEG_INCLUDE_DIR libavcodec/avcodec.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include)
find_library(AIFOOTBALL_DEMO_AVFORMAT_LIB avformat
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
find_library(AIFOOTBALL_DEMO_AVCODEC_LIB avcodec
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
find_library(AIFOOTBALL_DEMO_AVUTIL_LIB avutil
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
find_library(AIFOOTBALL_DEMO_SWSCALE_LIB swscale
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
target_include_directories(aifootball_demo PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/sdk/aifootball/include"
    "${AIFOOTBALL_DEMO_FFMPEG_INCLUDE_DIR}")
target_link_libraries(aifootball_demo PRIVATE
    aifootball_sdk
    ${AIFOOTBALL_DEMO_AVFORMAT_LIB}
    ${AIFOOTBALL_DEMO_AVCODEC_LIB}
    ${AIFOOTBALL_DEMO_AVUTIL_LIB}
    ${AIFOOTBALL_DEMO_SWSCALE_LIB})
