add_executable(aifootball_app
    main.cpp
    AppConfig.cpp
    VideoReader.cpp
    JsonlWriter.cpp
    VideoRenderer.cpp)
find_path(AIFOOTBALL_APP_FFMPEG_INCLUDE_DIR libavcodec/avcodec.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include)
find_library(AIFOOTBALL_APP_AVFORMAT_LIB avformat
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
find_library(AIFOOTBALL_APP_AVCODEC_LIB avcodec
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
find_library(AIFOOTBALL_APP_AVUTIL_LIB avutil
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
find_library(AIFOOTBALL_APP_SWSCALE_LIB swscale
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
target_include_directories(aifootball_app PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/sdk/aifootball/include"
    "${AIFOOTBALL_APP_FFMPEG_INCLUDE_DIR}")
target_link_libraries(aifootball_app PRIVATE
    aifootball_sdk
    JsonCpp::JsonCpp
    ${AIFOOTBALL_APP_AVFORMAT_LIB}
    ${AIFOOTBALL_APP_AVCODEC_LIB}
    ${AIFOOTBALL_APP_AVUTIL_LIB}
    ${AIFOOTBALL_APP_SWSCALE_LIB})
