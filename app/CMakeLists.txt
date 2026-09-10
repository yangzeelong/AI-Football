# --- Collect sources (excluding tensorrt/ unless CUDA+TRT are available) ---
file(GLOB_RECURSE ALL_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/**.cpp")

# --- CUDA detection ---
set(CUDA_FOUND FALSE)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    add_definitions(-DWITH_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    # Try modern CMake approach (CMake >= 3.17)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA compiler found via check_language")
        add_definitions(-DWITH_CUDA)
        set(CUDA_FOUND TRUE)
        set(CUDA_LIBRARIES CUDA::cudart)
        set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    else()
        message(WARNING "CUDA not found — TensorRT modules will not be compiled")
    endif()
endif()

# --- TensorRT detection ---
set(TRT_FOUND FALSE)
if(CUDA_FOUND)
    set(TRT_INCLUDE_SEARCH_PATHS
        /usr/include/x86_64-linux-gnu
        /usr/include
        /usr/local/include
        ${CUDA_INCLUDE_DIRS}
    )
    set(TRT_LIB_SEARCH_PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    )
    find_path(TRT_INCLUDE_DIR NvInfer.h PATHS ${TRT_INCLUDE_SEARCH_PATHS})
    find_library(TRT_NVINFER_LIB nvinfer PATHS ${TRT_LIB_SEARCH_PATHS})
    find_library(TRT_NVINFER_PLUGIN_LIB nvinfer_plugin PATHS ${TRT_LIB_SEARCH_PATHS})
    if(TRT_INCLUDE_DIR AND TRT_NVINFER_LIB)
        message(STATUS "TensorRT found:")
        message(STATUS "  Include: ${TRT_INCLUDE_DIR}")
        message(STATUS "  Library: ${TRT_NVINFER_LIB}")
        add_definitions(-DWITH_TENSORRT)
        set(TRT_FOUND TRUE)
    else()
        message(WARNING "TensorRT not found — building without TRT support")
    endif()
endif()

# --- Filter sources: drop inference/TensorRTEngine when TRT is unavailable ---
set(SRC_FILES ${ALL_SRC_FILES})
if(NOT TRT_FOUND)
    list(FILTER SRC_FILES EXCLUDE REGEX ".*/inference/TensorRTEngine\\.cpp$")
endif()

# --- FFmpeg detection (no pkg-config dependency) ---
set(FFMPEG_FOUND FALSE)
set(FFMPEG_LIB_SEARCH_PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib
    /usr/local/lib
    /usr/lib64
)
set(FFMPEG_INCLUDE_SEARCH_PATHS
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/include
)

find_library(AVFORMAT_LIB avformat PATHS ${FFMPEG_LIB_SEARCH_PATHS})
find_library(AVCODEC_LIB  avcodec  PATHS ${FFMPEG_LIB_SEARCH_PATHS})
find_library(AVUTIL_LIB   avutil   PATHS ${FFMPEG_LIB_SEARCH_PATHS})
find_library(SWSCALE_LIB  swscale  PATHS ${FFMPEG_LIB_SEARCH_PATHS})

find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h PATHS ${FFMPEG_INCLUDE_SEARCH_PATHS})
find_path(AVCODEC_INCLUDE_DIR  libavcodec/avcodec.h   PATHS ${FFMPEG_INCLUDE_SEARCH_PATHS})
find_path(AVUTIL_INCLUDE_DIR   libavutil/imgutils.h   PATHS ${FFMPEG_INCLUDE_SEARCH_PATHS})
find_path(SWSCALE_INCLUDE_DIR  libswscale/swscale.h   PATHS ${FFMPEG_INCLUDE_SEARCH_PATHS})

set(FFMPEG_LIBRARIES "")
set(FFMPEG_INCLUDE_DIRS "")
if(AVFORMAT_LIB AND AVCODEC_LIB AND AVUTIL_LIB AND SWSCALE_LIB)
    message(STATUS "FFmpeg found:")
    message(STATUS "  avformat: ${AVFORMAT_LIB}")
    message(STATUS "  avcodec : ${AVCODEC_LIB}")
    message(STATUS "  avutil  : ${AVUTIL_LIB}")
    message(STATUS "  swscale : ${SWSCALE_LIB}")
    list(APPEND FFMPEG_LIBRARIES ${AVFORMAT_LIB} ${AVCODEC_LIB} ${AVUTIL_LIB} ${SWSCALE_LIB})
    foreach(inc ${AVFORMAT_INCLUDE_DIR} ${AVCODEC_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${SWSCALE_INCLUDE_DIR})
        if(inc AND NOT "${inc}" IN_LIST FFMPEG_INCLUDE_DIRS)
            list(APPEND FFMPEG_INCLUDE_DIRS ${inc})
        endif()
    endforeach()
    set(FFMPEG_FOUND TRUE)
    add_definitions(-DWITH_FFMPEG)
else()
    message(WARNING "FFmpeg not found — VideoReader/VideoDecoder will be excluded from build")
endif()

# --- Filter sources: drop VideoReader/VideoDecoder when FFmpeg is unavailable ---
if(NOT FFMPEG_FOUND)
    list(FILTER SRC_FILES EXCLUDE REGEX ".*/VideoReader\\.cpp$")
    list(FILTER SRC_FILES EXCLUDE REGEX ".*/VideoDecoder\\.cpp$")
endif()

# --- Create target ---
add_executable(football_server ${SRC_FILES})

target_include_directories(football_server PRIVATE
    ${FFMPEG_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/module
    ${CMAKE_CURRENT_SOURCE_DIR}/module/common
    ${CMAKE_CURRENT_SOURCE_DIR}/module/video
    ${CMAKE_CURRENT_SOURCE_DIR}/module/detector
    ${CMAKE_CURRENT_SOURCE_DIR}/module/tracker
    ${CMAKE_CURRENT_SOURCE_DIR}/module/pose
    ${CMAKE_CURRENT_SOURCE_DIR}/module/smoother
    ${CMAKE_CURRENT_SOURCE_DIR}/module/output
    ${CMAKE_CURRENT_SOURCE_DIR}/module/inference
)

if(CUDA_FOUND)
    target_include_directories(football_server PRIVATE ${CUDA_INCLUDE_DIRS})
endif()
if(TRT_FOUND)
    target_include_directories(football_server PRIVATE ${TRT_INCLUDE_DIR})
endif()

target_link_libraries(football_server PRIVATE
    nexusflow
    ${FFMPEG_LIBRARIES}
)
if(CUDA_FOUND)
    target_link_libraries(football_server PRIVATE ${CUDA_LIBRARIES})
endif()
if(TRT_FOUND)
    target_link_libraries(football_server PRIVATE ${TRT_NVINFER_LIB})
    if(TRT_NVINFER_PLUGIN_LIB)
        target_link_libraries(football_server PRIVATE ${TRT_NVINFER_PLUGIN_LIB})
    endif()
endif()
