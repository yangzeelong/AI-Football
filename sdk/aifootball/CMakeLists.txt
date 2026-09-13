# AI-Football business modules and public SDK facade.
file(GLOB_RECURSE AIFOOTBALL_CPP_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

set(AIFOOTBALL_CUDA_AVAILABLE FALSE)
set(CUDA_FOUND FALSE)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "AI-Football SDK: CUDA found: ${CUDA_VERSION}")
    include(CheckLanguage)
    if((NOT CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER MATCHES "NOTFOUND") AND
       CUDA_NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER "${CUDA_NVCC_EXECUTABLE}" CACHE FILEPATH
            "CUDA compiler" FORCE)
    endif()
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(AIFOOTBALL_CUDA_AVAILABLE TRUE)
    endif()
else()
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CUDA_FOUND TRUE)
        set(AIFOOTBALL_CUDA_AVAILABLE TRUE)
        set(CUDA_LIBRARIES CUDA::cudart)
        set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
endif()

set(TRT_FOUND FALSE)
if(CUDA_FOUND)
    find_path(TRT_INCLUDE_DIR NvInfer.h
        PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include
              ${CUDA_INCLUDE_DIRS})
    find_library(TRT_NVINFER_LIB nvinfer
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
              ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    find_library(TRT_NVINFER_PLUGIN_LIB nvinfer_plugin
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
              ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    if(TRT_INCLUDE_DIR AND TRT_NVINFER_LIB)
        set(TRT_FOUND TRUE)
        message(STATUS "AI-Football SDK: TensorRT found: ${TRT_NVINFER_LIB}")
    else()
        message(WARNING "AI-Football SDK: TensorRT not found; inference backend disabled")
    endif()
endif()

set(FFMPEG_FOUND FALSE)
set(FFMPEG_LIBRARIES "")
set(FFMPEG_INCLUDE_DIRS "")
set(FFMPEG_LIB_SEARCH_PATHS
    /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib /usr/lib64)
set(FFMPEG_INCLUDE_SEARCH_PATHS
    /usr/include/x86_64-linux-gnu /usr/include /usr/local/include)
find_library(AVFORMAT_LIB avformat PATHS ${FFMPEG_LIB_SEARCH_PATHS})
find_library(AVCODEC_LIB avcodec PATHS ${FFMPEG_LIB_SEARCH_PATHS})
find_library(AVUTIL_LIB avutil PATHS ${FFMPEG_LIB_SEARCH_PATHS})
find_library(SWSCALE_LIB swscale PATHS ${FFMPEG_LIB_SEARCH_PATHS})
find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h PATHS ${FFMPEG_INCLUDE_SEARCH_PATHS})
find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h PATHS ${FFMPEG_INCLUDE_SEARCH_PATHS})
find_path(AVUTIL_INCLUDE_DIR libavutil/imgutils.h PATHS ${FFMPEG_INCLUDE_SEARCH_PATHS})
find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h PATHS ${FFMPEG_INCLUDE_SEARCH_PATHS})
if(AVFORMAT_LIB AND AVCODEC_LIB AND AVUTIL_LIB AND SWSCALE_LIB)
    set(FFMPEG_FOUND TRUE)
    list(APPEND FFMPEG_LIBRARIES
        ${AVFORMAT_LIB} ${AVCODEC_LIB} ${AVUTIL_LIB} ${SWSCALE_LIB})
    foreach(inc ${AVFORMAT_INCLUDE_DIR} ${AVCODEC_INCLUDE_DIR}
              ${AVUTIL_INCLUDE_DIR} ${SWSCALE_INCLUDE_DIR})
        if(inc AND NOT "${inc}" IN_LIST FFMPEG_INCLUDE_DIRS)
            list(APPEND FFMPEG_INCLUDE_DIRS "${inc}")
        endif()
    endforeach()
else()
    message(WARNING "AI-Football SDK: FFmpeg not found; video I/O disabled")
endif()

if(NOT TRT_FOUND)
    list(FILTER AIFOOTBALL_CPP_FILES EXCLUDE REGEX ".*/inference/TensorRTEngine\\.cpp$")
endif()

if(NOT (TRT_FOUND AND AIFOOTBALL_CUDA_AVAILABLE))
    list(FILTER AIFOOTBALL_CPP_FILES EXCLUDE REGEX ".*/detector/RFDetrCudaPreprocess\\.cu$")
else()
    list(APPEND AIFOOTBALL_CPP_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/detector/RFDetrCudaPreprocess.cu")
endif()

if(NOT FFMPEG_FOUND)
    list(FILTER AIFOOTBALL_CPP_FILES EXCLUDE REGEX ".*/video/VideoReader\\.cpp$")
    list(FILTER AIFOOTBALL_CPP_FILES EXCLUDE REGEX ".*/video/VideoDecoder\\.cpp$")
endif()

add_library(aifootball_sdk STATIC ${AIFOOTBALL_CPP_FILES})
add_library(Nexusflow::aifootball_sdk ALIAS aifootball_sdk)

target_include_directories(aifootball_sdk
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ${FFMPEG_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TRT_INCLUDE_DIR})

target_compile_definitions(aifootball_sdk PRIVATE
    $<$<BOOL:${FFMPEG_FOUND}>:WITH_FFMPEG>
    $<$<BOOL:${TRT_FOUND}>:WITH_TENSORRT>
    $<$<BOOL:${AIFOOTBALL_CUDA_AVAILABLE}>:WITH_CUDA>
    $<$<BOOL:${TRT_FOUND}>:$<$<BOOL:${AIFOOTBALL_CUDA_AVAILABLE}>:WITH_CUDA_KERNELS>>)

target_link_libraries(aifootball_sdk
    PUBLIC
        nexusflow
        ${FFMPEG_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${TRT_NVINFER_LIB})
if(TRT_NVINFER_PLUGIN_LIB)
    target_link_libraries(aifootball_sdk PUBLIC ${TRT_NVINFER_PLUGIN_LIB})
endif()

if(AIFOOTBALL_CUDA_AVAILABLE)
    set_target_properties(aifootball_sdk PROPERTIES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set_property(TARGET aifootball_sdk PROPERTY CUDA_ARCHITECTURES native)
    endif()
endif()

install(TARGETS aifootball_sdk
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
