add_executable(aifootball_demo main.cpp)
target_include_directories(aifootball_demo PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/sdk/aifootball/include")
target_link_libraries(aifootball_demo PRIVATE aifootball_sdk)
