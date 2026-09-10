file(GLOB_RECURSE SRC_FILES "**.cpp")
add_executable(nexusflow_how_to_use ${SRC_FILES})
target_link_libraries(nexusflow_how_to_use PRIVATE nexusflow)
