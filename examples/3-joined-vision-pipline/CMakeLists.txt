file(GLOB_RECURSE SRC_FILES "**.cpp")
add_executable(nexusflow_joined_pipeline ${SRC_FILES})
target_link_libraries(nexusflow_joined_pipeline PRIVATE nexusflow)
