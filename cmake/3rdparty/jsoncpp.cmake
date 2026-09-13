include(FetchContent)

set(JSONCPP_WITH_TESTS OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_EXAMPLE OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_OBJECT_ID OFF CACHE BOOL "" FORCE)

FetchContent_Declare(jsoncpp
    URL https://github.com/open-source-parsers/jsoncpp/archive/refs/tags/1.9.5.tar.gz)
FetchContent_MakeAvailable(jsoncpp)

# JsonCpp 1.9.5 exposes the build target as jsoncpp_static when consumed via
# FetchContent. Keep the application-facing target name consistent with the
# namespaced targets used by the rest of the project.
if(TARGET jsoncpp_static AND NOT TARGET JsonCpp::JsonCpp)
    add_library(JsonCpp::JsonCpp ALIAS jsoncpp_static)
endif()
