cmake_minimum_required(VERSION 3.0.0)
project(Tipousi VERSION 0.0.1)

# Set the default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build" FORCE)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/activation
    ${PROJECT_SOURCE_DIR}/include/loss
    ${PROJECT_SOURCE_DIR}/include/layer
    ${PROJECT_SOURCE_DIR}/include/graph
    ${PROJECT_SOURCE_DIR}/include/base
)

# Source files
file(GLOB SOURCES "src/*/*.cpp" tests/main.cpp)

include_directories("${CMAKE_SOURCE_DIR}/externals/eigen-3.4.0")

add_executable(Tipousi ${SOURCES})

# Ensure debug symbols are generated
set_target_properties(Tipousi PROPERTIES COMPILE_FLAGS "-g")
